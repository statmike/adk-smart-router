#!/usr/bin/env python
"""Build-order task 11 — measure G1 on the live A/B that `run_ab.py` produced.

G1 is the cost gate (DESIGN.md 10): **billed-equivalent input tokens per turn,
down 10% against the holdout.** Three things about that definition matter, and
all three are easy to get wrong.

**Billed-equivalent, not prompt tokens.** Gemini bills a cached input token at
about a tenth of an uncached one, so counting raw `usage_prompt_tokens` would
credit the router for cache hits it did not cause and punish it for cache breaks
it did. The measured quantity is `uncached + 0.10 x cached`, where
`uncached = usage_prompt_tokens - usage_cached_tokens`.

**Per turn, not per session.** Sessions have different lengths, and the router
changes nothing about how long a conversation runs. A per-session average would
mostly measure scenario mix.

**Against the holdout, not against history.** The control arm ran in the same
process, in the same hour, against the same model version, with the router
installed and declining to act. The corpus is not a valid baseline: it predates
the router and was generated under different conditions.

Which arm a session was in comes out of the router's own provenance
(`attributes.custom_metadata.smart_router[].decision_source`), not out of a hash
recomputed here. That makes this script a test of the provenance path as well as
a measurement: if the records are missing, it says so rather than reporting a
one-armed A/B.

Cost is four BigQuery reads, five with `--prunable`. Authentication is ADC only.

Usage:
    uv run python scripts/report_ab.py --prefix ab-20260901-
    uv run python scripts/report_ab.py --prefix ab-20260901- --agent support
    uv run python scripts/report_ab.py --prefix ab-20260901- --prunable book_callback,submit_review
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402

# Same defaults as everywhere else in the project (DESIGN.md 3.1).
DEFAULT_AGENT = "support"

# What a cached input token costs relative to an uncached one, on Gemini 2.5
# Flash. The gate is defined in these units, so this constant is part of the
# gate, not a display preference (DESIGN.md 10).
CACHED_TOKEN_WEIGHT = 0.10

# G1: billed-equivalent input tokens per turn, routed against holdout.
#
# Was -35%, which was a raw-prompt number wearing a billed-equivalent label. Once
# the cache guard measures the real prompt, the honest ceiling on the reference
# agent is around -13%: the router prunes on turn one, when the prompt is still
# under the cache floor, and then stops as conversation history carries it over.
# 10% of an enterprise input bill for a drop-in plugin is worth having, and a gate
# nothing can pass is not a gate (DESIGN.md 18.13).
G1_TARGET = -0.10

# Decision sources that mean the router acted on the request. Everything else is
# a recorded decision not to route, and a session containing one is neither arm.
#
# This list also held "grown" and "budget", from the pruner's deleted growth path.
# "budget" was a mislabelling as well as dead weight: it is a passthrough source, so a
# session that hit it would have been counted as routed while sending the full tool list.
# Neither value was ever written, so no historical run is scored differently by their
# removal (0 occurrences in 13,474 warehouse records).
ROUTED_SOURCES = ("fresh", "reused")


# --- What the warehouse gives back ----------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One arm's totals. Sums come from SQL; every rate is derived here."""

    name: str
    sessions: int
    turns: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    total_ms: int

    @property
    def billed_input(self) -> float:
        """Uncached tokens plus discounted cached ones. The unit G1 is defined in."""
        uncached = max(self.prompt_tokens - self.cached_tokens, 0)
        return uncached + CACHED_TOKEN_WEIGHT * self.cached_tokens

    @property
    def billed_per_turn(self) -> float:
        return self.billed_input / self.turns if self.turns else 0.0

    @property
    def prompt_per_turn(self) -> float:
        return self.prompt_tokens / self.turns if self.turns else 0.0

    @property
    def cached_share(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def completion_per_turn(self) -> float:
        return self.completion_tokens / self.turns if self.turns else 0.0

    @property
    def ms_per_turn(self) -> float:
        return self.total_ms / self.turns if self.turns else 0.0


@dataclass(frozen=True)
class Completeness:
    """How much of a planned run reached the warehouse. Counts from SQL, rates here.

    Three different things go missing and they do not cost the same:

      * **never started** — no event under that session id at all. Costs sample size
        and nothing else.
      * **failed** — the session produced events and then hit an error. It is in the
        arms table already, as a partial session, and partial sessions are dearer per
        turn than whole ones (see `completeness_note`).
      * **breaker** — the session never got an embedding, so it landed in `neither`.
        Whole and healthy, just not in an arm. The name is historical: it counts
        `timeout`, `error` and `breaker` together, because all three cost a session
        and only the router's own counters tell them apart.
    """

    planned: int
    started: int
    failed: int
    quota_failed: int
    breaker: int

    @property
    def never_started(self) -> int:
        return max(self.planned - self.started, 0)

    @property
    def landed(self) -> int:
        """Sessions that ran end to end. `failed` sessions started and did not finish."""
        return max(self.started - self.failed, 0)

    @property
    def lost(self) -> int:
        return max(self.planned - self.landed, 0)

    @property
    def landed_share(self) -> float:
        return self.landed / self.planned if self.planned > 0 else 0.0

    @property
    def quota_share(self) -> float:
        return self.quota_failed / self.planned if self.planned > 0 else 0.0


def infer_planned(top_index: int | None, started: int, override: int | None) -> int:
    """How many sessions the run meant to send.

    `--planned` wins, because it is the only source that can see a plan whose tail never
    started at all — those sessions leave no row anywhere. Otherwise the top session
    index carries it: run_ab.py:133 numbers sessions `{prefix}{index:05d}` contiguously
    from zero, so the highest index observed is the plan minus one.

    Never below `started`. An unparseable or renumbered session id would otherwise book a
    negative loss and report better-than-complete.
    """
    if override is not None:
        return max(override, started)
    if top_index is None:
        return started
    return max(top_index + 1, started)


@dataclass(frozen=True)
class PrunableTool:
    """One `prunable_by_default` name, and what each arm did with it.

    The routed arm cannot call a tool it was never sent, so a positive `routed_calls` on a
    name that is already in the config means the config and the deployment disagree. Before
    the name is set, both arms still declare the tool and both columns are just demand.
    """

    name: str
    holdout_calls: int
    routed_calls: int
    holdout_sessions: int
    routed_sessions: int


def relative_change(routed: Arm, holdout: Arm) -> float | None:
    """G1. Negative is cheaper. None when the baseline has nothing to divide by."""
    baseline = holdout.billed_per_turn
    if baseline <= 0:
        return None
    return (routed.billed_per_turn - baseline) / baseline


# --- Queries ---------------------------------------------------------------------

# One row per arm. The join is the whole query: sessions are labelled from the
# router's provenance, turns are counted from the LLM responses in those sessions.
#
# `tool_name IS NULL` keeps cache-serve records out. Both record shapes land in
# the same array with no discriminator field, and only the routing shape has a
# decision_source (provenance.py NOTE(contract)).
_ARMS_SQL = """
WITH decisions AS (
  SELECT
    e.session_id,
    JSON_VALUE(record, '$.decision_source') AS source
  FROM `{project}.{dataset}.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.agent = @agent
    AND STARTS_WITH(e.session_id, @prefix)
    AND JSON_VALUE(record, '$.tool_name') IS NULL
),
arms AS (
  SELECT
    session_id,
    CASE
      WHEN LOGICAL_OR(source = 'holdout') THEN 'holdout'
      WHEN LOGICAL_OR(source IN UNNEST(@routed_sources)) THEN 'routed'
      ELSE 'neither'
    END AS arm
  FROM decisions
  GROUP BY session_id
),
turns AS (
  SELECT
    session_id,
    COALESCE(usage_prompt_tokens, 0) AS prompt_tokens,
    COALESCE(usage_cached_tokens, 0) AS cached_tokens,
    COALESCE(usage_completion_tokens, 0) AS completion_tokens,
    COALESCE(total_ms, 0) AS total_ms
  FROM `{project}.{dataset}.v_llm_response`
  WHERE agent = @agent
    AND STARTS_WITH(session_id, @prefix)
    AND usage_prompt_tokens IS NOT NULL
)
SELECT
  arms.arm AS name,
  COUNT(DISTINCT arms.session_id) AS sessions,
  COUNT(turns.session_id) AS turns,
  SUM(turns.prompt_tokens) AS prompt_tokens,
  SUM(turns.cached_tokens) AS cached_tokens,
  SUM(turns.completion_tokens) AS completion_tokens,
  SUM(turns.total_ms) AS total_ms
FROM arms
JOIN turns USING (session_id)
GROUP BY name
ORDER BY name
"""

# Why each session landed where it did, and how large the kept sets were. Read as
# a sanity check on the arms table above, not as a metric.
_SOURCES_SQL = """
SELECT
  JSON_VALUE(record, '$.decision_source') AS source,
  COUNT(*) AS decisions,
  COUNT(DISTINCT e.session_id) AS sessions,
  AVG(ARRAY_LENGTH(JSON_QUERY_ARRAY(record, '$.kept_tools'))) AS avg_kept
FROM `{project}.{dataset}.agent_events` AS e,
  UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
WHERE e.agent = @agent
  AND STARTS_WITH(e.session_id, @prefix)
  AND JSON_VALUE(record, '$.tool_name') IS NULL
GROUP BY source
ORDER BY decisions DESC
"""


# When the router was routing, against how long the run lasted. A one-way `cache_watch`
# demotion silently converts the treatment arm into a second control arm, and every
# session after it lands in `neither`. DESIGN.md 18.21 is that run: routing stopped
# eleven minutes into fifty-five and the arms table gave no hint of it.
_TIMELINE_SQL = """
SELECT
  MIN(e.timestamp) AS run_start,
  MAX(e.timestamp) AS run_end,
  MAX(IF(JSON_VALUE(record, '$.decision_source') IN UNNEST(@routed_sources), e.timestamp, NULL))
    AS last_routed,
  COUNTIF(JSON_VALUE(record, '$.decision_source') = 'watch') AS watch_decisions
FROM `{project}.{dataset}.agent_events` AS e,
  UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
WHERE e.agent = @agent
  AND STARTS_WITH(e.session_id, @prefix)
  AND JSON_VALUE(record, '$.tool_name') IS NULL
"""


# How much of the run landed. The verdict above is a ratio and says nothing about the
# denominator behind it, so a run that lost a fifth of its sessions to 429s reads exactly
# like one that lost none (DESIGN.md 18.23 lost 40 of 600 and the report never said so).
#
# Nothing new is instrumented for this. `agent_events` already carries a per-session
# `status` and an `error_message` whose text contains `RESOURCE_EXHAUSTED` on a 429, and
# the zero-padded index at the end of the session id recovers the plan (`infer_planned`).
#
# Unfiltered by provenance on purpose: a session that died before the router ever ran has
# no `smart_router` metadata, and those are exactly the ones worth counting.
_COMPLETENESS_SQL = """
WITH per_session AS (
  SELECT
    e.session_id,
    MAX(SAFE_CAST(REGEXP_EXTRACT(e.session_id, r'(\\d+)$') AS INT64)) AS session_index,
    LOGICAL_OR(e.status = 'ERROR') AS failed,
    LOGICAL_OR(
      e.status = 'ERROR'
      AND REGEXP_CONTAINS(COALESCE(e.error_message, ''), 'RESOURCE_EXHAUSTED')
    ) AS quota_failed
  FROM `{project}.{dataset}.agent_events` AS e
  WHERE e.agent = @agent
    AND STARTS_WITH(e.session_id, @prefix)
  GROUP BY e.session_id
),
breaker AS (
  SELECT DISTINCT e.session_id
  FROM `{project}.{dataset}.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.agent = @agent
    AND STARTS_WITH(e.session_id, @prefix)
    AND JSON_VALUE(record, '$.tool_name') IS NULL
    -- One loss, three causes, and they had one name until 2026-09-05. A hot-path
    -- timeout, a call that raised, and a skip because the breaker was already open
    -- are separate rows now, and all three still mean the same thing here: the
    -- session never got a vector, so it is in neither arm.
    AND JSON_VALUE(record, '$.decision_source') IN ('error', 'timeout', 'breaker')
)
SELECT
  MAX(session_index) AS top_index,
  COUNT(*) AS started,
  COUNTIF(failed) AS failed,
  COUNTIF(quota_failed) AS quota_failed,
  (SELECT COUNT(*) FROM breaker) AS breaker
FROM per_session
"""


# Demand for the tools an operator has declared prunable, split by arm. This is the detector
# for `config.prunable_by_default`, and it is the only one: nothing else in the report can see
# a tool that was removed on purpose, because the arms table measures tokens and a censored
# tool costs nothing. Only `--prunable` runs it.
#
# OK calls only. A failed call is a tool the model reached for and did not get an answer from,
# which is a different fault, and mixing the two would let an unrelated outage read as demand.
_PRUNABLE_SQL = """
WITH decisions AS (
  SELECT
    e.session_id,
    JSON_VALUE(record, '$.decision_source') AS source
  FROM `{project}.{dataset}.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.agent = @agent
    AND STARTS_WITH(e.session_id, @prefix)
    AND JSON_VALUE(record, '$.tool_name') IS NULL
),
arms AS (
  SELECT
    session_id,
    CASE
      WHEN LOGICAL_OR(source = 'holdout') THEN 'holdout'
      WHEN LOGICAL_OR(source IN UNNEST(@routed_sources)) THEN 'routed'
      ELSE 'neither'
    END AS arm
  FROM decisions
  GROUP BY session_id
),
calls AS (
  SELECT session_id, tool_name
  FROM `{project}.{dataset}.v_tool_completed`
  WHERE agent = @agent
    AND STARTS_WITH(session_id, @prefix)
    AND status = 'OK'
    AND tool_name IN UNNEST(@tools)
)
SELECT
  calls.tool_name AS name,
  COUNTIF(arms.arm = 'holdout') AS holdout_calls,
  COUNTIF(arms.arm = 'routed') AS routed_calls,
  COUNT(DISTINCT IF(arms.arm = 'holdout', calls.session_id, NULL)) AS holdout_sessions,
  COUNT(DISTINCT IF(arms.arm = 'routed', calls.session_id, NULL)) AS routed_sessions
FROM arms
JOIN calls USING (session_id)
GROUP BY name
ORDER BY holdout_calls DESC, name
"""


# The control period of a sequential A/B, which ran with the router uninstalled and
# so left no provenance to join against. Every turn under the prefix is the arm.
# Read `--baseline-prefix` in the argument list for why that run exists at all.
_BASELINE_SQL = """
SELECT
  COUNT(DISTINCT session_id) AS sessions,
  COUNT(*) AS turns,
  SUM(COALESCE(usage_prompt_tokens, 0)) AS prompt_tokens,
  SUM(COALESCE(usage_cached_tokens, 0)) AS cached_tokens,
  SUM(COALESCE(usage_completion_tokens, 0)) AS completion_tokens,
  SUM(COALESCE(total_ms, 0)) AS total_ms
FROM `{project}.{dataset}.v_llm_response`
WHERE agent = @agent
  AND STARTS_WITH(session_id, @prefix)
  AND usage_prompt_tokens IS NOT NULL
"""


def _run(
    sql: str,
    project: str,
    dataset: str,
    agent: str,
    prefix: str,
    tools: Sequence[str] = (),
) -> list[dict[str, object]]:
    params: list[bigquery.query._AbstractQueryParameter] = [
        bigquery.ScalarQueryParameter("agent", "STRING", agent),
        bigquery.ScalarQueryParameter("prefix", "STRING", prefix),
    ]
    # Declared only when the statement uses it. BigQuery rejects a parameter list
    # that does not match the query.
    if "@routed_sources" in sql:
        params.append(bigquery.ArrayQueryParameter("routed_sources", "STRING", ROUTED_SOURCES))
    if "@tools" in sql:
        params.append(bigquery.ArrayQueryParameter("tools", "STRING", list(tools)))
    job = bigquery.Client(project=project).query(
        sql.format(project=project, dataset=dataset),
        job_config=bigquery.QueryJobConfig(query_parameters=params),
    )
    return [dict(row) for row in job.result()]


def fetch_arms(project: str, dataset: str, agent: str, prefix: str) -> list[Arm]:
    return [
        Arm(
            name=str(row["name"]),
            sessions=int(row["sessions"] or 0),
            turns=int(row["turns"] or 0),
            prompt_tokens=int(row["prompt_tokens"] or 0),
            cached_tokens=int(row["cached_tokens"] or 0),
            completion_tokens=int(row["completion_tokens"] or 0),
            total_ms=int(row["total_ms"] or 0),
        )
        for row in _run(_ARMS_SQL, project, dataset, agent, prefix)
    ]


def fetch_baseline(project: str, dataset: str, agent: str, prefix: str) -> Arm | None:
    """The control period as one arm. `None` when nothing ran under that prefix."""
    rows = _run(_BASELINE_SQL, project, dataset, agent, prefix)
    if not rows or not int(rows[0]["turns"] or 0):
        return None
    row = rows[0]
    return Arm(
        name="holdout",
        sessions=int(row["sessions"] or 0),
        turns=int(row["turns"] or 0),
        prompt_tokens=int(row["prompt_tokens"] or 0),
        cached_tokens=int(row["cached_tokens"] or 0),
        completion_tokens=int(row["completion_tokens"] or 0),
        total_ms=int(row["total_ms"] or 0),
    )


def fetch_sources(project: str, dataset: str, agent: str, prefix: str) -> list[dict[str, object]]:
    return _run(_SOURCES_SQL, project, dataset, agent, prefix)


def fetch_prunable(
    project: str, dataset: str, agent: str, prefix: str, tools: Sequence[str]
) -> list[PrunableTool]:
    """Per-arm demand for the declared-prunable tools. Empty list when none was named."""
    if not tools:
        return []
    return [
        PrunableTool(
            name=str(row["name"]),
            holdout_calls=int(row["holdout_calls"] or 0),
            routed_calls=int(row["routed_calls"] or 0),
            holdout_sessions=int(row["holdout_sessions"] or 0),
            routed_sessions=int(row["routed_sessions"] or 0),
        )
        for row in _run(_PRUNABLE_SQL, project, dataset, agent, prefix, tools)
    ]


def _sessions(count: int) -> str:
    """`1 session`, not `1 sessions`. The counts here are usually one or two."""
    return f"{count} session" if count == 1 else f"{count} sessions"


def prunable_note(called: Sequence[PrunableTool], declared: Sequence[str]) -> str | None:
    """The verdict on the declared list, or None when no name was declared.

    The failure this looks for is a name with holdout demand and no routed demand: the
    control arm's users wanted the tool and the treatment arm's users could not have it.

    It is a weak test and saying so is part of the output. The whole live signal on the
    reference agent was 3 holdout calls against 8 routed ones across 574 sessions, which no
    single run can separate from noise. Read it accumulated over runs, and read a silent
    report as "no evidence yet" rather than as "no harm done".
    """
    if not declared:
        return None
    wanted = [t for t in called if t.holdout_calls and not t.routed_calls]
    if not wanted:
        return (
            f"none of the {len(declared)} declared-prunable tools was called by the holdout "
            "arm alone. That is the absence of evidence, not evidence of absence: one run is "
            "far too small to clear a name."
        )
    names = ", ".join(f"{t.name} ({_sessions(t.holdout_sessions)})" for t in wanted)
    return (
        f"{len(wanted)} declared-prunable tools were called by the holdout arm and never by "
        f"the routed arm: {names}. Those are the sessions the prune cost, and the routed arm "
        "could not have recovered them within the turn. Take the names out of "
        "prunable_by_default, or accept the misses knowingly."
    )


def fetch_timeline(project: str, dataset: str, agent: str, prefix: str) -> dict[str, object]:
    rows = _run(_TIMELINE_SQL, project, dataset, agent, prefix)
    return rows[0] if rows else {}


def fetch_completeness(
    project: str, dataset: str, agent: str, prefix: str, planned: int | None = None
) -> Completeness:
    rows = _run(_COMPLETENESS_SQL, project, dataset, agent, prefix)
    row = rows[0] if rows else {}
    top = row.get("top_index")
    started = int(row.get("started") or 0)
    return Completeness(
        planned=infer_planned(None if top is None else int(top), started, planned),
        started=started,
        failed=int(row.get("failed") or 0),
        quota_failed=int(row.get("quota_failed") or 0),
        breaker=int(row.get("breaker") or 0),
    )


def completeness_line(c: Completeness) -> str:
    """One sentence of counts, no judgement. Empty for a run with no sessions at all."""
    if c.planned <= 0:
        return "no sessions under this prefix."
    return (
        f"{c.landed} of {c.planned} planned sessions landed ({c.landed_share:.0%}); "
        f"{c.never_started} never started, {c.failed} errored "
        f"({c.quota_failed} of those on quota)."
    )


def completeness_summary(c: Completeness) -> str | None:
    """The short form that rides along with the G1 verdict, so the two never separate."""
    if c.planned <= 0:
        return None
    summary = f"on {c.landed_share:.0%} of {c.planned} planned sessions"
    if c.breaker:
        # Only a routed session is ever embedded, so the breaker cannot take a holdout
        # session. Whatever it takes comes off the treatment arm's sample, not the
        # control's, which is why this is worth saying next to the number.
        summary += (
            f"; {c.breaker} more tripped the embedding breaker and fell into "
            "'neither', all off the routed side"
        )
    return summary


def completeness_note(c: Completeness) -> str | None:
    """A warning when enough of the run went missing to bend G1, else None.

    Five percent is the threshold, and the reason it is that low is that the loss is not
    neutral. A session that died partway still contributed its expensive uncached first
    turn and none of its cheap cached later ones, so an arm that lost sessions reads
    dearer per turn than it really is. Losses are rarely even between the arms either.
    """
    if c.planned <= 0 or c.landed_share >= 0.95:
        return None
    cause = (
        f" {c.quota_share:.0%} of the plan went to quota (429 RESOURCE_EXHAUSTED)."
        if c.quota_failed
        else " None of it was quota, so look elsewhere for the cause."
    )
    return (
        f"only {c.landed_share:.0%} of this run landed: {c.lost} of {c.planned} planned "
        f"sessions are missing or partial.{cause} A session that died partway still "
        "contributed its expensive uncached first turn and none of its cheap cached "
        "later ones, so whichever arm lost it reads dearer per turn than it is."
    )


def routing_stopped_early(timeline: dict[str, object]) -> str | None:
    """A warning if the router stopped routing well before the run ended, else None.

    Ten percent of the run is the threshold, which is generous: a run that ends while
    still routing leaves a few unrouted sessions at the tail, and one that demoted is
    usually missing most of its length. Anything in between deserves the warning anyway.
    """
    start, end = timeline.get("run_start"), timeline.get("run_end")
    last = timeline.get("last_routed")
    if start is None or end is None or last is None:
        return None
    run_s = (end - start).total_seconds()  # type: ignore[operator]
    routed_s = (last - start).total_seconds()  # type: ignore[operator]
    if run_s <= 0 or routed_s >= 0.9 * run_s:
        return None
    watch = int(timeline.get("watch_decisions") or 0)
    cause = (
        f" {watch} decisions were recorded as 'watch', so cache_watch demoted the agent."
        if watch
        else " No 'watch' decisions were recorded, so this is not a cache_watch demotion."
    )
    return (
        f"the router stopped routing {routed_s / 60:.0f} minutes into a "
        f"{run_s / 60:.0f}-minute run.{cause} Every session after that point is an "
        "unrouted control, so the routed arm below describes only the first "
        f"{routed_s / run_s:.0%} of the traffic."
    )


# --- The report -------------------------------------------------------------------


def report(
    arms: Sequence[Arm],
    sources: Sequence[dict[str, object]],
    timeline: dict[str, object] | None = None,
    baseline_prefix: str | None = None,
    completeness: Completeness | None = None,
    baseline_completeness: Completeness | None = None,
    prunable: Sequence[str] = (),
    prunable_calls: Sequence[PrunableTool] = (),
) -> bool:
    """Print both arms and the gate verdict. Returns whether G1 passed.

    Completeness is stated, never gated on. A thin run still gets its verdict, with the
    denominator attached to it; only the existing thin-arm cases return INCONCLUSIVE.

    The declared-prunable block is stated and never gated on either, for the opposite
    reason: it is far too underpowered to fail a run on, and a gate nobody trusts is a gate
    people learn to pass with `--prunable` left off.
    """
    by_name = {arm.name: arm for arm in arms}

    if baseline_prefix:
        print(
            f"\nSEQUENTIAL: the holdout row below is a separate control run, prefix "
            f"{baseline_prefix}. It shares this run's seed, so both periods sent the same "
            "conversations, and it ran with the router uninstalled."
        )

    stopped = routing_stopped_early(timeline or {})
    if stopped:
        print(f"\nWARNING: {stopped}")

    if completeness is not None:
        incomplete = completeness_note(completeness)
        if incomplete:
            print(f"\nWARNING: {incomplete}")
        if baseline_completeness is not None:
            missing = completeness_note(baseline_completeness)
            if missing:
                print(f"\nWARNING: control run {baseline_prefix}: {missing}")

        print("\ncompleteness")
        print(f"  this run  {completeness_line(completeness)}")
        if baseline_completeness is not None:
            print(f"  control   {completeness_line(baseline_completeness)}")
        if completeness.breaker:
            print(
                f"  {completeness.breaker} sessions tripped the embedding breaker and are in "
                "'neither' below. Only a routed session is ever embedded, so that loss comes "
                "off the routed arm's sample and never off the holdout's."
            )

    print("\ndecision sources")
    print(f"  {'source':<12} {'decisions':>10} {'sessions':>9} {'avg kept':>9}")
    for row in sources:
        avg = row.get("avg_kept")
        shown = f"{float(avg):.1f}" if avg is not None else "-"
        print(
            f"  {str(row['source']):<12} {int(row['decisions'] or 0):>10} "
            f"{int(row['sessions'] or 0):>9} {shown:>9}"
        )

    print("\narms")
    header = f"  {'arm':<9} {'sessions':>8} {'turns':>6} {'prompt/t':>9} {'cached':>7} "
    print(header + f"{'billed/t':>9} {'out/t':>7} {'ms/t':>7}")
    for name in ("holdout", "routed", "neither"):
        arm = by_name.get(name)
        if arm is None:
            continue
        print(
            f"  {arm.name:<9} {arm.sessions:>8} {arm.turns:>6} {arm.prompt_per_turn:>9.0f} "
            f"{arm.cached_share:>6.1%} {arm.billed_per_turn:>9.0f} "
            f"{arm.completion_per_turn:>7.0f} {arm.ms_per_turn:>7.0f}"
        )

    if prunable:
        print(f"\ndeclared prunable, and called anyway ({len(prunable)} tools declared)")
        if prunable_calls:
            print(f"  {'tool':<32} {'holdout':>9} {'routed':>9}")
            for tool in prunable_calls:
                print(f"  {tool.name:<32} {tool.holdout_calls:>9} {tool.routed_calls:>9}")
        else:
            print("  no arm called any of them.")
        note = prunable_note(prunable_calls, prunable)
        if note:
            print(f"\n  {note}")

    stray = by_name.get("neither")
    if stray is not None:
        print(
            f"\n  {stray.sessions} sessions are in neither arm. The router recorded a decision "
            "not to route them (low confidence, too few tools, no artifact, or an embedding "
            "failure). They are excluded from G1 and the source table above says which."
        )

    routed, holdout = by_name.get("routed"), by_name.get("holdout")
    if routed is None or holdout is None:
        missing = "routed" if routed is None else "holdout"
        print(f"\nINCONCLUSIVE: no {missing} arm. Check --prefix, and that BQAA flushed.")
        return False
    if routed.turns < 30 or holdout.turns < 30:
        print(
            f"\nINCONCLUSIVE: {routed.turns} routed and {holdout.turns} holdout turns. "
            "Too few to read a 10% difference. Run more sessions."
        )
        return False

    change = relative_change(routed, holdout)
    if change is None:
        print("\nINCONCLUSIVE: the holdout arm billed nothing, so there is no baseline.")
        return False

    print(
        f"\nG1  billed-equivalent input tokens per turn: "
        f"{holdout.billed_per_turn:.0f} -> {routed.billed_per_turn:.0f}, {change:+.1%}"
    )
    # Attached to the verdict rather than left in the block above, because the verdict is
    # the line that gets quoted into a doc or a chat message on its own.
    if completeness is not None:
        summary = completeness_summary(completeness)
        if summary:
            print(f"    {summary}")
    passed = change <= G1_TARGET
    print(f"    target {G1_TARGET:+.0%} or better: {'PASS' if passed else 'FAIL'}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", required=True, help="session id prefix, e.g. ab-20260901-")
    parser.add_argument(
        "--baseline-prefix",
        default=None,
        help=(
            "read the control arm from a separate run instead of from this run's holdout. "
            "For a sequential A/B, where the two periods do not share a process and so do "
            "not share prefix warmth"
        ),
    )
    parser.add_argument(
        "--planned",
        type=int,
        default=None,
        help=(
            "how many sessions the run meant to send. Only needed when the tail of a run "
            "never started at all, which leaves nothing in BigQuery to count; otherwise the "
            "plan is read off the highest session index. run_ab.py prints this flag for you"
        ),
    )
    parser.add_argument(
        "--prunable",
        default="",
        help=(
            "comma-separated tool names, the ones set in config.prunable_by_default. Adds one "
            "read that counts each name's calls per arm, which is the only way to see the harm "
            "a wrong name does: a censored tool costs no tokens, so it is invisible to G1. "
            "startup_checks logs the candidate list at boot"
        ),
    )
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    return parser.parse_args()


def parse_tools(raw: str) -> list[str]:
    """Split and clean a comma-separated tool list. Empty entries are dropped, not passed on."""
    return [name.strip() for name in raw.split(",") if name.strip()]


def main() -> int:
    args = parse_args()
    print(f"reading {args.project}.{args.dataset}, agent {args.agent}, prefix {args.prefix}")
    arms = fetch_arms(args.project, args.dataset, args.agent, args.prefix)
    sources = fetch_sources(args.project, args.dataset, args.agent, args.prefix)
    timeline = fetch_timeline(args.project, args.dataset, args.agent, args.prefix)
    completeness = fetch_completeness(
        args.project, args.dataset, args.agent, args.prefix, args.planned
    )
    if not arms:
        print("no rows. Either nothing ran under that prefix, or provenance never landed.")
        return 1

    baseline_completeness = None
    if args.baseline_prefix:
        baseline = fetch_baseline(args.project, args.dataset, args.agent, args.baseline_prefix)
        if baseline is None:
            print(f"no rows under baseline prefix {args.baseline_prefix}.")
            return 1
        arms = [arm for arm in arms if arm.name != "holdout"] + [baseline]
        # `--planned` describes this run's plan. The two periods of a sequential A/B send
        # the same conversations (run_ab.py, same seed), so it describes the control too.
        baseline_completeness = fetch_completeness(
            args.project, args.dataset, args.agent, args.baseline_prefix, args.planned
        )

    prunable = parse_tools(args.prunable)
    passed = report(
        arms,
        sources,
        timeline,
        args.baseline_prefix,
        completeness,
        baseline_completeness,
        prunable,
        fetch_prunable(args.project, args.dataset, args.agent, args.prefix, prunable),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
