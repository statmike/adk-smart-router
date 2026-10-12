#!/usr/bin/env python
"""The cache guard, asked of the whole fleet instead of one process.

`cache_watch.CacheWatch` decides whether pruning is costing an agent money, and it counts in
memory. That makes it fast and it makes it blind three ways: a thinly routed agent never
reaches a verdict in any single replica, a demotion earned in one replica does not reach the
others, and the flap bound resets whenever a process restarts. E4's measured case is
`ab-20260902`, which lost 13.8% with fifteen warm routed calls against a floor of fifty and
was never judged (DESIGN.md 19.75).

The fix is not to persist the counters. Every input the guard consumes is already in the
warehouse, written from every replica by the analytics plugin: the prompt size and cached
share from `v_llm_response`, the arm and the deciding replica from the provenance record on
`agent_events`. So this asks the same question of that data, and nothing new is written on
the serving path except one string per decision.

Two things are kept identical to the live guard on purpose. The rule is
`cache_watch.costs_more`, imported rather than restated, for the reason DESIGN.md 19.74
gives: a gate that scores a different cut from the one that serves is a second wrong answer.
And a session's first model call is excluded from both arms, because turn position dominates
a call's price and the two arms do not hold the same fraction of turn one (DESIGN.md 18.22).

What is new is the stratification. Replicas hold different mixes of routed and holdout
sessions, so pooling across them is the same paradox with a new grouping variable. The
verdict is taken on a directly standardised comparison, each replica weighed at its own
share of the traffic, and the pooled number is printed beside it so the gap is visible.

Cost is one BigQuery read. Authentication is ADC only.

Usage:
    uv run python scripts/fleet_cache_watch.py
    uv run python scripts/fleet_cache_watch.py --agent support --days 30
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from deployment import DEFAULT_DATASET, project_arg, require_project  # noqa: E402
from google.cloud import bigquery

from adk_smart_router.cache_watch import costs_more
from adk_smart_router.config import RouterConfig

DEFAULT_AGENT = "support"

# How far back to look. Long enough for a thin agent to accumulate a verdict, short enough
# that the model behind the prices has probably not changed underneath it. See `versions` in
# the report for whether the window straddles a republish.
DEFAULT_WINDOW_DAYS = 30

# The same labelling report_ab.py, replay_promotion.py and replay_cache_watch.py use: a
# session is pruned only if the router acted on it. Everything else it saw is unpruned.
ROUTED_SOURCES = ("fresh", "reused")

# The weight G1 is defined in (DESIGN.md 10), and the one CacheWatch bills at.
CACHED_TOKEN_WEIGHT = 0.10

# A replica that predates the provenance field, or a record that lost it. Grouped together
# rather than dropped: an old warehouse should still get a verdict, it just gets an
# unstratified one, which is what every warehouse got before this existed.
UNKNOWN_REPLICA = "unknown"

STRATA_SQL = """
WITH decisions AS (
  SELECT
    e.session_id,
    JSON_VALUE(record, '$.decision_source') AS source,
    JSON_VALUE(record, '$.replica_id') AS replica_id,
    JSON_VALUE(record, '$.model_version') AS model_version
  FROM `{project}.{dataset}.agent_events` AS e,
    UNNEST(JSON_QUERY_ARRAY(e.attributes, '$.custom_metadata.smart_router')) AS record
  WHERE e.agent = @agent
    AND e.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    AND JSON_VALUE(record, '$.tool_name') IS NULL
),
sessions AS (
  SELECT
    session_id,
    CASE
      WHEN LOGICAL_OR(source = 'holdout') THEN 'holdout'
      WHEN LOGICAL_OR(source IN UNNEST(@routed_sources)) THEN 'routed'
      ELSE 'neither'
    END AS arm,
    -- One routing decision is made per session, so both of these are constant within it in
    -- the ordinary case. A session is filed under the replica that decided it, which is the
    -- thing whose traffic mix the stratification is trying to hold still.
    ANY_VALUE(IFNULL(replica_id, '{unknown}')) AS replica_id,
    ANY_VALUE(model_version) AS model_version
  FROM decisions
  GROUP BY session_id
),
later_calls AS (
  SELECT session_id, prompt_tokens, cached_tokens
  FROM (
    SELECT
      session_id,
      COALESCE(usage_prompt_tokens, 0) AS prompt_tokens,
      LEAST(COALESCE(usage_cached_tokens, 0), COALESCE(usage_prompt_tokens, 0))
        AS cached_tokens,
      ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY timestamp) AS position
    FROM `{project}.{dataset}.v_llm_response`
    WHERE agent = @agent
      AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
      AND usage_prompt_tokens IS NOT NULL
  )
  -- Turn one is excluded from both arms, exactly as the live guard excludes it.
  WHERE position > 1
)
SELECT
  s.replica_id,
  s.arm,
  COUNT(*) AS calls,
  SUM((c.prompt_tokens - c.cached_tokens) + @weight * c.cached_tokens) AS billed,
  -- Routed sessions only ever, because a holdout decision is taken before any artifact is
  -- resolved and so carries no version at all. See `Verdict.versions`.
  COUNT(DISTINCT s.model_version) AS versions
FROM later_calls c
JOIN sessions s USING (session_id)
WHERE s.arm IN ('routed', 'holdout')
GROUP BY replica_id, arm
"""


@dataclass(frozen=True)
class Arm:
    """One side of the comparison, within one replica."""

    calls: int = 0
    billed: float = 0.0

    @property
    def per_call(self) -> float:
        return self.billed / self.calls if self.calls else 0.0


@dataclass(frozen=True)
class Stratum:
    """Both arms as one replica saw them."""

    replica_id: str
    pruned: Arm
    whole: Arm
    versions: int  # distinct artifact versions the routed arm ran on

    def judgeable(self, min_samples: int) -> bool:
        """Whether this replica has enough of both arms to contribute a comparison."""
        return self.pruned.calls >= min_samples and self.whole.calls >= min_samples


@dataclass(frozen=True)
class Verdict:
    """What the fleet says, and everything a reader needs to disbelieve it."""

    agent: str
    days: int
    min_samples: int
    margin: float
    strata: tuple[Stratum, ...]
    judged: tuple[Stratum, ...]
    pooled_pruned: float  # billed per call
    pooled_whole: float
    standardised_pruned: float | None  # None when no replica had enough of both arms
    standardised_whole: float | None
    harmful: bool

    @property
    def measured(self) -> bool:
        return self.standardised_pruned is not None

    @property
    def versions(self) -> int:
        """Distinct artifact versions behind the routed arm, across the whole window.

        More than one means the window straddles a republish and the comparison is about a
        mixture. It is reported rather than filtered on, because a holdout decision records
        no version, so keying both arms to one artifact would delete the control arm.
        """
        return max((s.versions for s in self.strata), default=0)


def combine(strata: tuple[Stratum, ...]) -> tuple[float, float]:
    """The pooled per-call cost of each arm: every call weighed the same, replicas ignored."""
    pruned = Arm(sum(s.pruned.calls for s in strata), sum(s.pruned.billed for s in strata))
    whole = Arm(sum(s.whole.calls for s in strata), sum(s.whole.billed for s in strata))
    return pruned.per_call, whole.per_call


def standardise(judged: tuple[Stratum, ...]) -> tuple[float, float]:
    """Each arm's per-call cost if every replica carried the same share of both arms.

    Direct standardisation: compare the arms inside a replica, where the mix is whatever it
    is, and weigh each replica by how much of the fleet's traffic it handled. A replica that
    is entirely one arm contributes nothing and is not in `judged` to begin with.

    With one replica this is the pooled number, which is the right way for it to degrade:
    there is no mix to correct for.
    """
    total = sum(s.pruned.calls + s.whole.calls for s in judged)
    weights = [(s.pruned.calls + s.whole.calls) / total for s in judged]
    pruned = sum(w * s.pruned.per_call for w, s in zip(weights, judged, strict=True))
    whole = sum(w * s.whole.per_call for w, s in zip(weights, judged, strict=True))
    return pruned, whole


def judge(
    agent: str, days: int, strata: tuple[Stratum, ...], min_samples: int, margin: float
) -> Verdict:
    """The comparison, on the standardised numbers, with the pooled ones kept for the report."""
    judged = tuple(s for s in strata if s.judgeable(min_samples))
    pooled_pruned, pooled_whole = combine(strata)
    if not judged:
        return Verdict(
            agent,
            days,
            min_samples,
            margin,
            strata,
            judged,
            pooled_pruned,
            pooled_whole,
            None,
            None,
            harmful=False,
        )

    pruned, whole = standardise(judged)
    # Rates against a notional one call each. The standardisation has already removed the
    # call counts, and `costs_more` cross-multiplies, so this asks exactly
    # `pruned >= (1 + margin) * whole` through the same function the live guard uses.
    harmful = costs_more(pruned, 1, whole, 1, margin)
    return Verdict(
        agent,
        days,
        min_samples,
        margin,
        strata,
        judged,
        pooled_pruned,
        pooled_whole,
        pruned,
        whole,
        harmful,
    )


def fetch(
    client: bigquery.Client, *, project: str, dataset: str, agent: str, days: int
) -> tuple[Stratum, ...]:
    """One row per (replica, arm), turned into one `Stratum` per replica."""
    sql = STRATA_SQL.format(project=project, dataset=dataset, unknown=UNKNOWN_REPLICA)
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("days", "INT64", days),
                bigquery.ScalarQueryParameter("weight", "FLOAT64", CACHED_TOKEN_WEIGHT),
                bigquery.ArrayQueryParameter("routed_sources", "STRING", list(ROUTED_SOURCES)),
            ]
        ),
    )
    arms: dict[str, dict[str, Arm]] = {}
    versions: dict[str, int] = {}
    for row in job.result():
        replica = row["replica_id"]
        arms.setdefault(replica, {})[row["arm"]] = Arm(row["calls"], float(row["billed"]))
        if row["arm"] == "routed":
            versions[replica] = row["versions"]
    return tuple(
        Stratum(
            replica_id=replica,
            pruned=sides.get("routed", Arm()),
            whole=sides.get("holdout", Arm()),
            versions=versions.get(replica, 0),
        )
        for replica, sides in sorted(arms.items())
    )


def measure(
    client: bigquery.Client,
    *,
    project: str | None = None,
    dataset: str = DEFAULT_DATASET,
    agent: str = DEFAULT_AGENT,
    days: int = DEFAULT_WINDOW_DAYS,
    min_samples: int | None = None,
    margin: float | None = None,
) -> Verdict:
    """One BigQuery read, then the verdict. What the publish gate calls.

    Keyword arguments rather than an `argparse.Namespace`, because the other caller is
    `publish_router.py`, and it should not have to grow a `--days` or a `--margin` of its own
    to borrow this. The two thresholds default to whatever the live guard is configured to
    serve.
    """
    cfg = RouterConfig()
    strata = fetch(
        client, project=require_project(project), dataset=dataset, agent=agent, days=days
    )
    return judge(
        agent,
        days,
        strata,
        cfg.cache_watch_min_samples if min_samples is None else min_samples,
        cfg.cache_watch_margin if margin is None else margin,
    )


def report(verdict: Verdict) -> None:
    """The table, then the sentence. Both numbers always, because the gap is the point."""
    print(f"\n{verdict.agent}: {verdict.days} days, later calls only\n")
    print("| replica | pruned calls | pruned/call | unpruned calls | unpruned/call | judged |")
    print("| --- | ---: | ---: | ---: | ---: | :---: |")
    for s in verdict.strata:
        mark = "yes" if s.judgeable(verdict.min_samples) else "no"
        print(
            f"| {s.replica_id} | {s.pruned.calls} | {s.pruned.per_call:.1f} | "
            f"{s.whole.calls} | {s.whole.per_call:.1f} | {mark} |"
        )

    print(f"\npooled          {verdict.pooled_pruned:.1f} against {verdict.pooled_whole:.1f}")
    if not verdict.measured:
        print(
            f"standardised    not measured: no replica has {verdict.min_samples} calls in "
            "both arms, so there is nothing to compare"
        )
        return

    assert verdict.standardised_pruned is not None  # narrowed by `measured`
    assert verdict.standardised_whole is not None
    ratio = verdict.standardised_pruned / verdict.standardised_whole - 1.0
    print(
        f"standardised    {verdict.standardised_pruned:.1f} against "
        f"{verdict.standardised_whole:.1f}, {ratio:+.1%}, over "
        f"{len(verdict.judged)} of {len(verdict.strata)} replicas"
    )
    if verdict.versions > 1:
        print(
            f"note            the routed arm ran on {verdict.versions} artifact versions in "
            "this window, so this compares a mixture"
        )
    print(
        f"verdict         {'pruning is costing this agent money' if verdict.harmful else 'ok'}"
        f", at a margin of {verdict.margin:.0%}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cfg = RouterConfig()
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--days", type=int, default=DEFAULT_WINDOW_DAYS)
    # Both default to what the live guard serves, so the two cannot drift into disagreeing.
    parser.add_argument("--min-samples", type=int, default=cfg.cache_watch_min_samples)
    parser.add_argument("--margin", type=float, default=cfg.cache_watch_margin)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verdict = measure(
        bigquery.Client(project=args.project),
        project=args.project,
        dataset=args.dataset,
        agent=args.agent,
        days=args.days,
        min_samples=args.min_samples,
        margin=args.margin,
    )
    report(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
