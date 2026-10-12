"""Can this warehouse train a router today, on the traffic it has already logged?

The router trains on history, not on live traffic. `train_router.sql` reads a window of the
analytics log and nothing in it requires the router to have been installed while that window
was being written, so an agent that has been running the BigQuery Agent Analytics plugin for a
month already owns the corpus. This module is how someone finds that out before provisioning
anything, cloning anything, or waiting.

It answers four questions, in the order that kills the idea fastest:

1. **Is the prompt text there?** Labels come from `JSON_VALUE(content, '$.text_summary')` on the
   base `agent_events` table, and whether that field is populated depends on the analytics
   plugin's own settings in a deployment this project has never seen. Empty means no router,
   whatever else is true, so it is checked first and reported as a share rather than a boolean.
2. **Are there enough labelled sessions, spread over enough tools?** A class the training run
   sees fewer than `min_class_rows` times cannot be learned, so the count that matters is not
   how many tools exist but how many clear the floor.
3. **P1: how many distinct tools does a session touch?** Session-stable pruning chooses once and
   never widens. A session that reaches past the kept set is a miss, so a wide session is a
   no-go for the whole idea (DESIGN.md 3.4).
4. **P2: how good is a frequency table?** If the N most-called tools already cover almost
   everything, the correct product is a YAML file and not this package (DESIGN.md 3.5).

What it deliberately does not answer is P3, the k-NN gate that decides whether a learned
classifier beats that frequency table. P3 needs embeddings, embeddings need the `AI.EMBED`
connection, and the connection is the thing `provision.sh` creates. So the split is not
arbitrary: everything here runs against an untouched warehouse with read access and nothing
else, and P3 is the first question that costs a setup step. `sql/preflight.sql` has it.

Two other differences from `preflight.sql`, both because the audience is different. There is no
synthetic-corpus discriminator here, since an adopter's own analytics log is their own traffic
and the `corpus-` prefix means nothing in it. And the window is a day count rather than a
floor timestamp, so it matches the `lookback` argument the training procedure actually takes.

Read-only. Nothing here writes, and nothing here runs at request time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "CORPUS_SQL",
    "SPREAD_SQL",
    "Corpus",
    "Finding",
    "Readiness",
    "Spread",
    "read",
    "verdict",
]

# P2's stop sign (DESIGN.md 3.5). At or above this, a frequency table is the product.
STATIC_CEILING = 0.95

# Both mirror a DECLARE in sql/train_router.sql, and they are here rather than there so the
# question "could this train" is asked with the numbers the training run would actually use.
# If either changes in the SQL, it changes here, and the test that reads both keeps them honest.
TRAIN_LOOKBACK_DAYS = 35
MIN_CLASS_ROWS = 10

# Tool-opening sessions the gate needs before its verdict means much. Derived rather than
# chosen: the training run holds 25% of sessions out, and DESIGN.md 9.4 measured 391 scorable
# eval rows as about +/-0.7 points of noise against a 0.98 threshold. Four hundred eval rows
# is therefore the point where the gate can tell a pass from a fail, and four hundred is a
# quarter of sixteen hundred.
ROWS_FOR_A_STABLE_GATE = 1600

# One method of google-cloud-bigquery, aliased so the client stays out of this module's
# signatures and a test can pass a fake. Same reason `openings` and `model_store` do it.
BigQueryClient = Any

# `{project}` and `{dataset}` are formatted in because BigQuery cannot parameterise a table
# name. `@agent`, `@days`, `@top_n` and `@floor` are real parameters.
#
# Every read below filters on `timestamp` and `agent`, and the base-table reads also filter on
# `event_type`, because `agent_events` is DAY-partitioned on timestamp and clustered on
# (event_type, agent, user_id). A read that skips them scans the whole table.
CORPUS_SQL = """
WITH prompts AS (
  -- SOURCE: BASE agent_events. v_user_message_received projects only the common columns and
  -- carries no prompt text (DESIGN 18.2), so the text has to come off the base table.
  -- `has_text` is the whole point of this CTE: rows are kept even when the text is missing,
  -- so the share of prompts that carry any can be reported rather than silently dropped.
  SELECT
    invocation_id,
    ANY_VALUE(session_id) AS session_id,
    MIN(timestamp) AS prompt_ts,
    LENGTH(IFNULL(ANY_VALUE(JSON_VALUE(content, '$.text_summary')), '')) BETWEEN 3 AND 4000
      AS has_text
  FROM `{project}.{dataset}.agent_events`
  WHERE event_type = 'USER_MESSAGE_RECEIVED'
    AND agent = @agent
    AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    AND is_truncated = FALSE
  GROUP BY invocation_id
),
openers AS (
  -- The first prompt of each session, which is the only prompt the router ever decides on: a
  -- session that has already sent a full tool list cannot prune afterwards (DESIGN 18.22).
  -- This is the same row train_router.sql trains on, so the counts below are training rows.
  SELECT
    session_id,
    ARRAY_AGG(STRUCT(invocation_id, has_text) ORDER BY prompt_ts, invocation_id LIMIT 1)[
      OFFSET(0)
    ] AS opener
  FROM prompts
  GROUP BY session_id
),
calls AS (
  -- SOURCE: v_tool_completed (a VIEW), which renames the base table's `$.tool` and `$.result`
  -- to `tool_name` and `tool_result`. The cache-hit marker is read in the view's spelling; the
  -- base-table spelling matches nothing and fails open (DESIGN 9.1, 11.5).
  SELECT session_id, timestamp, tool_name
  FROM `{project}.{dataset}.v_tool_completed`
  WHERE agent = @agent
    AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    AND status = 'OK'
    -- ADK control plane, never a routed tool (DESIGN 7.2).
    AND tool_name != 'transfer_to_agent'
    AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
),
first_call AS (
  -- The tool each session called first. Aggregated up front rather than looked up per session:
  -- BigQuery rejects a correlated subquery over another CTE that it cannot de-correlate, so
  -- this has to be a join, and a join is the cheaper shape anyway.
  SELECT
    session_id,
    ARRAY_AGG(tool_name ORDER BY timestamp LIMIT 1)[OFFSET(0)] AS tool_name
  FROM calls
  GROUP BY session_id
),
labelled AS (
  -- One row per session: the tool it opened with, or NULL for the ABSTAIN class. ABSTAIN is a
  -- label and never an action (D10), so a session that called nothing is a training row too.
  SELECT o.session_id, o.opener.has_text AS has_text, f.tool_name
  FROM openers o
  LEFT JOIN first_call f USING (session_id)
),
classes AS (
  SELECT tool_name, COUNT(*) AS rows_for_class
  FROM labelled
  WHERE has_text AND tool_name IS NOT NULL
  GROUP BY tool_name
),
-- P2, the static frequency baseline: rank tools by how often they open a session, keep the
-- top N, and ask what share of tool-opening sessions that set would have covered. Fitted and
-- scored on the same rows, which flatters it, which is the safe direction for a stop sign.
top_n AS (
  SELECT tool_name
  FROM classes
  ORDER BY rows_for_class DESC, tool_name
  LIMIT @top_n
)
SELECT
  (SELECT COUNT(*) FROM prompts) AS prompts,
  (SELECT COUNTIF(has_text) FROM prompts) AS prompts_with_text,
  (SELECT COUNT(*) FROM labelled) AS sessions,
  (SELECT COUNTIF(has_text) FROM labelled) AS trainable,
  (SELECT COUNTIF(has_text AND tool_name IS NOT NULL) FROM labelled) AS tool_opening,
  (SELECT COUNT(*) FROM calls) AS tool_calls,
  (SELECT COUNT(*) FROM classes) AS tools,
  (SELECT COUNTIF(rows_for_class >= @floor) FROM classes) AS tools_at_floor,
  (SELECT SUM(rows_for_class) FROM classes WHERE tool_name IN (SELECT tool_name FROM top_n))
    AS covered_by_top_n
"""

# P1, kept separate because it counts every tool call of a session rather than the one it
# opened with, so it shares no CTE with the corpus counts above and folding the two together
# would only make one long query out of two short ones.
SPREAD_SQL = """
WITH per_session AS (
  SELECT session_id, COUNT(DISTINCT tool_name) AS distinct_tools
  FROM `{project}.{dataset}.v_tool_completed`
  WHERE agent = @agent
    AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
    AND status = 'OK'
    AND tool_name != 'transfer_to_agent'
  GROUP BY session_id
)
SELECT
  COUNT(*) AS sessions,
  APPROX_QUANTILES(distinct_tools, 100)[SAFE_OFFSET(50)] AS p50,
  APPROX_QUANTILES(distinct_tools, 100)[SAFE_OFFSET(90)] AS p90,
  APPROX_QUANTILES(distinct_tools, 100)[SAFE_OFFSET(99)] AS p99,
  MAX(distinct_tools) AS widest
FROM per_session
"""


@dataclass(frozen=True)
class Corpus:
    """What the window holds, counted the way the training run would count it."""

    prompts: int = 0
    prompts_with_text: int = 0
    sessions: int = 0
    trainable: int = 0
    tool_opening: int = 0
    tool_calls: int = 0
    tools: int = 0
    tools_at_floor: int = 0
    covered_by_top_n: int = 0

    @property
    def text_share(self) -> float | None:
        """Share of first user messages that carry usable text, or None with no prompts."""
        return self.prompts_with_text / self.prompts if self.prompts else None

    @property
    def static_coverage(self) -> float | None:
        """P2: what a frequency table of the top N tools would have covered."""
        return self.covered_by_top_n / self.tool_opening if self.tool_opening else None


@dataclass(frozen=True)
class Spread:
    """P1: how many distinct tools a session touches."""

    sessions: int = 0
    p50: int = 0
    p90: int = 0
    p99: int = 0
    widest: int = 0


@dataclass(frozen=True)
class Readiness:
    """Everything the command prints, and nothing that decides what to print."""

    agent: str
    days: int
    top_n: int
    floor: int
    corpus: Corpus | None
    spread: Spread | None
    scanned: int = 0

    @property
    def found_the_log(self) -> bool:
        """False when the dataset has no analytics tables, which is its own diagnosis."""
        return self.corpus is not None


@dataclass(frozen=True)
class Finding:
    """One question, its answer, and what the answer costs.

    `standing` is three-valued on purpose. "stop" is the only one that means do not train, and
    "thin" is the interesting one: enough data to run, not enough for the result to be trusted,
    which is the state most adopters are actually in and the one a boolean would flatten.
    """

    standing: Literal["go", "thin", "stop"]
    title: str
    detail: str


def verdict(r: Readiness) -> tuple[Finding, ...]:
    """Turn the counts into findings, hardest blocker first.

    Order is deliberate. Missing prompt text cannot be fixed by waiting, a wide session cannot
    be fixed at all, and a thin corpus fixes itself, so they are reported in the order that
    tells someone whether to stop reading.
    """
    if r.corpus is None:
        return (
            Finding(
                "stop",
                "the analytics log is readable",
                "no agent_events table in this dataset. Either the BigQuery Agent Analytics "
                "plugin has never written here, or --dataset points somewhere else. It is the "
                "dataset the analytics plugin writes to, not one this project creates.",
            ),
        )

    c = r.corpus
    findings = [_text(c), _volume(c, r.floor)]
    if r.spread is not None:
        findings.append(_width(r.spread, r.top_n))
    findings.append(_static(c, r.top_n))
    return tuple(findings)


def _text(c: Corpus) -> Finding:
    """The one blocker that is somebody else's configuration rather than their traffic."""
    share = c.text_share
    if not c.prompts:
        return Finding(
            "stop",
            "first user messages carry text",
            "no USER_MESSAGE_RECEIVED events in the window at all. Check --agent against the "
            "name the analytics plugin writes, and --days against when it was installed.",
        )
    if share is None or share == 0:
        return Finding(
            "stop",
            "first user messages carry text",
            f"{c.prompts:,} prompts logged and none carries usable text at "
            f"$.text_summary. The labels are built from that field, so nothing here can train "
            f"until the analytics plugin is configured to log message content.",
        )
    if share < 0.99:
        return Finding(
            "thin",
            "first user messages carry text",
            f"{share:.1%} of {c.prompts:,} prompts carry usable text. The rest are dropped, which "
            f"is fine at this share and worth understanding below it: truncated events are "
            f"already excluded, so the remainder is empty or out of the 3 to 4000 character band.",
        )
    return Finding(
        "go",
        "first user messages carry text",
        f"{share:.1%} of {c.prompts:,} prompts carry text at $.text_summary, which is where the "
        f"labels come from.",
    )


def _volume(c: Corpus, floor: int) -> Finding:
    """Rows and classes. Neither number means anything without the other."""
    if not c.tools_at_floor:
        return Finding(
            "stop",
            "enough labelled sessions, over enough tools",
            f"{c.tool_opening:,} sessions opened with a tool and no tool reaches the {floor}-row "
            f"floor, so there is no class to learn. Widen --days, or wait.",
        )
    detail = (
        f"{c.trainable:,} trainable sessions, {c.tool_opening:,} of them opening with a tool over "
        f"{c.tools} distinct tools, {c.tools_at_floor} of which clear the {floor}-row floor. "
        f"The rest fold into ABSTAIN rather than blocking the run."
    )
    if c.tool_opening < ROWS_FOR_A_STABLE_GATE:
        return Finding(
            "thin",
            "enough labelled sessions, over enough tools",
            f"{detail} Under {ROWS_FOR_A_STABLE_GATE:,}, though, so the promotion gate will be "
            f"reading a held-out set too small to separate a pass from a fail cleanly. Training "
            f"works; trust the verdict less.",
        )
    return Finding("go", "enough labelled sessions, over enough tools", detail)


def _width(s: Spread, top_n: int) -> Finding:
    """P1. The gate on whether session-stable pruning can work here at all."""
    tool = "tool" if s.p50 == 1 else "tools"
    shape = (
        f"a session touches {s.p50} distinct {tool} at the median, {s.p90} at p90 and {s.p99} at "
        f"p99, widest {s.widest}, over {s.sessions:,} sessions that called anything"
    )
    if s.p90 > top_n:
        return Finding(
            "stop",
            f"sessions stay inside a menu of {top_n}",
            f"{shape}. The router chooses once and never widens, so a session past the kept set "
            f"is a miss and not a repair. Pruning does not pay on traffic this wide.",
        )
    return Finding("go", f"sessions stay inside a menu of {top_n}", f"{shape}.")


def _static(c: Corpus, top_n: int) -> Finding:
    """P2. The stop sign that says the correct product is a configuration file."""
    coverage = c.static_coverage
    if coverage is None:
        return Finding(
            "thin",
            "a frequency table is not already enough",
            "no session opened with a tool, so there is nothing to rank and nothing to beat.",
        )
    shape = (
        f"the {top_n} most-called tools would have covered {coverage:.1%} of tool-opening sessions"
    )
    if coverage >= STATIC_CEILING:
        return Finding(
            "stop",
            "a frequency table is not already enough",
            f"{shape}, at or above the {STATIC_CEILING:.0%} line. A static list of those tools is "
            f"the correct product here and it needs none of this package's machinery.",
        )
    return Finding(
        "go",
        "a frequency table is not already enough",
        f"{shape}, which leaves room for a model to beat. Whether it does is P3, and P3 needs "
        f"the AI.EMBED connection that provision.sh creates.",
    )


def read(
    project: str,
    dataset: str,
    agent: str,
    days: int,
    top_n: int,
    floor: int,
    client: BigQueryClient = None,
) -> Readiness:
    """Run both reads. Two scans of one partition range, and nothing is written.

    A missing table returns a `Readiness` with no corpus rather than raising, because "the
    analytics plugin has never written here" is the most likely answer for the person running
    this and it deserves a sentence, not a stack trace.
    """
    # Imported here so that `import adk_smart_router` costs an agent nothing at serving time.
    from google.cloud import bigquery

    reader = client or bigquery.Client(project=project)
    scanned = 0

    rows, read_bytes = _query(reader, CORPUS_SQL, project, dataset, agent, days, top_n, floor)
    scanned += read_bytes
    corpus = None if rows is None else Corpus(**{k: int(v or 0) for k, v in rows[0].items()})

    spread = None
    if corpus is not None:
        rows, read_bytes = _query(reader, SPREAD_SQL, project, dataset, agent, days, top_n, floor)
        scanned += read_bytes
        spread = None if rows is None else Spread(**{k: int(v or 0) for k, v in rows[0].items()})

    return Readiness(
        agent=agent,
        days=days,
        top_n=top_n,
        floor=floor,
        corpus=corpus,
        spread=spread,
        scanned=scanned,
    )


def _query(
    reader: BigQueryClient,
    sql: str,
    project: str,
    dataset: str,
    agent: str,
    days: int,
    top_n: int,
    floor: int,
) -> tuple[list[dict[str, Any]] | None, int]:
    """One read, with a missing table turned into None instead of an exception."""
    from google.api_core import exceptions as api
    from google.cloud import bigquery

    job = reader.query(
        sql.format(project=project, dataset=dataset),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", agent),
                bigquery.ScalarQueryParameter("days", "INT64", days),
                bigquery.ScalarQueryParameter("top_n", "INT64", top_n),
                bigquery.ScalarQueryParameter("floor", "INT64", floor),
            ]
        ),
    )
    try:
        rows = [dict(row) for row in job.result()]
    except api.NotFound:
        return None, 0
    return rows, int(job.total_bytes_processed or 0)
