#!/usr/bin/env python
"""What turn-one warmth costs the routed arm, and what it would be at production traffic.

A 50/50 A/B cannot fairly price a mechanism whose value is prefix reuse. The control
sends one tool list, so every control session warms the same prefix. The routed arm
spreads its sessions over the menu pool, so each of its prefixes sees a fraction of the
traffic. In `ab-v2-100tools` that is 289 arrivals against a median of 9. The treatment arm
is starved of exactly the resource it exists to consume, and `ROADMAP.md` has carried the
problem since the first run without a number attached to it.

This script attaches the number, from data already paid for. The idea is that arrivals per
prefix vary a great deal *within* the run: the busiest menu saw 30 sessions and the
quietest saw 1. That variation is a natural experiment, and it is immune to the day-to-day
swings in implicit caching that make any comparison across two runs worthless (§19.50).

So it groups every turn-one call by how long its prefix had been idle, and reads warmth
off the curve:

    gap         seconds since any session last used this exact prefix
    warm rate   share of turn-one calls that were served anything at all
    depth       how much was served, when something was

Then it reprices the routed arm at two counterfactuals. The first gives every routed
prefix the arrival rate the control actually had, which is what production traffic buys.
The second additionally gives it the control's warm rate, which is what a single-prefix
agent gets for free. The two bracket the answer; neither is a measurement, and the script
says so where it prints them.

Cost is two BigQuery reads and no model traffic. Authentication is ADC only.

Usage:
    uv run python scripts/study_prefix_arrivals.py --prefix ab-v2-100tools
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from google.cloud import bigquery

# Same defaults as everywhere else in the project (DESIGN.md 3.1).
DEFAULT_AGENT = "support"

# Kept identical to `report_ab.py` so the arms are the same arms.
CACHED_TOKEN_WEIGHT = 0.10
ROUTED_SOURCES = ("fresh", "reused")

# Idle-gap buckets, in seconds. The edges are round numbers chosen before the data was
# looked at, and the first one matters most: the control's traffic sits almost entirely
# inside it, so it is the bucket the counterfactual reads from.
BUCKET_EDGES: tuple[tuple[str, float], ...] = (
    ("<30s", 30.0),
    ("<2m", 120.0),
    ("<5m", 300.0),
    ("<15m", 900.0),
    ("15m+", float("inf")),
)

# The bucket a prefix's very first appearance falls in. It has no gap because nothing came
# before it, and lumping it in with "15m+" would be a different claim: a cold prefix is
# cold because it has never been sent, not because it was sent long ago.
FIRST = "first"


# --- The arithmetic, which needs no warehouse -------------------------------------


@dataclass(frozen=True)
class TurnOne:
    """One session's opening call, with the idle gap on the prefix it was sent under."""

    arm: str
    gap_s: float | None
    prompt_tokens: int
    cached_tokens: int


@dataclass(frozen=True)
class Totals:
    """One arm's whole-run input, for repricing turn one inside it."""

    arm: str
    sessions: int
    prompt_tokens: int
    cached_tokens: int

    @property
    def billed_input(self) -> float:
        uncached = max(self.prompt_tokens - self.cached_tokens, 0)
        return uncached + CACHED_TOKEN_WEIGHT * self.cached_tokens

    @property
    def billed_per_session(self) -> float:
        return self.billed_input / self.sessions if self.sessions else 0.0


@dataclass(frozen=True)
class Bucket:
    """What one idle-gap bucket of one arm looked like."""

    name: str
    calls: int
    warm_calls: int
    prompt_tokens: int
    cached_tokens: int

    @property
    def warm_rate(self) -> float:
        return self.warm_calls / self.calls if self.calls else 0.0

    @property
    def mean_prompt(self) -> float:
        return self.prompt_tokens / self.calls if self.calls else 0.0

    @property
    def mean_cached(self) -> float:
        """Averaged over every call in the bucket, warm or cold. What repricing needs."""
        return self.cached_tokens / self.calls if self.calls else 0.0

    @property
    def depth_when_warm(self) -> float:
        """Averaged over the warm calls only. Separates how often from how much."""
        return self.cached_tokens / self.warm_calls if self.warm_calls else 0.0


def bucket_of(gap_s: float | None) -> str:
    if gap_s is None:
        return FIRST
    for name, edge in BUCKET_EDGES:
        if gap_s < edge:
            return name
    return BUCKET_EDGES[-1][0]


def buckets_for(rows: list[TurnOne], arm: str) -> list[Bucket]:
    """Every bucket that has at least one call in it, in gap order with `first` leading."""
    order = [FIRST, *(name for name, _ in BUCKET_EDGES)]
    mine = [row for row in rows if row.arm == arm]
    out = []
    for name in order:
        in_bucket = [row for row in mine if bucket_of(row.gap_s) == name]
        if not in_bucket:
            continue
        out.append(
            Bucket(
                name=name,
                calls=len(in_bucket),
                warm_calls=sum(1 for row in in_bucket if row.cached_tokens > 0),
                prompt_tokens=sum(row.prompt_tokens for row in in_bucket),
                cached_tokens=sum(row.cached_tokens for row in in_bucket),
            )
        )
    return out


def busiest_bucket(buckets: list[Bucket]) -> Bucket | None:
    """The bucket holding the most calls. The control's is the arrival rate to aim at."""
    return max(buckets, key=lambda b: b.calls) if buckets else None


def observed_cached(rows: list[TurnOne], arm: str) -> int:
    return sum(row.cached_tokens for row in rows if row.arm == arm)


def reprice(totals: Totals, rows: list[TurnOne], target_cached: float) -> float:
    """Billed input per session if every turn one in `arm` had cached `target_cached`.

    Only turn one moves. Every later turn of a session caches against the same session's
    own history and the arrival rate on the shared prefix cannot reach it, which is why
    §18.18 could already price turns three onward and not the run.
    """
    mine = [row for row in rows if row.arm == totals.arm]
    if not mine or not totals.sessions:
        return totals.billed_per_session
    # Nothing can be served more of the prompt than the prompt contains.
    wanted = sum(min(target_cached, row.prompt_tokens) for row in mine)
    delta = wanted - observed_cached(rows, totals.arm)
    return (totals.billed_input - (1.0 - CACHED_TOKEN_WEIGHT) * delta) / totals.sessions


def change(routed: float, holdout: float) -> float | None:
    return (routed - holdout) / holdout if holdout else None


# --- The warehouse ----------------------------------------------------------------

# Sessions are labelled from the router's own provenance, exactly as `report_ab.py` labels
# them. A prefix is the kept set itself, sorted and hashed, rather than the top predicted
# label: two labels can select the same menu, and it is the menu the cache sees.
_ARMS_CTE = """
WITH decisions AS (
  SELECT
    e.session_id,
    JSON_VALUE(record, '$.decision_source') AS source,
    TO_JSON_STRING(ARRAY(
      SELECT tool FROM UNNEST(JSON_VALUE_ARRAY(record, '$.kept_tools')) AS tool ORDER BY tool
    )) AS menu
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
    END AS arm,
    CASE
      WHEN LOGICAL_OR(source = 'holdout') THEN 'control'
      ELSE TO_HEX(MD5(MAX(IF(source IN UNNEST(@routed_sources), menu, NULL))))
    END AS prefix_key
  FROM decisions
  GROUP BY session_id
)
"""

# One row per session, carrying its opening call and the idle time on its prefix. The gap
# is measured against every turn of every session on that prefix, not just opening ones: a
# long conversation keeps its own prefix alive for the next session that lands on it.
_TURN_ONE_SQL = (
    _ARMS_CTE
    + """,
calls AS (
  SELECT
    arms.arm,
    arms.prefix_key,
    v.session_id,
    v.timestamp AS ts,
    COALESCE(v.usage_prompt_tokens, 0) AS prompt_tokens,
    COALESCE(v.usage_cached_tokens, 0) AS cached_tokens,
    ROW_NUMBER() OVER (PARTITION BY v.session_id ORDER BY v.timestamp) AS turn
  FROM `{project}.{dataset}.v_llm_response` AS v
  JOIN arms USING (session_id)
  WHERE v.agent = @agent
    AND STARTS_WITH(v.session_id, @prefix)
    AND v.usage_prompt_tokens IS NOT NULL
    AND arms.arm != 'neither'
),
gaps AS (
  SELECT
    *,
    TIMESTAMP_DIFF(ts, LAG(ts) OVER (PARTITION BY prefix_key ORDER BY ts), SECOND) AS gap_s
  FROM calls
)
SELECT arm, gap_s, prompt_tokens, cached_tokens
FROM gaps
WHERE turn = 1
"""
)

# Whole-run input per arm, so turn one can be repriced inside a real total rather than
# reported as a rate on its own.
_TOTALS_SQL = (
    _ARMS_CTE
    + """
SELECT
  arms.arm,
  COUNT(DISTINCT v.session_id) AS sessions,
  SUM(COALESCE(v.usage_prompt_tokens, 0)) AS prompt_tokens,
  SUM(COALESCE(v.usage_cached_tokens, 0)) AS cached_tokens
FROM `{project}.{dataset}.v_llm_response` AS v
JOIN arms USING (session_id)
WHERE v.agent = @agent
  AND STARTS_WITH(v.session_id, @prefix)
  AND v.usage_prompt_tokens IS NOT NULL
  AND arms.arm != 'neither'
GROUP BY arm
"""
)

# How many sessions each prefix saw. Printed as the asymmetry itself, in one line.
_ARRIVALS_SQL = (
    _ARMS_CTE
    + """
SELECT arm, prefix_key, COUNT(*) AS sessions
FROM arms
WHERE arm != 'neither'
GROUP BY arm, prefix_key
ORDER BY sessions DESC
"""
)


def _config(agent: str, prefix: str) -> bigquery.QueryJobConfig:
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("agent", "STRING", agent),
            bigquery.ScalarQueryParameter("prefix", "STRING", prefix),
            bigquery.ArrayQueryParameter("routed_sources", "STRING", list(ROUTED_SOURCES)),  # type: ignore[arg-type]
        ]
    )


def fetch_turn_ones(
    client: bigquery.Client, dataset: str, agent: str, prefix: str
) -> list[TurnOne]:
    sql = _TURN_ONE_SQL.format(project=client.project, dataset=dataset)
    return [
        TurnOne(
            arm=str(row["arm"]),
            gap_s=None if row["gap_s"] is None else float(row["gap_s"]),
            prompt_tokens=int(row["prompt_tokens"]),
            cached_tokens=int(row["cached_tokens"]),
        )
        for row in client.query(sql, job_config=_config(agent, prefix)).result()
    ]


def fetch_totals(client: bigquery.Client, dataset: str, agent: str, prefix: str) -> list[Totals]:
    sql = _TOTALS_SQL.format(project=client.project, dataset=dataset)
    return [
        Totals(
            arm=str(row["arm"]),
            sessions=int(row["sessions"]),
            prompt_tokens=int(row["prompt_tokens"]),
            cached_tokens=int(row["cached_tokens"]),
        )
        for row in client.query(sql, job_config=_config(agent, prefix)).result()
    ]


def fetch_arrivals(
    client: bigquery.Client, dataset: str, agent: str, prefix: str
) -> dict[str, list[int]]:
    sql = _ARRIVALS_SQL.format(project=client.project, dataset=dataset)
    out: dict[str, list[int]] = {}
    for row in client.query(sql, job_config=_config(agent, prefix)).result():
        out.setdefault(str(row["arm"]), []).append(int(row["sessions"]))
    return out


# --- What it prints ---------------------------------------------------------------


def median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _bucket_rows(arm: str, buckets: list[Bucket]) -> list[str]:
    lines = [f"  {arm}"]
    for bucket in buckets:
        lines.append(
            f"    {bucket.name:<8}{bucket.calls:>6}{bucket.warm_rate:>9.0%}"
            f"{bucket.mean_prompt:>11,.0f}{bucket.mean_cached:>11,.0f}"
            f"{bucket.depth_when_warm:>11,.0f}"
        )
    return lines


def report(rows: list[TurnOne], totals: list[Totals], arrivals: dict[str, list[int]]) -> int:
    by_arm = {total.arm: total for total in totals}
    routed, holdout = by_arm.get("routed"), by_arm.get("holdout")
    if routed is None or holdout is None:
        print("Both arms are needed and at least one is missing. Arms found: ", end="")
        print(", ".join(sorted(by_arm)) or "none")
        return 1

    routed_arrivals = arrivals.get("routed", [])
    holdout_arrivals = arrivals.get("holdout", [])
    print(
        f"\nturn-one warmth by idle gap — {routed.sessions:,} routed, "
        f"{holdout.sessions:,} holdout\n"
    )
    print(
        f"  the asymmetry: {len(holdout_arrivals)} control prefix over "
        f"{sum(holdout_arrivals):,} sessions, against {len(routed_arrivals)} routed\n"
        f"  prefixes over {sum(routed_arrivals):,}, a median of "
        f"{median(routed_arrivals):,.0f} arrivals each."
    )

    print(f"\n    {'gap':<8}{'calls':>6}{'warm':>9}{'prompt':>11}{'cached':>11}{'when warm':>11}")
    routed_buckets = buckets_for(rows, "routed")
    holdout_buckets = buckets_for(rows, "holdout")
    for arm, buckets in (("holdout", holdout_buckets), ("routed", routed_buckets)):
        for line in _bucket_rows(arm, buckets):
            print(line)

    fast_routed = busiest_bucket(holdout_buckets)
    if fast_routed is None:
        print("\n  no control traffic to read an arrival rate from.")
        return 0
    match = next((b for b in routed_buckets if b.name == fast_routed.name), None)
    if match is None:
        print(f"\n  no routed traffic in the control's own bucket ({fast_routed.name}).")
        return 0

    control_calls = sum(bucket.calls for bucket in holdout_buckets)
    print("\n  what the routed arm would bill at the control's arrival rate")
    print(
        f"    the control's traffic sits in the {fast_routed.name} bucket: "
        f"{fast_routed.calls} of its {control_calls} opening calls.\n"
    )

    at_rate = reprice(routed, rows, match.mean_cached)
    at_parity = reprice(routed, rows, fast_routed.warm_rate * match.depth_when_warm)
    scenarios = (
        ("observed", routed.billed_per_session, "as the run actually ran"),
        ("at the control's arrival rate", at_rate, "read off the routed arm's fast bucket"),
        ("and at its warm rate too", at_parity, "an upper bound, not a measurement"),
    )
    print(f"    {'':<32}{'billed in / session':>20}{'vs holdout':>12}")
    print(f"    {'holdout':<32}{holdout.billed_per_session:>20,.0f}{'—':>12}")
    for label, value, note in scenarios:
        delta = change(value, holdout.billed_per_session)
        shown = "     —" if delta is None else f"{delta:+.1%}"
        print(f"    {label:<32}{value:>20,.0f}{shown:>12}  {note}")

    print(
        "\n    Neither scenario was run. Both hold everything except turn-one warmth fixed,\n"
        "    and turn one is the only turn the arrival rate can reach: every later turn\n"
        "    caches against its own session's history in both arms."
    )
    print()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prefix", required=True, help="session id prefix of the run")
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = bigquery.Client(project=args.project)
    rows = fetch_turn_ones(client, args.dataset, args.agent, args.prefix)
    totals = fetch_totals(client, args.dataset, args.agent, args.prefix)
    arrivals = fetch_arrivals(client, args.dataset, args.agent, args.prefix)
    return report(rows, totals, arrivals)


if __name__ == "__main__":
    sys.exit(main())
