#!/usr/bin/env python
"""Ask whether the published model still fits the surface it is serving, and say so out loud.

The router already has two ways of noticing it has gone stale, and neither is a detector.
`cache_watch` reads the bill, so it fires after the money is spent. `max_model_age_days` reads
the clock, so it fires whether or not anything has changed. This is the third: a signal readable
from the requests themselves (DESIGN.md 20.5).

**Unseen share gates. Confidence is reported and never gates.** §19.63 drove a real drift event
and measured both. The surface change from 25 tools to 100 cost 36 points of recall and moved
unseen share from 26.1% to 77.3% at the same instant, so the signal tracks the harm. A constructed
intent shift moved confidence by 0.25 for 1.3 points of recall, which is sensitive to change
rather than to harm. That belongs on a prompt, not on a trigger.

**Nothing here retrains anything.** It writes a `drift` row saying `due` or `ok` and returns.
`run_cycle.sh` is what turns that verdict into a retrain, and it reads the verdict through
`--exit-code`; the decision to run it at all stays with a scheduler, and deliberately not with
this script.

**The threshold is a two-point fit.** One real drift event and one real non-event is not a curve.
`--backfill` replays the agent's own history against a range of candidate margins so the default
can be argued with using the agent's own numbers rather than this project's.

Offline. It reads the warehouse and nothing else, adds no serving dependency, and runs nowhere
near the hot path. Authentication is ADC only.

Usage:
    uv run scripts/check_drift.py --agent support
    uv run scripts/check_drift.py --agent support --no-record      # look without writing
    uv run scripts/check_drift.py --agent support --estimate       # what would this scan cost
    uv run scripts/check_drift.py --agent support --backfill 30    # fit the margin on this agent
    uv run scripts/check_drift.py --agent support --exit-code      # exit 10 when a retrain is due
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from google.cloud import bigquery

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deployment import DEFAULT_DATASET, bucket_arg, project_arg  # noqa: E402

from adk_smart_router import drift, lifecycle  # noqa: E402
from adk_smart_router.config import load_config  # noqa: E402
from adk_smart_router.model_store import ModelStore  # noqa: E402

DEFAULT_AGENT = "support"

# The margins `--backfill` reports against. Wide on purpose: the point of the exercise is to see
# where an agent's own history separates, not to confirm the shipped default.
BACKFILL_MARGINS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)

# What `--exit-code` returns when a retrain is due. Not 1, because 1 is what this script exits
# with when it breaks, and a scheduler that cannot tell "the model has drifted" from "the check
# crashed" will retrain on a broken query (DESIGN.md 21.6). `run_cycle.sh` reads this.
DUE_EXIT_CODE = 10

# The declared surface, read from the tool list BQAA serializes into every LLM_REQUEST row.
#
# A *union* over the window rather than the newest request, because with the router installed a
# routed request carries the menu it was pruned to and not the surface it was pruned from. Every
# declared tool still appears somewhere: the holdout arm, declines and fail-open sessions all ship
# the whole list. The union is therefore the surface, and its one bias is the safe direction — a
# tool retired mid-window stays in the union and makes the reading conservative, while a tool
# added mid-window shows up at once, which is the case that costs recall.
#
# One day by default. This column is the expensive one in the warehouse: BQAA writes the entire
# serialized tool list into every row, 94 KB each on the 100-tool surface, and the surface is a
# property of the deployment rather than of a week of traffic. Partition, event type and agent
# are all pruned; widen the window only when you mean to pay for it.
SURFACE_SQL = """
SELECT DISTINCT JSON_VALUE(tool, '$.name') AS tool_name
FROM `{project}.{dataset}.agent_events`,
  UNNEST(JSON_QUERY_ARRAY(attributes, '$.tools')) AS tool
WHERE event_type = 'LLM_REQUEST'
  AND agent = @agent
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
  AND JSON_VALUE(tool, '$.name') IS NOT NULL
"""

# The same surface, cut by day, for --backfill. Same cost per day as the query above.
SURFACE_BY_DAY_SQL = """
SELECT
  DATE(timestamp) AS day,
  ARRAY_AGG(DISTINCT JSON_VALUE(tool, '$.name') IGNORE NULLS) AS tools
FROM `{project}.{dataset}.agent_events`,
  UNNEST(JSON_QUERY_ARRAY(attributes, '$.tools')) AS tool
WHERE event_type = 'LLM_REQUEST'
  AND agent = @agent
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
GROUP BY day
ORDER BY day
"""

# Top-1 mass per routed decision, for the model that is live right now. Pinned to one version
# because a quantile taken across two generations describes neither of them.
CONFIDENCE_SQL = """
SELECT confidence
FROM `{project}.{dataset}.v_router_decisions`
WHERE agent = @agent
  AND model_version = @version
  AND decision_class = 'routed'
  AND confidence IS NOT NULL
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
ORDER BY timestamp DESC
LIMIT @limit
"""


def query(
    client: bigquery.Client,
    sql: str,
    params: list[bigquery.ScalarQueryParameter],
    estimate: bool = False,
) -> tuple[list[Any], int]:
    """Run one query, or price it. Returns the rows and the bytes it scanned.

    The bytes are reported rather than hidden because one of these queries reads the column
    that holds the whole serialized tool list, and an operator running this on a schedule
    should be able to see what that costs before it becomes a line on an invoice.
    """
    config = bigquery.QueryJobConfig(query_parameters=params, dry_run=estimate)
    job = client.query(sql, job_config=config)
    if estimate:
        return [], int(job.total_bytes_processed or 0)
    rows = list(job.result())
    return rows, int(job.total_bytes_processed or 0)


def gigabytes(scanned: int) -> str:
    return f"{scanned / 1024**3:.2f} GB"


def read_surface(client: bigquery.Client, args: argparse.Namespace) -> tuple[set[str], int]:
    """Every tool name declared to the model over the window."""
    rows, scanned = query(
        client,
        SURFACE_SQL.format(project=args.project, dataset=args.dataset),
        [
            bigquery.ScalarQueryParameter("agent", "STRING", args.agent),
            bigquery.ScalarQueryParameter("days", "INT64", args.surface_days),
        ],
        args.estimate,
    )
    return {row["tool_name"] for row in rows}, scanned


def read_confidences(
    client: bigquery.Client, args: argparse.Namespace, version: str
) -> tuple[list[float], int]:
    """Recent top-1 mass, newest first. Empty when the router has not routed anything yet."""
    rows, scanned = query(
        client,
        CONFIDENCE_SQL.format(project=args.project, dataset=args.dataset),
        [
            bigquery.ScalarQueryParameter("agent", "STRING", args.agent),
            bigquery.ScalarQueryParameter("version", "STRING", version),
            bigquery.ScalarQueryParameter("days", "INT64", args.days),
            bigquery.ScalarQueryParameter("limit", "INT64", args.max_sessions),
        ],
        args.estimate,
    )
    return [float(row["confidence"]) for row in rows], scanned


def report(now: drift.Signals, baseline: drift.Signals | None, verdict: drift.Verdict) -> None:
    """Print the reading. Every line degrades to a stated absence rather than to a zero."""
    print("\n==> surface")
    print(f"    declared        {now.declared} tools")
    print(f"    no class for    {now.unclassed} ({now.unseen_share:.1%})")
    if baseline is None:
        print("    at publish      not recorded; this artifact predates the baseline")
    else:
        moved = now.unseen_share - baseline.unseen_share
        print(f"    at publish      {baseline.unseen_share:.1%} ({moved:+.1%} since)")

    print("\n==> confidence, reported and never gating")
    if now.sessions == 0:
        print("    no routed sessions in the window, so there is nothing to take a quantile of")
    else:
        print(f"    p50             {now.confidence_p50:.3f} over {now.sessions} sessions")
        print(f"    p10             {now.confidence_p10:.3f}")
        if baseline is not None and baseline.confidence_p50 is not None:
            print(f"    at publish      p50 {baseline.confidence_p50:.3f}")

    print(f"\n==> {'RETRAIN DUE' if verdict.due else 'no retrain due'}")
    print(f"    {verdict.reason}")


def backfill(client: bigquery.Client, args: argparse.Namespace, model: Any) -> int:  # noqa: ANN401
    """Replay the agent's own surface history against a range of candidate margins.

    The shipped default is fitted to one drift event and one non-event, which is not a curve.
    This is how an adopter argues with it using their own traffic: a margin that fires on half
    the days in a quiet month is too tight, and one that never fires on a month containing a
    known surface change is too loose.

    Every day is scored against the *current* model's classes and the current baseline, so this
    answers "would this margin have flagged the surface moving" and not "what would a model
    trained that week have thought". The second question needs a model per week and is what
    `backtest.sql` is for.
    """
    rows, scanned = query(
        client,
        SURFACE_BY_DAY_SQL.format(project=args.project, dataset=args.dataset),
        [
            bigquery.ScalarQueryParameter("agent", "STRING", args.agent),
            bigquery.ScalarQueryParameter("days", "INT64", args.backfill),
        ],
        args.estimate,
    )
    if args.estimate:
        print(f"==> --backfill {args.backfill} days would scan {gigabytes(scanned)}")
        return 0
    if not rows:
        print(f"no LLM_REQUEST rows for {args.agent} in the last {args.backfill} days.")
        return 0

    baseline = model.metadata.drift_baseline
    if baseline is None:
        print("this artifact has no baseline, so there is nothing to measure a margin against.")
        print("Republish to set one, then come back.")
        return 1

    classes = model.classes
    print(f"==> {len(rows)} days, scanned {gigabytes(scanned)}")
    print(f"    baseline unseen share {baseline.unseen_share:.1%}\n")
    header = "    day          declared  unseen   " + "  ".join(
        f"{margin:.0%}".rjust(5) for margin in BACKFILL_MARGINS
    )
    print(header)
    fires = dict.fromkeys(BACKFILL_MARGINS, 0)
    for row in rows:
        today = drift.signals(list(row["tools"]), classes)
        marks = []
        for margin in BACKFILL_MARGINS:
            due = drift.verdict(today, baseline, margin=margin).due
            fires[margin] += due
            marks.append(("DUE" if due else "-").rjust(5))
        print(
            f"    {row['day']}   {today.declared:>7}  {today.unseen_share:>6.1%}   "
            + "  ".join(marks)
        )
    print("\n    days flagged")
    for margin in BACKFILL_MARGINS:
        note = "  <- the shipped default" if margin == drift.DEFAULT_UNSEEN_SHARE_MARGIN else ""
        print(f"    {margin:>5.0%}  {fires[margin]:>3} of {len(rows)}{note}")
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset, gcs_bucket=args.bucket)

    model = ModelStore(cfg).load(args.agent)
    if model is None:
        print(f"{args.agent} has no published router, so there is nothing to check.")
        return 0

    client = bigquery.Client(project=args.project)
    if args.backfill:
        return backfill(client, args, model)

    surface, surface_bytes = read_surface(client, args)
    confidences, confidence_bytes = read_confidences(client, args, model.metadata.version)
    if args.estimate:
        print(f"==> the surface read would scan {gigabytes(surface_bytes)}")
        print(f"==> the confidence read would scan {gigabytes(confidence_bytes)}")
        return 0

    if not surface:
        # No LLM_REQUEST rows means no requests, not a surface of nothing. Measuring that as
        # a fully covered surface would report `ok` on an agent nobody is using.
        print(f"no LLM_REQUEST rows for {args.agent} in the last {args.surface_days} days,")
        print("so the declared surface cannot be read. Nothing was recorded.")
        return 0

    now = drift.signals(surface, model.classes, confidences)
    baseline = model.metadata.drift_baseline
    verdict = drift.verdict(
        now,
        baseline,
        margin=args.margin,
        age_days=model.age_days,
        max_age_days=cfg.max_model_age_days,
    )

    print(f"==> {args.agent}, version {model.metadata.version}, {model.age_days:.0f} days old")
    print(f"    scanned {gigabytes(surface_bytes + confidence_bytes)}")
    report(now, baseline, verdict)

    due = DUE_EXIT_CODE if (args.exit_code and verdict.outcome == "due") else 0
    if args.no_record:
        return due
    metrics = now.as_dict()
    metrics["signal"] = verdict.signal
    if baseline is not None:
        metrics["unseen_share_baseline"] = baseline.unseen_share
    lifecycle.record(
        args.project,
        args.dataset,
        lifecycle.Step(
            cycle_id=args.cycle_id,
            step="drift",
            agent=args.agent,
            outcome=verdict.outcome,
            environment=cfg.environment,
            reason=verdict.reason,
            version=model.metadata.version,
            metrics=metrics,
            source="check_drift.py",
        ),
    )
    return due


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--bucket", **bucket_arg())
    parser.add_argument(
        "--cycle-id",
        default=lifecycle.new_cycle_id(),
        help="the cycle this check belongs to; one is minted if it is left out",
    )
    parser.add_argument(
        "--surface-days",
        type=int,
        default=1,
        help="how far back to read the declared tool list. One day is usually right: the "
        "surface is a property of the deployment, and this is the expensive read",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="how far back to read routing confidence, which is reported and never gates",
    )
    parser.add_argument("--max-sessions", type=int, default=20_000)
    parser.add_argument(
        "--margin",
        type=float,
        default=drift.DEFAULT_UNSEEN_SHARE_MARGIN,
        help="how far unseen share may drift above its own baseline before a retrain is due. "
        f"Defaults to {drift.DEFAULT_UNSEEN_SHARE_MARGIN}, which is a two-point fit; use "
        "--backfill to fit it on this agent instead",
    )
    parser.add_argument(
        "--backfill",
        type=int,
        default=0,
        help="replay this many days of surface history against a range of candidate margins, "
        "and record nothing",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="price the reads and exit without running them or recording anything",
    )
    parser.add_argument(
        "--no-record", action="store_true", help="print the reading without writing a ledger row"
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help=f"exit {DUE_EXIT_CODE} instead of 0 when a retrain is due, so a scheduler script "
        "can branch on it. Off by default, because a verdict is not a failure",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
