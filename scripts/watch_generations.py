#!/usr/bin/env python
"""Withdraw a published generation that turned out worse than not routing at all.

This is the guard that lets the retrain trigger be automatic. An automatic deploy without an
automatic rollback is the one ordering that can make an agent worse, so this is built first and
the scheduler comes after it (DESIGN.md 21.5, 21.9).

**The test is the generation against its own concurrent holdout arm**, not against the generation
before it. The previous generation ran on different traffic at a different time; the holdout arm
is running right now, on the same traffic, and is exactly "what this agent costs without the
plugin". That is the contract in 21.1 written as arithmetic. It reads `g1_later` rather than `g1`,
because turn one bills about three times a continuation and pruning pushes that one call the wrong
way, which on this project's own warehouse is a nineteen-point difference between the two figures.

**Withdrawing is one small reversible write.** A `withdrawn.json` in the version's directory makes
`model_store` stop counting it as published, `max(complete)` falls back to the generation before
it, and every serving process follows within `refresh_interval_s` with no redeploy. `--restore`
deletes the marker and puts the version back. Nothing is ever deleted.

**It records every time it looks**, including the times it does nothing. A watch that writes a row
only when it acts is indistinguishable from a watch that is not running.

Offline. It reads the warehouse and writes one small object, adds no serving dependency, and runs
nowhere near the hot path. Authentication is ADC only.

Usage:
    uv run scripts/watch_generations.py --agent support
    uv run scripts/watch_generations.py --agent support --dry-run   # judge, change nothing
    uv run scripts/watch_generations.py --agent support --restore 20260905T171738Z
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from google.cloud import bigquery, storage

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deployment import DEFAULT_DATASET, bucket_arg, project_arg  # noqa: E402

from adk_smart_router import generation_watch, lifecycle  # noqa: E402
from adk_smart_router.config import load_config  # noqa: E402
from adk_smart_router.model_store import WITHDRAWN_NAME  # noqa: E402

DEFAULT_AGENT = "support"

# The current generation and its two arms. `is_current` rather than the newest version, because
# what is serving is what can be withdrawn: taking an already-superseded version out of the bucket
# changes nothing and would spend the rate limit doing it.
CURRENT_SQL = """
SELECT
  version,
  routed_later_turns,
  routed_later_billed_per_turn,
  holdout_later_turns,
  holdout_later_billed_per_turn,
  g1,
  g1_later,
  live_hours
FROM `{project}.{dataset}.v_router_generations`
WHERE agent = @agent AND is_current
ORDER BY published_at DESC
LIMIT 1
"""

# How many times this agent has already been withdrawn inside the window. Counted from the ledger
# rather than from the bucket, because the bucket says which versions are out and the ledger says
# when they went out, and the rate limit is a question about time.
WITHDRAWALS_SQL = """
SELECT COUNT(*) AS n
FROM `{project}.{dataset}.router_lifecycle`
WHERE recorded_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
  AND recorded_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @hours HOUR)
  AND agent = @agent
  AND step = 'withdraw'
  AND outcome = 'ok'
"""


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset, gcs_bucket=args.bucket)
    bucket = storage.Client(project=args.project).bucket(cfg.gcs_bucket)

    if args.restore:
        return restore(bucket, args, cfg.environment)

    client = bigquery.Client(project=args.project)
    row = current_generation(client, args)
    if row is None:
        print(f"{args.agent} has no current published generation, so there is nothing to judge.")
        return 0

    arms = generation_watch.Arms(
        routed_turns=int(row["routed_later_turns"] or 0),
        routed_billed_per_turn=_maybe_float(row["routed_later_billed_per_turn"]),
        holdout_turns=int(row["holdout_later_turns"] or 0),
        holdout_billed_per_turn=_maybe_float(row["holdout_later_billed_per_turn"]),
    )
    verdict = generation_watch.verdict(
        arms,
        min_later_turns=args.min_later_turns,
        margin=args.margin,
        withdrawals_in_window=withdrawals_in_window(client, args),
        max_withdrawals=args.max_withdrawals,
    )

    report(args.agent, row, arms, verdict)

    if verdict.withdraw and not args.dry_run:
        withdraw(bucket, args.agent, row["version"], verdict, cfg.environment)

    if args.dry_run or args.no_record:
        return 0
    record(args, cfg.environment, row["version"], verdict)
    return 0


# --- Reading ----------------------------------------------------------------------


def current_generation(client: Any, args: argparse.Namespace) -> dict[str, Any] | None:  # noqa: ANN401
    """The generation serving right now, or None if the agent has never published one."""
    job = client.query(
        CURRENT_SQL.format(project=args.project, dataset=args.dataset),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("agent", "STRING", args.agent)]
        ),
    )
    rows = [dict(row) for row in job.result()]
    return rows[0] if rows else None


def withdrawals_in_window(client: Any, args: argparse.Namespace) -> int:  # noqa: ANN401
    """How many withdrawals this agent has already had in the window. Zero on a missing ledger.

    A missing table is the normal state on an agent that has never retrained, and reading it as
    an error would make the very first withdrawal impossible.
    """
    from google.api_core import exceptions as api

    job = client.query(
        WITHDRAWALS_SQL.format(project=args.project, dataset=args.dataset),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("agent", "STRING", args.agent),
                bigquery.ScalarQueryParameter("hours", "INT64", args.window_hours),
            ]
        ),
    )
    try:
        return int(next(iter(job.result()))["n"])
    except api.NotFound:
        return 0


# --- Acting -----------------------------------------------------------------------


def withdraw(
    bucket: Any,  # noqa: ANN401
    agent: str,
    version: str,
    verdict: generation_watch.Verdict,
    environment: str,
) -> None:
    """Write the marker. One small object, and the version stops being served.

    The marker holds why, not just that. It is the copy of the reason that travels with the
    artifact, so somebody looking at the bucket six months later does not have to find the ledger
    row to know what happened, and re-publishing over a withdrawn version cannot quietly erase it.
    """
    blob = bucket.blob(f"routers/{agent}/{version}/{WITHDRAWN_NAME}")
    blob.upload_from_string(
        json.dumps(
            {
                "agent": agent,
                "version": version,
                "environment": environment,
                "withdrawn_at": dt.datetime.now(dt.UTC).isoformat(),
                "reason": verdict.reason,
                "metrics": verdict.metrics,
                "source": "watch_generations.py",
            },
            indent=2,
            sort_keys=True,
        ),
        content_type="application/json",
    )
    print(f"\n==> withdrew {version}. Serving falls back on its next re-poll.")
    print(f"    to undo: uv run scripts/watch_generations.py --agent {agent} --restore {version}")


def restore(bucket: Any, args: argparse.Namespace, environment: str) -> int:  # noqa: ANN401
    """Delete the marker, putting the version back in the running.

    The reverse of `withdraw` and deliberately as cheap. A withdrawal that could only be undone by
    re-publishing would make the watch something people switch off rather than something they
    trust, which defeats the point of having it.
    """
    blob = bucket.blob(f"routers/{args.agent}/{args.restore}/{WITHDRAWN_NAME}")
    if not blob.exists():
        print(f"{args.agent} version {args.restore} is not withdrawn. Nothing to do.")
        return 0
    blob.delete()
    print(f"==> restored {args.restore}. Serving picks it up again on its next re-poll.")
    if args.no_record:
        return 0
    lifecycle.record(
        args.project,
        args.dataset,
        lifecycle.Step(
            cycle_id=args.cycle_id,
            step="withdraw",
            agent=args.agent,
            outcome="declined",
            environment=environment,
            reason=f"withdrawal marker on {args.restore} deleted by hand; the version is back.",
            version=args.restore,
            source="watch_generations.py --restore",
        ),
    )
    return 0


def record(
    args: argparse.Namespace,
    environment: str,
    version: str,
    verdict: generation_watch.Verdict,
) -> None:
    """One ledger row per look, whatever the verdict was."""
    lifecycle.record(
        args.project,
        args.dataset,
        lifecycle.Step(
            cycle_id=args.cycle_id,
            step="withdraw",
            agent=args.agent,
            outcome=verdict.outcome,
            environment=environment,
            reason=verdict.reason,
            version=version,
            metrics=dict(verdict.metrics),
            source="watch_generations.py",
        ),
    )


# --- Saying so --------------------------------------------------------------------


def report(
    agent: str,
    row: dict[str, Any],
    arms: generation_watch.Arms,
    verdict: generation_watch.Verdict,
) -> None:
    hours = row["live_hours"]
    print(f"==> {agent}, version {row['version']}, live {hours} hours")
    print(
        f"    routed    {_rate(arms.routed_billed_per_turn)} over {arms.routed_turns} later turns"
    )
    print(
        f"    holdout   {_rate(arms.holdout_billed_per_turn)} over {arms.holdout_turns} later turns"
    )
    print(f"    g1_later  {_pct(arms.g1_later)}   ·   g1 pooled {_pct(_maybe_float(row['g1']))}")
    print(f"\n    {verdict.outcome}: {verdict.reason}")


def _rate(value: float | None) -> str:
    return "  n/a  " if value is None else f"{value:7.1f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def _maybe_float(value: Any) -> float | None:  # noqa: ANN401
    return None if value is None else float(value)


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
        "--margin",
        type=float,
        default=generation_watch.DEFAULT_MARGIN,
        help="how much dearer per later turn than its own holdout arm a generation may be "
        f"before it is withdrawn. Defaults to {generation_watch.DEFAULT_MARGIN}, the same "
        "margin cache_watch uses",
    )
    parser.add_argument(
        "--min-later-turns",
        type=int,
        default=generation_watch.DEFAULT_MIN_LATER_TURNS,
        help="priced later turns each arm needs before the comparison is trusted. Defaults to "
        f"{generation_watch.DEFAULT_MIN_LATER_TURNS}, which is a stated prior and not a fit",
    )
    parser.add_argument(
        "--max-withdrawals",
        type=int,
        default=generation_watch.DEFAULT_MAX_WITHDRAWALS,
        help="withdrawals allowed per agent per window, so a bad measurement cannot walk an "
        "agent back through several generations before anyone reads the ledger",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=generation_watch.DEFAULT_WINDOW_HOURS,
        help="how far back --max-withdrawals counts",
    )
    parser.add_argument(
        "--restore",
        metavar="VERSION",
        help="delete this version's withdrawal marker and put it back in the running",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="judge and print, withdrawing nothing and recording nothing",
    )
    parser.add_argument("--no-record", action="store_true", help="act without writing a ledger row")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
