#!/usr/bin/env python
"""Write one row into the retrain ledger, from a shell.

`train` runs as a BigQuery stored procedure and `deploy` is whatever an adopter uses to put
a new artifact in front of traffic. Neither is a Python script this project owns, so neither
can call `lifecycle.record` the way tuning and publishing do. This is how they say what they
did (DESIGN.md 20.3).

It always exits 0. A ledger that can fail a retrain, or a deploy, is a worse trade than a gap
in a chart, so a row that does not land is reported on stderr and forgiven. Authentication is
ADC only.

Usage:
    uv run scripts/record_step.py --step train --outcome ok --run-id "$after" \
        --cycle-id "$CYCLE_ID" --reason 'the gate passed'
    uv run scripts/record_step.py --step deploy --outcome ok --version 20260908T110000Z \
        --reason 'rolled out to prod'
    uv run scripts/record_step.py --ensure-table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402

from adk_smart_router import lifecycle  # noqa: E402

DEFAULT_AGENT = "support"


def parse_metrics(raw: str | None) -> dict[str, Any] | None:
    """Read `--metrics`, or None. A malformed value is dropped rather than fatal.

    The caller is a shell script partway through a cycle. Refusing the whole row over a
    quoting mistake would lose the outcome as well, which is the part worth keeping.
    """
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        print(f"warning: --metrics is not JSON, dropping it: {raw!r}", file=sys.stderr)
        return None
    if not isinstance(value, dict):
        print(f"warning: --metrics is not a JSON object, dropping it: {raw!r}", file=sys.stderr)
        return None
    return value


def main() -> int:
    args = parse_args()
    if args.ensure_table:
        # `dashboard.sql` will not deploy while `router_lifecycle` is missing, because BigQuery
        # resolves a view's references when it creates the view. Every other way of creating the
        # table writes a row into it, and an adopter who has not retrained yet has no row to
        # write. This is the way that does not invent one.
        lifecycle.ensure_table(args.project, args.dataset)
        print(f"{args.project}.{args.dataset}.{lifecycle.TABLE} is ready.")
        return 0
    if not args.step or not args.outcome:
        print("error: --step and --outcome are required", file=sys.stderr)
        return 2
    step = lifecycle.Step(
        cycle_id=args.cycle_id or lifecycle.new_cycle_id(),
        step=args.step,
        agent=args.agent,
        outcome=args.outcome,
        environment=args.environment,
        reason=args.reason,
        run_id=args.run_id,
        version=args.version,
        metrics=parse_metrics(args.metrics),
        source=args.source,
    )
    if not lifecycle.record(args.project, args.dataset, step):
        print("warning: the ledger row was not recorded; see the log above", file=sys.stderr)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--environment", default="prod")
    # Not `required`, because `--ensure-table` is a mode that records nothing. main() checks.
    parser.add_argument("--step", default=None, choices=lifecycle.STEPS)
    parser.add_argument("--outcome", default=None, choices=lifecycle.OUTCOMES)
    parser.add_argument(
        "--ensure-table",
        action="store_true",
        help="create the ledger table if it is missing, write no row, and exit",
    )
    parser.add_argument(
        "--cycle-id",
        default=None,
        help="the cycle this step belongs to; one is minted if it is left out",
    )
    parser.add_argument("--reason", default=None, help="why this step ended the way it did")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--metrics", default=None, help="a JSON object of numbers to keep")
    parser.add_argument(
        "--source",
        default="record_step.py",
        help="what wrote this row, when something other than this script is calling it",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
