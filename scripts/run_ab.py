#!/usr/bin/env python
"""Build-order task 11 — drive live A/B traffic through the reference agent.

Everything measured so far is offline. `backtest.sql` replays history, and
`check_router.py` scores the published artifact against held-out rows. Neither
one changes a request, so neither can say what the router does to the bill.

This runs the reference agent with the router installed and the rollout split
turned on, so two arms of traffic go through the same process at the same time:

  * **routed** — the tool menu is pruned once per session.
  * **holdout** — the router records that it saw the session and returns without
    touching the request. The permanent control arm (DESIGN.md 11.4).

The split is the plugin's own, not this script's. `rollout` hashes
(app_name, user_id, session_id) under a holdout salt, so which arm a session
lands in is decided inside the router and recorded in its provenance. This script
never assigns an arm, which is why it cannot get the assignment wrong.

Measurement is a separate script. `report_ab.py` reads the arms back out of
BigQuery and computes G1. Nothing here computes a metric.

Concurrent or sequential
------------------------
The split above is concurrent, and for prefix-cache economics that is not a fair
test. Both arms share one process and one project, so the control arm's single
unpruned prefix is warmed by every holdout session *and* by every session the
router declined, while the routed arm's several menus are warmed only by their
own share of the traffic. The control gets warmth the treatment paid for.

For a fair test, run two periods instead:

    ... --seed 42 --label 42a --route-fraction 1 --holdout-fraction 0 --no-cache-watch
    ... --seed 42 --label 42b --route-fraction 0

Same seed, so both periods send the same conversations; different labels, so the
two populations sit under different session prefixes. `report_ab.py --baseline-prefix`
compares them (DESIGN.md 18.23).

What is and is not being measured
---------------------------------
`cacheable_tools` is empty in the reference agent's configuration, so the tool
result cache is inert in both arms. This A/B isolates the effect of pruning the
tool menu, which is what G1 is defined on.

Cost and restartability
-----------------------
The default 200 sessions of about 3 turns is roughly 600 turns, a few dollars of
Flash. The script refuses to start without `--confirm`, prints a running
estimate, and journals each finished session so a restart skips what was already
paid for. Authentication is ADC only.

Usage:
    uv run python scripts/run_ab.py --dry-run
    uv run python scripts/run_ab.py --confirm --limit 4     # pilot, then check BigQuery
    uv run python scripts/run_ab.py --confirm
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# The customer simulator, the scenario blindness checks and the journal all come
# from the corpus generator rather than being written twice. This is the same
# traffic, driven the same way; the only difference is that the router is on.
from generate_corpus import (  # noqa: E402
    CUSTOMER_MODEL,
    KNOWN_TOOL_NAMES,
    PROJECT,
    SCENARIO_DIR,
    VERTEX_LOCATION,
    LeakError,
    SessionPlan,
    Spend,
    append_journal,
    assert_scenarios_are_blind,
    build_plan,
    estimate_usd,
    live_tool_names,
    load_json,
    load_root_agent,
    read_journal,
    run_session,
)

DEFAULT_JOURNAL_DIR = REPO_ROOT / ".ab_runs"

SESSIONS = int(os.environ.get("SMART_ROUTER_AB_SESSIONS", "200"))
SEED = int(os.environ.get("SMART_ROUTER_AB_SEED", "20260901"))
CONCURRENCY = int(os.environ.get("SMART_ROUTER_AB_CONCURRENCY", "4"))

# Half and half. The production default is 0.05, because there a holdout is a
# permanent tax on a system you believe works. Here the holdout *is* the
# experiment, and an even split is the cheapest way to a given confidence.
HOLDOUT_FRACTION = float(os.environ.get("SMART_ROUTER_AB_HOLDOUT", "0.5"))


def session_prefix(label: str) -> str:
    """`ab-`, never `corpus-`. The two populations must not mix in a query.

    Corpus sessions were generated with the router uninstalled. A query that
    swept both prefixes would silently pool unrouted history into the control
    arm, which is the one mistake that would make the whole measurement lie.
    """
    return f"ab-{label}-"


def plan_for(seed: int, sessions: int, label: str) -> list[SessionPlan]:
    """The corpus generator's plan, re-labelled into the `ab-` namespace.

    `seed` picks the conversations and `label` names the run. They are separate
    so a sequential A/B can send the *same* traffic through two runs with two
    prefixes: `--seed 42 --label 42a` then `--seed 42 --label 42b`. Same personas,
    same scenarios, same turn counts, two populations BigQuery can tell apart.
    """
    planned = build_plan(
        load_json(SCENARIO_DIR / "personas.json")["personas"],
        load_json(SCENARIO_DIR / "scenarios.json")["scenarios"],
        sessions,
        seed,
    )
    return [
        dataclasses.replace(p, session_id=f"{session_prefix(label)}{p.index:05d}") for p in planned
    ]


def journal_path(explicit: str | None, label: str) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return DEFAULT_JOURNAL_DIR / f"seed-{label}.jsonl"


async def run_all(
    plan: Sequence[SessionPlan],
    args: argparse.Namespace,
    tool_names: frozenset[str],
    journal: Path,
    spend: Spend,
) -> None:
    """Drive the planned sessions with the router installed, journalling as they land."""
    from google import genai
    from google.adk.runners import InMemoryRunner
    from reference_agent.run_agent import build_app, warmup_router

    app = build_app(
        analytics=True,
        route_fraction=args.route_fraction,
        holdout_fraction=args.holdout_fraction,
        cache_watch=not args.no_cache_watch,
    )
    runner = InMemoryRunner(app=app)
    client = genai.Client(vertexai=True, project=PROJECT, location=VERTEX_LOCATION)

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    total = len(plan)

    async def one(item: SessionPlan) -> None:
        async with semaphore:
            try:
                record = await run_session(
                    item, runner, client, args.customer_model, tool_names, spend
                )
            except LeakError:
                raise
            except Exception as exc:  # one bad session must not end the run
                spend.errors += 1
                print(f"  session {item.session_id} failed: {type(exc).__name__}: {exc}")
                return
            async with write_lock:
                append_journal(journal, record)
                done = spend.sessions
                if done % 25 == 0 or done == total:
                    print(
                        f"  {done}/{total} sessions, {spend.turns} turns, ~${spend.usd:.2f} spent"
                    )

    try:
        # Before any traffic, and once for the whole run. A cold embedding client
        # overruns the hot-path budget, and one embedding failure pins a session to
        # passthrough — which here would look like a routed session behaving like a
        # control (DESIGN.md 18.10).
        await warmup_router(app)
        await asyncio.gather(*(one(p) for p in plan))
    finally:
        # A run that loses a fifth of its sessions to embedding failures still finishes,
        # and until now it finished quietly: the counters existed but nothing read them,
        # so the loss was only visible days later in BigQuery. Say it at the console
        # while the operator is still watching.
        # Duck-typed off the app for the same reason `warmup_router` is: this script
        # must keep working against an app with no router in it.
        for plugin in app.plugins or ():
            stats = getattr(plugin, "embedder_stats", None)
            if stats is not None:
                print(f"  embedder: {stats}")
                print(f"  fail-open: {plugin.failopen_counts}")
        # BQAA only flushes its buffered rows on close.
        await runner.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", type=int, default=SESSIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--label", default=None, help="run name for the session prefix (default: the seed)"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap sessions run this invocation")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--route-fraction", type=float, default=1.0)
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument(
        "--no-cache-watch",
        action="store_true",
        help=(
            "disable cache_watch for this run. Required with --holdout-fraction 0: "
            "the watch compares pruned calls against unpruned ones, and with no holdout "
            "its unpruned arm is whatever the router happened to decline"
        ),
    )
    parser.add_argument("--customer-model", default=CUSTOMER_MODEL)
    parser.add_argument("--journal", default=None, help="resume file (default .ab_runs/)")
    parser.add_argument("--dry-run", action="store_true", help="plan and price it, call nothing")
    parser.add_argument("--confirm", action="store_true", help="required: this spends real money")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    root_agent = load_root_agent(required=not args.dry_run)
    personas = load_json(SCENARIO_DIR / "personas.json")["personas"]
    scenarios = load_json(SCENARIO_DIR / "scenarios.json")["scenarios"]
    tool_names = KNOWN_TOOL_NAMES | (live_tool_names(root_agent) if root_agent else frozenset())
    assert_scenarios_are_blind(personas, scenarios, tool_names)

    label = args.label or str(args.seed)
    journal = journal_path(args.journal, label)
    already = {record["session_id"] for record in read_journal(journal)}
    plan = [p for p in plan_for(args.seed, args.sessions, label) if p.session_id not in already]
    if args.limit is not None:
        plan = plan[: args.limit]

    turns = sum(p.turns for p in plan)
    print(f"A/B run, prefix {session_prefix(label)}")
    print(f"  route_fraction   {args.route_fraction}")
    print(f"  holdout_fraction {args.holdout_fraction}")
    print(f"  cache_watch      {'off' if args.no_cache_watch else 'on'}")
    print(f"  {len(plan)} sessions to run ({len(already)} already journalled), ~{turns} turns")
    print(f"  estimated cost   ~${estimate_usd(turns):.2f} (an upper bound: routing spends less)")
    print(f"  journal          {journal}")

    if args.dry_run:
        return 0
    if not plan:
        print("nothing to run.")
        return 0
    if not args.confirm:
        print("\nrefusing to spend without --confirm.")
        return 2

    spend = Spend()
    asyncio.run(run_all(plan, args, tool_names, journal, spend))
    print(
        f"\ndone: {spend.sessions} sessions, {spend.turns} turns, "
        f"{spend.errors} failed, ~${spend.usd:.2f} spent"
    )
    # `--planned` carries the size of the whole plan, not of this invocation. A run whose
    # tail never started — killed, or resumed and never finished — leaves no rows for those
    # sessions, so BigQuery cannot infer the plan and the report would quietly call a
    # truncated run complete. This script is the only thing that knows the true number.
    print("measure it with:")
    print(
        f"  uv run python scripts/report_ab.py --prefix {session_prefix(label)} "
        f"--planned {args.sessions}"
    )
    return 1 if spend.sessions == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
