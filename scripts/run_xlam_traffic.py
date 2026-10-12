#!/usr/bin/env python
"""E1 — drive xLAM's own queries through the xLAM agent, live.

`build_xlam_agent.py` produced an agent whose 100 tool declarations are somebody else's
words and a file of 15,786 queries that fall entirely on that surface. This sends them.

It does two jobs, because they are the same job with the router turned off or on:

  * **corpus pass** (`--route-fraction 0`, the default) writes the training data. The
    router must be off here. Routing while generating labels would prune the menu Gemini
    chooses from, so the labels would record the router's guesses and the next model would
    train on its own output (DESIGN.md 3.3).
  * **A/B pass** (`--route-fraction 1`) sends the same shaped traffic with the router
    installed and the plugin's own holdout split on. `report_ab.py --agent xlam` reads the
    arms back out and computes G1. Nothing here computes a metric.

Why sessions are one query long
-------------------------------
Because xLAM rows are one query long. There is no conversation in the dataset and
inventing follow-ups would put our words back into the traffic, which is the one thing E1
exists to avoid. A session is therefore a real user message, the model's tool call, and
the model reading the result: two or three model calls, only the first of which opens the
session. That is enough for the cache watch and for G1, both of which separate opening
calls from later ones anyway (DESIGN.md 19.28).

It is not enough for a multi-turn saving, and nobody should read one off this run. What it
measures is whether a menu predicted from a real first message covers what that message
turns out to need, priced on a real bill.

Which queries, and why the two passes cannot share any
------------------------------------------------------
The 15,786 queries are split once, by a stable hash of the xLAM row id, into a `train`
pool and an `eval` pool. The corpus pass draws from `train` and the A/B draws from `eval`,
so no query the router was trained on can turn up in the run that grades it. The split is
on the row id rather than on the draw, because two seeded samples of different sizes from
the same list overlap by chance, and a few dozen memorised queries in a 600-session A/B is
exactly the kind of contamination that reads as a good result.

Within a pool, sampling is a seeded RNG without replacement. Without replacement because a
duplicate query would warm its own prefix and flatter whichever arm drew it twice. Seeded
so that a rerun sends the same traffic and two A/B passes can be read against each other.

Cost and restartability
-----------------------
Roughly 2.5 model calls a session. The script refuses to start without `--confirm`, prints
a running estimate, and journals each finished session so a restart skips what was already
paid for. Authentication is ADC only.

Usage:
    uv run python scripts/run_xlam_traffic.py --dry-run
    uv run python scripts/run_xlam_traffic.py --confirm --limit 4 --label pilot
    uv run python scripts/run_xlam_traffic.py --confirm --sessions 2000 --label corpus
    uv run python scripts/run_xlam_traffic.py --confirm --sessions 600 --label ab1 \
        --pool eval --route-fraction 1
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

if TYPE_CHECKING:
    from google.adk.runners import InMemoryRunner

# The journal, the spend counter and the price table come from the corpus generator rather
# than being written a second time. Same accounting, same restart semantics.
from generate_corpus import (  # noqa: E402
    PROJECT,
    VERTEX_LOCATION,
    Spend,
    append_journal,
    estimate_usd,
    read_journal,
)

QUERIES = REPO_ROOT / "examples" / "xlam_agent" / "queries.jsonl"
DEFAULT_JOURNAL_DIR = REPO_ROOT / ".ab_runs"

SESSIONS = int(os.environ.get("XLAM_SESSIONS", "600"))
SEED = int(os.environ.get("XLAM_SEED", "20260906"))
CONCURRENCY = int(os.environ.get("XLAM_CONCURRENCY", "4"))
HOLDOUT_FRACTION = float(os.environ.get("XLAM_HOLDOUT", "0.5"))

# A single query rarely needs more than a handful of calls, and a session that keeps going
# past this is a loop rather than a conversation. It bounds the bill per session.
MAX_CALLS = 8

# One row in five is held out of training. A fifth of 15,786 is about 3,150 queries, which
# is five A/B runs of the size we expect to need before any of them repeats a query.
EVAL_SHARE = 5


@dataclass(frozen=True)
class Session:
    """One planned session: one xLAM query, sent once."""

    index: int
    session_id: str
    user_id: str
    row_id: str
    query: str
    tools: tuple[str, ...]  # what xLAM says answers it, carried for the journal only


def load_queries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `scripts/build_xlam_agent.py --fetch` first.")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def session_prefix(label: str) -> str:
    """`xlam-`, never `ab-` or `corpus-`.

    A separate namespace from the reference agent's, so no query can ever pool the two
    fixtures into one population. The agent name in BigQuery differs too, and either alone
    would be enough; both is cheap.
    """
    return f"xlam-{label}-"


def pool_of(row: Mapping[str, object]) -> str:
    """Which side of the train/eval split a query falls on, forever.

    SHA-256 of the row id rather than `hash()`, which is salted per process and would put
    the same query in the training pool one day and the evaluation pool the next.
    """
    digest = hashlib.sha256(str(row["id"]).encode()).hexdigest()
    return "eval" if int(digest[:8], 16) % EVAL_SHARE == 0 else "train"


def plan_for(
    rows: Sequence[dict[str, object]], sessions: int, seed: int, label: str, pool: str
) -> list[Session]:
    """Sample `sessions` queries from one pool, without replacement, deterministically.

    The user id varies with the index because the rollout hash reads
    (app_name, user_id, session_id). One user id for the whole run would still split, but
    it would split on session id alone, which is a thinner hash than the one that ships.
    """
    rng = random.Random(seed)
    eligible = [row for row in rows if pool_of(row) == pool]
    chosen = rng.sample(eligible, min(sessions, len(eligible)))
    prefix = session_prefix(label)
    return [
        Session(
            index=i,
            session_id=f"{prefix}{i:05d}",
            user_id=f"xlam-user-{i % 97:03d}",
            row_id=str(row["id"]),
            query=str(row["query"]),
            tools=tuple(row["tools"]),  # type: ignore[arg-type]
        )
        for i, row in enumerate(chosen)
    ]


def journal_path(explicit: str | None, label: str) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return DEFAULT_JOURNAL_DIR / f"xlam-{label}.jsonl"


async def run_session(item: Session, runner: InMemoryRunner, spend: Spend) -> dict[str, object]:
    """Send one query and record what the agent did with it.

    The record is for restart and for eyeballing, not for measurement. Everything the
    analysis reads is written to BigQuery by the analytics plugin, from inside the process,
    which is the only account of the run that cannot drift from what was billed.
    """
    from google.genai import types

    await runner.session_service.create_session(
        app_name="xlam", user_id=item.user_id, session_id=item.session_id
    )
    message = types.Content(role="user", parts=[types.Part(text=item.query)])
    called: list[str] = []
    answer = ""
    async for event in runner.run_async(
        user_id=item.user_id, session_id=item.session_id, new_message=message
    ):
        spend.add_usage(getattr(event, "usage_metadata", None))
        for call in event.get_function_calls():
            called.append(call.name)
        if event.is_final_response() and event.content and event.content.parts:
            answer = "".join(part.text or "" for part in event.content.parts).strip()
        if len(called) > MAX_CALLS:
            break

    spend.turns += 1
    spend.sessions += 1
    return {
        "session_id": item.session_id,
        "row_id": item.row_id,
        "query": item.query,
        "wanted": list(item.tools),
        "called": called,
        "answer": answer[:400],
    }


async def run_all(
    plan: Sequence[Session], args: argparse.Namespace, journal: Path, spend: Spend
) -> None:
    """Drive the planned sessions, journalling as they land."""
    from google.adk.runners import InMemoryRunner
    from xlam_agent.run_agent import build_app, warmup_router

    app = build_app(
        analytics=True,
        route_fraction=args.route_fraction,
        holdout_fraction=args.holdout_fraction,
        cache_watch=not args.no_cache_watch,
    )
    runner = InMemoryRunner(app=app)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    total = len(plan)

    async def one(item: Session) -> None:
        async with semaphore:
            try:
                record = await run_session(item, runner, spend)
            except Exception as exc:  # one bad session must not end the run
                spend.errors += 1
                print(f"  session {item.session_id} failed: {type(exc).__name__}: {exc}")
                return
            async with write_lock:
                append_journal(journal, record)
                done = spend.sessions
                if done % 25 == 0 or done == total:
                    print(f"  {done}/{total} sessions, ~${spend.usd:.2f} spent")

    try:
        # Once, before any traffic. A cold embedding client overruns the hot-path budget,
        # and one embedding failure pins a session to passthrough, which would look like a
        # routed session behaving like a control (DESIGN.md 18.10).
        await warmup_router(app)
        await asyncio.gather(*(one(item) for item in plan))
    finally:
        for plugin in app.plugins or ():
            stats = getattr(plugin, "embedder_stats", None)
            if stats is not None:
                print(f"  embedder: {stats}")
                print(f"  fail-open: {plugin.failopen_counts}")
        # The analytics plugin only flushes its buffered rows on close.
        await runner.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", type=int, default=SESSIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--label", default=None, help="run name for the session prefix")
    parser.add_argument("--limit", type=int, default=None, help="cap sessions run this invocation")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument(
        "--pool",
        choices=("train", "eval"),
        default="train",
        help="train for a corpus pass, eval for an A/B. They share no queries.",
    )
    parser.add_argument("--route-fraction", type=float, default=0.0)
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument(
        "--no-cache-watch",
        action="store_true",
        help="disable cache_watch for this run, as a sequential A/B with no holdout must",
    )
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--journal", default=None, help="resume file (default .ab_runs/)")
    parser.add_argument("--dry-run", action="store_true", help="plan and price it, call nothing")
    parser.add_argument("--confirm", action="store_true", help="required: this spends real money")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    label = args.label or str(args.seed)
    journal = journal_path(args.journal, label)
    already = {record["session_id"] for record in read_journal(journal)}

    rows = load_queries(args.queries)
    planned = plan_for(rows, args.sessions, args.seed, label, args.pool)
    plan = [s for s in planned if s.session_id not in already]
    if args.limit is not None:
        plan = plan[: args.limit]

    # Two to three model calls a session: the one that decides to call a tool, the one that
    # reads the result, and sometimes a third when the query needs two tools.
    turns = int(len(plan) * 2.5)
    print(f"xLAM traffic, prefix {session_prefix(label)}, agent xlam")
    print(f"  pool             {args.pool}")
    print(f"  route_fraction   {args.route_fraction}")
    print(f"  holdout_fraction {args.holdout_fraction}")
    print(f"  cache_watch      {'off' if args.no_cache_watch else 'on'}")
    print(f"  {len(plan)} sessions to run ({len(already)} already journalled), ~{turns} calls")
    print(f"  estimated cost   ~${estimate_usd(turns):.2f} (an upper bound: routing spends less)")
    print(f"  project {PROJECT}, {VERTEX_LOCATION}, journal {journal}")

    if args.dry_run or not plan:
        return 0
    if not args.confirm:
        print("\nRefusing to run without --confirm. This spends real money.")
        return 1

    spend = Spend()
    asyncio.run(run_all(plan, args, journal, spend))
    print(f"\n{spend.sessions} sessions, {spend.errors} failed, ~${spend.usd:.2f} spent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
