#!/usr/bin/env python
"""V2-2(a) — how high can the menu pool's ceiling go, and what does raising it cost?

DESIGN.md 19.9 found a ceiling nobody had priced. An oracle that hands every session the
best of the eight published menus recalls 0.9856, and gate criterion 1 asks for 0.98. A
perfect predictor would have half a point of headroom over the bar. That is not the
model's fault and no amount of retraining moves it, because the pool is chosen before the
model ever runs.

So this script asks the pool's own question: for a given menu size and count, what is the
best set recall any router could reach, and what does that ceiling cost in saving? The two
move against each other. Wider menus hold more of what a session wants and leave fewer
tokens behind, and the whole point of the router is the tokens left behind.

`menus.choose` does the choosing, so re-running this scores the code that is published.
Pools are fit on the training split and scored on the held-out one, because a pool fit and
scored on the same sessions measures memorisation.

Cost is two BigQuery reads and 25 token counts. No embedding calls and no model. ADC only.

Usage:
    uv run python scripts/tune_menus.py
    uv run python scripts/tune_menus.py --sizes 7 11 15 --counts 4 8 12 16
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from deployment import DEFAULT_DATASET, bucket_arg, project_arg  # noqa: E402
from measure_menus import declaration_tokens  # noqa: E402
from score_sets import (  # noqa: E402
    DEFAULT_AGENT,
    Arm,
    Session,
    fetch_sessions,
    menu_best,
    score,
)

from adk_smart_router.config import FRAMEWORK_TOOLS, load_config  # noqa: E402
from adk_smart_router.menus import choose  # noqa: E402
from adk_smart_router.model_store import ModelStore  # noqa: E402

Menu = frozenset[str]


def ceiling(
    name: str,
    sessions: Sequence[Session],
    pool: Sequence[Menu],
    live: Menu,
    protected: Menu,
    tokens: dict[str, int],
) -> Arm:
    """Score a pool at its oracle: every session gets the menu that suits it best.

    `menu_best` and `score` are score_sets' own, so this measures a pool under exactly the
    definitions DESIGN.md 19.4 gates on and the arithmetic that is already under test.
    """
    arm = Arm(name)
    for session in sessions:
        score(arm, session, menu_best(session, pool, live, protected), live, tokens)
    return arm


def report(arms: Sequence[tuple[Arm, float]], billed: float, sessions: int) -> None:
    print(f"\n{sessions} held-out sessions, {billed:,.0f} billed-equivalent tokens\n")
    print(
        f"  {'pool':<18}{'recall':>9}{'complete':>10}{'exact':>8}{'pred G1':>10}{'avg tools':>11}"
    )
    for arm, width in arms:
        print(
            f"  {arm.name:<18}"
            f"{arm.mean(arm.recall_sum, arm.recall_n):>9.4f}"
            f"{arm.complete / arm.sessions if arm.sessions else 0:>10.1%}"
            f"{arm.exact / arm.sessions if arm.sessions else 0:>8.1%}"
            f"{-arm.removed_billed / billed if billed else 0:>10.1%}"
            f"{width:>11.1f}"
        )
    print(
        "\n  Every row is an oracle: the ceiling of that pool, not a number a router reaches.\n"
        "  recall     mean share of a session's tools its best menu held.\n"
        "  pred G1    DESIGN.md 19.5 criterion 3, at that ceiling. Steady state only.\n"
        "  avg tools  mean menu size, which is what the recall is being bought with."
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset, gcs_bucket=args.bucket)

    tokens = asyncio.run(declaration_tokens())
    live = frozenset(tokens)
    protected = (FRAMEWORK_TOOLS | cfg.never_prune) & live
    print(f"{len(live)} live tools, {sum(tokens.values())} declaration tokens")

    everything = fetch_sessions(
        args.project, args.dataset, args.agent, cfg.embedding_dims, args.session_prefix, False
    )
    held_out = fetch_sessions(
        args.project, args.dataset, args.agent, cfg.embedding_dims, args.session_prefix, True
    )
    if not everything or not held_out:
        print(f"\nno sessions under prefix {args.session_prefix!r}.", file=sys.stderr)
        return 1

    eval_ids = {s.session_id for s in held_out}
    train = [s for s in everything if s.session_id not in eval_ids]
    # `choose` wants one sequence of calls per session. Under the set objective the call
    # order and the repeats carry no information, so a sorted set is the whole input.
    corpus = [sorted(s.tools) for s in train if s.tools]
    print(f"{len(train)} training sessions ({len(corpus)} of them called a tool), ", end="")
    print(f"{len(held_out)} held out")

    arms: list[tuple[Arm, float]] = []
    model = ModelStore(cfg).load(args.agent)
    if model is not None:
        published = list(model.menus)
        arms.append(
            (
                ceiling(
                    f"published ({len(published)})", held_out, published, live, protected, tokens
                ),
                width(published),
            )
        )
        print(f"published pool: {len(published)} menus, {width(published):.1f} tools each")

    for size in args.sizes:
        for count in args.counts:
            pool = list(choose(corpus, size=size, count=count))
            name = f"{len(pool)} x {size}"
            arms.append((ceiling(name, held_out, pool, live, protected, tokens), width(pool)))

    report(arms, sum(s.billed for s in held_out), len(held_out))
    print(
        f"\n  for reference, the full list is {sum(tokens.values())} tokens over {len(live)} tools"
    )
    print(f"  and a session's true set averages {mean_set(held_out):.1f} of them")
    return 0


def width(pool: Sequence[Menu]) -> float:
    return sum(len(menu) for menu in pool) / len(pool) if pool else 0.0


def mean_set(sessions: Sequence[Session]) -> float:
    return sum(len(s.tools) for s in sessions) / len(sessions) if sessions else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--bucket", **bucket_arg())
    parser.add_argument("--session-prefix", default="corpus")
    parser.add_argument("--sizes", type=int, nargs="+", default=[7, 11, 15])
    parser.add_argument("--counts", type=int, nargs="+", default=[4, 8, 12, 16])
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
