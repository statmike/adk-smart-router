#!/usr/bin/env python
"""E7 — how wide does a menu have to be, on traffic nobody here wrote?

`max_tools_kept` is 11. That number was fitted on our own corpus, which is the weakest
possible basis for it: we wrote the tools, we wrote the queries, and we picked the value that
looked best against both. `score_xlam.py` already showed the rule survives on somebody else's
data. It scored one width. This sweeps the width and reports the frontier.

There are two frontiers here and the comparison between them is the point.

**A fixed cap.** Keep the `N` best-supported tools, whatever the query. This is what the
plugin does when `keep_probability_mass` is loose, and it is the thing `max_tools_kept`
names. Its cost is flat and its recall is whatever that width buys on the hardest query.

**An adaptive width.** Keep tools until their support covers `p` of the total, however many
that takes. This is what the plugin does when `max_tools_kept` is loose, and it is the shape
the project actually wants: predict the right one or more tools without deciding in advance
how many that is. Its cost varies per query, so it is only comparable to a fixed cap at equal
*mean* width, which is how the verdict section reads it.

Both are scored on the same votes, so the tables can be read against each other line by line.
Nothing here is a new prediction rule; `votes` and the tie-break come from
`score_unseen_intent.py` and the surfaces from `score_xlam.py`, unchanged, so a difference
between this and those tables is a difference in width and nothing else.

**What this does not sweep.** The pooled arm. Menu pools have to be refitted for every width,
which is the slow part of `score_xlam.py`, and the question here is the width rather than the
pool. The pooled arm's number at width 11 stays where it is, in DESIGN.md 19.44.

Cost is nothing. The vectors are the ones `score_xlam.py --confirm` already paid for and
cached; if that cache is missing this script says so and stops rather than quietly spending.

Usage:
    uv run python scripts/sweep_menu_size.py
    uv run python scripts/sweep_menu_size.py --surfaces 100
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from score_unseen_intent import Case, Menu, Regime, normalise, recall, votes
from score_xlam import CORPUS_PATH, ID_PATH, VECTOR_PATH, load_corpus, say, surface_of

from adk_smart_router.menus import pad, popularity

# Matched to score_xlam.py so the two sets of tables are comparable.
NEIGHBOURS = 15
SURFACES = (25, 100, 400, 0)  # 0 means every tool
REGIMES = (
    Regime("block the row", lambda case: case.session_id),
    Regime("block the opening", lambda case: normalise(case.opening)),
)

# The widths to price. Dense where the shipped value lives, thin out past it: the interesting
# part of the curve is the knee, and nobody is going to declare fifty tools to save tokens.
SIZES = (1, 2, 3, 5, 8, 11, 15, 20, 25, 30, 40, 50)

# Support fractions for the adaptive arm. 1.0 keeps every tool that got a single weak vote,
# which is the uncapped ceiling and the honest asymptote of the whole exercise.
MASSES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)

# The recall a menu has to reach before the verdict section will call a width sufficient.
# Chosen to match the publish gate in DESIGN.md 19.5 rather than derived here.
TARGET_RECALL = 0.95

BLOCK = 512


@dataclass
class Arm:
    """One point on a frontier: what a width bought, averaged over every query."""

    label: str
    recall: float = 0.0
    no_miss: float = 0.0
    width: float = 0.0


def ordered(weighted: dict[str, float], ranked: Sequence[str]) -> list[tuple[str, float]]:
    """Tools by support, best first, with `voted_menu`'s exact tie-break.

    Repeated here as a sorted list rather than called through `voted_menu` because the sweep
    needs every prefix of this order and re-sorting per width would be the whole runtime. The
    key is copied character for character; the test pins the two against each other.
    """
    order = {tool: place for place, tool in enumerate(ranked)}
    return sorted(
        weighted.items(), key=lambda item: (-item[1], order.get(item[0], len(order)), item[0])
    )


def mass_prefix(support: Sequence[tuple[str, float]], mass: float) -> int:
    """How many tools it takes to cover `mass` of the total support.

    At least one whenever anything was predicted, because a menu of nothing is a passthrough
    and this arm is being priced as a menu.
    """
    total = sum(weight for _, weight in support)
    if total <= 0:
        return len(support)
    running = 0.0
    for taken, (_, weight) in enumerate(support, start=1):
        running += weight
        if running >= mass * total:
            return taken
    return len(support)


def sweep(
    cases: Sequence[Case],
    vectors: np.ndarray,
    regime: Regime,
    ranked: Sequence[str],
    neighbours: int,
) -> tuple[list[Arm], list[Arm]]:
    """Score every fixed width and every support fraction, on one pass over the votes."""
    keys = np.array([regime.key(case) for case in cases])
    tools = [case.tools for case in cases]
    fixed = [Arm(f"top {size}") for size in SIZES]
    adaptive = [Arm(f"mass {mass:g}") for mass in MASSES]

    say(f"sweeping {len(cases):,} cases, {regime.name}")
    for start in range(0, len(cases), BLOCK):
        if start and start % (BLOCK * 20) == 0:
            say(f"  {start:,} of {len(cases):,}")
        block = vectors[start : start + BLOCK] @ vectors.T
        for offset, similarity in enumerate(block):
            index = start + offset
            support = ordered(votes(similarity, keys == keys[index], tools, neighbours), ranked)
            wanted = tools[index]

            for arm, size in zip(fixed, SIZES, strict=True):
                menu = frozenset(tool for tool, _ in support[:size])
                arm.recall += recall(menu, wanted)
                arm.no_miss += float(wanted <= menu)
                arm.width += len(menu)

            for arm, mass in zip(adaptive, MASSES, strict=True):
                menu = frozenset(tool for tool, _ in support[: mass_prefix(support, mass)])
                arm.recall += recall(menu, wanted)
                arm.no_miss += float(wanted <= menu)
                arm.width += len(menu)

    n = len(cases)
    for arm in (*fixed, *adaptive):
        arm.recall /= n
        arm.no_miss /= n
        arm.width /= n
    return fixed, adaptive


def static_arms(ranked: Sequence[str], cases: Sequence[Case]) -> list[Arm]:
    """The popularity baseline at every width: one menu for everybody, no prediction."""
    arms = []
    for size in SIZES:
        menu = frozenset(pad([], ranked, size))
        scores = [recall(menu, case.tools) for case in cases]
        arms.append(
            Arm(
                f"top {size}",
                sum(scores) / len(cases),
                sum(1.0 for case in cases if case.tools <= menu) / len(cases),
                float(len(menu)),
            )
        )
    return arms


def interpolate(baseline: Sequence[Arm], width: float) -> float:
    """What the static baseline gets at this mean width, straight-lined between its points.

    The adaptive arm lands on fractional widths that no static menu has, so comparing it to
    the static number at the nearest integer would flatter whichever side rounded its way.
    A straight line between the two neighbouring static widths is the fairest cheap answer,
    and it is only ever used for the comparison column.
    """
    points = sorted((arm.width, arm.recall) for arm in baseline)
    if not points or width <= points[0][0]:
        return points[0][1] if points else 0.0
    for (left_w, left_r), (right_w, right_r) in zip(points, points[1:], strict=False):
        if width <= right_w:
            span = right_w - left_w
            return left_r if span == 0 else left_r + (right_r - left_r) * (width - left_w) / span
    return points[-1][1]


def report_frontier(title: str, arms: Sequence[Arm], tools: int, baseline: Sequence[Arm]) -> None:
    print(f"\n**{title}**\n")
    print("| arm | mean width | prune | recall | no miss | over static |")
    print("| --- | --- | --- | --- | --- | --- |")
    for arm in arms:
        lift = arm.recall - interpolate(baseline, arm.width)
        print(
            f"| {arm.label} | {arm.width:.1f} | {1 - arm.width / tools:.1%} |"
            f" {arm.recall:.4f} | {arm.no_miss:.4f} | {lift:+.4f} |"
        )


def report_static(arms: Sequence[Arm], tools: int) -> None:
    print("\n**static popularity, no prediction**\n")
    print("| arm | width | recall | no miss |")
    print("| --- | --- | --- | --- |")
    for arm in arms:
        print(f"| {arm.label} | {arm.width:.0f} | {arm.recall:.4f} | {arm.no_miss:.4f} |")


def first_reaching(arms: Sequence[Arm], target: float) -> Arm | None:
    """The narrowest arm that clears the recall bar, or nothing if none does."""
    return next((arm for arm in sorted(arms, key=lambda a: a.width) if arm.recall >= target), None)


def dominance(fixed: Sequence[Arm], adaptive: Sequence[Arm]) -> tuple[int, int]:
    """How many fixed caps the adaptive frontier beats at the fixed cap's own mean width.

    The narrowest-arm-clearing-the-bar comparison is one point on two curves and it can be
    won by whichever curve happened to land closer to the bar. This walks the whole frontier
    instead: for every fixed cap, what would an adaptive rule of the same average cost have
    got? Points where the two are within a thousandth count as neither, because a difference
    that small is corpus noise rather than a property of the rule.
    """
    ahead = behind = 0
    for arm in fixed:
        gap = interpolate(adaptive, arm.width) - arm.recall
        if abs(gap) < 1e-3:
            continue
        ahead += gap > 0
        behind += gap < 0
    return ahead, behind


def saturation_width(fixed: Sequence[Arm]) -> int:
    """The narrowest cap whose menu no wider cap changes.

    Past this point the cap is inert: the predictor is not proposing enough tools to fill it,
    so raising it declares nothing new and lowering it is the only thing that does anything.
    Measured on width rather than recall, since a cap that stops growing has stopped binding
    whether or not the recall it bought was any good.

    The widest cap swept always matches itself, so this always answers. Whether that answer
    means anything is a separate question the caller asks by checking if recall was still
    climbing at that width.
    """
    widest = fixed[-1].width
    for arm, size in zip(fixed, SIZES, strict=True):
        if abs(arm.width - widest) < 5e-2:
            return size
    return SIZES[-1]


def report_verdict(width: int, fixed: Sequence[Arm], adaptive: Sequence[Arm]) -> None:
    label = "every tool" if width == 0 else f"top {width}"
    cheapest_fixed = first_reaching(fixed, TARGET_RECALL)
    cheapest_adaptive = first_reaching(adaptive, TARGET_RECALL)

    print(f"\n**verdict, surface {label}**\n")
    if cheapest_fixed is None and cheapest_adaptive is None:
        print(f"  Neither shape reaches {TARGET_RECALL:.2f} recall at any width swept here.")
        print(f"  The widest fixed cap tried was {max(SIZES)} and it got {fixed[-1].recall:.4f}.")
        return
    for name, arm in (("fixed cap", cheapest_fixed), ("adaptive", cheapest_adaptive)):
        if arm is None:
            print(f"  {name:<10} never reaches {TARGET_RECALL:.2f} at the widths swept.")
        else:
            print(
                f"  {name:<10} {arm.label:<10} mean width {arm.width:5.1f}, recall {arm.recall:.4f}"
            )
    if cheapest_fixed is not None and cheapest_adaptive is not None:
        saved = cheapest_fixed.width - cheapest_adaptive.width
        better = "adaptive" if saved > 0 else "the fixed cap"
        print(f"\n  At equal recall, {better} is narrower by {abs(saved):.1f} tools.")

    ahead, behind = dominance(fixed, adaptive)
    print("\n  Read the other way, at each fixed cap's own mean width the adaptive frontier is")
    print(f"  ahead on {ahead} of {ahead + behind} points and behind on {behind}.")

    saturated = saturation_width(fixed)
    still_climbing = fixed[-1].recall - fixed[-2].recall > 1e-3
    if still_climbing:
        print(f"  The cap was still binding at {SIZES[-1]}, the widest tried, so the frontier")
        print("  has not been walked to its end here.")
    else:
        print(f"  The cap stops binding at {saturated}: every wider cap keeps the same tools.")


def load_vectors() -> tuple[list[str], np.ndarray]:
    if not (VECTOR_PATH.exists() and ID_PATH.exists()):
        raise SystemExit(
            f"{VECTOR_PATH} is missing. This sweep reuses the vectors score_xlam.py cached;\n"
            "run `uv run python scripts/score_xlam.py --confirm` first if you mean to spend."
        )
    return json.loads(ID_PATH.read_text()), np.load(VECTOR_PATH)


def main() -> int:
    args = parse_args()
    queries = load_corpus(CORPUS_PATH)
    ids, vectors = load_vectors()

    by_id = {q.row_id: q for q in queries}
    embedded = [by_id[i] for i in ids if i in by_id]
    lookup = {q.row_id: n for n, q in enumerate(embedded)}
    vectors = vectors[[n for n, i in enumerate(ids) if i in by_id]]
    print(f"xlam-function-calling-60k: {len(embedded):,} embedded queries")
    print(f"widths {SIZES}")
    print(f"support fractions {MASSES}")

    for width in args.surfaces:
        surface: Menu = surface_of(embedded, width)
        keep = [q for q in embedded if q.tools and q.tools <= surface]
        if len(keep) < NEIGHBOURS * 2:
            print(f"\nskipping the top-{width} surface: only {len(keep)} queries fit it.")
            continue

        cases = [Case(q.row_id, q.tools, q.text, "", "") for q in keep]
        subset = vectors[[lookup[q.row_id] for q in keep]]
        ranked = popularity([sorted(case.tools) for case in cases])
        label = "every tool" if width == 0 else f"top {width}"
        print(f"\n### surface: {label} ({len(surface):,} tools, {len(cases):,} queries)")
        static = static_arms(ranked, cases)
        report_static(static, len(surface))

        for regime in REGIMES:
            fixed, adaptive = sweep(cases, subset, regime, ranked, args.neighbours)
            report_frontier(f"fixed cap, {regime.name}", fixed, len(surface), static)
            report_frontier(f"adaptive width, {regime.name}", adaptive, len(surface), static)
            report_verdict(width, fixed, adaptive)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument(
        "--surfaces",
        type=int,
        nargs="+",
        default=list(SURFACES),
        help="surface widths to sweep; 0 means every tool",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
