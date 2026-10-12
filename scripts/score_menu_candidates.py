#!/usr/bin/env python
"""Does merging menu candidates beat padding them, once cache warmth is priced?

`DESIGN.md` 19.42 measured half of this and stopped, deliberately. On a wide flat surface
the menu pool collapses — 0.2390 oracle ceiling on xLAM's full 3,179 tools — and the cause
is `menus.pad`: it fills every candidate's spare slots from the same global popularity
ranking, so 24 menus of 11 hold 91 distinct tools between their 264 slots. Building
candidates by merging co-occurring sessions instead raised the ceiling to 0.3967 at every
tool and to 0.7830 at 400. That measurement was in-sample, from a throwaway clustering, and
it priced coverage alone.

Coverage alone is the wrong question, and that is why nothing was changed. A menu pool does
not exist to cover sessions; a bespoke set covers better than any pool and 18.16 measured it
losing money. The pool exists so that a handful of prefixes are each sent often enough to
stay in the model's implicit cache. A pool anchored on the popular head is a pool whose
prefixes are reused. Spreading it out is the bespoke-set failure again at a coarser grain,
and no amount of recall makes up for a prefix that opens cold every time.

So this script scores both together, on the same corpus, out of fold:

    coverage      set recall of the menu each query actually snaps to, via `menus.rank`
    reuse         how the corpus's traffic distributes across the pool's menus
    warmth        what that distribution is worth against the measured residency curve
    billed        the two combined, in billed-equivalent input tokens per session

The last column is the answer. It is the same quantity `predict_saving.py` predicts and the
same one the G1 gate measures, so a win here is a win in the units the project already
commits to.

What is assumed, and why it is stated rather than hidden
--------------------------------------------------------
xLAM is a corpus of queries with no timestamps, so arrival density has to be supplied rather
than measured. It is swept over four rates spanning two orders of magnitude, because the
whole trade depends on it: at a high enough rate every menu stays warm and coverage wins
outright, and at a low enough rate nothing stays warm and the pool's shape stops mattering.
Where the crossover sits is the finding.

Gaps between sends of one menu are taken as exponential at that menu's own rate, and the
residency curve is integrated against that distribution rather than evaluated at the mean.
Evaluating at the mean is wrong in a direction that flatters the spread-out pool, because
the curve is concave in log gap.

Declaration cost is 143 tokens per tool, the reference agent's measured mean (14,324 over
100). xLAM ships no declaration text, so any per-tool figure is borrowed; this one at least
comes from a surface that was measured rather than guessed.

Cost and credentials
--------------------
No embedding spend. The 60,000 vectors `score_xlam.py` paid for are read from the cached
`.npy`, or from BigQuery if the local copy is gone. Application Default Credentials.

Usage:
    uv run python scripts/score_menu_candidates.py                  # every surface
    uv run python scripts/score_menu_candidates.py --surfaces 400   # one of them
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from deployment import DEFAULT_DATASET, project_arg  # noqa: E402
from predict_saving import CACHED_RATE, MEASURED_CURVE, ResidencyCurve  # noqa: E402
from score_unseen_intent import (  # noqa: E402
    NEIGHBOURS,
    Case,
    Menu,
    Regime,
    Vote,
    fold_of,
    recall,
    votes,
)
from score_xlam import (  # noqa: E402
    BLOCK,
    CORPUS_PATH,
    MENU_COUNT,
    POOL_FOLDS,
    SURFACES,
    VECTOR_PATH,
    embed,
    load_corpus,
    say,
    surface_of,
)

from adk_smart_router.config import load_config  # noqa: E402
from adk_smart_router.menus import bucket, choose, pad, popularity, rank  # noqa: E402

# The reference agent's measured mean, 14,324 declaration tokens over 100 tools. Borrowed,
# because xLAM ships no declaration text, and said out loud in the docstring.
TOKENS_PER_TOOL = 143

# The system instruction a session carries whatever its menu is. The reference agent's,
# measured 2026-09-01. It is in the prefix, so it is in the cache and in the bill.
SYSTEM_TOKENS = 1_777

# Turns per hour across the whole corpus. Two orders of magnitude, because the crossover
# this script is looking for is a function of arrival density and nothing else.
RATES = (40.0, 400.0, 4_000.0, 40_000.0)

# One regime, not two. `score_xlam.py` runs both and they agree to four decimal places on
# xLAM, so the second one costs an hour and says nothing.
REGIME = Regime("block the row", lambda case: case.session_id)

# (label, merge the candidates, refill the winners to `size`). The third arm exists so a
# merged win can be attributed: `merged` changes both what a menu contains and how wide it
# is, and `merged, refilled` holds the width equal to the shipped arm.
ARMS = (
    ("padded as shipped", False, False),
    ("merged", True, False),
    ("merged, refilled", True, True),
)


@dataclass(frozen=True)
class Fitted:
    """One candidate strategy's pool, and what the corpus did with it."""

    name: str
    coverage: float  # mean set recall of the menu each query snapped to
    sends: list[Counter[Menu]]  # one send distribution per fold, never pooled

    @property
    def menus(self) -> float:
        """Menus actually sent, per fold.

        Per fold and not pooled, because a deployment fits one pool and sends from it. Five
        fold-pools that disagree would otherwise read as one pool with five times the spread,
        and the pricing below would charge a candidate strategy for the harness's folds
        rather than for its own prefixes.
        """
        return sum(len(fold) for fold in self.sends) / len(self.sends)

    @property
    def distinct_tools(self) -> float:
        return sum(len({t for menu in fold for t in menu}) for fold in self.sends) / len(self.sends)

    @property
    def mean_width(self) -> float:
        total = sum(sum(fold.values()) for fold in self.sends)
        if not total:
            return 0.0
        return sum(len(m) * n for fold in self.sends for m, n in fold.items()) / total


# --- Fitting the two pools -------------------------------------------------------------


def fit(
    cases: list[Case], size: int, count: int, folds: int, merged: bool, refill: bool
) -> list[tuple[Menu, ...]]:
    """One pool per fold, each fitted without that fold's queries.

    The only difference between the arms is which candidates `choose` picks from, and whether
    the winners' spare slots are then filled. The greedy selection, the scoring and the fold
    blocking are identical, so anything the table shows is the candidates and not the fit.

    `refill` exists to separate two effects that the merged arm changes at once. A merged
    bucket is usually narrower than `size`, so it is both differently composed and cheaper,
    and a win could be either. Filling the winners back out to `size` from the popularity
    ranking holds the token cost equal to the padded arm and leaves only the composition.
    """
    pools = []
    for fold in range(folds):
        training = [
            sorted(case.tools)
            for case in cases
            if case.tools and fold_of(REGIME.key(case), folds) != fold
        ]
        candidates = bucket(training, size) if merged else None
        chosen = choose(training, size=size, count=count, candidates=candidates)
        if refill:
            ranked = popularity(training)
            chosen = tuple(frozenset(pad(sorted(menu), ranked, size)) for menu in chosen)
        pools.append(tuple(chosen))
    return pools


def snap(
    cases: list[Case],
    vectors: np.ndarray,
    pools: list[tuple[Menu, ...]],
    neighbours: int,
) -> tuple[float, list[Counter[Menu]]]:
    """Send every query to the menu it would get, and record both recall and the menu.

    The same block loop `score_xlam.score_surface` runs, kept here rather than imported
    because that one returns a summary and this one needs the assignment itself: the send
    distribution is half of what is being measured.

    Kept per fold. A query is scored against the pool that did not see it, which is what
    makes the coverage number honest, but each of those pools is a whole deployment as far as
    the cache is concerned and mixing their sends together invents prefixes nobody sends.
    """
    keys = np.array([REGIME.key(case) for case in cases])
    tools = [case.tools for case in cases]
    covered = 0.0
    sends: list[Counter[Menu]] = [Counter() for _ in pools]

    for start in range(0, len(cases), BLOCK):
        if start and start % (BLOCK * 20) == 0:
            say(f"    {start:,} of {len(cases):,}")
        block = vectors[start : start + BLOCK] @ vectors.T
        for offset, similarity in enumerate(block):
            index = start + offset
            weighted = votes(similarity, keys == keys[index], tools, neighbours)
            fold = fold_of(keys[index], len(pools))
            chosen = rank([Vote(t, w) for t, w in weighted.items()], pools[fold])
            covered += recall(chosen or frozenset(), tools[index])
            if chosen is not None:
                sends[fold][chosen] += 1
    return covered / len(cases), sends


# --- Pricing what the pool does to the cache -------------------------------------------


def expected_warmth(rate_per_hour: float, curve: ResidencyCurve, prefix_tokens: int) -> float:
    """Mean cached share of a prefix sent at `rate_per_hour`, over exponential gaps.

    Integrated rather than evaluated at the mean gap. The curve is concave in log gap, so
    reading it at the mean overstates warmth, and it overstates it most for the menus that
    are sent least, which are exactly the ones a spread-out pool creates. Doing it the cheap
    way would have flattered the arm this script exists to be sceptical of.

    A hundred log-spaced points from a millisecond to a day, weighted by the exponential's
    mass in each interval. Deterministic, and accurate to well under a point against a
    fine grid, which is far below anything the curve itself is known to.
    """
    if rate_per_hour <= 0:
        return curve.floor
    lam = rate_per_hour / 3600.0
    edges = np.geomspace(1e-3, 86_400.0, 101)
    mass = np.exp(-lam * edges[:-1]) - np.exp(-lam * edges[1:])
    mids = np.sqrt(edges[:-1] * edges[1:])
    warm = np.array([curve.warmth(float(t), prefix_tokens) for t in mids])
    # Whatever mass falls beyond a day sits on the curve's flat tail, so it takes the floor.
    beyond = float(np.exp(-lam * edges[-1]))
    return float((mass * warm).sum() + beyond * curve.warmth(edges[-1], prefix_tokens))


def billed_tokens(fitted: Fitted, rate_per_hour: float, curve: ResidencyCurve) -> float:
    """Billed-equivalent input tokens for one session's turn-one prefix, on this pool.

    A cached token bills at a tenth of an uncached one, so the bill is the prefix scaled by
    `warm * 0.10 + (1 - warm)`. Each menu is priced at its own send rate and its own size,
    then averaged over the sessions that actually landed on it, which is the only weighting
    that answers what the corpus costs.

    Fold by fold, at the full arrival rate in each. A fold holds a fifth of the corpus but
    stands for a whole deployment, so its menus' *shares* are what transfers and its counts
    are not. Pricing a fold at a fifth of the rate would make every pool look colder than it
    is, and would penalise whichever strategy happens to produce more menus twice over.
    """
    bills = []
    for fold in fitted.sends:
        total = sum(fold.values())
        if not total:
            continue
        bill = 0.0
        for menu, count in fold.items():
            prefix = SYSTEM_TOKENS + len(menu) * TOKENS_PER_TOOL
            warm = expected_warmth(rate_per_hour * count / total, curve, prefix)
            bill += count * prefix * (warm * CACHED_RATE + (1 - warm))
        bills.append(bill / total)
    return sum(bills) / len(bills) if bills else 0.0


def full_list_billed(tools: int, rate_per_hour: float, curve: ResidencyCurve) -> float:
    """The same quantity for declaring everything: one prefix, warmed by every session."""
    prefix = SYSTEM_TOKENS + tools * TOKENS_PER_TOOL
    warm = expected_warmth(rate_per_hour, curve, prefix)
    return prefix * (warm * CACHED_RATE + (1 - warm))


# --- Reporting --------------------------------------------------------------------------


def report(width: int, tools: int, cases: int, arms: list[Fitted], curve: ResidencyCurve) -> None:
    label = "every tool" if width == 0 else f"top {width}"
    print(f"\n### surface: {label} ({tools:,} tools, {cases:,} queries)\n")

    print("| candidates | coverage | menus | distinct tools | mean width |")
    print("| --- | --- | --- | --- | --- |")
    for arm in arms:
        print(
            f"| {arm.name} | {arm.coverage:.4f} | {arm.menus:.1f} | {arm.distinct_tools:.0f} |"
            f" {arm.mean_width:.1f} |"
        )

    print("\nbilled-equivalent turn-one prefix tokens, against declaring everything\n")
    header = " | ".join(f"{rate:,.0f}/h" for rate in RATES)
    print(f"| candidates | {header} |")
    print("| --- |" + " --- |" * len(RATES))
    for arm in arms:
        cells = []
        for rate in RATES:
            full = full_list_billed(tools, rate, curve)
            cells.append(f"{billed_tokens(arm, rate, curve) / full - 1:+.1%}")
        print(f"| {arm.name} | " + " | ".join(cells) + " |")


# --- Main ---------------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset)
    size = args.menu_size or cfg.max_tools_kept
    queries = load_corpus(CORPUS_PATH)

    print(f"xlam-function-calling-60k: {len(queries):,} queries, menus of {size}")
    print("  candidates    padded as shipped, against merged by `menus.bucket`")
    print(
        f"  priced at     {TOKENS_PER_TOOL} tokens/tool over a {SYSTEM_TOKENS:,}-token instruction"
    )
    print("  residency     the shipped curve, measured on the ab-v2-100tools run")

    if not VECTOR_PATH.exists():
        print(f"\n{VECTOR_PATH} is missing; reading the vectors back from BigQuery.")
    connection = cfg.bq_connection or f"{args.project}.us.smart_router_conn"
    ids, vectors = embed(
        queries, args.project, args.dataset, cfg.embedding_model, cfg.embedding_dims, connection
    )

    by_id = {q.row_id: q for q in queries}
    embedded = [by_id[i] for i in ids if i in by_id]
    lookup = {q.row_id: n for n, q in enumerate(embedded)}
    vectors = vectors[[n for n, i in enumerate(ids) if i in by_id]]

    for width in args.surfaces:
        surface = surface_of(embedded, width)
        keep = [q for q in embedded if q.tools and q.tools <= surface]
        if len(keep) < NEIGHBOURS * 2:
            print(f"\nskipping the top-{width} surface: only {len(keep)} queries fit it.")
            continue

        cases = [Case(q.row_id, q.tools, q.text, "", "") for q in keep]
        subset = vectors[[lookup[q.row_id] for q in keep]]
        say(f"surface {width or 'all'}: {len(surface):,} tools, {len(cases):,} queries")

        arms = []
        for name, merged, refill in ARMS:
            say(f"  fitting {args.folds} pools, {name}")
            pools = fit(cases, size, args.menu_count, args.folds, merged, refill)
            say(f"  snapping {len(cases):,} queries, {name}")
            coverage, sends = snap(cases, subset, pools, args.neighbours)
            arms.append(Fitted(name, coverage, sends))

        report(width, len(surface), len(cases), arms, MEASURED_CURVE)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--menu-size", type=int, default=None, help="defaults to max_tools_kept")
    parser.add_argument("--menu-count", type=int, default=MENU_COUNT)
    parser.add_argument("--folds", type=int, default=POOL_FOLDS)
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument(
        "--surfaces",
        type=lambda v: tuple(int(w) for w in v.split(",")),
        default=SURFACES,
        help="comma-separated tool counts; 0 means every tool",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
