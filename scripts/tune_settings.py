#!/usr/bin/env python
"""V2-7(a) — choose the router's settings by pricing them, against the agent's own history.

Four choices decide what the router does, and all four were once a human reading a table:
`--menu-size`, `--menu-count` and `--menu-candidates` at publish, and `miss_cost_tokens` in
the config. V2-4 moved three of them at once, by hand, because the tool surface went from 25
tools to 100. An installed plugin cannot do that, so it has to be able to do this instead:
sweep the settings against its own logged sessions, price each one, and pick.

`tune_menus.py` already sweeps the pool shape and it is not enough, for two reasons that
both push the same way.

    it scores an oracle    every session gets the best menu in the pool, so the number it
                           reports is the pool's ceiling rather than anything the real
                           model reaches. Ceilings rise with menu count monotonically.
    it prices tokens only  its `pred G1` is declaration tokens removed over tokens billed.
                           That term is real and it is not the big one. DESIGN.md 19.14
                           measured a pool of menus losing more to cold prefixes than the
                           prune saved, and a token-only objective cannot see that at all.

So both of its blind spots flatter wide pools, and it was used to pick a wide pool. This
script closes both. Menus are assigned by the **published model**, through the same
`pruner.decide` the plugin calls, and the result is priced by **`predict_saving.predict`**,
which replays the arrivals against the measured residency curve and so charges a pool for
the warmth it gives away. What comes out is a frontier of predicted G1 against menu recall,
and a choice: the best saving that still clears the recall bar.

Two disciplines matter enough to state.

Pools are fit on the training split and everything is scored on the held-out one. A pool fit
and scored on the same sessions measures memorisation, and menu choosing is exactly the kind
of procedure that would score well on it.

Scoring a quarter of the traffic would make every menu look colder than it will be, because
a quarter of the arrivals is a quarter of the warming. That is a bias against menu count
specifically, so it is corrected rather than tolerated: the replay runs at
`1 / held-out share` times the sampled rate, which is the rate the full traffic really had.

Cost is three BigQuery reads and one count_tokens call. No training, no model calls, and no
writes unless `--out` is given. Authentication is ADC only.

    uv run python scripts/tune_settings.py
    uv run python scripts/tune_settings.py --sizes 7 11 15 --counts 8 16 24 --out settings.json

`--out` writes what was chosen and the evidence for it. `publish_router.py --settings` reads
that file, so the loop closes without anyone retyping a number.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from deployment import DEFAULT_DATASET, bucket_arg, project_arg  # noqa: E402
from measure_menus import declaration_tokens  # noqa: E402
from predict_saving import (  # noqa: E402
    MEASURED_CURVE,
    SYSTEM_TOKENS,
    Prediction,
    ResidencyCurve,
    Traffic,
    fetch_traffic,
    load_curve,
    predict,
    surface_mismatch,
)
from publish_router import DEFAULT_MIN_MENU_RECALL  # noqa: E402
from score_sets import (  # noqa: E402
    DEFAULT_AGENT,
    Arm,
    Session,
    fetch_sessions,
    menu_always,
    menu_router,
    score,
    tok,
)

from adk_smart_router import lifecycle  # noqa: E402
from adk_smart_router.config import RouterConfig, load_config  # noqa: E402
from adk_smart_router.menus import bucket, choose  # noqa: E402
from adk_smart_router.model_store import ModelStore  # noqa: E402
from adk_smart_router.predictor import Predictor  # noqa: E402
from adk_smart_router.pruner import PredictionLike, contained_mass  # noqa: E402

Menu = frozenset[str]
# Fitted pool and the menu each session was assigned from it, per pool shape. Shared across
# sweeps so the stability check does not refit what it is deliberately holding fixed. The shape
# is (size, count, merged), because all three change what `fit_pool` returns.
ShapeCache = dict[tuple[int, int, bool], tuple[tuple[Menu, ...], dict[str, "Menu | None"]]]

# The grid. Sizes run from well under the published 11 up to 35, which is where recall
# crosses 0.98 on this agent. The gate no longer asks for 0.98, but the grid still reaches it,
# because a sweep that stops below a bar can only report that the bar is unreachable, which is
# what DESIGN.md 19.16 concluded from a single pool and what `report_floors` exists to avoid
# repeating for whatever bar comes next. Counts step either side of the published 24. Two miss
# prices, the shipped 1600 and the 2278 a miss actually measured, are enough to show the axis is
# inert; `report_miss_price` says why swinging it wider would change nothing.
SIZES = (7, 11, 15, 25, 35)
COUNTS = (8, 16, 24, 32)
MISS_COSTS = (1_600, 2_278)

# How the pool's candidate menus are built, swept rather than decided here. `menus.pad` gives
# each session its own set plus filler from the popularity ranking; `menus.bucket` merges
# co-occurring sets instead. DESIGN.md 19.47 measured both on 60,000 third-party queries and
# found merging worth 20 points of coverage on a 100-tool surface for half a point of cache
# warmth, and worth slightly less than nothing on a 25-tool one. Which side of that an agent
# lands on depends on its own tool count and traffic, so it is priced here like everything else.
MERGES = (False, True)

# The recall a pool must clear to be eligible. This is a policy input, not a measurement: it
# says what the operator will trade, and `report_floors` exists because the answer moves when
# that willingness does. DESIGN.md 19.16 found 0.98 unreachable on a 100-tool surface, where
# the published pool measures 0.9110, so this has been anchored to what shipped rather than
# to what was hoped for: clear the bar means at least as good as the pool in production.
#
# Imported from the publisher since 2026-09-05 rather than declared again. The two files
# disagreed for a month, which meant the bar a model faced depended on which of them the
# operator happened to run.
MIN_MENU_RECALL = DEFAULT_MIN_MENU_RECALL

# The bars the sensitivity table walks. Wide enough to bracket the shipped 0.90, the 0.98 it
# replaced, and the point where the bar stops binding at all, because a frontier that only
# shows the answer near one bar cannot tell you whether the bar is the thing deciding.
FLOORS = (0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98)


@dataclass(frozen=True)
class Candidate:
    """One setting of the four things this script exists to choose."""

    menu_size: int
    menu_count: int
    miss_cost: int
    merged: bool = False

    @property
    def candidates(self) -> str:
        """The name this arm goes by in the report and in the settings file."""
        return "merged" if self.merged else "padded"


@dataclass(frozen=True)
class Scored:
    """A candidate and what it is predicted to do on traffic it was not fitted to."""

    candidate: Candidate
    menus: int  # what `choose` actually returned, which can be fewer than asked for
    width: float
    recall: float  # the gate's definition, pinned to the label set; this is what the bar reads
    honest: float  # the same over every live tool the session wanted, blind spots included
    pruned_share: float
    g1: float
    warm: float
    floor: float = MIN_MENU_RECALL

    @property
    def eligible(self) -> bool:
        return self.recall >= self.floor


# --- The sweep ------------------------------------------------------------------------


def fit_pool(
    train: Sequence[Session], size: int, count: int, merged: bool = False
) -> tuple[Menu, ...]:
    """Choose a pool from the training sessions, with the shipped chooser.

    `choose` wants one sequence of calls per session. Under the set objective the call order
    and the repeats carry no information, so a sorted set is the whole input, which is the
    same reduction `tune_menus.py` makes.

    `merged` swaps what the chooser gets to choose between. Padding builds one candidate per
    session and fills its spare slots from the global popularity ranking; bucketing merges
    co-occurring sets into candidates that fit in `size`. The greedy set cover afterwards is
    the same either way, which is what makes the two comparable at all.
    """
    corpus = [sorted(session.tools) for session in train if session.tools]
    offered = bucket(corpus, size) if merged else None
    return tuple(choose(corpus, size=size, count=count, candidates=offered))


def assign(
    ranked: dict[str, Sequence[PredictionLike]],
    pool: Sequence[Menu],
    live: Menu,
    known: Menu,
    cfg: RouterConfig,
) -> dict[str, Menu | None]:
    """The menu the pruner picks per session, before the decline rule has an opinion.

    Separated from the decline step because the model's ranking and the pool decide this,
    and neither depends on the miss price. Sweeping the price therefore re-runs only the
    cheap half.
    """
    return {sid: menu_always(preds, live, known, cfg, pool) for sid, preds in ranked.items()}


def decline(
    ranked: dict[str, Sequence[PredictionLike]],
    chosen: dict[str, Menu | None],
    live: Menu,
    tokens: dict[str, int],
    cfg: RouterConfig,
) -> dict[str, Menu | None]:
    """Keep each menu only where the prune pays at this miss price. `pruner.worth_pruning`."""
    return {sid: menu_router(ranked[sid], menu, live, tokens, cfg) for sid, menu in chosen.items()}


def measure_recall(
    held_out: Sequence[Session],
    shipped: dict[str, Menu | None],
    live: Menu,
    known: Menu,
    tokens: dict[str, int],
) -> tuple[float, float, float]:
    """Set recall two ways, and the share of sessions actually pruned. DESIGN.md 19.4.

    `score_sets.score` does the counting, so both numbers come out of the arithmetic that is
    already under test. A declined session ships the full list and scores a free 1.0, which
    is why the pruned share is reported beside them: a pool that declines everything has
    perfect recall and is worth nothing.

    The two differ only in what a session is credited with wanting, and the gap is the whole
    reason both are here. The publish gate pins wanted sets to the **label set**, on the
    argument that a tool the booster has no class for is one no menu was ever allowed to
    hold, so counting it would grade the pool for the model's blind spot. That is fair for
    choosing between pools, and it is also 31 of this agent's 100 live tools, every one of
    which is a real miss when a session reaches for it. So the gate's definition decides
    eligibility, because it owns the bar, and the honest one is printed next to it.
    """
    gate, honest = Arm("gate"), Arm("all tools")
    for session in held_out:
        if session.session_id not in shipped:
            continue
        menu = shipped[session.session_id]
        score(gate, replace(session, tools=session.tools & known), menu, live, tokens)
        score(honest, session, menu, live, tokens)
    return (
        gate.mean(gate.recall_sum, gate.recall_n),
        honest.mean(honest.recall_sum, honest.recall_n),
        gate.pruned / gate.sessions if gate.sessions else 0.0,
    )


def price(
    traffic: Traffic,
    shipped: dict[str, Menu | None],
    full_prefix: int,
    tokens: dict[str, int],
    curve: ResidencyCurve,
    rate: float,
) -> Prediction:
    """Predicted G1 for a fully rolled-out router shipping these menus.

    `route_fraction=1.0` because this is a settings choice and not an experiment design: the
    question is what the agent should run, not what an A/B would report about it.
    """
    prefix_tokens = {
        menu: SYSTEM_TOKENS + tok(menu, tokens) for menu in shipped.values() if menu is not None
    }
    return predict(traffic, shipped, full_prefix, prefix_tokens, curve, 1.0, rate)


def sweep(
    ranked: dict[str, Sequence[PredictionLike]],
    train: Sequence[Session],
    held_out: Sequence[Session],
    traffic: Traffic,
    live: Menu,
    known: Menu,
    tokens: dict[str, int],
    cfg: RouterConfig,
    curve: ResidencyCurve,
    rate: float,
    grid: Sequence[Candidate],
    floor: float,
    shape_cache: ShapeCache | None = None,
) -> list[Scored]:
    """Score every candidate, refitting the pool only when the pool shape changes.

    The model's ranking is computed once by the caller and the pool once per shape, because
    neither depends on the miss price. Only the decline step and the pricing run per price,
    which is what keeps a grid of several dozen candidates to a few seconds of arithmetic.

    `shape_cache` can be passed in to share fitted pools across several calls, which is what
    the stability check needs: it scores the same pools on two halves of the held-out set, and
    refitting them each time would triple the only expensive step for no change in the answer.

    Only `held_out` decides which sessions count. Pricing is restricted to the same ones, so
    recall and G1 always describe the same traffic rather than drifting apart if a caller ever
    passes a `ranked` covering more sessions than it is scoring.
    """
    full_prefix = SYSTEM_TOKENS + sum(tokens.values())
    scored: list[Scored] = []
    shape_cache = {} if shape_cache is None else shape_cache
    scoring = {session.session_id for session in held_out}

    for candidate in grid:
        shape = (candidate.menu_size, candidate.menu_count, candidate.merged)
        if shape not in shape_cache:
            pool = fit_pool(train, candidate.menu_size, candidate.menu_count, candidate.merged)
            # `max_tools_kept` is parity with the menu size: a menu wider than the pruner
            # keeps is one the pruner would trim, so sweeping one without the other would
            # score a pool the router could never ship.
            sized = replace(cfg, max_tools_kept=candidate.menu_size)
            shape_cache[shape] = (pool, assign(ranked, pool, live, known, sized))
        pool, chosen = shape_cache[shape]
        if not pool:
            continue

        priced = replace(
            cfg, max_tools_kept=candidate.menu_size, miss_cost_tokens=candidate.miss_cost
        )
        shipped = {
            sid: menu
            for sid, menu in decline(ranked, chosen, live, tokens, priced).items()
            if sid in scoring
        }
        recall, honest, pruned_share = measure_recall(held_out, shipped, live, known, tokens)
        prediction = price(traffic, shipped, full_prefix, tokens, curve, rate)
        scored.append(
            Scored(
                candidate=candidate,
                menus=len(pool),
                width=sum(len(menu) for menu in pool) / len(pool),
                recall=recall,
                honest=honest,
                pruned_share=pruned_share,
                g1=prediction.rollout_g1,
                warm=prediction.warm_openings,
                floor=floor,
            )
        )
    return scored


def halves(held_out: Sequence[Session]) -> tuple[list[Session], list[Session]]:
    """Split the held-out sessions in two, by a hash of the id so the split is stable.

    Stable matters: a random split would let a rerun turn an unstable recommendation into a
    stable one by chance, which is the opposite of what a stability check is for.
    """
    left = [s for s in held_out if zlib.crc32(s.session_id.encode()) % 2 == 0]
    right = [s for s in held_out if zlib.crc32(s.session_id.encode()) % 2 == 1]
    return left, right


def agrees(scored: Sequence[Scored], best: Scored | None) -> bool:
    """Whether a candidate is still the choice on a set it was not chosen on."""
    winner = pick(scored)
    return best is not None and winner is not None and winner.candidate == best.candidate


def report_stability(left: Sequence[Scored], right: Sequence[Scored], best: Scored | None) -> bool:
    """Would this recommendation replicate? Print the answer and return it.

    A sweep over a few dozen candidates on a few hundred sessions will always return a winner,
    and the winner will always look like a finding. Splitting the held-out set and asking
    whether the same candidate wins on both halves costs nothing (the pools are already fitted)
    and is the difference between a recommendation and a coincidence. No distributional
    assumption is involved: it is the question "would this replicate?" asked literally.

    A disagreement does not mean the winner is wrong. It means the sweep cannot tell it apart
    from its neighbours on this much evidence, which is exactly when a settings change should
    not be made automatically.

    Compare the two halves with each other and not with the whole set. A half is under-warmed
    relative to a whole for the reason `report` gives, so its G1 is a few points shy; the
    candidate it picks is comparable, its level is not.
    """
    on_left, on_right = pick(left), pick(right)
    print("\nwould this replicate? the held-out set, split in two and swept again")
    for name, winner in (("first half", on_left), ("second half", on_right)):
        if winner is None:
            print(f"  {name:>12}: nothing cleared the bar")
            continue
        shape = (
            f"{winner.candidate.menu_count} x {winner.candidate.menu_size}"
            f" {winner.candidate.candidates}"
        )
        print(f"  {name:>12}: {shape} at {winner.candidate.miss_cost}, {winner.g1:.2%} predicted")

    stable = agrees(left, best) and agrees(right, best)
    if stable:
        print("  both halves pick what the whole set picked, so the choice is not a coincidence.")
    else:
        print(
            "  the halves disagree, which means the sweep cannot tell the winner from its\n"
            "  neighbours on this much evidence. Read the frontier and decide by hand; nothing\n"
            "  will be written."
        )
    return stable


def decline_threshold(
    ranked: dict[str, Sequence[PredictionLike]],
    chosen: dict[str, Menu | None],
    live: Menu,
    tokens: dict[str, int],
    cfg: RouterConfig,
) -> list[float]:
    """The miss price at which each session would stop pruning, one per session.

    V2-6 established that a knob is only worth tuning where its threshold sits inside the
    data, and the miss price is a knob like any other, so the tuner owes itself the same
    check. Rearranging `pruner.worth_pruning` for the price gives it directly:

        declines when  miss_cost  >=  cached_rate * tokens removed * turns / P(miss)

    A session the model is sure about has P(miss) near zero and so a threshold near
    infinity: no price declines it, and it is left out rather than counted as a very large
    number, because the median of a list of infinities says nothing.
    """
    thresholds = []
    for sid, menu in chosen.items():
        if menu is None:
            continue
        saved = tok(live - menu, tokens)
        risk = 1.0 - contained_mass(ranked[sid], menu)
        if risk <= 0.0:
            continue
        thresholds.append(cfg.cached_token_multiplier * saved * cfg.expected_session_turns / risk)
    return sorted(thresholds)


def pick(scored: Sequence[Scored]) -> Scored | None:
    """The largest saving among the candidates that clear the recall bar.

    Ties break towards fewer menus and then a smaller size, which is the conservative
    direction: a smaller pool holds its warmth better on traffic thinner than what was
    sampled here, and nothing in the sweep can see traffic it was not shown.
    """
    eligible = [s for s in scored if s.eligible]
    if not eligible:
        return None
    return min(eligible, key=lambda s: (s.g1, s.menus, s.candidate.menu_size))


# --- Reporting ------------------------------------------------------------------------


def report(scored: Sequence[Scored], best: Scored | None, current: Scored | None) -> None:
    print(f"\n{len(scored)} candidates, best predicted saving that clears the recall bar\n")
    print(
        f"  {'size':>5} {'count':>6} {'miss':>6} {'built':>7} {'menus':>6} {'width':>6}"
        f" {'recall':>8} {'all':>8} {'pruned':>8} {'warm':>7} {'pred G1':>9}"
    )
    for row in sorted(scored, key=lambda s: s.g1):
        mark = "*" if best is not None and row is best else " "
        gate = " " if row.eligible else "x"
        print(
            f" {mark}{row.candidate.menu_size:>5} {row.candidate.menu_count:>6}"
            f" {row.candidate.miss_cost:>6} {row.candidate.candidates:>7}"
            f" {row.menus:>6} {row.width:>6.1f}"
            f" {row.recall:>8.4f}{gate}{row.honest:>7.4f} {row.pruned_share:>8.1%}"
            f" {row.warm:>7.1%} {row.g1:>9.2%}"
        )

    floor = scored[0].floor if scored else MIN_MENU_RECALL
    print(f"\n  x  recall below the {floor:.4f} bar, so not eligible whatever it saves")
    print("  *  chosen")
    print(
        "\n  recall   mean share of a session's tools the menu it shipped held, over the tools\n"
        "           the booster has a class for. The publish gate's definition, so it is the\n"
        "           one the bar reads. A declined session scores 1.0, so read it with `pruned`.\n"
        "  all      the same over every live tool the session wanted. The gap is the model's\n"
        "           blind spot, which no pool can cover and every user still feels.\n"
        "  built    how the pool's candidate menus were built: `padded` gives each session its\n"
        "           own set plus popular filler, `merged` fuses co-occurring sets instead.\n"
        "           DESIGN.md 19.47 on why this is swept rather than fixed.\n"
        "  pruned   share of sessions that shipped a menu at all.\n"
        "  warm     share of routed openings landing on a prefix still cached. This is what\n"
        "           menu count costs and what a token-only objective cannot see.\n"
        "  pred G1  billed tokens against the same traffic with the router uninstalled.\n"
        "           Pessimistic by about three points, and read the note below before quoting it."
    )
    print(
        "\n  Note on the level, not the order. Replaying a sample at the full traffic's arrival\n"
        "  rate restores the rate but not the volume, and every prefix is cold on its first\n"
        "  arrival however fast the sessions come. So a quarter sample under-warms everything:\n"
        "  DESIGN.md 19.21 measured the same pools at 425, 850 and 1700 sessions and saw warmth\n"
        "  climb about five points and G1 improve three to four across the board. The effect is\n"
        "  near enough uniform that it did not reorder any candidate, which is why the ranking\n"
        "  above can be trusted and the level above should be quoted as a floor."
    )

    if all(row.recall == row.honest for row in scored):
        print(
            "\n  recall and all agree exactly on this agent, which is worth knowing: every tool\n"
            "  any session here called is one the booster has a class for, so the gate's pinning\n"
            "  removes nothing and the bar is not being flattered. Expect them to part on an\n"
            "  agent whose tool surface has moved since the model was last trained."
        )

    if best is None:
        print("\nno candidate clears the recall bar. Lower it, or widen the grid.")
        return

    print(
        f"\nchosen: {best.candidate.menu_count} menus of {best.candidate.menu_size},"
        f" {best.candidate.candidates} candidates,"
        f" a miss priced at {best.candidate.miss_cost}"
    )
    print(f"        {best.g1:.2%} predicted, recall {best.recall:.4f}")
    if current is not None:
        print(
            f"current: {current.candidate.menu_count} x {current.candidate.menu_size}"
            f" {current.candidate.candidates} at {current.candidate.miss_cost},"
            f" {current.g1:.2%} predicted, recall {current.recall:.4f}"
        )
        moved = best.candidate != current.candidate
        print(
            f"         the sweep would {'move' if moved else 'keep'} the current settings", end=""
        )
        print(f", worth {best.g1 - current.g1:+.2%} of G1" if moved else "")


def report_floors(scored: Sequence[Scored], floors: Sequence[float]) -> None:
    """Which candidate wins at each recall bar, because the bar is the whole argument.

    Everything else here is measured. The floor is not: it is a statement about how much
    recall an operator will sell for how much saving, and DESIGN.md 19.16 already showed the
    0.98 the publish gate ships cannot be met on a surface this wide. Printing the choice as
    a function of the bar is what turns "pick a number" into a decision someone can actually
    make, and it is the table ROADMAP's open `--min-menu-recall` item has been waiting for.
    """
    print("\nhow the choice moves with the bar")
    print(f"  {'bar':>6} {'eligible':>9} {'choice':>12} {'recall':>8} {'pred G1':>9}")
    for floor in floors:
        at = [replace(row, floor=floor) for row in scored]
        winner = pick(at)
        if winner is None:
            # Say how far short the field fell rather than only that it did. A pool can miss
            # a bar by less than the table's own rounding, and a bare "none" next to a row
            # printing the bar exactly reads as a bug in the tuner instead of a close call.
            best = max((row.recall for row in at), default=0.0)
            print(
                f"  {floor:>6.2f} {0:>9} {'none':>12} {'-':>8} {'-':>9}"
                f"   short by {floor - best:.5f}"
            )
            continue
        shape = (
            f"{winner.candidate.menu_count} x {winner.candidate.menu_size}"
            f" {winner.candidate.candidates[:3]}"
        )
        print(
            f"  {floor:>6.2f} {sum(1 for row in at if row.eligible):>9} {shape:>12}"
            f" {winner.recall:>8.4f} {winner.g1:>9.2%}"
        )
    print(
        "\n  Read down the rows: every point of recall bought costs saving, and the exchange\n"
        "  rate is what the bar is really choosing. Nothing above measures which rate is right."
    )


def report_miss_price(thresholds: Sequence[float], prices: Sequence[int], pruned: int) -> None:
    """Say plainly whether the miss price is a dial on this agent or a decoration.

    A swept dimension that changes no outcome is worse than one left alone, because the
    table then reads as though four prices were compared on their merits. V2-6's rule
    applies: report where the threshold sits relative to the data, and let that decide
    whether the knob is worth anyone's attention.
    """
    if not thresholds:
        print("\nmiss price: no session carries any risk of a miss, so no price declines any.")
        return
    lowest = min(thresholds)
    bites = [p for p in prices if p > lowest]
    print(f"\nmiss price: {pruned} sessions prune at the published pool.")
    print(
        f"  the cheapest session declines once a miss is priced above {lowest:,.0f} tokens,"
        f"\n  and the median one above {thresholds[len(thresholds) // 2]:,.0f}."
    )
    if bites:
        print(f"  prices in this grid that reach that: {bites}. The dimension is live.")
    else:
        print(
            f"  every price swept here ({min(prices):,}-{max(prices):,}) is below it, so all of"
            "\n  them prune everything and the column is inert. DESIGN.md 19.1 measured a real"
            "\n  miss at 2,278 tokens, so no honest price declines anything on this surface. The"
            "\n  prune is simply too large to gamble against, which is 19.20's finding on the"
            "\n  other half of the same rule."
        )


def write_settings(path: str, agent: str, best: Scored, context: dict) -> None:
    """Record the choice and the evidence, in the shape `publish_router.py --settings` reads.

    The evidence is written beside the values on purpose. A number a machine picked and
    cannot justify is worse than one a human picked and cannot justify, because at least the
    human can be asked.
    """
    payload = {
        "agent": agent,
        "chosen_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "menu_size": best.candidate.menu_size,
        "menu_count": best.candidate.menu_count,
        "menu_candidates": best.candidate.candidates,
        "miss_cost_tokens": best.candidate.miss_cost,
        "evidence": {
            "predicted_g1": round(best.g1, 5),
            "menu_recall": round(best.recall, 5),
            "min_menu_recall": best.floor,
            "pruned_share": round(best.pruned_share, 5),
            "warm_openings": round(best.warm, 5),
            **context,
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote the chosen settings to {path}")


def shape(candidate: Candidate | None) -> dict | None:
    """A candidate as the ledger stores it, or None when there was no recommendation."""
    if candidate is None:
        return None
    return {
        "menu_size": candidate.menu_size,
        "menu_count": candidate.menu_count,
        "menu_candidates": candidate.candidates,
        "miss_cost_tokens": candidate.miss_cost,
    }


def record_tune(
    args: argparse.Namespace,
    outcome: str,
    reason: str,
    before: Scored | None = None,
    after: Scored | None = None,
    evidence: Scored | None = None,
) -> None:
    """Write this sweep into the retrain ledger. Best effort, and never fatal.

    A run without `--out` recommends nothing to anybody, so it records nothing, the same
    way a `--dry-run` publish does. The rest of the time this is the only durable record
    the tuner leaves: the settings file it writes is overwritten by the next cycle, and a
    decline writes no file at all, so *the tuner looked and had nothing to say* is
    otherwise invisible a week later (DESIGN.md 20.1).

    A decline is the interesting row. It is a normal, healthy answer meaning the previous
    settings still stand, and it reads nothing like a gate rejecting a model, which is why
    the ledger keeps `declined` and `rejected` apart.

    `after` is what stands when the step finishes, so a decline leaves it null rather than
    writing the shape it declined to adopt. `evidence` is separate because that shape is
    still worth keeping: a recommendation that keeps arriving and keeps failing to replicate
    is a different story from a sweep that finds nothing at all.
    """
    if not args.out:
        return
    metrics = None
    if evidence is not None:
        metrics = {
            "predicted_g1": round(evidence.g1, 5),
            "menu_recall": round(evidence.recall, 5),
            "pruned_share": round(evidence.pruned_share, 5),
            "min_menu_recall": evidence.floor,
            "proposed": shape(evidence.candidate),
        }
    lifecycle.record(
        args.project,
        args.dataset,
        lifecycle.Step(
            cycle_id=args.cycle_id,
            step="tune",
            agent=args.agent,
            outcome=outcome,
            reason=reason,
            settings_before=shape(before.candidate if before else None),
            settings_after=shape(after.candidate if after else None),
            metrics=metrics,
            source="tune_settings.py",
        ),
    )


# --- Main -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    cfg = load_config(bq_project=args.project, bq_dataset=args.dataset, gcs_bucket=args.bucket)

    store = ModelStore(cfg)
    model = store.load(args.agent)
    if model is None:
        # A first cycle is not a failure. The sweep scores candidate pool shapes by replaying
        # held-out traffic through the published model's menus, so a new agent has nothing to
        # sweep against and the honest answer is the same one an inconclusive sweep gives:
        # decline, write no settings, and let the publisher ship on the defaults. The next
        # cycle has a model to measure against and can recommend properly.
        if not store.published_versions(args.agent):
            print(f"{args.agent} has no published router yet, so there is nothing to sweep")
            print("against. Declining: publish on the defaults and tune the next cycle.")
            record_tune(args, "declined", "no published router to sweep against; first cycle")
            return 0
        print("ModelStore refused to load an artifact; see the warning above.", file=sys.stderr)
        return 1

    tokens = asyncio.run(declaration_tokens())
    live = frozenset(tokens)
    full_prefix = SYSTEM_TOKENS + sum(tokens.values())
    print(f"loaded {args.agent} version {model.metadata.version}, {len(model.menus)} menus")
    print(f"       {len(live)} live tools, {sum(tokens.values())} declaration tokens")

    everything = fetch_sessions(
        args.project, args.dataset, args.agent, cfg.embedding_dims, args.session_prefix, False
    )
    held_out = fetch_sessions(
        args.project, args.dataset, args.agent, cfg.embedding_dims, args.session_prefix, True
    )
    if not everything or not held_out:
        print(f"no sessions under prefix {args.session_prefix!r}.", file=sys.stderr)
        return 1

    eval_ids = {s.session_id for s in held_out}
    train = [s for s in everything if s.session_id not in eval_ids]
    share = len(held_out) / len(everything)
    print(f"       {len(train)} training sessions, {len(held_out)} held out ({share:.0%})")

    traffic = fetch_traffic(
        args.project, args.dataset, args.agent, args.session_prefix, full_prefix
    )
    # The same refusal predict_saving.py makes, for the same reason and against the same
    # arithmetic. Without it the sweep pooled corpus-20260831 with corpus-20260904, whose
    # turn-one prompts differ by 14,000 tokens because the surface grew between them, and
    # the frontier printed G1 figures in the hundreds of per cent.
    complaint = surface_mismatch(traffic, args.session_prefix, full_prefix)
    if complaint:
        print(f"\n{complaint}", file=sys.stderr)
        return 1
    billed = {s.session_id for s in traffic.sessions}
    scoreable = [s for s in held_out if s.session_id in billed]
    if not scoreable:
        print("no held-out session appears in the billing table.", file=sys.stderr)
        return 1

    curve = load_curve(args.residency_curve) if args.residency_curve else MEASURED_CURVE
    source = args.residency_curve or "the shipped default, measured on a routed run"
    # The held-out split is a sample of the traffic, so replaying it at its own arrival rate
    # would under-warm every prefix by exactly the sampling factor. Undo that.
    print(f"       residency curve: {source}; replayed at {1.0 / share:.1f}x the sampled rate")

    predictor = Predictor(model)
    ranked = {s.session_id: predictor.predict(s.vector) for s in scoreable}
    known = frozenset(model.classes)

    grid = [
        Candidate(size, count, cost, merged)
        for size in args.sizes
        for count in args.counts
        for cost in args.miss_costs
        for merged in (MERGES if args.candidates == "both" else (args.candidates == "merged",))
    ]

    # One cache of fitted pools across every sweep below. Pools are fit on the training split
    # and never depend on which held-out sessions are being scored, so the stability check and
    # the incumbent's score reuse them rather than refitting the only expensive step.
    cache: ShapeCache = {}

    def run(sessions: Sequence[Session], candidates: Sequence[Candidate]) -> list[Scored]:
        # The rate correction follows whatever subset is being scored. Any sample of the
        # traffic under-warms every prefix by exactly its sampling factor, so the replay runs
        # at the reciprocal of the share this subset is of the whole.
        return sweep(
            ranked,
            train,
            sessions,
            traffic,
            live,
            known,
            tokens,
            cfg,
            curve,
            len(everything) / len(sessions) if sessions else 1.0,
            candidates,
            args.min_menu_recall,
            cache,
        )

    scored = run(scoreable, grid)
    if not scored:
        print("no candidate produced a pool; check the grid and the corpus.", file=sys.stderr)
        return 1

    # What is running today, scored the same way, so the table has a baseline and the
    # recommendation has something to be a recommendation against. `merged` defaults to false
    # because padding is what every artifact published before 2026-09-05 was built with, and
    # the metadata does not record it; a wrong guess here mislabels the baseline row only.
    today = Candidate(cfg.max_tools_kept, len(model.menus), cfg.miss_cost_tokens)
    current = next((s for s in scored if s.candidate == today), None)
    if current is None:
        current = next(iter(run(scoreable, [today])), None)

    best = pick(scored)
    report(scored, best, current)
    report_floors(scored, FLOORS)

    left, right = halves(scoreable)
    stable = report_stability(run(left, grid), run(right, grid), best)

    # Whether the miss price was worth sweeping at all, measured on the published pool so
    # the answer describes the agent rather than whichever candidate happened to win.
    published = assign(ranked, model.menus, live, known, cfg)
    report_miss_price(
        decline_threshold(ranked, published, live, tokens, cfg),
        args.miss_costs,
        sum(1 for menu in published.values() if menu is not None),
    )

    if args.out and best is None:
        print(f"\nnothing cleared the bar, so {args.out} was not written.")
        record_tune(
            args,
            "declined",
            f"no candidate cleared the {DEFAULT_MIN_MENU_RECALL} recall bar",
            before=current,
        )
    elif args.out and not (stable or args.write_unstable):
        # ROADMAP's condition on this increment: a tuner that moves settings on a slow week is
        # worse than one that never runs. The refusal is the feature, so it is loud.
        print(
            f"\n{args.out} was not written: the recommendation did not replicate on both\n"
            "halves of the held-out set. Gather more traffic, or pass --write-unstable if you\n"
            "have read the frontier and want it anyway."
        )
        record_tune(
            args,
            "declined",
            "the recommendation did not replicate on both halves of the held-out set",
            before=current,
            evidence=best,
        )
    elif args.out and best is not None:
        record_tune(
            args,
            "ok",
            f"menu_size {best.candidate.menu_size}, menu_count {best.candidate.menu_count}, "
            f"{best.candidate.candidates}, replicated on both halves"
            if stable
            else "written unstable at the operator's request",
            before=current,
            after=best,
            evidence=best,
        )
        write_settings(
            args.out,
            args.agent,
            best,
            {
                "session_prefix": args.session_prefix,
                "held_out_sessions": len(scoreable),
                "training_sessions": len(train),
                "residency_curve": source,
                "model_version": model.metadata.version,
                "replicated_on_both_halves": stable,
            },
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--bucket", **bucket_arg())
    parser.add_argument(
        "--cycle-id",
        default=lifecycle.new_cycle_id(),
        help="the retrain cycle this sweep belongs to, as passed down by retrain.sh. Run on "
        "its own, tuning mints one and becomes a single-step cycle",
    )
    parser.add_argument(
        "--session-prefix",
        # Dated, for the reason predict_saving.py's default is dated: a bare "corpus" also
        # matches corpus-20260831, which ran on the 25-tool surface, and pooling the two
        # poisons every token figure in the sweep.
        default="corpus-20260904",
        help="only read sessions whose id starts with this. Must be UNROUTED traffic that"
        " ran on the tool surface the agent declares today: a routed session's tool set is"
        " censored by the prune it received, and an older surface prices nothing.",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--counts", type=int, nargs="+", default=list(COUNTS))
    parser.add_argument("--miss-costs", type=int, nargs="+", default=list(MISS_COSTS))
    parser.add_argument(
        "--candidates",
        choices=("both", "padded", "merged"),
        default="both",
        help="how the pool's candidate menus are built. Both arms are swept by default; pin "
        "one to halve the grid. DESIGN.md 19.47.",
    )
    parser.add_argument("--min-menu-recall", type=float, default=MIN_MENU_RECALL)
    parser.add_argument(
        "--residency-curve",
        default=None,
        help="JSON from `measure_residency.py --out`. Defaults to the shipped curve, which"
        " is right until this agent has routed traffic of its own to measure.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the chosen settings and their evidence here, for"
        " `publish_router.py --settings` to read.",
    )
    parser.add_argument(
        "--write-unstable",
        action="store_true",
        help="write --out even when the recommendation did not replicate on both halves of"
        " the held-out set. For reading the frontier by hand, not for a schedule.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
