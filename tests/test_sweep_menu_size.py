"""Tests for the menu-width sweep's arithmetic.

`scripts/sweep_menu_size.py` reports a frontier, and a frontier is only useful if the points
on it are comparable. Three things can quietly make them incomparable, and each one has a
case here.

The sweep reimplements the shipped ranking as a sorted list so it can take every prefix
cheaply. If that order drifts from `voted_menu`'s, the sweep is measuring a different rule
than the one the plugin runs and the whole table is about somebody else's router.

The adaptive arm lands on fractional mean widths that no fixed cap has, so it is compared to
a straight line through the static baseline. Getting that interpolation wrong moves the
comparison column without moving anything real.

And the verdict picks the narrowest arm clearing the recall bar, which has to be narrowest by
width rather than first in the list.
"""

from __future__ import annotations

import random

from score_unseen_intent import voted_menu
from sweep_menu_size import (
    SIZES,
    Arm,
    dominance,
    first_reaching,
    interpolate,
    mass_prefix,
    ordered,
    saturation_width,
)

RANKED = ["popular", "common", "rare", "obscure"]


def test_the_sweeps_order_is_the_shipped_order_at_every_width():
    """The claim the whole script rests on, checked against `voted_menu` rather than asserted.

    Ties are the part that can drift: at fifteen neighbours a long tail scores one weak vote
    each, so most of the order is decided by the tie-break and not by the weights.
    """
    rng = random.Random(20260906)
    for _ in range(500):
        weighted = {tool: rng.choice([1.0, 1.0, 2.0, 0.5]) for tool in RANKED}
        support = ordered(weighted, RANKED)
        for size in range(len(RANKED) + 2):
            assert frozenset(tool for tool, _ in support[:size]) == voted_menu(
                weighted, RANKED, size
            )


def test_support_covers_the_fraction_asked_for_and_stops():
    support = [("a", 6.0), ("b", 3.0), ("c", 1.0)]

    assert mass_prefix(support, 0.5) == 1  # 6 of 10 already clears half
    assert mass_prefix(support, 0.6) == 1
    assert mass_prefix(support, 0.7) == 2  # 6 is short of 7, 9 clears it
    assert mass_prefix(support, 1.0) == 3


def test_a_support_of_nothing_asks_for_everything_rather_than_dividing_by_zero():
    """Every neighbour was blocked, so there is no ranking to trust and no prefix to take."""
    assert mass_prefix([], 0.9) == 0
    assert mass_prefix([("a", 0.0), ("b", 0.0)], 0.9) == 2


def test_the_baseline_is_read_between_its_points_not_at_the_nearest_one():
    """Halfway between a static width of 1 and 3 is halfway between their recalls."""
    baseline = [Arm("top 1", recall=0.20, width=1.0), Arm("top 3", recall=0.60, width=3.0)]

    assert interpolate(baseline, 2.0) == 0.40
    assert interpolate(baseline, 1.5) == 0.30


def test_the_baseline_is_flat_outside_the_widths_that_were_measured():
    """An arm narrower than the narrowest static menu, or wider than the widest, has no
    bracketing pair. Extrapolating would invent a number; the nearest measured one is honest."""
    baseline = [Arm("top 1", recall=0.20, width=1.0), Arm("top 3", recall=0.60, width=3.0)]

    assert interpolate(baseline, 0.4) == 0.20
    assert interpolate(baseline, 99.0) == 0.60


def test_the_verdict_takes_the_narrowest_qualifying_arm_not_the_first_one():
    """The adaptive arms are listed by support fraction, and width does not have to follow it.

    Reading the list in order would report the first arm over the bar as the cheapest, which
    is the claim the verdict is making and is not the same thing.
    """
    arms = [
        Arm("mass 0.9", recall=0.96, width=8.0),
        Arm("mass 0.95", recall=0.97, width=6.0),
        Arm("mass 0.5", recall=0.80, width=2.0),
    ]

    assert first_reaching(arms, 0.95) == arms[1]


def test_nothing_qualifying_is_reported_as_nothing():
    """The verdict says so in words. A silent fallback to the widest arm would read as a pass."""
    assert first_reaching([Arm("top 1", recall=0.5, width=1.0)], 0.95) is None


def test_dominance_compares_at_equal_cost_not_equal_label():
    """`top 3` and `mass 0.8` are not comparable by name; they are comparable by mean width."""
    fixed = [Arm("top 1", recall=0.50, width=1.0), Arm("top 3", recall=0.80, width=2.0)]
    adaptive = [Arm("mass 0.5", recall=0.60, width=1.0), Arm("mass 0.9", recall=0.90, width=2.0)]

    assert dominance(fixed, adaptive) == (2, 0)
    assert dominance(adaptive, fixed) == (0, 2)


def test_a_tie_within_noise_counts_for_neither_side():
    """Two rules that agree to three decimals have not been separated by this corpus, and
    calling that a win would turn rounding into a recommendation."""
    fixed = [Arm("top 1", recall=0.8000, width=1.0)]
    adaptive = [Arm("mass 0.5", recall=0.8004, width=1.0)]

    assert dominance(fixed, adaptive) == (0, 0)


def test_saturation_is_the_first_cap_the_predictor_cannot_fill():
    """Widths stop growing once the vote list runs out, and every wider cap is then inert."""
    arms = [Arm(f"top {size}", width=min(float(size), 2.7)) for size in SIZES]

    assert saturation_width(arms) == 3


def test_a_cap_still_binding_at_the_widest_width_saturates_there_rather_than_nowhere():
    """The widest cap always matches itself, so this always answers. Whether the answer means
    anything is the verdict's job, and it decides that from the recall column."""
    arms = [Arm(f"top {size}", width=float(size)) for size in SIZES]

    assert saturation_width(arms) == SIZES[-1]
