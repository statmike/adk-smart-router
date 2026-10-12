"""The P5 miss pricing, which is the only part of `cost_model.py` with an argument in it.

Everything else the script computes is guarded by its own `--check`: the numbers are
recorded next to the model and drift fails the run. P5 is different because it makes a
claim — that the break-even miss rate does not depend on which retry model you believe —
and a claim needs a test rather than a recorded value.
"""

from __future__ import annotations

import pytest
from cost_model import (
    MEASURED_MENU_RECALL,
    PUBLISHED_MENU_RECALL_BAR,
    Scenario,
    breakeven_miss_rate,
    expected_with_misses,
    policy_p0,
    policy_p2,
    total,
)

SCENARIO = Scenario()
P0 = total(policy_p0(SCENARIO), SCENARIO)
P2 = total(policy_p2(SCENARIO), SCENARIO)


def test_perfect_recall_costs_exactly_what_p2_costs() -> None:
    assert expected_with_misses(P2, 0.0) == P2


def test_the_break_even_is_the_same_under_both_retry_models() -> None:
    """The docstring's claim, and the reason the number is worth quoting at all.

    One model retries routed and pays a geometric series. The other assumes the retry runs
    unrouted, so it pays `routed + m * unrouted` once. They disagree everywhere except at the
    break-even, where both equal the unrouted total.
    """
    m = breakeven_miss_rate(P2, P0)
    assert expected_with_misses(P2, m) == pytest.approx(P0)
    assert P2 + m * P0 == pytest.approx(P0)

    # Away from it they really do differ, or the agreement above would be trivial.
    assert expected_with_misses(P2, 0.3) != pytest.approx(P2 + 0.3 * P0)


def test_a_miss_rate_of_one_is_refused_rather_than_dividing_by_zero() -> None:
    with pytest.raises(ValueError, match="miss_rate"):
        expected_with_misses(P2, 1.0)
    with pytest.raises(ValueError, match="miss_rate"):
        expected_with_misses(P2, -0.1)


def test_the_measured_miss_rate_is_far_inside_the_break_even() -> None:
    """The finding the section is written around, held as a test so it cannot quietly invert.

    If a future scenario ever makes the measured miss rate the binding constraint, the
    prose in `render` stops being true and this fails first.
    """
    breakeven = breakeven_miss_rate(P2, P0)
    assert breakeven / 10 > 1.0 - MEASURED_MENU_RECALL
    assert breakeven / 2 > 1.0 - PUBLISHED_MENU_RECALL_BAR
