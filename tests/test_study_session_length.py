"""The session-length study's arithmetic, checked without a network.

`study_session_length.py` reads BigQuery, but the finding it reports rests on three pure
functions: `prune_value` and `break_even`, which restate the decline rule so its margin can
be measured rather than only its verdict, and `regret`, which prices the constant against
the truth. These hold all three, plus the k-nearest-neighbour probe.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from study_session_length import Case, break_even, held_out, knn_lengths, prune_value, regret

from adk_smart_router.config import RouterConfig
from adk_smart_router.pruner import worth_pruning

CFG = RouterConfig(miss_cost_tokens=1_600, expected_session_turns=4.6)


def case(turns: int, saved: int, mass: float, session_id: str = "s", vector=(1.0, 0.0)) -> Case:
    return Case(session_id=session_id, turns=turns, saved=saved, mass=mass, vector=vector)


# --- prune_value and break_even -------------------------------------------------------


def test_the_value_is_the_rule_with_the_comparison_taken_out():
    """Whatever the inputs, positive value and `worth_pruning` must agree. Never drift.

    `worth_pruning` takes its length from the config rather than an argument, so the config
    is what varies here. That is the same length reaching the same arithmetic by the two
    routes, which is the only way the comparison means anything.
    """
    for saved in (100, 800, 4_000):
        for mass in (0.0, 0.5, 0.97, 1.0):
            for turns in (1, 4, 10):
                pays = prune_value(case(turns, saved, mass), turns, CFG) > 0
                assert pays == worth_pruning(
                    saved, mass, replace(CFG, expected_session_turns=turns)
                )


def test_a_longer_session_is_worth_more_to_prune():
    subject = case(1, 1_000, 0.5)
    assert prune_value(subject, 10, CFG) > prune_value(subject, 1, CFG)


def test_a_certain_session_pays_however_short_it_is():
    """Mass of 1.0 is no chance of a miss, so there is no cost to weigh the saving against."""
    assert break_even(case(1, 10, 1.0), CFG) == 0.0
    assert prune_value(case(1, 10, 1.0), 1, CFG) > 0


def test_removing_nothing_never_breaks_even():
    assert break_even(case(4, 0, 0.5), CFG) == float("inf")


def test_the_break_even_is_where_the_value_crosses_zero():
    subject = case(4, 1_000, 0.5)
    point = break_even(subject, CFG)
    assert prune_value(subject, point, CFG) == pytest.approx(0.0)
    assert prune_value(subject, point * 1.01, CFG) > 0 > prune_value(subject, point * 0.99, CFG)


def test_a_bigger_prune_breaks_even_sooner():
    assert break_even(case(4, 4_000, 0.5), CFG) < break_even(case(4, 500, 0.5), CFG)


# --- regret ---------------------------------------------------------------------------


def test_knowing_the_length_is_worth_nothing_when_it_changes_no_decision():
    """The reference agent's shape: a prune so large it pays at any length anyone runs."""
    cases = [case(turns, 8_000, 0.6, f"s{turns}") for turns in range(1, 11)]
    assert break_even(cases[0], CFG) < 1  # pays back before the shortest session ends
    assert regret(cases, CFG) == (0.0, 0)


def test_a_session_shorter_than_the_constant_expects_is_pruned_when_it_should_not_be():
    """One turn, a prune that needs two. The constant says yes and the session says no."""
    subject = case(1, 8_000, 0.0)
    assert 1 < break_even(subject, CFG) < CFG.expected_session_turns
    cost, flips = regret([subject], CFG)
    assert flips == 1
    assert cost == pytest.approx(-prune_value(subject, 1, CFG))
    assert cost > 0


def test_a_session_longer_than_the_constant_expects_is_declined_when_it_should_not_be():
    """The mirror: the constant is too pessimistic and a prune that would have paid is lost."""
    subject = case(10, 2_000, 0.0)
    assert CFG.expected_session_turns < break_even(subject, CFG) < 10
    cost, flips = regret([subject], CFG)
    assert flips == 1
    assert cost == pytest.approx(prune_value(subject, 10, CFG))


def test_regret_never_goes_below_zero():
    """It is measured against the best decision available, so nothing can beat that."""
    mixed = [
        case(turns, saved, mass, f"s{turns}-{saved}-{mass}")
        for turns in (1, 3, 10)
        for saved in (200, 1_000, 8_000)
        for mass in (0.0, 0.6, 1.0)
    ]
    assert regret(mixed, CFG)[0] >= 0.0
    for one in mixed:
        assert regret([one], CFG)[0] >= 0.0


def test_nothing_to_decide_costs_nothing():
    assert regret([], CFG) == (0.0, 0)


# --- the signal probe -------------------------------------------------------------------


def test_neighbours_in_embedding_space_are_what_gets_averaged():
    """Two clusters of different lengths. A test point in one must not be told the other's."""
    train = [case(2, 0, 0.0, f"a{i}", vector=(1.0, 0.01 * i)) for i in range(5)] + [
        case(9, 0, 0.0, f"b{i}", vector=(0.01 * i, 1.0)) for i in range(5)
    ]
    guessed = knn_lengths(train, [case(0, 0, 0.0, "t", vector=(1.0, 0.0))], k=3)
    assert guessed.tolist() == [2.0]


def test_asking_for_more_neighbours_than_exist_uses_all_of_them():
    train = [case(4, 0, 0.0, "a", vector=(1.0, 0.0)), case(6, 0, 0.0, "b", vector=(0.0, 1.0))]
    guessed = knn_lengths(train, [case(0, 0, 0.0, "t", vector=(1.0, 1.0))], k=99)
    assert guessed.tolist() == [5.0]


def test_a_zero_vector_does_not_divide_by_zero():
    train = [case(4, 0, 0.0, "a", vector=(0.0, 0.0))]
    assert np.isfinite(knn_lengths(train, [case(0, 0, 0.0, "t", vector=(0.0, 0.0))], k=1)).all()


def test_the_split_is_a_stable_quarter():
    ids = [f"session-{i}" for i in range(2_000)]
    out = [i for i in ids if held_out(i)]
    assert 0.2 < len(out) / len(ids) < 0.3
    assert [i for i in ids if held_out(i)] == out  # same answer every time it is asked
