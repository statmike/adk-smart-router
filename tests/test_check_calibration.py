"""The calibration check's arithmetic, without a warehouse.

`check_calibration.py` reads BigQuery and GCS, but the three questions it exists to answer
are pure functions of the cases it builds: does the score order the misses, does the stated
probability match the observed rate, and could the decline rule ever act on the difference.
These hold all three, plus the case building that feeds them.
"""

from __future__ import annotations

import pytest
from check_calibration import (
    Bucket,
    Case,
    billed_saving,
    build_cases,
    calibrate,
    calibration_error,
    discrimination,
    hazard,
    headroom,
    reliability,
)
from score_sets import Session

from adk_smart_router.config import RouterConfig

CFG = RouterConfig(
    miss_cost_tokens=1_600,
    expected_session_turns=4.6,
    max_tools_kept=3,
    # Five tools is under the shipped qualification floor of twenty, and `decide` would
    # pass every one of these fixtures straight through before the probability was read.
    min_tools_to_route=2,
)
TOKENS = {"a": 100, "b": 100, "c": 100, "d": 100, "e": 100}
LIVE = frozenset(TOKENS)


def session(session_id: str, tools: set[str]) -> Session:
    return Session(
        session_id=session_id,
        vector=[1.0, 0.0],
        tools=frozenset(tools),
        turns=4,
        billed=10_000.0,
    )


class Pred:
    """The shape `pruner` reads off a prediction: a label and a probability."""

    def __init__(self, label: str, probability: float) -> None:
        self.label = label
        self.probability = probability


def case(
    predicted: float,
    missed: bool,
    saving: float = 1_600.0,
    menu: frozenset[str] | None = frozenset({"a"}),
) -> Case:
    """A case whose prune is worth `saving` billed tokens; at a price of 1,600 it flips at 1.0."""
    return Case("s", menu, predicted, missed, saving)


# --- billed_saving and the flip point -----------------------------------------------------


def test_a_saving_is_the_tokens_removed_at_the_cached_rate_on_every_turn():
    """0.1 * 1000 * 4.6 = 460 billed-equivalent tokens."""
    assert billed_saving(1_000, CFG) == pytest.approx(460.0)


def test_the_flip_point_is_the_probability_where_the_prune_stops_paying():
    assert case(0.0, False, saving=460.0).flip_at(1_600.0) == pytest.approx(460.0 / 1_600.0)


def test_a_prune_that_saves_more_than_a_miss_costs_declines_at_no_probability():
    assert case(0.0, False, saving=46_000.0).flip_at(1_600.0) > 1.0


def test_a_free_miss_is_never_worth_declining_for():
    assert case(0.0, False, saving=460.0).flip_at(0.0) == float("inf")


# --- build_cases ------------------------------------------------------------------------


def test_a_session_gets_the_menu_the_shipped_pruner_picks_and_the_truth_beside_it():
    pool = [frozenset({"a", "b"})]
    sessions = [session("s1", {"a", "c"})]
    ranked = {"s1": [Pred("a", 0.8), Pred("b", 0.2)]}
    (built,) = build_cases(sessions, ranked, pool, LIVE, LIVE, TOKENS, CFG)
    assert built.menu == frozenset({"a", "b"})
    assert built.missed  # the session went on to call "c", which the menu does not hold
    assert built.predicted_miss == 0.0  # the model put all its mass on the menu, and was wrong


def test_a_session_the_router_would_not_prune_is_kept_and_marked_rather_than_dropped():
    """Dropping it would calibrate the rule on the population the rule already accepted."""
    (built,) = build_cases([session("s1", {"a"})], {"s1": []}, [], LIVE, LIVE, TOKENS, CFG)
    assert built.menu is None


def test_a_tool_the_agent_no_longer_offers_is_not_a_miss():
    """`live` is the surface today; a session that called a retired tool cannot miss on it."""
    sessions = [session("s1", {"a", "gone"})]
    ranked = {"s1": [Pred("a", 1.0)]}
    (built,) = build_cases(sessions, ranked, [frozenset({"a", "b"})], LIVE, LIVE, TOKENS, CFG)
    assert not built.missed


# --- reliability ------------------------------------------------------------------------


def test_each_bucket_reports_what_was_promised_against_what_happened():
    cases = [case(0.05, False), case(0.05, False), case(0.95, True), case(0.95, False)]
    low, high = reliability(cases, buckets=10)
    assert (low.n, low.predicted, low.observed) == (2, 0.05, 0.0)
    assert (high.n, high.predicted, high.observed) == (2, 0.95, 0.5)


def test_a_certain_miss_lands_in_the_last_bucket_rather_than_falling_off_the_end():
    (row,) = reliability([case(1.0, True)], buckets=10)
    assert (row.low, row.n) == (0.9, 1)


def test_empty_buckets_are_left_out_of_the_table():
    assert len(reliability([case(0.05, False), case(0.95, True)], buckets=10)) == 2


def test_the_sessions_the_router_declined_are_not_in_the_table():
    """They were never scored on the probability, so they cannot say whether it was right."""
    assert reliability([case(0.5, True, menu=None)], buckets=10) == []


# --- calibration_error ------------------------------------------------------------------


def test_a_model_that_is_right_about_its_own_odds_has_no_calibration_error():
    honest = [case(0.5, True), case(0.5, False)]
    assert calibration_error(reliability(honest, buckets=10)) == 0.0


def test_the_error_is_weighted_by_how_many_sessions_stood_behind_each_promise():
    """One bucket off by 0.9 with nine sessions must outweigh one off by 0.1 with one."""
    crowded = [case(0.95, False) for _ in range(9)] + [case(0.05, False)]
    assert calibration_error(reliability(crowded, buckets=10)) == pytest.approx(0.86)


def test_nothing_to_score_is_no_error_rather_than_a_division():
    assert calibration_error([]) == 0.0


# --- discrimination ---------------------------------------------------------------------


def test_a_score_that_ranks_every_miss_above_every_hit_is_perfect():
    assert discrimination([case(0.9, True), case(0.1, False)]) == 1.0


def test_a_score_that_says_the_same_thing_about_everyone_is_a_coin():
    assert discrimination([case(0.5, True), case(0.5, False)]) == 0.5


def test_a_backwards_score_scores_below_a_coin():
    assert discrimination([case(0.1, True), case(0.9, False)]) == 0.0


def test_a_corpus_where_nothing_missed_has_no_pair_to_order():
    assert discrimination([case(0.1, False), case(0.9, False)]) is None


# --- headroom ---------------------------------------------------------------------------


def test_a_prune_too_large_to_decline_cannot_be_changed_by_recalibrating_it():
    assert headroom([case(0.5, False, saving=2_240.0)], price=1_600.0)[0] == 0


def test_the_reachable_ones_are_counted_and_the_rest_priced_as_a_multiple():
    """Flip at 0.6 against a stated 0.2 means the probability must triple before it matters."""
    reachable, median = headroom([case(0.2, False, saving=960.0)], price=1_600.0)
    assert (reachable, median) == (1, pytest.approx(3.0))


def test_a_session_the_model_is_certain_about_is_left_out_of_the_multiple():
    """Dividing by a stated miss of zero is infinity, which is true and unprintable."""
    cases = [case(0.0, False, saving=960.0), case(0.2, False, saving=960.0)]
    assert headroom(cases, price=1_600.0)[1] == pytest.approx(3.0)


# --- calibrate --------------------------------------------------------------------------


def test_a_prediction_is_mapped_to_what_actually_happened_to_its_bucket():
    rows = [Bucket(0.0, 0.1, 10, 0.05, 0.0), Bucket(0.9, 1.0, 10, 0.95, 0.3)]
    assert calibrate(rows, 0.95) == 0.3


def test_a_prediction_no_bucket_covers_is_left_where_it_is():
    """The table only holds buckets that had sessions in them, so gaps are real."""
    assert calibrate([Bucket(0.0, 0.1, 10, 0.05, 0.0)], 0.5) == 0.5


def test_a_certain_miss_is_mapped_by_the_last_bucket_rather_than_falling_through():
    assert calibrate([Bucket(0.0, 0.1, 1, 0.05, 0.0), Bucket(0.9, 1.0, 1, 0.95, 0.4)], 1.0) == 0.4


# --- hazard -----------------------------------------------------------------------------


def test_the_two_columns_agree_while_the_prune_is_too_large_to_decline():
    """Both the stated and the true probability are below a flip point above 1.0."""
    rows = reliability([case(0.9, False) for _ in range(4)], buckets=10)
    assert hazard([case(0.9, False)], rows, price=1_600.0, multiples=[1.0]) == [(1.0, 0, 0)]


def test_the_columns_part_once_the_price_is_high_enough_to_decline_on():
    """Stated 0.9 against an observed 0.0: raising the price declines a prune that would pay."""
    rows = reliability([case(0.9, False) for _ in range(4)], buckets=10)
    (_, raw, true) = hazard([case(0.9, False)], rows, price=1_600.0, multiples=[100.0])[0]
    assert (raw, true) == (1, 0)


def test_the_sessions_the_router_never_pruned_are_not_swept():
    rows = reliability([case(0.9, False)], buckets=10)
    assert hazard([case(0.9, False, menu=None)], rows, 1_600.0, [100.0]) == [(100.0, 0, 0)]
