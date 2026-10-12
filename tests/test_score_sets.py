"""Tests for score_sets.py. Fully offline: no BigQuery client and no model is ever built.

The SQL is not tested here; it is checked by running it, and the run is recorded in
DESIGN.md 19.8. What is tested is the arithmetic behind the four metrics, and the wiring
that feeds the shipped decline rule, because a metric that is quietly wrong is worse than
no metric: it is the thing the next design decision gets made on.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from score_sets import (
    Arm,
    Session,
    fbeta,
    menu_always,
    menu_best,
    menu_router,
    menu_true,
    score,
    tok,
)

from adk_smart_router.config import RouterConfig
from adk_smart_router.predictor import Prediction
from adk_smart_router.pruner import ABSTAIN

# Three cheap tools and one expensive one, so token efficiency and plain precision cannot
# accidentally agree. `heavy` alone is worth more than the other three together.
TOKENS = {"a": 10, "b": 10, "c": 10, "heavy": 70}
LIVE = frozenset(TOKENS)


def session(*tools: str, turns: int = 3, billed: float = 1000.0) -> Session:
    return Session("s1", [0.0], frozenset(tools), turns, billed)


def scored(*tools: str, menu: frozenset[str] | None, turns: int = 3) -> Arm:
    arm = Arm("test")
    score(arm, session(*tools, turns=turns), menu, LIVE, TOKENS)
    return arm


# --- Recall ---------------------------------------------------------------------


def test_a_menu_holding_every_wanted_tool_scores_one():
    arm = scored("a", "b", menu=frozenset({"a", "b", "c"}))
    assert arm.recall_sum == pytest.approx(1.0)
    assert arm.complete == 1


def test_a_menu_missing_half_the_wanted_tools_scores_half():
    assert scored("a", "b", menu=frozenset({"a"})).recall_sum == pytest.approx(0.5)


def test_a_session_that_called_no_tool_has_no_recall_rather_than_a_zero():
    """Undefined, not 0.0. Averaging a zero in would punish exactly the sessions where
    pruning is pure profit (DESIGN.md 19.4)."""
    arm = scored(menu=frozenset({"a"}))
    assert arm.recall_n == 0
    assert arm.recall_sum == 0.0
    assert arm.complete == 1  # nothing was wanted, so nothing was missed


def test_a_wanted_tool_that_is_not_live_cannot_count_against_the_menu():
    """A tool the agent no longer declares is not a miss the router could have avoided."""
    assert scored("a", "retired", menu=frozenset({"a"})).recall_sum == pytest.approx(1.0)


# --- F-beta ---------------------------------------------------------------------


def test_a_perfect_menu_scores_a_perfect_fbeta():
    assert scored("a", "b", menu=frozenset({"a", "b"})).fbeta_sum == pytest.approx(1.0)


def test_missing_a_tool_costs_far_more_than_carrying_a_spare():
    """The whole point of beta 8: a miss is priced near 75 times an extra (DESIGN.md 19.1),
    so the penalty for width has to be nearly invisible next to the penalty for a gap."""
    # Three spare tools out of four declared, so precision is 0.25 and recall is perfect.
    spare = scored("a", menu=frozenset({"a", "b", "c", "heavy"})).fbeta_sum
    # One tool of two missed, so precision is perfect and recall is 0.5.
    missed = scored("a", "b", menu=frozenset({"a"})).fbeta_sum
    assert spare == pytest.approx(0.9559, abs=1e-4)
    assert missed == pytest.approx(0.5039, abs=1e-4)


def test_fbeta_is_undefined_on_a_session_that_wanted_nothing():
    """Same denominator as recall, so the two are always averaged over the same sessions."""
    arm = scored(menu=frozenset({"a"}))
    assert arm.recall_n == 0
    assert arm.fbeta_sum == 0.0


def test_a_menu_that_contains_nothing_wanted_scores_zero_rather_than_dividing_by_zero():
    assert scored("a", menu=frozenset({"b"})).fbeta_sum == 0.0


def test_fbeta_matches_the_textbook_formula():
    """A hand-checked value, so a refactor of the expression cannot quietly change it."""
    assert fbeta(0.5, 0.25) == pytest.approx(65 * 0.25 * 0.5 / (64 * 0.25 + 0.5))
    assert fbeta(1.0, 1.0) == pytest.approx(1.0)


# --- Declining ------------------------------------------------------------------


def test_declining_ships_the_full_list_and_scores_a_free_one():
    arm = scored("a", "b", menu=None)
    assert arm.declined == 1
    assert arm.recall_sum == pytest.approx(1.0)
    assert arm.efficiency_sum == 0.0
    assert arm.removed_billed == 0.0


def test_a_declined_session_is_kept_out_of_the_pruned_only_tallies():
    """The whole reason the second table exists: at a 36% routed share the blended recall
    is mostly measuring the decline rule."""
    arm = Arm("test")
    score(arm, session("a", "b"), None, LIVE, TOKENS)
    score(arm, session("a", "b"), frozenset({"a"}), LIVE, TOKENS)
    assert arm.recall_sum == pytest.approx(1.5)  # blended: one free 1.0 and one 0.5
    assert arm.pruned == 1
    assert arm.pruned_recall_sum == pytest.approx(0.5)
    assert arm.pruned_complete == 0


# --- Token efficiency -----------------------------------------------------------


def test_efficiency_is_one_when_everything_removable_is_removed():
    assert scored("a", menu=frozenset({"a"})).efficiency_sum == pytest.approx(1.0)


def test_efficiency_counts_tokens_not_tools():
    """Removing `heavy` alone beats removing all three cheap tools. That is the whole
    reason the metric is denominated in tokens (DESIGN.md 19.4)."""
    drop_heavy = scored("a", menu=frozenset({"a", "b", "c"}))
    drop_cheap = scored("a", menu=frozenset({"a", "heavy"}))
    assert drop_heavy.efficiency_sum == pytest.approx(70 / 90)
    assert drop_cheap.efficiency_sum == pytest.approx(20 / 90)


def test_a_session_wanting_everything_has_no_efficiency_to_measure():
    """Nothing was removable, so removing nothing is not a failure to remove."""
    arm = scored(*LIVE, menu=LIVE)
    assert arm.efficiency_n == 0


# --- Predicted saving -----------------------------------------------------------


def test_removed_tokens_are_billed_at_a_tenth_and_once_per_turn():
    """The core arithmetic of DESIGN.md 19.5 criterion 3: you save a tenth of what you
    remove, because the declarations you removed were cached anyway."""
    arm = scored("a", menu=frozenset({"a"}), turns=4)
    assert arm.removed_billed == pytest.approx(0.1 * 90 * 4)


def test_a_session_with_no_llm_turns_contributes_no_saving():
    assert scored("a", menu=frozenset({"a"}), turns=0).removed_billed == 0.0


# --- Exact set match ------------------------------------------------------------


def test_exact_match_ignores_tools_that_are_not_live():
    assert scored("a", "retired", menu=frozenset({"a"})).exact == 1


def test_a_menu_with_one_spare_tool_is_not_an_exact_match():
    assert scored("a", menu=frozenset({"a", "b"})).exact == 0


# --- The oracle arms ------------------------------------------------------------


def test_the_best_menu_is_the_one_covering_most_of_what_was_wanted():
    pool = [frozenset({"a", "b"}), frozenset({"c", "heavy"})]
    assert menu_best(session("c"), pool, LIVE, frozenset()) == frozenset({"c", "heavy"})


def test_a_tie_between_menus_falls_to_the_smaller_one():
    """Otherwise the oracle quietly rewards width and overstates the pool's ceiling."""
    pool = [frozenset({"a", "b", "c"}), frozenset({"a"})]
    assert menu_best(session("a"), pool, LIVE, frozenset()) == frozenset({"a"})


def test_an_empty_pool_declines_rather_than_inventing_a_menu():
    assert menu_best(session("a"), [], LIVE, frozenset()) is None


def test_both_oracles_carry_the_protected_tools():
    protected = frozenset({"heavy"})
    assert menu_true(session("a"), LIVE, protected) == frozenset({"a", "heavy"})
    pool = [frozenset({"a"})]
    assert menu_best(session("a"), pool, LIVE, protected) == frozenset({"a", "heavy"})


# --- The two router arms ----------------------------------------------------------
#
# The decline rule itself lives in `pruner.worth_pruning` and is proven in test_pruner.py.
# What is tested here is the wiring: that the arm feeds that rule the right token counts,
# the right probability mass, and the corpus's own session length.


class FakePredictor:
    """Just enough of `Predictor` for the router arms: one fixed ranking."""

    def __init__(self, mass: dict[str, float]) -> None:
        self._ranked = [Prediction(label, p) for label, p in mass.items()]

    def predict(self, vector: Sequence[float]) -> list[Prediction]:
        return list(self._ranked)


def routed(
    mass: dict[str, float],
    pool: Sequence[frozenset[str]],
    *,
    turns: float = 1.0,
    miss_cost: int = 100,
    known: frozenset[str] = LIVE,
    never_prune: frozenset[str] = frozenset(),
) -> frozenset[str] | None:
    """The script's own loop: pick a menu, then ask whether sending it pays."""
    cfg = RouterConfig(
        min_tools_to_route=1,  # the fixture has four tools; the shipped floor is twenty
        never_prune=never_prune,
        miss_cost_tokens=miss_cost,
        expected_session_turns=turns,
    )
    ranked = FakePredictor(mass).predict([0.0])
    return menu_router(ranked, menu_always(ranked, LIVE, known, cfg, pool), LIVE, TOKENS, cfg)


CHEAP_TOOLS = [frozenset({"a", "b", "c"})]  # leaves `heavy` behind, so 70 tokens removed


def test_a_small_removal_is_not_worth_an_uncertain_menu():
    """0.1 * 70 * 1 = 7 saved against 0.5 * 100 = 50 at risk. Decline."""
    assert routed({"a": 0.5, "heavy": 0.5}, CHEAP_TOOLS) is None


def test_the_same_menu_routes_once_the_session_is_long_enough():
    """The saving is per turn and the risk is not, so length alone flips the decision:
    0.1 * 70 * 20 = 140 against the same 50."""
    assert routed({"a": 0.5, "heavy": 0.5}, CHEAP_TOOLS, turns=20) == frozenset({"a", "b", "c"})


def test_certainty_makes_even_a_tiny_saving_worth_taking():
    """Nothing outside the menu has any mass, so there is no risk to weigh 7 tokens against."""
    assert routed({"a": 1.0}, CHEAP_TOOLS) == frozenset({"a", "b", "c"})


def test_abstain_mass_counts_as_contained():
    """DESIGN.md 19.5: a session the model is confident needs no tool is a session where
    every removal is free. Here 0.1 * 90 = 9 saved beats 0.1 * 50 = 5 at risk only because
    ABSTAIN's 0.9 is on the menu's side; count it as a miss and the arm declines."""
    pool = [frozenset({"a"})]
    assert routed({ABSTAIN: 0.9, "heavy": 0.1}, pool, miss_cost=50) == frozenset({"a"})
    assert routed({"b": 0.9, "heavy": 0.1}, pool, miss_cost=50) is None


def test_no_prediction_declines():
    assert routed({}, CHEAP_TOOLS) is None


def test_no_pool_falls_back_to_the_pruner_own_set():
    """The published artifact always carries menus, but the arm is the shipped code and the
    shipped code has a bespoke-set path behind it (DESIGN.md 18.16)."""
    assert routed({"a": 1.0}, []) == frozenset({"a"})


def test_unknown_and_protected_tools_ride_along():
    """`c` is not a class the model knows and `b` is never prunable, so neither is the
    menu's to drop even though the chosen menu holds only `a`."""
    menu = routed({"a": 1.0}, [frozenset({"a"})], known=LIVE - {"c"}, never_prune=frozenset({"b"}))
    assert menu == frozenset({"a", "b", "c"})


def test_a_menu_the_pruner_declined_is_never_second_guessed():
    """`menu_always` returning None is a passthrough, not a cheap menu with no tokens in it."""
    assert menu_router([Prediction("a", 1.0)], None, LIVE, TOKENS, RouterConfig()) is None


# --- Token arithmetic -----------------------------------------------------------


def test_an_unknown_tool_costs_nothing_rather_than_raising():
    """The declaration sizes come from the live agent and the tool sets come from history,
    so a retired tool appears in one and not the other."""
    assert tok(frozenset({"a", "retired"}), TOKENS) == 10
