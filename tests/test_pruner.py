"""Exhaustive tests for the pruning rule.

`pruner` is pure, so everything here is plain arithmetic on frozensets. The two
DESIGN.md 7.3 worked examples appear as named tests with their arithmetic spelled
out, because those numbers are the entire justification for the floor guard.

Predictions are a local NamedTuple stub: `pruner.Prediction` is a Protocol, so
this file has no dependency on `predictor.py`.
"""

from __future__ import annotations

import random
from typing import NamedTuple

from adk_smart_router.config import FRAMEWORK_TOOLS, RouterConfig
from adk_smart_router.pruner import (
    PASSTHROUGH_SOURCES,
    SessionDecision,
    breaks_cache,
    contained_mass,
    decide,
    top_k_by_mass,
    worth_pruning,
)

TRANSFER = "transfer_to_agent"


class Pred(NamedTuple):
    """Structural match for pruner.Prediction."""

    label: str
    probability: float


def cfg(**overrides: object) -> RouterConfig:
    """A config that routes small tool sets, so tests can use readable fixtures."""
    base: dict[str, object] = {
        "min_tools_to_route": 3,
        "max_tools_kept": 5,
        "keep_probability_mass": 0.90,
    }
    base.update(overrides)
    return RouterConfig(**base)  # type: ignore[arg-type]


def tools(*names: str) -> frozenset[str]:
    return frozenset(names)


LIVE = tools("a", "b", "c", "d", "e", "f", TRANSFER)
KNOWN = tools("a", "b", "c", "d", "e", "f", "ABSTAIN")


# --- top_k_by_mass ------------------------------------------------------------


def test_top_k_stops_once_mass_is_reached():
    preds = [Pred("a", 0.6), Pred("b", 0.35), Pred("c", 0.05)]
    # 0.6 < 0.9, 0.6 + 0.35 = 0.95 >= 0.9 -> stop after two.
    assert top_k_by_mass(preds, 0.90, 5) == ("a", "b")


def test_top_k_is_hard_capped():
    preds = [Pred(chr(ord("a") + i), 0.1) for i in range(10)]
    assert len(top_k_by_mass(preds, 0.99, 3)) == 3


def test_top_k_never_returns_empty_for_non_empty_preds():
    preds = [Pred("a", 0.01)]
    # Mass is never reached and the cap is degenerate; one label still comes back.
    assert top_k_by_mass(preds, 0.90, 0) == ("a",)
    assert top_k_by_mass(preds, 0.90, 5) == ("a",)


def test_top_k_empty_preds():
    assert top_k_by_mass([], 0.90, 5) == ()


def test_top_k_breaks_ties_by_label_ascending():
    preds = [Pred("z", 0.25), Pred("a", 0.25), Pred("m", 0.25), Pred("b", 0.25)]
    assert top_k_by_mass(preds, 0.90, 4) == ("a", "b", "m", "z")
    # Deterministic regardless of input order.
    assert top_k_by_mass(list(reversed(preds)), 0.90, 4) == ("a", "b", "m", "z")


def test_top_k_orders_by_descending_probability():
    preds = [Pred("low", 0.1), Pred("high", 0.8), Pred("mid", 0.1)]
    assert top_k_by_mass(preds, 1.0, 5)[0] == "high"


# --- Rule 1: transfer_to_agent is never pruned --------------------------------


def test_transfer_to_agent_survives_every_prediction_vector():
    c = cfg()
    rng = random.Random(11)
    labels = sorted(KNOWN - {"ABSTAIN"})
    for _ in range(200):
        preds = sorted(
            (Pred(name, rng.random()) for name in labels),
            key=lambda p: -p.probability,
        )
        result = decide(preds, LIVE, KNOWN, None, c)
        if result.kept is not None:
            assert TRANSFER in result.kept


def test_never_prune_names_are_protected_like_framework_tools():
    c = cfg(never_prune=frozenset({"audit_log"}))
    live = LIVE | {"audit_log"}
    known = KNOWN | {"audit_log"}
    result = decide([Pred("a", 0.99)], live, known, None, c)
    assert result.kept is not None
    assert {"audit_log", TRANSFER} <= result.kept


# --- Unknown tools ------------------------------------------------------------


def test_unknown_tools_are_always_kept():
    c = cfg()
    live = LIVE | {"brand_new_mcp_tool", "another_new_one"}
    result = decide([Pred("a", 0.99)], live, KNOWN, None, c)
    assert result.kept is not None
    assert {"brand_new_mcp_tool", "another_new_one"} <= result.kept


def test_a_tool_the_operator_declared_prunable_stops_riding_through_on_the_passthrough():
    c = cfg(prunable_by_default=frozenset({"dead_tool"}))
    live = LIVE | {"dead_tool", "brand_new_mcp_tool"}
    result = decide([Pred("a", 0.99)], live, KNOWN, None, c)
    assert result.kept is not None
    assert "dead_tool" not in result.kept
    # Only the declared name goes. Every other classless tool still rides through, because
    # nothing about the operator's judgement on one tool says anything about another.
    assert "brand_new_mcp_tool" in result.kept


def test_a_tool_the_model_predicts_survives_being_declared_prunable():
    """The knob subtracts from the passthrough and from nothing else. A tool the model
    asked for by name is kept on the prediction, whatever the config says about silence."""
    c = cfg(prunable_by_default=frozenset({"a"}))
    result = decide([Pred("a", 0.99)], LIVE, KNOWN, None, c)
    assert result.kept is not None
    assert "a" in result.kept


def test_a_tool_in_both_never_prune_and_prunable_by_default_is_kept():
    """The contradiction resolves in the safe direction, and `startup_checks` reports it."""
    c = cfg(never_prune=frozenset({"audit_log"}), prunable_by_default=frozenset({"audit_log"}))
    live = LIVE | {"audit_log"}
    result = decide([Pred("a", 0.99)], live, KNOWN, None, c)
    assert result.kept is not None
    assert "audit_log" in result.kept


def test_declaring_the_framework_tool_prunable_does_not_disable_delegation():
    c = cfg(prunable_by_default=frozenset({TRANSFER}))
    result = decide([Pred("a", 0.99)], LIVE, KNOWN, None, c)
    assert result.kept is not None
    assert TRANSFER in result.kept


def test_the_shipped_default_names_nothing_and_so_prunes_nothing_extra():
    """The mechanism is inert until an operator acts, which is the point of shipping it empty."""
    assert RouterConfig().prunable_by_default == frozenset()
    live = LIVE | {"brand_new_mcp_tool", "another_new_one"}
    result = decide([Pred("a", 0.7), Pred("b", 0.2)], live, KNOWN, None, cfg())
    assert result.kept is not None
    assert {"brand_new_mcp_tool", "another_new_one"} <= result.kept


def test_abstain_is_not_treated_as_a_known_label():
    # "ABSTAIN" is stripped from `known`, but it is not a live tool either, so it
    # cannot leak into `unknown`.
    c = cfg()
    result = decide([Pred("a", 0.99)], LIVE, KNOWN, None, c)
    assert result.kept is not None
    assert "ABSTAIN" not in result.kept


# --- No prediction at all (rule 2) --------------------------------------------
#
# This branch used to catch weak predictions and a top-ranked ABSTAIN as well. It does
# not any more: how sure the model is decides nothing on its own, because a doubtful
# menu that removes a lot of tokens is still worth sending. `worth_pruning` weighs the
# two against each other, and the caller asks it (DESIGN.md 19.5). What is left here is
# the case with nothing to weigh, where there is no prediction to prune towards.


def test_empty_preds_passes_through_as_low_confidence():
    result = decide([], LIVE, KNOWN, None, cfg())
    assert result.kept is None
    assert result.source == "lowconf"
    assert result.low_conf_turns == 1
    assert result.terminal_passthrough is False


def test_a_weak_prediction_still_produces_a_menu():
    # Under the old rule this was a passthrough at any probability below 0.55.
    result = decide([Pred("a", 0.30), Pred("b", 0.20)], LIVE, KNOWN, None, cfg())
    assert result.kept == frozenset({"a", "b", TRANSFER})
    assert result.source == "fresh"


def test_abstain_as_top_one_no_longer_vetoes_the_prune():
    # A session the model is sure wants no tool is the cheapest prune there is: every
    # removal is free and nothing can be missed. ABSTAIN is not a live tool, so the mass
    # walk keeps nothing but the protected framework tool.
    preds = [Pred("ABSTAIN", 0.99), Pred("a", 0.01)]
    result = decide(preds, LIVE, KNOWN, None, cfg())
    assert result.kept == frozenset({TRANSFER})
    assert result.source == "fresh"


def test_abstain_with_menus_still_lands_on_a_menu():
    # Serving always has menus, and no menu contains ABSTAIN, so the session gets the one
    # its remaining mass points at rather than a tool list of nothing.
    preds = [Pred("ABSTAIN", 0.95), Pred("d", 0.04), Pred("a", 0.01)]
    result = decide(preds, LIVE, KNOWN, None, cfg(), MENUS)
    assert result.kept == frozenset({"b", "d", "e", TRANSFER})


def test_no_prediction_once_then_twice_becomes_terminal():
    c = cfg()

    first = decide([], LIVE, KNOWN, None, c)
    assert (first.kept, first.low_conf_turns, first.terminal_passthrough) == (None, 1, False)

    second = decide([], LIVE, KNOWN, first, c)
    assert (second.kept, second.low_conf_turns, second.terminal_passthrough) == (None, 2, True)


def test_no_prediction_does_not_narrow_a_frozen_kept_set():
    # Rule 2: uncertainty widens the aperture even when a kept set already exists.
    c = cfg()
    routed = decide([Pred("a", 0.99)], LIVE, KNOWN, None, c)
    assert routed.kept is not None

    blind = decide([], LIVE, KNOWN, routed, c)
    assert blind.kept is None
    assert blind.source == "lowconf"


def test_terminal_passthrough_is_sticky_against_a_confident_prediction():
    c = cfg()
    terminal = SessionDecision(kept=None, low_conf_turns=2, terminal_passthrough=True)
    result = decide([Pred("a", 0.999)], LIVE, KNOWN, terminal, c)
    assert result.kept is None
    assert result.terminal_passthrough is True
    assert result.source == "lowconf"


def test_terminal_passthrough_beats_even_the_small_check():
    # The terminal check is first, so it wins regardless of anything downstream.
    c = cfg(min_tools_to_route=100)
    terminal = SessionDecision(kept=None, low_conf_turns=5, terminal_passthrough=True)
    assert decide([Pred("a", 0.9)], LIVE, KNOWN, terminal, c).source == "lowconf"


def test_probability_alone_decides_nothing():
    # The same prediction at four confidences produces the same menu. What separates a
    # prune from a passthrough is now what the menu saves, which `decide` cannot see.
    c = cfg()
    kept = {decide([Pred("a", p)], LIVE, KNOWN, None, c).kept for p in (0.01, 0.3, 0.55, 0.99)}
    assert kept == {frozenset({"a", TRANSFER})}


# --- Qualification floor ------------------------------------------------------


def test_too_few_live_tools_passes_through_as_small():
    c = cfg(min_tools_to_route=20)
    result = decide([Pred("a", 0.99)], LIVE, KNOWN, None, c)
    assert result.kept is None
    assert result.source == "small"


def test_a_small_passthrough_clears_the_low_confidence_counter():
    # A turn the model was never asked about is not evidence that the session is
    # off-distribution, and `terminal_passthrough` is irreversible once it latches.
    c = cfg(min_tools_to_route=20)
    prior = SessionDecision(kept=frozenset({"a"}), low_conf_turns=1)
    result = decide([Pred("a", 0.99)], LIVE, KNOWN, prior, c)
    assert (result.kept, result.low_conf_turns, result.terminal_passthrough) == (None, 0, False)


# --- Fresh and reused ---------------------------------------------------------


def test_fresh_decision_keeps_the_mass_walk_plus_protected():
    c = cfg()
    preds = [Pred("a", 0.7), Pred("b", 0.25), Pred("c", 0.05)]
    result = decide(preds, LIVE, KNOWN, None, c)
    assert result.kept == frozenset({"a", "b", TRANSFER})
    assert result.source == "fresh"


def test_same_top_label_on_a_later_turn_is_reused_with_no_cache_break():
    c = cfg()
    first = decide([Pred("a", 0.99)], LIVE, KNOWN, None, c)
    assert first.kept == frozenset({"a", TRANSFER})

    second = decide([Pred("a", 0.97)], LIVE, KNOWN, first, c)
    assert second.kept == first.kept
    assert second.source == "reused"


def test_a_brand_new_top_label_does_not_widen_a_session_that_already_chose():
    # The set is chosen once. `b` was pruned away on turn one and stays pruned away,
    # because widening mid-session breaks the prefix this session is being billed under.
    c = cfg()
    first = decide([Pred("a", 0.99)], LIVE, KNOWN, None, c)
    second = decide([Pred("b", 0.99), Pred("c", 0.98)], LIVE, KNOWN, first, c)
    assert second.kept == first.kept
    assert second.source == "reused"


def test_a_kept_set_survives_any_number_of_new_intents():
    # The deleted growth path allowed three widenings before pinning to full. There is
    # now no number of new intents that moves the declared list.
    c = cfg()
    decision = decide([Pred("a", 0.99)], LIVE, KNOWN, None, c)
    first_kept = decision.kept
    sources = [decision.source]
    for label in ("b", "c", "d", "e", "f"):
        decision = decide([Pred(label, 0.99)], LIVE, KNOWN, decision, c)
        sources.append(decision.source)
    assert decision.kept == first_kept
    assert sources == ["fresh", "reused", "reused", "reused", "reused", "reused"]


# --- No-op --------------------------------------------------------------------


def test_kept_equal_to_live_is_a_noop():
    c = cfg(max_tools_kept=10)
    live = tools("a", "b", "c", TRANSFER)
    known = tools("a", "b", "c")
    preds = [Pred("a", 0.34), Pred("b", 0.33), Pred("c", 0.33)]
    result = decide(preds, live, known, None, c)
    assert result.kept is None
    assert result.source == "noop"


def test_all_live_tools_unknown_is_a_noop_not_a_prune():
    # Every live tool is unknown, so `kept` == `live` and there is nothing to prune.
    c = cfg()
    known = tools("x", "y", "z")
    result = decide([Pred("x", 0.99)], LIVE, known, None, c)
    assert result.kept is None
    assert result.source == "noop"


def test_a_noop_clears_the_low_confidence_counter():
    # Same reason as the small passthrough: this turn had a confident prediction, it just
    # had nothing to prune, so it is not evidence for pinning the session to full.
    c = cfg(max_tools_kept=10)
    live = tools("a", "b", "c", TRANSFER)
    known = tools("a", "b", "c")
    prior = SessionDecision(kept=None, low_conf_turns=1)
    preds = [Pred("a", 0.34), Pred("b", 0.33), Pred("c", 0.33)]
    result = decide(preds, live, known, prior, c)
    assert result.source == "noop"
    assert result.low_conf_turns == 0


# --- Canonical menus ----------------------------------------------------------
#
# With menus, the pruner stops inventing a tool set per session and picks the closest of a
# few published ones instead. The tools kept barely change; what changes is how many
# distinct declaration lists the cache has to hold, which is what decided G1 (DESIGN.md
# 18.16). `menus.py` owns the choosing and the ranking; what these tests pin down is how
# `decide` uses them, and in particular that a session snaps to exactly one menu and then
# stops asking.

MENUS = (tools("a", "b"), tools("c", "d"), tools("b", "d", "e"))


def test_a_fresh_decision_snaps_to_the_menu_holding_the_most_mass():
    c = cfg()
    preds = [Pred("a", 0.7), Pred("b", 0.2), Pred("c", 0.1)]
    result = decide(preds, LIVE, KNOWN, None, c, MENUS)
    assert result.kept == frozenset({"a", "b", TRANSFER})
    assert result.source == "fresh"


def test_a_menu_beats_the_mass_walk_even_when_it_keeps_a_different_tool():
    """The bespoke set would be {c, TRANSFER}. The menu brings `d` along, which is the
    price of a shared prefix and the reason menus are measured before they are shipped."""
    c = cfg()
    preds = [Pred("c", 0.95), Pred("a", 0.05)]
    assert decide(preds, LIVE, KNOWN, None, c, MENUS).kept == frozenset({"c", "d", TRANSFER})


def test_a_menu_naming_a_tool_that_is_not_installed_keeps_the_rest():
    """Menus are published from a corpus that may be older than the deployment."""
    c = cfg()
    result = decide([Pred("a", 0.9)], LIVE, KNOWN, None, c, (tools("a", "ghost"),))
    assert result.kept == frozenset({"a", TRANSFER})


def test_a_menu_of_nothing_installed_still_leaves_the_control_plane():
    c = cfg()
    result = decide([Pred("a", 0.9)], LIVE, KNOWN, None, c, (tools("ghost", "spectre"),))
    assert result.kept == frozenset({TRANSFER})


def test_unknown_tools_are_added_to_the_menu_not_filtered_by_it():
    """`g` is live but was never in training, so no menu can name it. Uncertainty widens
    the aperture here exactly as it does everywhere else (DESIGN.md 7.2, rule 2)."""
    c = cfg()
    live = LIVE | {"g"}
    result = decide([Pred("a", 0.9)], live, KNOWN, None, c, MENUS)
    assert result.kept == frozenset({"a", "b", "g", TRANSFER})


def test_a_session_does_not_hop_menus_when_its_intent_changes():
    """The session was on {a, b} and now wants `d`, which {b, d, e} would have covered.
    It stays on {a, b} regardless: the hop was the growth path, and the growth path is
    gone. What a hop would have cost is a cold prefix; what staying costs is the miss,
    and `worth_pruning` priced that miss before the first prune happened."""
    c = cfg()
    first = decide([Pred("a", 0.7), Pred("b", 0.2)], LIVE, KNOWN, None, c, MENUS)
    assert first.kept == frozenset({"a", "b", TRANSFER})

    second = decide([Pred("d", 0.9), Pred("b", 0.05)], LIVE, KNOWN, first, c, MENUS)
    assert second.kept == first.kept
    assert second.source == "reused"


def test_a_repeat_of_the_same_intent_still_costs_nothing():
    c = cfg()
    first = decide([Pred("a", 0.7), Pred("b", 0.2)], LIVE, KNOWN, None, c, MENUS)
    second = decide([Pred("b", 0.8), Pred("a", 0.1)], LIVE, KNOWN, first, c, MENUS)
    assert second.kept == first.kept
    assert second.source == "reused"


def test_a_menu_covering_everything_live_is_a_noop():
    c = cfg()
    result = decide([Pred("a", 0.9)], LIVE, KNOWN, None, c, (LIVE - {TRANSFER},))
    assert result.kept is None
    assert result.source == "noop"


def test_no_menus_is_the_old_behaviour_unchanged():
    """The parameter defaults to empty, so an artifact published before menus existed
    still routes. Same predictions as the snap test above.

    The walk stops at `b` and `c` is dropped, because 0.7 + 0.2 is the whole 0.90 the
    threshold asks for. This test used to assert that `c` was kept, which was not the rule
    but the float: accumulating those two left to right gives 0.8999999999999999, a hair
    under the threshold, so the walk took one more label than it was entitled to.
    `pruner.MASS_TOLERANCE` closes that, and this vector is the plainest case of it there
    is, which is a fair measure of how often it was happening.
    """
    c = cfg()
    preds = [Pred("a", 0.7), Pred("b", 0.2), Pred("c", 0.1)]
    assert decide(preds, LIVE, KNOWN, None, c).kept == frozenset({"a", "b", TRANSFER})


# --- worth_pruning: the decline rule ------------------------------------------
#
# The rule that replaced `min_confidence_to_route` (DESIGN.md 19.5). Like the floor guard
# it is a pure function of numbers the caller measured, because only the caller can see
# the request the tokens are in. The defaults it reads are a miss at 1,600 billed tokens,
# a cached token at a tenth of an uncached one, and 4.6 model calls a session, so the left
# side of the comparison is `0.1 * saved * 4.6` and the right is `(1 - mass) * 1600`.


def test_a_certain_prune_pays_for_almost_nothing():
    # Contained mass of 1.0 makes a miss impossible, so any saving at all wins.
    assert worth_pruning(1, 1.0, cfg()) is True


def test_a_prune_that_removes_nothing_never_pays():
    # Zero saving cannot beat a positive risk, and ties go to not pruning.
    assert worth_pruning(0, 0.99, cfg()) is False
    assert worth_pruning(0, 1.0, cfg()) is False


def test_the_break_even_point_is_where_the_two_sides_meet():
    # A 5% miss chance prices the risk at 80 tokens, which needs 174 removed to beat.
    # 0.1 * 173 * 4.6 = 79.58, and 0.1 * 174 * 4.6 = 80.04.
    assert worth_pruning(173, 0.95, cfg()) is False
    assert worth_pruning(174, 0.95, cfg()) is True


def test_a_dearer_miss_declines_more():
    # The same menu and the same odds, priced twice. This is the whole frontier.
    assert worth_pruning(400, 0.95, cfg(miss_cost_tokens=1600)) is True
    assert worth_pruning(400, 0.95, cfg(miss_cost_tokens=4000)) is False


def test_a_longer_session_prunes_more():
    # The saving accrues every turn and the risk is paid once, so length only helps.
    assert worth_pruning(100, 0.90, cfg(expected_session_turns=4.6)) is False
    assert worth_pruning(100, 0.90, cfg(expected_session_turns=20.0)) is True


def test_confidence_and_size_trade_against_each_other():
    # A doubtful prune that removes a lot beats a near-certain one that removes little,
    # which is the thing a confidence threshold could not express.
    doubtful_but_large = worth_pruning(2000, 0.80, cfg())
    certain_but_tiny = worth_pruning(20, 0.99, cfg())
    assert (doubtful_but_large, certain_but_tiny) == (True, False)


# --- contained_mass -----------------------------------------------------------


def test_contained_mass_adds_up_what_survived():
    preds = [Pred("a", 0.5), Pred("b", 0.3), Pred("c", 0.2)]
    assert contained_mass(preds, tools("a", "b")) == 0.8


def test_abstain_counts_as_contained():
    # Nothing can be missed by a session that calls no tool, so its mass is not risk.
    preds = [Pred("ABSTAIN", 0.5), Pred("a", 0.25), Pred("b", 0.25)]
    assert contained_mass(preds, tools("a")) == 0.75


def test_contained_mass_of_nothing_is_zero():
    assert contained_mass([], tools("a")) == 0.0
    assert contained_mass([Pred("a", 1.0)], frozenset()) == 0.0


# --- breaks_cache: the floor guard --------------------------------------------
#
# The guard is a pure function of three numbers and knows nothing about tools. That is
# the correction: v1 asked it about a tool count and a config constant, which cannot see
# the conversation history that actually carries a prompt over the floor (DESIGN.md
# 18.13). Both token counts here are whole prompts.


def test_pruning_across_the_floor_is_refused():
    # Cacheable at 2600, not cacheable at 2000, so the prune turns the discount off and
    # every kept token goes back to billing at ten times what it did.
    assert breaks_cache(2600, 2000, 2048) is True


def test_pruning_that_stays_above_the_floor_is_allowed():
    assert breaks_cache(2600, 2200, 2048) is False


def test_pruning_below_a_floor_nothing_reached_is_allowed():
    # Neither arm was ever cacheable, so there is no discount to lose.
    assert breaks_cache(2600, 2000, 4096) is False


def test_exactly_on_the_floor_counts_as_cacheable():
    assert breaks_cache(2048, 2047, 2048) is True
    assert breaks_cache(2049, 2048, 2048) is False


def test_an_unknown_floor_means_no_opinion():
    # None is "no guard possible", not "no floor". Refusing every prune on an
    # unrecognised model would turn the router off for every non-Gemini deployment.
    assert breaks_cache(9999, 10, None) is False


def test_the_guard_no_longer_lives_in_decide():
    """`decide` prunes an 8-tool set that a 2048 floor would once have protected.

    The same numbers as the DESIGN.md 7.3 worked example. Whether applying this pays is
    now the plugin's question, asked per turn against the measured request, so the rule
    module answers only "what would we keep".
    """
    c = cfg(
        max_tools_kept=5,
        min_tools_to_route=3,
        keep_probability_mass=1.0,
    )
    live = tools("t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7")
    preds = [Pred(f"t{i}", 0.125) for i in range(8)]

    result = decide(preds, live, live, None, c)
    assert result.kept is not None
    assert len(result.kept) == 5
    assert result.source == "fresh"


# --- Property test ------------------------------------------------------------


def test_property_decide_never_returns_an_empty_or_out_of_range_kept_set():
    """~2000 pseudo-random prediction vectors over randomised sessions.

    Two invariants: a non-None `kept` is never empty, and it never contains a name
    that is not live. Both are DESIGN.md 7.2 rule 3 in different words.

    Half the sessions carry menus, drawn without regard for what is live, because a
    published artifact can name tools this deployment does not have.
    """
    rng = random.Random(20260831)
    universe = [f"tool_{i}" for i in range(24)]

    for _ in range(2000):
        live = frozenset(rng.sample(universe, rng.randint(1, 24))) | {TRANSFER}
        known = frozenset(rng.sample(universe, rng.randint(0, 24))) | {"ABSTAIN"}
        label_pool = [*universe, "ABSTAIN", TRANSFER, "ghost_tool"]
        menu_pool = [*universe, "ghost_tool"]
        menus = tuple(
            frozenset(rng.sample(menu_pool, rng.randint(1, 8)))
            for _ in range(rng.randint(0, 6) if rng.random() < 0.5 else 0)
        )
        c = cfg(
            min_tools_to_route=rng.randint(0, 8),
            max_tools_kept=rng.randint(1, 8),
            keep_probability_mass=rng.choice([0.5, 0.9, 0.99, 1.0]),
        )
        prior: SessionDecision | None = None
        for _turn in range(rng.randint(1, 5)):
            n = rng.randint(0, 6)
            preds = sorted(
                (Pred(rng.choice(label_pool), rng.random()) for _ in range(n)),
                key=lambda p: -p.probability,
            )
            decision = decide(preds, live, known, prior, c, menus)

            assert decision.source in PASSTHROUGH_SOURCES | {"fresh", "reused"}
            if decision.kept is None:
                assert decision.source in PASSTHROUGH_SOURCES
            else:
                assert decision.kept, "a non-None kept set must never be empty"
                assert decision.kept <= live, "kept must never name a tool that is not live"
                assert decision.kept < live, "a kept set equal to live should have no-opped"
                assert TRANSFER in decision.kept
            prior = decision


def test_property_operator_safety_configuration_survives_every_decision():
    """~2000 randomised sessions that also randomise `never_prune` and `prunable_by_default`.

    The walk above pins `kept <= live` but holds the operator's configuration fixed, which
    leaves the two settings an operator actually reaches for untested against random input.
    `never_prune` is the answer `docs/threat-model.md` gives to prompt injection: put the
    escalation and audit tools in it and no message can talk the classifier into dropping
    them. That answer is only worth giving if it holds unconditionally.

    The guarantee is about what the model is *shown*, not about `kept`, so it has two arms. A
    passthrough decision declares everything and satisfies it trivially; a `kept` set has to
    contain the protected tools outright. Stating it as "declared" rather than "in kept" is
    what makes it true on every branch instead of most of them.

    `prunable_by_default` is the opposite setting and gets the opposite invariant: it is
    subtracted from the ride-along only, so it must never be able to remove a tool the model
    asked for, and never beat `never_prune` when an operator sets both.
    """
    rng = random.Random(20260906)
    universe = [f"tool_{i}" for i in range(24)]

    for _ in range(2000):
        live = frozenset(rng.sample(universe, rng.randint(1, 24))) | {TRANSFER}
        known = frozenset(rng.sample(universe, rng.randint(0, 24))) | {"ABSTAIN"}
        # Drawn from the whole universe rather than from `live`, because an operator names
        # tools in configuration and a deployment can be missing any of them.
        never = frozenset(rng.sample(universe, rng.randint(0, 6)))
        prunable = frozenset(rng.sample(universe, rng.randint(0, 6)))
        label_pool = [*universe, "ABSTAIN", TRANSFER, "ghost_tool"]
        c = cfg(
            min_tools_to_route=rng.randint(0, 8),
            max_tools_kept=rng.randint(1, 8),
            keep_probability_mass=rng.choice([0.5, 0.9, 0.99, 1.0]),
            never_prune=never,
            prunable_by_default=prunable,
        )
        prior: SessionDecision | None = None
        for _turn in range(rng.randint(1, 5)):
            n = rng.randint(0, 6)
            preds = sorted(
                (Pred(rng.choice(label_pool), rng.random()) for _ in range(n)),
                key=lambda p: -p.probability,
            )
            decision = decide(preds, live, known, prior, c)

            # What the model is shown: the whole live set on a passthrough, else `kept`.
            declared = live if decision.kept is None else decision.kept
            assert (never & live) <= declared, "a never_prune tool was withheld from the model"
            assert (FRAMEWORK_TOOLS & live) <= declared, "a framework tool was withheld"
            assert (never & prunable & live) <= declared, "prunable_by_default beat never_prune"

            if decision.source == "fresh":
                # `prunable_by_default` is subtracted from the ride-along and nowhere else, so
                # a tool the model actually ranked into the menu has to survive it. Only
                # checked on `fresh`: a `reused` set was chosen by an earlier turn's
                # predictions and owes nothing to this turn's.
                asked = frozenset(top_k_by_mass(preds, c.keep_probability_mass, c.max_tools_kept))
                assert (asked & live) <= declared, "a predicted tool was dropped as prunable"
            prior = decision
