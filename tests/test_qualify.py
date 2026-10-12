"""The bands, their boundaries, and the two agents they were calibrated on.

`qualify` is the only thing in this project that answers before any money is spent, so it
is also the only thing whose wrong answer costs a user a provisioned GCP project and a
corpus. Two of these tests are the whole calibration: the agent that lost money must not
come out of here as `pays`, and the agent that won must not come out as `thin`. The rest
pin the boundaries, because every band edge is a place where a rounding change could move
an agent from a warning to a refusal without anyone noticing.
"""

from __future__ import annotations

import pytest

from adk_smart_router.config import RouterConfig
from adk_smart_router.qualify import (
    CACHE_FLOOR_HEADROOM,
    PAYS_ABOVE,
    UNKNOWN_MODEL_FLOOR,
    Verdict,
    qualify,
)

CFG = RouterConfig()
FLASH = "gemini-2.5-flash"  # cache floor 2,048, so a prefix is judged against 2,560
GEMINI_3 = "gemini-3-pro"  # cache floor 4,096, judged against 4,608


def billed(prompt: int) -> float:
    """The denominator the bands divide by, spelled out rather than imported."""
    return 0.05 * prompt + 0.10 * 0.95 * prompt


# --- The two measured agents ------------------------------------------------------


def test_the_hundred_tool_agent_that_billed_23_percent_less_qualifies_as_pays():
    # DESIGN.md 19.17: steady prompt 16,681 tokens, corrected prune 7,894, measured G1 -23.4%.
    # 34.0% against its real bill of 2,323; 32.6% here, because `billed()` estimates 2,419.
    verdict = qualify(FLASH, 16_681, 7_894, CFG)
    assert verdict.band == "pays"
    assert verdict.proceeds
    assert verdict.steady_saving == pytest.approx(0.326, abs=0.005)


def test_the_twenty_five_tool_agent_that_billed_22_percent_more_is_only_thin_not_pays():
    # DESIGN.md 18.24: steady prompt 4,002 tokens, prune 912, and the whole run cost +22.4%.
    # 12.9% against its real bill of 705; 15.7% here, because `billed()` estimates 580. The
    # estimate flatters the agent that lost, which is why the bar is not drawn at 0.15.
    verdict = qualify(FLASH, 4_002, 912, CFG)
    assert verdict.band == "thin"
    assert verdict.steady_saving == pytest.approx(0.157, abs=0.005)


def test_a_thin_verdict_still_proceeds_because_the_evidence_for_it_is_one_agent():
    assert qualify(FLASH, 4_002, 912, CFG).proceeds


# --- The trap: the only refusal ---------------------------------------------------


def test_a_prune_that_carries_a_cacheable_prompt_under_the_floor_is_a_trap():
    verdict = qualify(FLASH, 3_000, 600, CFG)  # 2,400 after the prune, retained size 2,560
    assert verdict.band == "trap"
    assert not verdict.proceeds


def test_the_trap_reports_a_loss_rather_than_a_saving():
    verdict = qualify(FLASH, 3_000, 600, CFG)
    assert verdict.steady_saving < 0
    assert verdict.saved_tokens < 0
    # What it costs is the whole pruned prompt at full rate against a mostly cached one.
    assert verdict.saved_tokens == round(billed(3_000) - 2_400)


def test_a_prune_landing_exactly_on_the_retained_size_is_not_a_trap_because_it_still_caches():
    verdict = qualify(FLASH, 3_000, 440, CFG)  # 2,560 after the prune, exactly the retained size
    assert verdict.band != "trap"


def test_a_prompt_exactly_at_the_retained_size_can_still_be_trapped():
    verdict = qualify(FLASH, 2_560, 1, CFG)
    assert verdict.band == "trap"


def test_the_trap_names_the_measured_loss_so_the_refusal_is_arguable():
    assert "docs/scenarios.md#the-trap" in qualify(FLASH, 3_000, 600, CFG).reason


# --- Under the floor: nothing caches, so pruning is free --------------------------


def test_a_prompt_wholly_under_the_cache_floor_is_uncached_and_proceeds():
    verdict = qualify(FLASH, 2_000, 800, CFG)
    assert verdict.band == "uncached"
    assert verdict.proceeds


def test_the_uncached_band_counts_every_removed_token_at_full_rate():
    verdict = qualify(FLASH, 2_000, 800, CFG)
    assert verdict.saved_tokens == 800
    assert verdict.steady_saving == pytest.approx(0.4)


def test_one_token_under_the_retained_size_is_uncached_and_one_token_over_it_is_not():
    assert qualify(FLASH, 2_559, 500, CFG).band == "uncached"
    assert qualify(FLASH, 2_560, 500, CFG).band == "trap"


def test_the_uncached_band_says_it_has_never_been_measured():
    assert "measured" in qualify(FLASH, 2_000, 800, CFG).reason


# --- The bar between thin and pays ------------------------------------------------


def test_the_bar_falls_between_two_prunes_ten_tokens_apart():
    prompt = 100_000
    # The prune at which r * removed / billed(prompt) is exactly PAYS_ABOVE. Ten tokens
    # either side of it is 0.007% of the saving, and the band has to change across that.
    on_the_bar = round(PAYS_ABOVE * billed(prompt) / CFG.cached_token_multiplier)
    assert qualify(FLASH, prompt, on_the_bar + 10, CFG).band == "pays"
    assert qualify(FLASH, prompt, on_the_bar - 10, CFG).band == "thin"


def test_the_saving_is_a_tenth_of_the_removed_tokens_because_they_were_already_cached():
    verdict = qualify(FLASH, 16_681, 7_894, CFG)
    assert verdict.saved_tokens == round(0.10 * 7_894)


def test_a_cheaper_cached_token_makes_pruning_worth_less_not_more():
    dearer = qualify(FLASH, 16_681, 7_894, RouterConfig(cached_token_multiplier=0.25))
    assert dearer.steady_saving > qualify(FLASH, 16_681, 7_894, CFG).steady_saving


def test_removing_nothing_is_thin_rather_than_an_error():
    verdict = qualify(FLASH, 16_681, 0, CFG)
    assert verdict.band == "thin"
    assert verdict.saved_tokens == 0


# --- Which floor gets used --------------------------------------------------------


def test_a_gemini_3_agent_is_judged_against_its_own_larger_floor():
    # 3,000 tokens is comfortably cached on 2.5 and not cacheable at all on 3.
    assert qualify(FLASH, 3_000, 200, CFG).band == "thin"
    assert qualify(GEMINI_3, 3_000, 200, CFG).band == "uncached"


def test_an_unrecognised_model_is_judged_against_the_largest_published_floor():
    verdict = qualify("some-other-llm", UNKNOWN_MODEL_FLOOR - 1, 100, CFG)
    assert verdict.floor == UNKNOWN_MODEL_FLOOR
    assert verdict.band == "uncached"


def test_an_unrecognised_model_says_so_rather_than_quietly_assuming():
    assert "floor assumed" in qualify("some-other-llm", 5_000, 1_000, CFG).reason
    assert "floor assumed" not in qualify(FLASH, 5_000, 1_000, CFG).reason


def test_a_fully_qualified_vertex_model_path_still_resolves_its_floor():
    verdict = qualify("publishers/google/models/gemini-2.5-flash", 16_681, 7_894, CFG)
    assert verdict.floor == 2_048
    assert verdict.band == "pays"


# --- The margin over the floor ----------------------------------------------------


def test_the_size_the_bands_are_judged_against_is_the_floor_plus_the_measured_margin():
    """DESIGN.md 19.50: a prefix has to clear its published floor before it is actually kept."""
    for model, floor in ((FLASH, 2_048), (GEMINI_3, 4_096)):
        verdict = qualify(model, 20_000, 1_000, CFG)
        assert verdict.floor == floor
        assert verdict.retained == floor + CACHE_FLOOR_HEADROOM


def test_the_reported_floor_stays_the_published_one_so_the_margin_can_be_audited():
    """Reporting only the sum would hide which half is Google's number and which is ours."""
    verdict = qualify(FLASH, 20_000, 1_000, CFG)
    assert verdict.floor < verdict.retained


def test_a_prefix_the_prune_pushes_into_the_margin_is_warned_about():
    """The margin moved this warning's trigger up by 512 tokens.

    A prefix of 2,800 falling to 2,500 used to pass silently: both sides were over 2,048. The
    sweep put 2,281 tokens at a 1% hit rate, the same as a prefix half a thousand tokens under
    the floor, so 2,500 is now understood to be cold and the warning fires.
    """
    verdict = qualify(FLASH, 12_000, 300, CFG, history_tokens=9_200)  # prefix 2,800 -> 2,500
    assert "no session can ever open on another session's cache" in verdict.reason


def test_a_prefix_already_inside_the_margin_before_the_prune_is_reported_as_the_agent_s_shape():
    """2,300 tokens clears the published floor and never caches, pruned or not.

    The warning above is about a prune breaking cross-session caching, and there is nothing
    here for it to break. The reader still has a prefix that will never serve another session,
    so it is reported, in the other message and with the causation the right way round.
    """
    verdict = qualify(FLASH, 12_000, 300, CFG, history_tokens=9_700)  # prefix 2,300 -> 2,000
    assert 2_048 <= 2_300 < verdict.retained
    assert "before any prune" in verdict.reason
    assert "no session can ever open on another session's cache" not in verdict.reason


def test_an_already_cold_prefix_does_not_cap_the_verdict_because_both_arms_pay_it():
    """Capping here would charge the router for a cold turn one it did not cause.

    The same prompt and prune read `pays` with no history split. Splitting it so the shared
    prefix falls inside the margin must not move the band, because an unrouted session of this
    agent opens exactly as cold.
    """
    verdict = qualify(FLASH, 6_000, 2_200, CFG, history_tokens=4_000)  # prefix 2,000, cold
    assert verdict.band == "pays"
    assert verdict.proceeds
    assert "before any prune" in verdict.reason
    assert "no turn-one penalty on top" in verdict.reason


def test_the_two_cold_prefix_messages_never_both_fire():
    """One says the prune did it and the other says it did not, so they are exclusive."""
    for history in range(0, 12_000, 250):
        reason = qualify(FLASH, 12_000, 300, CFG, history_tokens=history).reason
        broke = "no session can ever open on another session's cache" in reason
        shaped = "before any prune" in reason
        assert not (broke and shaped), history


def test_the_margin_is_quoted_in_every_message_that_turns_on_it():
    """A refusal that names a size the user cannot find in Google's docs has to explain it."""
    trapped = qualify(FLASH, 3_000, 600, CFG)
    uncached = qualify(FLASH, 2_000, 800, CFG)
    for verdict in (trapped, uncached):
        assert f"{verdict.retained:,}" in verdict.reason
        assert "probe_prefix_headroom.py" in verdict.reason


# --- Inputs that cannot describe a request ----------------------------------------


@pytest.mark.parametrize(
    ("prompt", "removed"),
    [(0, 0), (-1, 0), (1_000, -1), (1_000, 1_001)],
)
def test_sizes_that_cannot_describe_a_real_request_raise_rather_than_being_absorbed(
    prompt, removed
):
    with pytest.raises(ValueError):
        qualify(FLASH, prompt, removed, CFG)


def test_a_verdict_is_frozen_so_a_caller_cannot_edit_the_answer_it_prints():
    verdict = qualify(FLASH, 16_681, 7_894, CFG)
    assert isinstance(verdict, Verdict)
    with pytest.raises(AttributeError):
        verdict.band = "trap"  # type: ignore[misc]


# --- The second floor: the prefix two sessions share --------------------------------


def test_a_prefix_that_prunes_under_the_floor_is_capped_at_thin_however_good_the_saving():
    """The within-session test passes and the cross-session one does not.

    A 6,000-token prompt of which 3,000 is history has a 3,000-token shared prefix, and a
    2,200-token prune takes that to 800. Nothing about the whole prompt notices: it is still
    3,800 tokens after the prune, well over the floor, and the saving clears the bar. But no
    session can open on another session's cache, because all they share is under the floor.
    """
    verdict = qualify(FLASH, 6_000, 2_200, CFG, history_tokens=3_000)
    assert verdict.band == "thin"
    assert verdict.proceeds, "this is a warning, not the measured trap"
    assert "clears the 25% bar on its own" in verdict.reason
    assert "no session can ever open on another session's cache" in verdict.reason

    # Without the history split there is nothing to notice, and it reads pays.
    assert qualify(FLASH, 6_000, 2_200, CFG).band == "pays"


def test_the_same_agent_with_a_bigger_instruction_clears_both_floors():
    """A bigger shared prefix passes both tests, which is the arithmetic `qualify` implements.

    Note what this does not say. DESIGN.md 19.50 swept the margin over the floor and found that
    clearing it is necessary and not sufficient: no prefix at any size cached on more than 28%
    of its reads. So passing here means the prefix is big enough to be kept, not that it will
    be. `qualify` tests the size because that is what it can compute without traffic.
    """
    verdict = qualify(FLASH, 6_000, 2_200, CFG, history_tokens=1_000)  # prefix 5,000 -> 2,800
    assert verdict.band == "pays"
    assert "open on another session's cache" not in verdict.reason


def test_a_thin_saving_and_a_cold_prefix_are_reported_together_rather_than_one_hiding_the_other():
    verdict = qualify(FLASH, 12_000, 300, CFG, history_tokens=9_200)  # prefix 2,800 -> 2,500
    assert verdict.band == "thin"
    assert "under the 25% bar" in verdict.reason
    assert "no session can ever open on another session's cache" in verdict.reason


def test_the_measured_trap_still_wins_over_the_cross_session_warning():
    """Both fire on a small agent, and the refusal is the one that has to be reported."""
    verdict = qualify(FLASH, 3_000, 600, CFG, history_tokens=0)
    assert verdict.band == "trap"
    assert not verdict.proceeds


@pytest.mark.parametrize("history", [-1, 12_001])  # negative, and larger than the prompt
def test_history_outside_the_prompt_raises_rather_than_producing_a_negative_prefix(history):
    with pytest.raises(ValueError):
        qualify(FLASH, 12_000, 2_000, CFG, history_tokens=history)


@pytest.mark.parametrize("prompt", [82, 3_000, 12_000, 100_000])
def test_an_agent_with_nothing_to_prune_is_told_that_and_not_given_a_band_verdict(prompt):
    """The two-tool agent's answer, and the one most first-time askers will see.

    Every band is a statement about what the prune costs or earns, and a menu wider than the
    agent's tool list makes no prune. The floor arithmetic still runs on the untouched prompt
    and produces a true sentence about a cut that did not happen, which is the misleading
    part. Checked at four prompt sizes so it does not depend on which band it would have
    fallen into: `uncached`, `trap`, `thin` and `pays` in order.
    """
    verdict = qualify(FLASH, prompt, 0, CFG)
    assert verdict.band == "thin"
    assert verdict.proceeds
    assert verdict.saved_tokens == 0
    assert verdict.steady_saving == 0.0
    assert "nothing to prune" in verdict.reason
    assert "cache floor" not in verdict.reason


def test_the_cross_session_warning_does_not_recommend_padding_the_instruction():
    """It used to, and DESIGN.md 19.43 measured that advice failing on both counts.

    A padded agent opened warm in 1 session of 11 against an unpruned control's 7, and the
    padding cost 16.6% of the saving on every turn to buy that. The message now says what the
    arithmetic says: prune, accept the cold first turn, and keep more tools only if turn-one
    latency is worth more than the tokens. This test exists so the advice cannot drift back.
    """
    verdict = qualify(FLASH, 6_000, 2_200, CFG, history_tokens=3_000)
    assert "no session can ever open on another session's cache" in verdict.reason
    assert "costs turn one and only turn one" in verdict.reason
    assert "Padding the system instruction back over the floor does not fix it" in verdict.reason
    assert "Lengthen" not in verdict.reason
    assert "Keeping more tools" in verdict.reason
