"""The whole-system ledger's arithmetic, checked without touching the warehouse.

The script's claim is that G1 hides two things, so what has to be right is the accounting
that reveals them: the per-session totals, the significance test that turns "the router
does not lengthen conversations" from an assumption into a measurement, and the response
counterfactual an adopter reads when their tools return more than ours do.

One test carries the real run. If the numbers on `ab-v2-100tools` ever stop reproducing
G1's -23.4%, the two scripts have diverged and this is the cheaper place to find out.
"""

from __future__ import annotations

import pytest
from report_ledger import (
    CACHED_TOKEN_WEIGHT,
    Arm,
    change,
    chars_test,
    normal_cdf,
    report,
    response_leverage,
    turns_test,
)


def an_arm(
    name: str = "routed",
    *,
    sessions: int = 100,
    turns: int = 400,
    prompt_tokens: int = 1_000_000,
    cached_tokens: int = 500_000,
    completion_tokens: int = 20_000,
    calls: int = 150,
    result_chars: int = 20_000,
    mean_turns: float = 4.0,
    sd_turns: float = 1.0,
    mean_chars: float = 200.0,
    sd_chars: float = 50.0,
) -> Arm:
    return Arm(
        name=name,
        sessions=sessions,
        turns=turns,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        calls=calls,
        result_chars=result_chars,
        mean_turns=mean_turns,
        sd_turns=sd_turns,
        mean_chars=mean_chars,
        sd_chars=sd_chars,
    )


# --- The unit G1 is defined in ----------------------------------------------------


def test_a_cached_token_is_billed_at_a_tenth_of_an_uncached_one():
    arm = an_arm(prompt_tokens=1_000, cached_tokens=1_000)
    assert arm.billed_input == pytest.approx(100.0)


def test_billed_input_never_goes_negative_when_cached_exceeds_prompt():
    """It should not happen and the warehouse is not ours, so the clamp is checked."""
    arm = an_arm(prompt_tokens=100, cached_tokens=500)
    assert arm.billed_input == pytest.approx(CACHED_TOKEN_WEIGHT * 500)


def test_the_per_turn_row_is_the_same_quantity_report_ab_calls_g1():
    arm = an_arm(prompt_tokens=1_000, cached_tokens=0, turns=4)
    assert arm.billed_per_turn == pytest.approx(250.0)


def test_the_whole_system_row_is_input_plus_output():
    arm = an_arm(prompt_tokens=1_000, cached_tokens=0, completion_tokens=200, sessions=10)
    assert arm.input_per_session == pytest.approx(100.0)
    assert arm.output_per_session == pytest.approx(20.0)
    assert arm.whole_per_session == pytest.approx(120.0)


@pytest.mark.parametrize("field", ["sessions", "turns"])
def test_an_empty_arm_divides_by_nothing_rather_than_raising(field):
    arm = an_arm(**{field: 0})
    assert arm.billed_per_turn >= 0.0
    assert arm.whole_per_session >= 0.0


def test_change_against_a_zero_baseline_is_unknown_rather_than_infinite():
    assert change(5.0, 0.0) is None
    assert change(8.0, 10.0) == pytest.approx(-0.2)


# --- Does the router lengthen conversations? --------------------------------------


def test_the_measured_run_reads_as_no_difference_in_length():
    """`ab-v2-100tools`, 2026-09-06. The assumption report_ab.py makes, checked."""
    routed = an_arm("routed", sessions=285, mean_turns=4.6702, sd_turns=1.1912)
    holdout = an_arm("holdout", sessions=289, mean_turns=4.6367, sd_turns=1.2732)
    t, p = turns_test(routed, holdout)
    assert t == pytest.approx(0.33, abs=0.01)
    assert p == pytest.approx(0.745, abs=0.005)
    assert p > 0.05


def test_a_real_lengthening_is_detected_rather_than_shrugged_at():
    """Half a turn more per session on the same spread is the failure this looks for."""
    routed = an_arm("routed", sessions=285, mean_turns=5.14, sd_turns=1.19)
    holdout = an_arm("holdout", sessions=289, mean_turns=4.64, sd_turns=1.27)
    _, p = turns_test(routed, holdout)
    assert p < 0.001


def test_two_arms_with_no_spread_are_not_a_significance_test():
    routed = an_arm("routed", sd_turns=0.0, mean_turns=4.0)
    holdout = an_arm("holdout", sd_turns=0.0, mean_turns=5.0)
    assert turns_test(routed, holdout) is None


def test_a_single_session_arm_is_not_a_significance_test_either():
    assert turns_test(an_arm("routed", sessions=1), an_arm("holdout")) is None


# --- Does the router change what the tools send back? -----------------------------


def test_the_heavy_run_moved_response_size_by_less_than_its_own_noise():
    """`ab-v2-heavy`, 2026-09-06. The routed arm carried 9.9% more tool response a session
    while making 4.8% fewer calls, which reads as a mechanism until it is tested. Response
    size per session has a fat tail here, p95 is four times the median, and a gap this size
    is what that tail produces on a few hundred sessions."""
    routed = an_arm("routed", sessions=280, mean_chars=2798.9, sd_chars=2600.0)
    holdout = an_arm("holdout", sessions=311, mean_chars=2547.5, sd_chars=2400.0)

    _, p = chars_test(routed, holdout)

    assert p > 0.05


def test_a_response_channel_shift_large_enough_to_matter_is_detected():
    routed = an_arm("routed", sessions=280, mean_chars=3400.0, sd_chars=2600.0)
    holdout = an_arm("holdout", sessions=311, mean_chars=2547.5, sd_chars=2400.0)

    _, p = chars_test(routed, holdout)

    assert p < 0.001


def test_the_two_tests_read_different_columns():
    """One Welch, two questions. Wiring both to the same field would make the report say
    the response channel was quiet whenever the turn count was."""
    routed = an_arm("routed", mean_turns=4.0, sd_turns=1.0, mean_chars=9_000.0, sd_chars=100.0)
    holdout = an_arm("holdout", mean_turns=4.0, sd_turns=1.0, mean_chars=200.0, sd_chars=100.0)

    assert turns_test(routed, holdout)[1] > 0.05
    assert chars_test(routed, holdout)[1] < 0.001


@pytest.mark.parametrize(("z", "expected"), [(0.0, 0.5), (1.96, 0.975), (-1.96, 0.025)])
def test_the_normal_approximation_is_the_normal_distribution(z, expected):
    assert normal_cdf(z) == pytest.approx(expected, abs=0.001)


# --- The tool-response channel ----------------------------------------------------


def test_the_response_channel_is_worth_nothing_when_both_arms_call_the_same_tools():
    routed = an_arm("routed", sessions=100, calls=150)
    holdout = an_arm("holdout", sessions=100, calls=150)
    assert response_leverage(routed, holdout, 5_000) == pytest.approx(0.0)


def test_calls_the_router_prevents_are_a_saving_and_calls_it_adds_are_a_cost():
    holdout = an_arm("holdout", sessions=100, calls=150)
    fewer = an_arm("routed", sessions=100, calls=100)
    more = an_arm("routed", sessions=100, calls=200)
    saves = response_leverage(fewer, holdout, 5_000)
    costs = response_leverage(more, holdout, 5_000)
    assert saves < 0 < costs
    assert saves == pytest.approx(-costs)


def test_a_bigger_response_moves_the_channel_proportionally():
    """The point of the flag: the channel scales with what an adopter's tools return."""
    holdout = an_arm("holdout", sessions=100, calls=150)
    routed = an_arm("routed", sessions=100, calls=100)
    small = response_leverage(routed, holdout, 500)
    large = response_leverage(routed, holdout, 5_000)
    assert large == pytest.approx(10.0 * small)


def test_a_response_size_of_zero_is_not_a_counterfactual():
    assert response_leverage(an_arm("routed"), an_arm("holdout"), 0) is None


# --- What it prints ---------------------------------------------------------------


def _responses(calls: float = 903) -> dict[str, float]:
    return {"calls": calls, "mean": 145.8, "p50": 95, "p95": 333, "max": 703}


def test_the_report_reproduces_the_measured_run(capsys):
    """End to end on the real totals, so the printed number cannot drift from G1's."""
    routed = Arm(
        name="routed",
        sessions=285,
        turns=1_331,
        prompt_tokens=11_162_239,
        cached_tokens=9_769_636,
        completion_tokens=44_061,
        calls=430,
        result_chars=61_861,
        mean_turns=4.6702,
        sd_turns=1.1912,
        mean_chars=217.1,
        sd_chars=180.0,
    )
    holdout = Arm(
        name="holdout",
        sessions=289,
        turns=1_340,
        prompt_tokens=22_352_172,
        cached_tokens=21_376_791,
        completion_tokens=45_181,
        calls=437,
        result_chars=65_688,
        mean_turns=4.6367,
        sd_turns=1.2732,
        mean_chars=227.3,
        sd_chars=190.0,
    )
    assert report([routed, holdout], _responses(), 5_000) == 0
    printed = capsys.readouterr().out
    assert "no measurable difference" in printed
    assert "the honest number" in printed
    # The two rows the script exists to put side by side.
    assert "-23.4%" in printed and "-22.5%" in printed


def test_a_one_armed_run_is_refused_rather_than_reported(capsys):
    assert report([an_arm("routed")], _responses(), 5_000) == 1
    assert "at least one is missing" in capsys.readouterr().out


def test_a_run_with_no_tool_calls_says_so_instead_of_dividing_by_it(capsys):
    report([an_arm("routed"), an_arm("holdout")], _responses(calls=0), 5_000)
    assert "nothing to price" in capsys.readouterr().out


def test_turns_per_session_is_printed_to_enough_places_to_see_the_difference(capsys):
    """Two arms 0.033 turns apart print as 4.6 and 4.7 at one decimal, which is a lie."""
    routed = an_arm("routed", mean_turns=4.6702)
    holdout = an_arm("holdout", mean_turns=4.6367)
    report([routed, holdout], _responses(), 5_000)
    printed = capsys.readouterr().out
    assert "4.670" in printed and "4.637" in printed
