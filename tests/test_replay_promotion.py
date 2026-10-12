"""Tests for the promotion policy's arithmetic and its bookkeeping.

`scripts/replay_promotion.py` needs BigQuery. What it does with the rows is a small amount of
arithmetic and a larger amount of deciding what counts, and the second part is where this
particular script went wrong twice before it produced a number worth reading.

Both failures had the same shape: a window disappeared from the replay instead of appearing
with a bad answer. A prefix regex dropped the one run every published figure comes from, and
an inner join dropped the sequential control period. Neither raised anything. So the cases
below spend as much effort on what lands in the denominator as on the interval itself.
"""

from __future__ import annotations

import random

from replay_promotion import (
    Session,
    Verdict,
    Window,
    best_powered,
    decide,
    effect,
    interval,
    per_turn,
)


def verdict(**kwargs: object) -> Verdict:
    """A verdict with everything defaulted, so a case names only what it is about."""
    fields: dict[str, object] = {
        "prefix": "ab-test-",
        "routed": 0,
        "holdout": 0,
        "neither": 0,
        "silent": 0,
        "unseen": 0,
        "effect": 0.0,
        "low": 0.0,
        "high": 0.0,
        "action": "hold",
    }
    fields.update(kwargs)
    return Verdict(**fields)  # type: ignore[arg-type]


def test_billing_is_pooled_over_turns_not_averaged_over_sessions():
    """A ten-turn session and a two-turn session are not one vote each.

    Sessions run to different lengths, the router changes none of that, and averaging
    per-session first would let the scenario mix move G1 without any billing changing.
    """
    sessions = [
        Session("routed", turns=10, billed=1000.0),
        Session("routed", turns=2, billed=600.0),
    ]

    assert per_turn(sessions) == 1600.0 / 12
    # The per-session average would be (100 + 300) / 2 = 200, which is a different number.
    assert per_turn(sessions) != 200.0


def test_no_turns_is_zero_rather_than_a_division_error():
    assert per_turn([]) == 0.0
    assert per_turn([Session("routed", turns=0, billed=0.0)]) == 0.0


def test_the_effect_is_signed_so_that_cheaper_is_negative():
    routed = [Session("routed", turns=10, billed=900.0)]
    holdout = [Session("holdout", turns=10, billed=1000.0)]

    assert effect(routed, holdout) == -0.10
    assert effect(holdout, routed) > 0


def test_an_empty_baseline_scores_nothing_rather_than_infinity():
    """No holdout turns means no denominator. Zero is the reading that keeps the policy on
    hold; a division would crash the replay on the one-armed windows, which are three of
    twelve."""
    assert effect([Session("routed", turns=5, billed=500.0)], []) == 0.0


def test_the_policy_ramps_only_when_the_whole_interval_clears_the_target():
    assert decide(low=-0.30, high=-0.15, target=-0.10) == "ramp"
    # The point estimate beats the target and the interval does not. This is the case the
    # whole experiment exists to catch.
    assert decide(low=-0.30, high=-0.05, target=-0.10) == "hold"


def test_the_policy_demotes_only_when_the_whole_interval_is_a_loss():
    assert decide(low=0.10, high=0.25, target=-0.10) == "demote"
    assert decide(low=-0.02, high=0.25, target=-0.10) == "hold"


def test_a_real_but_unproven_win_is_a_hold_and_not_a_ramp():
    """Straddling zero and straddling the target are both holds, and they are the common
    outcome. A policy that treated "probably helping" as "ramp" would be acting on the point
    estimate with extra steps."""
    assert decide(low=-0.18, high=0.03, target=-0.10) == "hold"


def test_an_arm_missing_makes_the_interval_infinite_and_the_policy_hold():
    rng = random.Random(0)
    low, high = interval([Session("routed", turns=5, billed=500.0)], [], 10, 0.90, rng)

    assert (low, high) == (float("-inf"), float("inf"))
    assert decide(low, high, -0.10) == "hold"


def test_the_interval_narrows_as_the_arms_grow():
    """The bootstrap has to actually respond to sample size, or the size question below it is
    answering nothing. Same effect, same spread, twenty times the sessions."""
    rng = random.Random(20260906)

    def arms(n: int) -> tuple[list[Session], list[Session]]:
        spread = [80.0, 100.0, 120.0]
        routed = [Session("routed", 1, spread[i % 3] * 0.8) for i in range(n)]
        holdout = [Session("holdout", 1, spread[i % 3]) for i in range(n)]
        return routed, holdout

    small_low, small_high = interval(*arms(10), 400, 0.90, rng)
    big_low, big_high = interval(*arms(200), 400, 0.90, rng)

    assert big_high - big_low < small_high - small_low


def test_a_session_that_billed_no_turn_is_counted_but_not_scored():
    """It has no per-turn cost to contribute, so averaging it in would be dividing by a turn
    that never happened. Leaving it out of the count entirely is how a stalled run looks
    healthy."""
    window = Window(
        "ab-test-",
        (
            Session("routed", turns=4, billed=400.0),
            Session("routed", turns=0, billed=0.0),
        ),
    )

    assert len(window.arm("routed")) == 1
    assert window.silent == 1


def test_a_window_with_no_provenance_is_kept_and_marked_rather_than_dropped():
    """`ab-20260910b` is the sequential run's control period, 300 sessions with the router
    uninstalled. It bills turns and writes no decision, so it belongs to no arm. An inner join
    made it vanish, and a promoter that cannot see an experimental design should say so."""
    window = Window("ab-test-", tuple(Session("unseen", turns=5, billed=500.0) for _ in range(3)))

    assert window.arm("routed") == []
    assert window.arm("holdout") == []
    assert window.unseen == 3
    assert verdict(unseen=3).unreadable
    assert not verdict(unseen=3).two_armed


def test_a_one_armed_window_is_not_confused_with_an_unreadable_one():
    """One is a window the policy read and declined to act on. The other it never saw. Both
    hold, and reporting them as the same thing hides which of the two we keep producing."""
    one_armed = verdict(routed=83, neither=217)

    assert not one_armed.two_armed
    assert not one_armed.unreadable


def test_the_size_question_goes_to_the_best_powered_window_not_the_largest_one():
    """Power is set by the smaller arm. A 500-session window split 490/10 tells you less than
    a 100-session window split evenly, and picking by total would choose the first."""
    lopsided = Window(
        "ab-lopsided-",
        tuple(Session("routed", 1, 100.0) for _ in range(490))
        + tuple(Session("holdout", 1, 100.0) for _ in range(10)),
    )
    balanced = Window(
        "ab-balanced-",
        tuple(Session("routed", 1, 100.0) for _ in range(50))
        + tuple(Session("holdout", 1, 100.0) for _ in range(50)),
    )

    assert best_powered([lopsided, balanced]) is balanced


def test_no_two_armed_window_means_no_size_question():
    """Rather than pooling one-armed windows into a comparison that never ran."""
    one_armed = Window("ab-one-", (Session("routed", 1, 100.0),))

    assert best_powered([one_armed]) is None
    assert best_powered([]) is None
