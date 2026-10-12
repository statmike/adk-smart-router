"""Tests for the cache-watch replay's bookkeeping.

The class under replay is the shipped one and has its own tests. What this file pins is
everything around it: which calls count as which arm, what a window's outcome is scored
against, and what a demotion is priced at. Those are the parts a replay can get quietly
wrong, because a mislabelled arm still produces a full table and the table still reads as
a verdict on the watch.

The first two tests are the ones that matter. An earlier version of this script scored a
window by pooling every unpruned call it could find, declined sessions included, and that
flipped six of twelve windows from a win to a loss.
"""

from __future__ import annotations

import pytest
from replay_cache_watch import Call, Outcome, per_turn, price, replay, truth, warm_truth


def call(arm: str, prompt: int, cached: int = 0, first: bool = False, seconds: float = 0.0) -> Call:
    return Call(
        session_id="s00001",
        window="w-",
        arm=arm,
        first_call=first,
        prompt_tokens=prompt,
        cached_tokens=cached,
        seconds=seconds,
    )


def outcome(truth_value: float | None, detected: int | None) -> Outcome:
    return Outcome(
        window="w-",
        calls=1,
        truth=truth_value,
        warm_truth=truth_value,
        warm_routed=0,
        detected_at=detected,
        detected_seconds=None,
        demotions=0,
        repromotions=0,
        demoted_calls=0,
        cost=None,
    )


def test_only_a_routed_call_counts_as_pruned():
    """What the watch is told. A declined session kept every tool it was offered, so it is
    evidence about the unpruned price and not about pruning."""
    assert call("routed", 100).pruned is True
    assert call("holdout", 100).pruned is False
    assert call("neither", 100).pruned is False


def test_billing_discounts_the_cached_share():
    """G1's definition: uncached plus a tenth of cached."""
    assert call("routed", 1_000, cached=500).billed == pytest.approx(550.0)


def test_a_window_is_scored_against_its_holdout_and_not_against_declined_sessions():
    """The defect this file exists for. Routed and holdout are level here, so the honest
    verdict is zero. The declined sessions are much cheaper, and pooling them into the
    baseline would report pruning as a heavy loss on a window where it broke even."""
    calls = [call("routed", 1_000), call("holdout", 1_000), call("neither", 200)]

    assert truth(calls) == pytest.approx(0.0)


def test_a_window_with_no_holdout_is_unscoreable_rather_than_zero():
    """The sequential control of DESIGN 18.24 ran its arms as separate prefixes, so some
    windows have no baseline inside themselves. Scoring one anyway would invent a verdict."""
    assert truth([call("routed", 1_000), call("neither", 200)]) is None
    assert truth([call("holdout", 1_000)]) is None


def test_a_losing_window_scores_positive_and_a_winning_one_negative():
    assert truth([call("routed", 1_200), call("holdout", 1_000)]) == pytest.approx(0.2)
    assert truth([call("routed", 800), call("holdout", 1_000)]) == pytest.approx(-0.2)


def test_warm_calls_are_scored_apart_from_the_calls_that_open_a_session():
    """Where the two columns come from, and why they can disagree. Pruning is winning on
    every warm call here and losing badly on the openers, which is DESIGN 18.23's window in
    miniature: pooled it reads as a loss, and the watch, which counts only warm calls, is
    looking at a win."""
    calls = [
        call("routed", 600),
        call("holdout", 1_000),
        call("routed", 4_000, first=True),
        call("holdout", 1_000, first=True),
    ]

    assert warm_truth(calls) == pytest.approx(-0.4)
    assert truth(calls) == pytest.approx(1.3)


def test_warm_truth_needs_both_arms_warm():
    assert warm_truth([call("routed", 600), call("holdout", 1_000, first=True)]) is None


def test_per_turn_is_the_arms_mean_and_missing_arms_say_so():
    calls = [call("routed", 100), call("routed", 300)]

    assert per_turn(calls, "routed") == pytest.approx(200.0)
    assert per_turn(calls, "holdout") is None


def test_a_demotion_is_priced_at_the_gap_it_would_have_paid():
    """Each demoted call would have billed the holdout rate instead of the routed one."""
    calls = [call("routed", 800), call("holdout", 1_000)]

    assert price(calls, demoted_calls=10) == pytest.approx(2_000.0)


def test_a_demotion_on_a_losing_window_is_priced_as_a_saving():
    """Same arithmetic, opposite sign. The policy did not change; the window did."""
    calls = [call("routed", 1_200), call("holdout", 1_000)]

    assert price(calls, demoted_calls=10) == pytest.approx(-2_000.0)


def test_nothing_demoted_has_no_price():
    assert price([call("routed", 800), call("holdout", 1_000)], demoted_calls=0) is None


def test_a_verdict_needs_both_a_demotion_and_a_baseline():
    assert outcome(0.2, detected=5).right is True
    assert outcome(-0.2, detected=5).right is False
    assert outcome(0.2, detected=None).right is None
    assert outcome(None, detected=5).right is None


def test_a_window_that_never_loses_is_never_demoted():
    """The false-positive floor. Pruning is cheaper on every call here, so a watch that
    fired would be firing on noise."""
    calls = [call("routed" if i % 2 else "holdout", 600 if i % 2 else 1_000) for i in range(400)]

    result = replay("w-", calls, cooloff=500, agent="support")

    assert result.detected_at is None
    assert result.demotions == 0


def test_a_window_that_loses_badly_is_demoted_and_the_demotion_is_scored_right():
    """The other end. Pruning costs two thirds more on every call, which is well past
    anything the watch should tolerate."""
    calls = [call("routed" if i % 2 else "holdout", 1_800 if i % 2 else 1_000) for i in range(400)]

    result = replay("w-", calls, cooloff=500, agent="support")

    assert result.detected_at is not None
    assert result.right is True
    assert result.demoted_calls > 0


def test_the_cooloff_does_not_move_detection():
    """The claim the sweep rests on. The first demotion happens when the evidence arrives,
    and at that point no cool-off has started, so every setting detects on the same call."""
    calls = [call("routed" if i % 2 else "holdout", 1_800 if i % 2 else 1_000) for i in range(400)]

    detections = {replay("w-", calls, c, "support").detected_at for c in (100, 500, 2_000)}

    assert len(detections) == 1


def test_a_longer_cooloff_cannot_flap_more_than_a_shorter_one():
    """What the sweep is actually choosing between. Holding an agent down longer can only
    remove re-promotions from a fixed stream of calls, never add them."""
    calls = [
        call("routed" if i % 2 else "holdout", 1_800 if i % 2 else 1_000) for i in range(4_000)
    ]

    short = replay("w-", calls, cooloff=100, agent="support")
    long = replay("w-", calls, cooloff=2_000, agent="support")

    assert long.repromotions <= short.repromotions
