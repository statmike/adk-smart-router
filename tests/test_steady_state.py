"""Tests for steady_state.py: the size an agent's prompts settle at.

The module is four lines of arithmetic, and the reason it exists is the interesting part.
The floor guard used to be answered against whichever turn happened to be in front of it,
which on turn one is the shortest a session will ever be. These tests pin the two
properties the guard depends on: no opinion until there is evidence, and once there is
evidence, the mean rather than the first sample.
"""

from __future__ import annotations

from adk_smart_router.steady_state import PromptSizes


def test_no_opinion_until_min_samples() -> None:
    """None means "measure the request yourself", never "the prompts are small"."""
    sizes = PromptSizes(min_samples=3)
    sizes.observe("support", 5000)
    sizes.observe("support", 5000)

    assert sizes.typical("support") is None
    assert sizes.samples("support") == 2

    sizes.observe("support", 5000)
    assert sizes.typical("support") == 5000


def test_an_unseen_agent_has_no_opinion_either() -> None:
    sizes = PromptSizes(min_samples=1)
    assert sizes.typical("nobody") is None
    assert sizes.samples("nobody") == 0


def test_the_mean_runs_above_a_first_turn() -> None:
    """The whole point: a session's later turns pull the estimate above its first one.

    A guard answered on 800 tokens and a guard answered on 3600 give opposite answers
    against a 2048 floor, and only one of them is a statement about the session.
    """
    sizes = PromptSizes(min_samples=4)
    for tokens in (800, 2400, 4800, 6400):
        sizes.observe("support", tokens)

    assert sizes.typical("support") == 3600


def test_agents_are_counted_separately() -> None:
    sizes = PromptSizes(min_samples=1)
    sizes.observe("support", 1000)
    sizes.observe("billing", 9000)

    assert sizes.typical("support") == 1000
    assert sizes.typical("billing") == 9000


def test_unreported_usage_is_dropped_rather_than_counted_as_zero() -> None:
    """A response without usage metadata reports 0, and averaging that in is a lie."""
    sizes = PromptSizes(min_samples=2)
    sizes.observe("support", 4000)
    sizes.observe("support", 0)
    sizes.observe("support", -1)

    assert sizes.samples("support") == 1
    assert sizes.typical("support") is None

    sizes.observe("support", 4000)
    assert sizes.typical("support") == 4000


def test_the_agent_table_is_bounded() -> None:
    """A process that sees a long tail of agent names must not grow without bound.

    Least recently observed is evicted first, so the agents actually taking traffic keep
    their estimates.
    """
    sizes = PromptSizes(min_samples=1, capacity=2)
    sizes.observe("a", 1000)
    sizes.observe("b", 2000)
    sizes.observe("a", 1000)  # touches "a", so "b" is now the oldest
    sizes.observe("c", 3000)

    assert sizes.typical("a") == 1000
    assert sizes.typical("c") == 3000
    assert sizes.typical("b") is None
