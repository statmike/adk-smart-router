"""Tests for the heavy tool variant.

One of these matters more than the rest. The E2 experiment is only interpretable if the
heavy surface declares itself to the model exactly as the ordinary one does, because that
is what lets the same router artifact serve both and makes the difference between the two
runs the payload rather than the prompt. So the declaration comparison is the test to
break if this module is ever changed.

The rest pin the shape: writes stay small, errors stay small, and the delays are real
enough to be awaited but small enough that the suite does not notice them.
"""

from __future__ import annotations

import asyncio

from google.adk.tools import FunctionTool
from reference_agent.heavy_tools import (
    DOCUMENT_FIELD,
    document,
    document_chars,
    heavier,
    heavy,
    is_write,
    latency_seconds,
    subject_of,
)
from reference_agent.tools import ALL_TOOLS


def declaration(func):  # noqa: ANN001, ANN201 - whatever ADK builds, we only compare it
    return FunctionTool(func)._get_declaration()


def test_wrapping_does_not_change_a_single_declaration():
    """The load-bearing claim. If this fails, the E2 arms are not comparable and the
    published menus no longer describe the surface they are pruning."""
    for plain, wrapped in zip(ALL_TOOLS, heavy(ALL_TOOLS), strict=True):
        assert declaration(plain) == declaration(wrapped), plain.__name__


def test_a_read_comes_back_with_a_document():
    def get_billing_policy() -> dict[str, str]:
        """Read something."""
        return {"policy": "returns within 30 days"}

    result = asyncio.run(heavier(get_billing_policy)())

    assert result["policy"] == "returns within 30 days"
    assert len(result[DOCUMENT_FIELD]) > 500


def test_a_write_comes_back_unchanged():
    """A real write returns a receipt, not a knowledge-base article. Padding one would
    hand the router a saving it did not earn, since avoiding a write is not the thing
    pruning does."""

    def update_email(address: str) -> dict[str, bool]:
        """Change something."""
        return {"updated": True}

    assert asyncio.run(heavier(update_email)("a@b.c")) == {"updated": True}


def test_an_error_is_never_padded():
    """A failed lookup is cheap in a real system. Making it expensive here would inflate
    the cost of every wrong tool call and flatter the router."""

    def get_order_status(order_id: str) -> dict[str, str]:
        """Read something."""
        return {"error": "no such order"}

    assert asyncio.run(heavier(get_order_status)("nope")) == {"error": "no such order"}


def test_a_tool_that_returns_something_other_than_a_dict_is_left_alone():
    def get_count() -> int:
        """Read something."""
        return 7

    assert asyncio.run(heavier(get_count)()) == 7


def test_size_and_latency_are_fixed_by_the_name():
    """A rerun has to reproduce itself, or the two arms of a sequential A/B differ by
    something other than the thing under test."""
    assert document_chars("get_order_status") == document_chars("get_order_status")
    assert latency_seconds("get_order_status") == latency_seconds("get_order_status")


def test_size_and_latency_are_not_the_same_coin_flip():
    """Two digests, so the slowest tools are not also the wordiest ones by construction."""
    sizes = {name: document_chars(name) for name in (f"get_thing_{i}" for i in range(60))}
    delays = {name: latency_seconds(name) for name in sizes}
    biggest = max(sizes, key=lambda n: sizes[n])

    assert latency_seconds(biggest) < max(delays.values())


def test_writes_are_slower_than_the_fastest_reads():
    reads = [latency_seconds(f"get_thing_{i}") for i in range(40)]
    writes = [latency_seconds(f"update_thing_{i}") for i in range(40)]

    assert min(writes) > min(reads)


def test_the_document_is_about_the_tool_it_came_from():
    """Not required for the arithmetic, but a payload that talks about nothing would
    change how the model behaves for a reason unrelated to its size."""
    assert "purchase order" in document("get_purchase_order_status", 1_200)


def test_the_document_is_at_least_the_size_asked_for():
    """Sentences do not divide evenly into a target, so it overshoots rather than
    undershoots. Undershooting would quietly report a lighter surface than was run."""
    for chars in (1_200, 6_000):
        assert len(document("get_thing", chars)) >= chars


def test_no_document_at_all_when_the_size_is_zero():
    assert document("get_thing", 0) == ""


def test_the_verb_decides_read_from_write():
    assert is_write("update_email")
    assert is_write("cancel_order")
    assert not is_write("get_order_status")
    assert not is_write("list_orders")


def test_the_subject_drops_the_verb_and_the_trailing_field():
    assert subject_of("get_purchase_order_status") == "purchase order"
    assert subject_of("list_orders") == "orders"


def test_a_one_word_tool_still_has_a_subject():
    """`subject_of` is fed the name minus its verb, and some tools are only a verb."""
    assert subject_of("escalate") == "account"


def test_some_reads_return_no_document_at_all():
    """Not every read on a real surface is wordy, and a variant where all of them are
    would measure a shape nobody deploys."""
    reads = [f"get_thing_{i}" for i in range(60)]
    sizes = {document_chars(name) for name in reads}

    assert 0 in sizes
    assert len(sizes) > 1
