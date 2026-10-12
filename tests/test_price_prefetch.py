"""Tests for the prefetch pricing arithmetic.

`scripts/price_prefetch.py` needs BigQuery and a published artifact. What it does with what
those return is arithmetic, and the arithmetic is the whole result: E8's conclusion is a
comparison between two numbers, so a sign error in either one changes the recommendation.

The cases below are the three ways this particular calculation goes wrong. Counting a tool
slower than the model as hiding all of its own time overstates the win. Dividing by a zero
hit rate crashes instead of concluding. And a hit rate below the shipping bar is unreachable
at any latency, which reads as "buy a slower tool" if the ceiling is not checked.
"""

from __future__ import annotations

import price_prefetch
from price_prefetch import Row, hidden_ms, required_tool_ms, top1_accuracy


class FakePred:
    """Stands in for the predictor's ranked output; only `.label` is read."""

    def __init__(self, label: str) -> None:
        self.label = label


class FakePredictor:
    """Returns a canned ranking per vector, keyed on the vector's first element."""

    def __init__(self, rankings: dict[float, list[str]]) -> None:
        self.rankings = rankings

    def predict(self, vector: list[float]) -> list[FakePred]:
        return [FakePred(name) for name in self.rankings[vector[0]]]


def test_a_tool_slower_than_the_model_hides_only_the_model():
    """The turn waits for the tool either way, so the excess is not a saving."""
    assert hidden_ms(1.0, tool_ms=9000.0, model_ms=3000.0) == 3000.0


def test_a_tool_faster_than_the_model_hides_all_of_itself():
    assert hidden_ms(1.0, tool_ms=200.0, model_ms=3000.0) == 200.0


def test_a_miss_hides_nothing_and_the_hit_rate_scales_the_rest():
    assert hidden_ms(0.0, tool_ms=200.0, model_ms=3000.0) == 0.0
    assert hidden_ms(0.25, tool_ms=200.0, model_ms=3000.0) == 50.0


def test_a_hit_rate_below_the_bar_is_unreachable_at_any_latency():
    """Prefetch hides at most `hit` of a turn, so a 4% hit rate cannot hide 5% of one.

    Reporting a finite requirement here would say "find a slower tool", which is wrong: no
    tool is slow enough. This is the same shape as the `worth_pruning` ceiling in H1.
    """
    assert required_tool_ms(0.04, model_ms=3000.0, worth=0.05) == float("inf")
    assert required_tool_ms(0.0, model_ms=3000.0, worth=0.05) == float("inf")


def test_the_required_latency_is_the_one_that_actually_hits_the_bar():
    """At a 10% hit rate, hiding 5% of a 3,000 ms turn needs a 1,500 ms tool."""
    need = required_tool_ms(0.10, model_ms=3000.0, worth=0.05)

    assert need == 1500.0
    assert hidden_ms(0.10, need, 3000.0) / 3000.0 == 0.05


def test_top1_counts_only_the_first_ranked_tool():
    """Prefetch runs one tool. Being right in second place is a miss, not a partial hit."""
    predictor = FakePredictor({1.0: ["get_order", "refund"], 2.0: ["refund", "get_order"]})
    rows = [Row("get_order", [1.0]), Row("get_order", [2.0])]

    assert top1_accuracy(predictor, rows) == (1, 2)


def test_an_empty_ranking_is_not_scored_either_way():
    """No ranking is a passthrough: nothing is prefetched, so there is nothing to be right or
    wrong about. Counting it as a miss would understate the hit rate of the prefetches that
    actually happen, which is the number being priced."""
    predictor = FakePredictor({1.0: ["get_order"], 2.0: []})
    rows = [Row("get_order", [1.0]), Row("get_order", [2.0])]

    assert top1_accuracy(predictor, rows) == (1, 1)


def test_the_argument_ceiling_is_the_measured_one():
    """A drifting constant here silently changes E8's verdict, so it is pinned to DESIGN 19.56."""
    assert price_prefetch.GROUNDED_ARGUMENTS == 0.252
