"""Tests for the censoring simulation's mechanics.

`scripts/simulate_censoring.py` trains real models, so what is worth pinning is not the numbers
it produces but the loop that produces them. If the censoring step does not censor, or the
ratchet does not withhold, the run still finishes and still prints a tidy table, and the table
says the loop is safe because the loop was never wired up.

Three pieces carry the argument. Serving has to truncate a routed session's labels to its menu
and leave an unrouted session alone. The rare-class floor has to be the amplifier it is claimed
to be. And a model down to nothing has to report a collapse rather than raise.
"""

from __future__ import annotations

import random

import pytest
from simulate_censoring import (
    ABSTAIN,
    MIN_CLASS_ROWS,
    Case,
    Model,
    balanced,
    floor_rare,
    rows,
    score,
    serve,
)

CATALOGUE = ["get_order", "refund", "escalate", "lookup_faq"]


def case(session: str, wanted: set[str], held_out: bool = False) -> Case:
    return Case(session_id=session, vector=[0.5], wanted=frozenset(wanted), held_out=held_out)


class FixedModel:
    """Serves the same menu to everybody, so a test can state the menu instead of training one."""

    def __init__(self, menu: list[str]) -> None:
        self.menu = menu

    def menus(self, cases, width):  # noqa: ANN001, ANN201 - a stand-in for Model.menus
        return [self.menu[:width] for _ in cases]


def test_a_routed_session_only_records_the_tools_it_was_left_with():
    """The whole mechanism. A tool pruned off the menu could not be called, so it does not
    appear in the next generation's labels, which is how the loop is supposed to bite."""
    pool = [case("s1", {"get_order", "escalate"})]
    model = FixedModel(["get_order", "refund"])

    observed, routed = serve(model, pool, 1.0, 0.0, 2, CATALOGUE, random.Random(0))

    assert observed["s1"] == frozenset({"get_order"})
    assert routed == {"s1"}


def test_an_unrouted_session_records_the_truth():
    """This is the holdout arm, and it is the only thing standing between the ratchet and
    starvation. A bug that censored it too would make every arm identical."""
    pool = [case("s1", {"get_order", "escalate"})]
    model = FixedModel(["refund"])

    observed, routed = serve(model, pool, 0.0, 0.0, 2, CATALOGUE, random.Random(0))

    assert observed["s1"] == frozenset({"get_order", "escalate"})
    assert routed == set()


def test_a_routed_session_that_wanted_nothing_on_the_menu_records_nothing():
    """Not a partial credit and not a drop. It becomes an ABSTAIN row, which teaches the next
    model that this opening wants no tools, and that is the compounding step."""
    pool = [case("s1", {"escalate"})]
    model = FixedModel(["get_order", "refund"])

    observed, _ = serve(model, pool, 1.0, 0.0, 2, CATALOGUE, random.Random(0))

    assert observed["s1"] == frozenset()
    _, labels = rows(pool, observed)
    assert labels == [ABSTAIN]


def test_exploration_can_recover_a_tool_the_model_has_given_up_on():
    """With epsilon at 1 every menu carries one extra tool, so over enough sessions the tool the
    model never proposes is still observed. That is the only route back for a starved class."""
    pool = [case(f"s{i}", {"escalate"}) for i in range(200)]
    model = FixedModel(["get_order", "refund"])

    observed, _ = serve(model, pool, 1.0, 1.0, 2, CATALOGUE, random.Random(20260906))

    recovered = sum(1 for tools in observed.values() if tools)
    assert 0 < recovered < len(pool)  # roughly one in four, since escalate is one of four tools


def test_no_exploration_never_recovers_it():
    pool = [case(f"s{i}", {"escalate"}) for i in range(200)]
    model = FixedModel(["get_order", "refund"])

    observed, _ = serve(model, pool, 1.0, 0.0, 2, CATALOGUE, random.Random(20260906))

    assert all(tools == frozenset() for tools in observed.values())


def test_a_class_under_the_floor_loses_every_row_at_once():
    """The amplifier. Losing rows one at a time is a slope; crossing the floor is a cliff, and
    the cliff is what turns a slow decline into a tool that vanishes between generations."""
    vectors = [[0.0]] * (MIN_CLASS_ROWS + MIN_CLASS_ROWS - 1)
    labels = ["common"] * MIN_CLASS_ROWS + ["rare"] * (MIN_CLASS_ROWS - 1)

    _, kept = floor_rare(vectors, labels)

    assert set(kept) == {"common"}
    assert len(kept) == MIN_CLASS_ROWS


def test_a_class_exactly_at_the_floor_survives():
    """The trainer's comparison is `n < min_class_rows`, so the floor itself is inside."""
    labels = ["common"] * MIN_CLASS_ROWS + ["rare"] * MIN_CLASS_ROWS
    _, kept = floor_rare([[0.0]] * len(labels), labels)

    assert set(kept) == {"common", "rare"}


def test_abstain_is_never_floored_away():
    """A corpus can legitimately be mostly abstentions, and a model that cannot say "keep
    nothing" would prune on every message including the ones it has no opinion about."""
    labels = ["common"] * MIN_CLASS_ROWS + [ABSTAIN]
    _, kept = floor_rare([[0.0]] * len(labels), labels)

    assert ABSTAIN in kept


def test_class_weights_give_every_class_the_same_pull():
    """BQML's auto_class_weights, and the reason a starved class can climb back. A rare tool
    with one row weighs as much in total as a common one with nine."""
    weights = balanced(["common"] * 9 + ["rare"])

    assert sum(weights[:9]) == pytest.approx(sum(weights[9:]))


def test_a_model_with_one_class_serves_it_rather_than_crashing():
    """Total collapse is a possible outcome of the censored arm, and the tables should be able
    to show it. A softmax needs two classes, so this path never reaches the booster."""
    model = Model([[0.1], [0.2]], ["get_order", "get_order"])

    assert model.menus([case("s1", {"get_order"})], 11) == [["get_order"]]


def test_a_model_that_only_learned_to_abstain_proposes_nothing():
    """ABSTAIN is not a tool the prompt can carry, so a model that knows only ABSTAIN has no
    menu to serve. Recall against the holdout then goes to zero, which is the honest reading."""
    model = Model([[0.1], [0.2]], [ABSTAIN, ABSTAIN])
    exam = [case("h1", {"get_order"}, held_out=True)]

    assert model.menus(exam, 11) == [[]]
    assert score(model, exam, 11) == (0.0, 0)


def test_probabilities_cover_every_class_and_sum_to_one():
    """The drift experiment reads the shape of this distribution rather than its top few
    names, so a map missing a class would quietly understate how flat the model has gone."""
    model = Model([[0.1], [0.9]], ["get_order", "refund"])

    probs = model.probabilities([case("s1", {"get_order"})])[0]

    assert set(probs) == {"get_order", "refund"}
    assert sum(probs.values()) == pytest.approx(1.0)


def test_a_model_with_nothing_to_rank_reports_a_flat_distribution():
    """A collapsed model is not an error, and `1 / len(classes)` on an empty class list
    would be a division by zero rather than a reading."""
    assert Model([], []).probabilities([case("s1", {"get_order"})]) == [{}]


def test_recall_is_all_or_nothing_per_session():
    """One missing tool costs the turn regardless of how many were right, so half a set is not
    half a point."""
    model = FixedModel(["get_order"])
    exam = [
        case("h1", {"get_order"}, held_out=True),
        case("h2", {"get_order", "refund"}, held_out=True),
    ]

    assert score(model, exam, 11) == (0.5, 1)


def test_sessions_that_wanted_nothing_are_left_out_of_recall():
    """Every menu covers the empty set, so counting abstentions would report the abstention
    rate dressed up as accuracy."""
    model = FixedModel(["get_order"])
    exam = [case("h1", set(), held_out=True), case("h2", {"refund"}, held_out=True)]

    recall, _ = score(model, exam, 11)

    assert recall == 0.0
