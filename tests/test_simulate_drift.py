"""Tests for the drift experiment's signals.

The experiment's conclusion is about which signal moves under which kind of drift, so
what has to be right is that each signal measures what its name says. A novelty score
that quietly returned the same number for familiar and unfamiliar openings would still
print a tidy table, and the table would say novelty does not work.

The era split gets a test of its own because an off-by-one on the changeover day would
put the new surface's first sessions in the old era, which is the one mistake that would
make the whole comparison lie in the direction it was hoping for.
"""

from __future__ import annotations

from simulate_censoring import Model
from simulate_drift import (
    Case,
    Era,
    cluster,
    novelty,
    recall_of,
    split_eras,
    top1_mass,
    unseen_share,
)


def case(session: str, vector: list[float], wanted: set[str], held_out: bool = False) -> Case:
    return Case(session_id=session, vector=vector, wanted=frozenset(wanted), held_out=held_out)


def model_over(tools: list[str]) -> Model:
    """A model whose class list is exactly `tools`, which is all most of these need."""
    return Model([[float(i)] for i in range(len(tools))], tools)


def test_unseen_share_counts_the_tools_the_model_has_no_class_for():
    """Surface drift's whole detector. A plugin can compute this from the request alone,
    which is why it beats anything that has to wait for a bill."""
    model = model_over(["get_order", "refund"])

    assert unseen_share(model, frozenset({"get_order", "refund"})) == 0.0
    assert unseen_share(model, frozenset({"get_order", "refund", "new_tool"})) == 1 / 3


def test_unseen_share_of_an_empty_surface_is_zero_rather_than_an_error():
    assert unseen_share(model_over(["get_order"]), frozenset()) == 0.0


def test_novelty_of_a_set_against_itself_is_nothing():
    """The floor. If this were not zero the signal would read drift on every healthy run."""
    train = [case("a", [1.0, 0.0], {"x"}), case("b", [0.0, 1.0], {"x"})]

    assert novelty(train, train) < 1e-6


def test_novelty_rises_for_an_opening_the_training_set_never_saw():
    """Orthogonal to everything fitted is cosine distance 1, the top of the scale."""
    train = [case("a", [1.0, 0.0, 0.0], {"x"})]
    strange = [case("z", [0.0, 0.0, 1.0], {"x"})]

    assert novelty(train, strange) > 0.9


def test_novelty_finds_the_nearest_neighbour_not_the_average():
    """One familiar opening among many strange ones still scores as familiar. Averaging to
    a centroid instead would call a multi-topic corpus novel to all of its own traffic."""
    train = [
        case("a", [1.0, 0.0, 0.0], {"x"}),
        case("b", [0.0, 1.0, 0.0], {"x"}),
        case("c", [0.0, 0.0, 1.0], {"x"}),
    ]

    assert novelty(train, [case("z", [0.0, 1.0, 0.0], {"x"})]) < 1e-6


def test_novelty_with_nothing_to_compare_against_is_zero():
    assert novelty([], [case("z", [1.0], {"x"})]) == 0.0
    assert novelty([case("a", [1.0], {"x"})], []) == 0.0


def test_a_zero_vector_does_not_divide_by_its_own_length():
    """Embedding failures are recorded as zeros in places, and a crash here would take out
    the whole run for one bad row."""
    assert novelty([case("a", [0.0, 0.0], {"x"})], [case("z", [0.0, 0.0], {"x"})]) == 1.0


def test_top1_mass_is_one_when_the_model_has_a_single_class():
    """A collapsed model is certain about everything, which is the honest reading: it is
    not confused, it simply has one answer. Novelty is the signal that catches that one."""
    assert top1_mass(model_over(["get_order"]), [case("a", [0.0], {"get_order"})]) == 1.0


def test_top1_mass_with_no_cases_is_zero_rather_than_an_error():
    assert top1_mass(model_over(["get_order", "refund"]), []) == 0.0


def test_recall_is_all_or_nothing_per_session():
    model = model_over(["get_order", "refund"])
    exam = [case("a", [0.0], {"get_order"}), case("b", [0.0], {"get_order", "missing"})]

    assert recall_of(model, exam, 11) == 0.5


def test_sessions_that_wanted_nothing_are_left_out_of_recall():
    """Every menu covers the empty set, so counting them would report the abstention rate."""
    model = model_over(["get_order", "refund"])

    assert recall_of(model, [case("a", [0.0], set())], 11) == 0.0


def test_a_narrow_menu_costs_recall():
    """The width argument has to reach the menu, or every condition would score the same."""
    model = Model([[0.0], [0.0], [1.0]], ["get_order", "refund", "escalate"])
    exam = [case("a", [0.0], {"get_order", "refund"})]

    assert recall_of(model, exam, 2) >= recall_of(model, exam, 1)


def test_the_changeover_day_belongs_to_the_new_surface():
    """An off-by-one here would leak the new surface's first sessions into the training
    era, and the experiment would report a model that had already seen the drift."""
    loaded = [
        (case("old", [0.0], {"x"}), "2026-09-03"),
        (case("new", [0.0], {"y"}), "2026-09-04"),
        (case("newer", [0.0], {"z"}), "2026-09-05"),
    ]

    old, new = split_eras(loaded, "2026-09-04")

    assert [c.session_id for c in old.cases] == ["old"]
    assert [c.session_id for c in new.cases] == ["new", "newer"]


def test_an_era_reports_every_tool_its_sessions_touched():
    era = Era("x", (case("a", [0.0], {"one", "two"}), case("b", [0.0], {"two", "three"})))

    assert era.tools == frozenset({"one", "two", "three"})


def test_clustering_assigns_every_case_to_exactly_one_cluster():
    cases = [case(f"s{i}", [float(i % 3), float(i % 5)], {"x"}) for i in range(30)]

    assignments = cluster(cases, 3)

    assert len(assignments) == len(cases)
    assert set(assignments) <= {0, 1, 2}


def test_clustering_separates_two_obvious_groups():
    """Not a claim about k-means quality. It is a claim that the withheld cluster in the
    intent-drift condition is a topic rather than a random quarter of the corpus, which is
    the difference between constructing a drift and constructing nothing."""
    left = [case(f"l{i}", [1.0, 0.0], {"x"}) for i in range(10)]
    right = [case(f"r{i}", [0.0, 1.0], {"x"}) for i in range(10)]

    assignments = cluster(left + right, 2)

    assert len(set(assignments[:10])) == 1
    assert len(set(assignments[10:])) == 1
    assert assignments[0] != assignments[10]


def test_clustering_is_deterministic():
    """Seeded from the first k cases in corpus order, so a rerun withholds the same topic."""
    cases = [case(f"s{i}", [float(i % 4), float(i % 7)], {"x"}) for i in range(40)]

    assert cluster(cases, 4) == cluster(cases, 4)
