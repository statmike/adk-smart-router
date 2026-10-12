"""The gate for the cut the pruner actually makes when an artifact ships no menus.

`train_router.sql` has always reported `coverage`, a strict top-N cut with ABSTAIN stripped.
Without menus the router serves `pruner.top_k_by_mass`, which walks the ranking keeping labels
until their cumulative probability reaches `keep_probability_mass` and walks over ABSTAIN on
the way. Those are different sets, the second is usually smaller, and on the reference corpus
the published number was six points above the served one. `coverage_mass` measures the served
rule and `publish_router.gate_mass_coverage` holds a menu-less artifact to it (DESIGN.md 19.74).

Two things are worth testing and one is not. The SQL cannot run here, so what is checked is
that the rule it spells out is the rule `top_k_by_mass` implements: the predicate is
transcribed into Python, cross-checked against the shipped function over random rankings, and
the committed SQL is read as text for the three places it deliberately differs from the
top-N predicate above it. The gate itself is a plain function and is tested directly.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pytest
from publish_router import GATE_METRIC_COLUMNS, Refused, gate_mass_coverage
from test_publish_router import run

from adk_smart_router.predictor import Prediction
from adk_smart_router.pruner import ABSTAIN, MASS_TOLERANCE, top_k_by_mass

SQL = (
    Path(__file__).resolve().parents[1] / "src" / "adk_smart_router" / "sql" / "train_router.sql"
).read_text()

MENU = frozenset({"get_invoice", "cancel_order"})


def args(**overrides: object) -> argparse.Namespace:
    fields: dict[str, object] = {"skip_recall_gate": False, "min_menu_recall": 0.90}
    fields.update(overrides)
    return argparse.Namespace(**fields)


# --- The rule the SQL claims to implement -----------------------------------------


def covered_mass(preds: list[Prediction], label: str, top_n: int, mass: float) -> bool:
    """The `covered_mass` predicate from `train_router.sql`, transcribed line for line.

    Not a call into the SQL, which needs BigQuery. It is here so the rule has a second
    statement that can be compared to the first, and the comparison is the next test.

    `MASS_TOLERANCE` is in the SQL too, and it is what makes the comparison meaningful:
    this transcription and the pruner sum the same floats in different orders, so without
    the slack they disagree wherever the mass before a label is exactly the threshold.
    """
    for hit in preds:
        if hit.label != label:
            continue
        before = [
            p
            for p in preds
            if p.probability > hit.probability
            or (p.probability == hit.probability and p.label < label)
        ]
        return len(before) < top_n and sum(p.probability for p in before) < mass - MASS_TOLERANCE
    return False


def rankings(seed: int, labels: int) -> list[Prediction]:
    """A random normalised ranking, sorted the way `predictor.predict` returns one."""
    rng = random.Random(seed)
    names = [f"tool_{i}" for i in range(labels - 1)] + [ABSTAIN]
    # Deliberately coarse, so exact ties happen and the tie-break gets exercised.
    weights = [rng.randint(1, 6) for _ in names]
    total = sum(weights)
    preds = [
        Prediction(label=n, probability=w / total) for n, w in zip(names, weights, strict=True)
    ]
    return sorted(preds, key=lambda p: (-p.probability, p.label))


def test_the_transcribed_predicate_keeps_exactly_what_the_pruner_keeps() -> None:
    """The whole point of `coverage_mass` is that it scores the served set.

    If these two ever disagree the number is just a second wrong answer, so the agreement is
    checked over rankings with ties in them rather than asserted in a comment.
    """
    for seed in range(200):
        preds = rankings(seed, labels=14)
        kept = set(top_k_by_mass(preds, 0.90, 11))
        by_predicate = {p.label for p in preds if covered_mass(preds, p.label, 11, 0.90)}
        assert kept == by_predicate, f"seed {seed}: {sorted(kept)} vs {sorted(by_predicate)}"


def test_the_two_agree_when_the_mass_lands_exactly_on_the_threshold() -> None:
    """The case CPython 3.12 exposed, pinned so it cannot come back quietly.

    Ten equal labels put exactly 0.90 ahead of the tenth, so the rule stops at the ninth and
    the tenth is not kept. Three summations have to reach that conclusion: the pruner's
    running total, the transcription above, and Python's own `sum` inside it, which changed
    its answer in 3.12 when it gained compensated summation. Before `MASS_TOLERANCE` the
    pruner kept ten and the SQL scored nine, on the same vector.
    """
    preds = [Prediction(label=f"tool_{i}", probability=0.1) for i in range(10)]
    nine = {f"tool_{i}" for i in range(9)}
    assert set(top_k_by_mass(preds, 0.90, 11)) == nine
    assert {p.label for p in preds if covered_mass(preds, p.label, 11, 0.90)} == nine


def test_a_label_the_model_has_no_class_for_is_not_covered() -> None:
    """A tool absent from the ranking is a miss, not a silently-true EXISTS."""
    preds = rankings(0, labels=6)
    assert not covered_mass(preds, "a_tool_nobody_trained_on", 11, 0.90)


# --- The three places the SQL differs from the top-N predicate above it -------------


def _predicate(name: str) -> str:
    start = SQL.index(f") AS {name}")
    return SQL[SQL.rindex("EXISTS(", 0, start) : start]


def test_the_mass_predicate_does_not_strip_abstain_and_says_why() -> None:
    """`covered_model` filters ABSTAIN out of the ranking. The pruner does not.

    Copying that filter here would quietly credit the model with a slot it spends, which
    measured on the reference corpus is 0.4 points of the gap.
    """
    assert "ABSTAIN" in _predicate("covered_model")
    assert "ABSTAIN" not in _predicate("covered_mass")
    assert "ABSTAIN is NOT stripped" in SQL


def test_the_mass_predicate_breaks_ties_the_way_the_pruner_breaks_them() -> None:
    """`top_k_by_mass` sorts by label ascending within a tie, so the SQL has to as well.

    Counting a tie as covered on both sides would make the number optimistic exactly where
    the model is least sure.
    """
    predicate = _predicate("covered_mass")
    # Once for the rank subquery and once for the mass subquery. Both walk the same order.
    assert predicate.count("other.label < hit.label") == 2
    assert "other.label < hit.label" not in _predicate("covered_model")


def test_the_mass_predicate_stops_on_the_mass_before_the_label_not_including_it() -> None:
    """The label that crosses the threshold is kept, which is what the pruner does.

    `< @mass` on the sum of everything ranked ahead is that rule. `<= @mass` including the
    label itself would drop the tool that carries the session over the line.
    """
    assert "< @mass" in _predicate("covered_mass")


def test_the_mass_predicate_carries_the_tolerance_the_pruner_carries() -> None:
    """One number in two languages, so the comparison has to be written out in both.

    BigQuery has no way to read `pruner.MASS_TOLERANCE`, so the SQL spells the literal and
    this joins them. Dropping it on either side reopens the disagreement above.
    """
    assert MASS_TOLERANCE == 1e-9
    assert "< @mass - 1e-9" in _predicate("covered_mass")


def test_the_declared_mass_matches_the_config_the_router_serves() -> None:
    """The SQL's default and `RouterConfig.keep_probability_mass` are one number in two files.

    Nothing joins them at run time, so they are joined here. `startup_checks.artifact_skew`
    catches the drift afterwards, for an operator who already published.
    """
    from adk_smart_router.config import RouterConfig

    declared = re.search(r"DECLARE keep_probability_mass FLOAT64 DEFAULT ([\d.]+);", SQL)
    assert declared is not None
    assert float(declared.group(1)) == RouterConfig().keep_probability_mass


# --- What the publisher does with the number ---------------------------------------


def test_the_run_record_is_asked_for_the_served_coverage_and_its_setting() -> None:
    assert "coverage_mass" in GATE_METRIC_COLUMNS
    assert "keep_probability_mass" in GATE_METRIC_COLUMNS


def test_an_artifact_with_menus_is_gated_by_menu_recall_instead(capsys) -> None:
    """Menus are taken whole, so neither cut runs and this gate has no subject.

    `gate_menu_recall` is criterion 1 on that path and it is measured against the rule that
    serves, which is the reason menu artifacts were never part of this problem.
    """
    gate_mass_coverage((MENU,), run(gate_metrics={"coverage_mass": 0.10}), args())
    assert capsys.readouterr().out == ""


def test_skipping_the_recall_gate_skips_this_one_too(capsys) -> None:
    """One flag, one meaning: do not hold this publish to a recall number."""
    gate_mass_coverage((), run(gate_metrics={"coverage_mass": 0.10}), args(skip_recall_gate=True))
    assert "skipped" in capsys.readouterr().out


def test_a_run_trained_before_the_column_existed_gates_on_nothing(capsys) -> None:
    """NULL in BigQuery is dropped from `gate_metrics`, and unmeasured is not zero.

    Refusing here would make every artifact trained before this change unpublishable, which
    is a worse failure than the one being fixed.
    """
    gate_mass_coverage((), run(gate_metrics={"coverage": 0.9852}), args())
    assert "predates coverage_mass" in capsys.readouterr().out


def test_a_menu_less_artifact_below_the_threshold_is_refused() -> None:
    metrics = {"coverage": 0.9889, "coverage_mass": 0.8100, "keep_probability_mass": 0.90}
    with pytest.raises(Refused, match="0.8100 is below 0.9000"):
        gate_mass_coverage((), run(gate_metrics=metrics), args())


def test_the_refusal_says_which_rule_the_number_came_from() -> None:
    """An operator reading it has a passing `coverage` on screen from the training job.

    Without the name of the rule the two numbers just look like a contradiction.
    """
    metrics = {"coverage_mass": 0.8100}
    with pytest.raises(Refused, match="top_k_by_mass"):
        gate_mass_coverage((), run(gate_metrics=metrics), args())


def test_a_menu_less_artifact_above_the_threshold_passes_and_shows_both_numbers(capsys) -> None:
    """The reference artifact's own numbers: 0.9889 by the strict cut, 0.9263 as served.

    Both are printed because the gap is the finding, and a publisher who sees only the
    number that passed learns nothing from it.
    """
    metrics = {"coverage": 0.9889, "coverage_mass": 0.9263, "keep_probability_mass": 0.90}
    gate_mass_coverage((), run(gate_metrics=metrics), args())

    out = capsys.readouterr().out
    assert "0.9263" in out
    assert "0.9889" in out
    assert "0.90" in out
