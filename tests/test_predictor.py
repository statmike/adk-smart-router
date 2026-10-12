"""Offline tests for predictor.py. No GCS, no model store, no network.

The fake booster records the DMatrix it is handed so the tests can assert the
positional `f0..fN` feature naming from DESIGN.md D5 directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import xgboost

from adk_smart_router.predictor import Prediction, Predictor


@dataclass
class FakeModel:
    """Stands in for model_store.RouterModel: only these two attributes are used."""

    classes: tuple[str, ...]
    booster_path: str = "/nonexistent/model.ubj"


@dataclass
class FakeBooster:
    """Returns a canned output and remembers what it was asked to score."""

    output: Any = None
    error: Exception | None = None
    seen: list[Any] = field(default_factory=list)

    def predict(self, matrix: Any) -> Any:
        self.seen.append(matrix)
        if self.error is not None:
            raise self.error
        return self.output


def make(classes: tuple[str, ...], **kwargs: Any) -> tuple[Predictor, FakeBooster]:
    booster = FakeBooster(**kwargs)
    return Predictor(FakeModel(classes), booster=booster), booster


def test_ranks_descending_by_probability():
    predictor, _ = make(("a", "b", "c"), output=np.array([[0.2, 0.7, 0.1]]))

    result = predictor.predict([0.0] * 4)

    assert result == (
        Prediction("b", pytest.approx(0.7)),
        Prediction("a", pytest.approx(0.2)),
        Prediction("c", pytest.approx(0.1)),
    )


def test_ties_break_by_label_ascending():
    predictor, _ = make(("zulu", "alpha", "mike"), output=np.array([[0.5, 0.5, 0.5]]))

    assert [p.label for p in predictor.predict([1.0, 2.0])] == ["alpha", "mike", "zulu"]


def test_feature_names_are_positional():
    predictor, booster = make(("a", "b"), output=np.array([[0.4, 0.6]]))

    predictor.predict([0.1, 0.2, 0.3, 0.4, 0.5])

    matrix = booster.seen[0]
    assert matrix.feature_names == ["f0", "f1", "f2", "f3", "f4"]
    assert (matrix.num_row(), matrix.num_col()) == (1, 5)


def test_width_mismatch_returns_empty():
    predictor, _ = make(("a", "b", "c"), output=np.array([[0.5, 0.5]]))

    assert predictor.predict([0.0, 1.0]) == ()


def test_raising_booster_returns_empty():
    predictor, _ = make(("a", "b"), error=RuntimeError("booster is toast"))

    assert predictor.predict([0.0, 1.0]) == ()


def test_garbage_output_returns_empty():
    predictor, _ = make(("a", "b"), output="not an array")

    assert predictor.predict([0.0, 1.0]) == ()


def test_multi_row_output_returns_empty():
    predictor, _ = make(("a", "b"), output=np.array([[0.4, 0.6], [0.1, 0.9]]))

    assert predictor.predict([0.0, 1.0]) == ()


def test_non_finite_probability_returns_empty():
    predictor, _ = make(("a", "b"), output=np.array([[float("nan"), 1.0]]))

    assert predictor.predict([0.0, 1.0]) == ()


def test_empty_vector_returns_empty_without_calling_the_booster():
    predictor, booster = make(("a", "b"), output=np.array([[0.4, 0.6]]))

    assert predictor.predict([]) == ()
    assert booster.seen == []


def test_two_class_scalar_probability():
    predictor, _ = make(("no", "yes"), output=np.array([0.8]))

    assert predictor.predict([0.0, 1.0]) == (
        Prediction("yes", pytest.approx(0.8)),
        Prediction("no", pytest.approx(0.2)),
    )


def test_two_class_scalar_margin_is_squashed():
    predictor, _ = make(("no", "yes"), output=np.array([2.0]))

    result = predictor.predict([0.0, 1.0])

    assert [p.label for p in result] == ["yes", "no"]
    assert result[0].probability == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))
    assert sum(p.probability for p in result) == pytest.approx(1.0)


def test_extreme_margin_does_not_overflow():
    predictor, _ = make(("no", "yes"), output=np.array([-10_000.0]))

    result = predictor.predict([0.0, 1.0])

    assert result[0].label == "no"
    assert result[0].probability == pytest.approx(1.0)


def test_flat_row_of_class_width_is_accepted():
    predictor, _ = make(("a", "b", "c"), output=np.array([0.1, 0.6, 0.3]))

    assert [p.label for p in predictor.predict([0.0])] == ["b", "c", "a"]


def test_classes_property_mirrors_the_model():
    predictor, _ = make(("a", "b", "c"))

    assert predictor.classes == ("a", "b", "c")


def test_loads_a_real_booster_from_disk_and_predicts(tmp_path):
    """End-to-end over a genuine .bst trained here, so the load path is covered."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 8))
    labels = np.arange(60) % 3
    trained = xgboost.train(
        {"objective": "multi:softprob", "num_class": 3, "max_depth": 2, "verbosity": 0},
        xgboost.DMatrix(features, label=labels, feature_names=[f"f{i}" for i in range(8)]),
        num_boost_round=3,
    )
    path = tmp_path / "model.ubj"
    trained.save_model(str(path))

    predictor = Predictor(FakeModel(("read", "write", "ABSTAIN"), booster_path=str(path)))
    result = predictor.predict(features[0].tolist())

    probabilities = [p.probability for p in result]
    assert len(result) == 3
    assert {p.label for p in result} == {"read", "write", "ABSTAIN"}
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)
    assert probabilities == sorted(probabilities, reverse=True)


def test_vector_length_mismatch_against_the_real_booster_returns_empty(tmp_path):
    rng = np.random.default_rng(1)
    trained = xgboost.train(
        {"objective": "multi:softprob", "num_class": 3, "max_depth": 2, "verbosity": 0},
        xgboost.DMatrix(
            rng.normal(size=(30, 8)),
            label=np.arange(30) % 3,
            feature_names=[f"f{i}" for i in range(8)],
        ),
        num_boost_round=2,
    )
    path = tmp_path / "model.ubj"
    trained.save_model(str(path))
    predictor = Predictor(FakeModel(("a", "b", "c"), booster_path=str(path)))

    assert predictor.predict([0.0] * 5) == ()


def _tiny_booster_path(tmp_path):
    """A real 3-class booster on disk, small enough to build in a test."""
    rng = np.random.default_rng(7)
    trained = xgboost.train(
        {"objective": "multi:softprob", "num_class": 3, "max_depth": 2, "verbosity": 0},
        xgboost.DMatrix(
            rng.normal(size=(30, 8)),
            label=np.arange(30) % 3,
            feature_names=[f"f{i}" for i in range(8)],
        ),
        num_boost_round=2,
    )
    path = tmp_path / "model.ubj"
    trained.save_model(str(path))
    return str(path)


def test_the_booster_is_pinned_to_one_thread(tmp_path):
    """Left to itself XGBoost forks an OpenMP team per call, over every core it sees.

    For one row of 768 floats that costs far more than the arithmetic it parallelises:
    41.0 ms a call against 1.8 ms pinned, on a 32-core box (DESIGN.md 19.54). The
    routing decision is always one row, so there is nothing here to parallelise.
    """
    predictor = Predictor(FakeModel(("a", "b", "c"), booster_path=_tiny_booster_path(tmp_path)))

    config = json.loads(predictor._booster.save_config())  # noqa: SLF001

    assert config["learner"]["generic_param"]["nthread"] == "1"


def test_the_matrix_is_pinned_too(monkeypatch):
    """DMatrix builds its own thread team and does not inherit the booster's.

    It is the larger of the two terms, so pinning only the booster leaves most of the
    cost in place: 12.7 ms against 1.8 ms with both pinned.
    """
    predictor, _ = make(("a", "b"), output=np.array([[0.4, 0.6]]))
    seen: dict[str, Any] = {}
    real = xgboost.DMatrix

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(predictor._xgb, "DMatrix", spy)  # noqa: SLF001
    predictor.predict([0.1, 0.2])

    assert seen["nthread"] == 1


def test_feature_names_follow_a_vector_that_changes_width():
    """The names are cached across calls, so a width change has to invalidate them.

    Nothing in serving changes the embedding width mid-process, but a stale name list
    is the failure DESIGN.md D5 is about: too few names is an error, and the wrong ones
    would permute the inputs silently.
    """
    predictor, booster = make(("a", "b"), output=np.array([[0.4, 0.6]]))

    predictor.predict([0.1, 0.2, 0.3])
    predictor.predict([0.1, 0.2])

    assert booster.seen[0].feature_names == ["f0", "f1", "f2"]
    assert booster.seen[1].feature_names == ["f0", "f1"]
