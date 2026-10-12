"""Turns one embedding vector into a ranked list of tool names.

Owns the in-memory XGBoost booster for a single agent. The non-obvious rule:
feature names are positional `f0..f{n-1}`, never semantic. `EXPORT MODEL` renames
features by position and the training SQL flattens the embedding to `f0..f767` to
match (DESIGN.md D5), so any other naming either errors out or, far worse,
permutes the inputs silently. The export was checked and does emit exactly
`f0..f767` in order (DESIGN.md 18.8).

The order of `model.classes` is the other half of that contract. It is zipped
against the probability vector by position, so it must be the order the exported
assets list, not the sorted order BigQuery records. `scripts/publish_router.py`
is what guarantees that (DESIGN.md 18.8).

`predict()` never raises. An empty tuple is the only failure signal, because the
plugin's response to "no prediction" and to "prediction blew up" is identical:
pass the full tool set through (DESIGN.md 11.2).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from adk_smart_router.model_store import RouterModel

logger = logging.getLogger(__name__)

# Margins beyond this saturate the logistic anyway; clamping keeps math.exp from
# overflowing on a pathological booster output.
_MARGIN_CLAMP = 60.0


@dataclass(frozen=True)
class Prediction:
    """One class label and its probability for the current prompt."""

    label: str
    probability: float


class Predictor:
    """One loaded booster for one agent. Construct once per model version."""

    def __init__(self, model: RouterModel, booster: Any | None = None) -> None:  # noqa: ANN401
        # `booster` is Any by contract: xgboost.Booster is untyped, and tests
        # inject a fake that only needs a .predict().
        # Lazy: importing the package must not pull in the xgboost native library
        # on processes that never route.
        import xgboost

        self._xgb: Any = xgboost
        self._classes: tuple[str, ...] = tuple(model.classes)
        # NOTE(contract): a corrupt booster raises here rather than degrading to
        # (). That matches DESIGN.md 11.2 ("booster corrupt | xgboost raises |
        # @fail_open -> passthrough"); only predict() is total.
        if booster is not None:
            self._booster: Any = booster
        else:
            # nthread=1, and it is worth 23x. XGBoost defaults to every core it can see
            # and forks an OpenMP team per call, which for one row of 768 floats costs far
            # more than the arithmetic it parallelises. Measured on a 32-core box against
            # the shipped 74-class booster: 41.0 ms a call at the default, 1.8 ms pinned,
            # bit-identical output (DESIGN.md 19.54). It also stops a routing decision
            # briefly seizing every core of a box that is serving other traffic.
            self._booster = xgboost.Booster(params={"nthread": 1}, model_file=model.booster_path)
        # Built once. Positional names are a fixed function of the model, and rebuilding
        # 768 strings on every turn is pure waste on the hot path.
        self._feature_names: list[str] | None = None
        self._warned = False

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    def predict(self, vector: Sequence[float]) -> tuple[Prediction, ...]:
        """Rank every class for `vector`, highest probability first. () on failure."""
        if not vector or not self._classes:
            return ()

        if self._feature_names is None or len(self._feature_names) != len(vector):
            self._feature_names = [f"f{i}" for i in range(len(vector))]

        try:
            matrix = self._xgb.DMatrix(
                np.asarray([list(vector)], dtype=np.float32),
                feature_names=self._feature_names,
                # Same reason as the booster's, and it is the larger of the two terms.
                # DMatrix builds its own OpenMP team and does not inherit the booster's.
                nthread=1,
            )
            probabilities = _probability_row(self._booster.predict(matrix), len(self._classes))
        except Exception as exc:
            self._warn_once("smart_router: prediction failed (%s), passing through", exc)
            return ()

        if probabilities is None:
            self._warn_once(
                "smart_router: booster output does not match %d classes, passing through",
                len(self._classes),
            )
            return ()

        ranked = [
            Prediction(label, probability)
            for label, probability in zip(self._classes, probabilities, strict=True)
        ]
        # Ties break by label so two replicas holding the same model prune the
        # same way, which is what keeps the prefix cache warm across a restart.
        ranked.sort(key=lambda p: (-p.probability, p.label))
        return tuple(ranked)

    def _warn_once(self, message: str, *args: object) -> None:
        """First failure is a warning; the rest are debug. This runs every turn."""
        if self._warned:
            logger.debug(message, *args)
            return
        self._warned = True
        logger.warning(message, *args)


def _probability_row(raw: npt.ArrayLike, n_classes: int) -> list[float] | None:
    """Normalise a booster output to one finite probability per class, or None."""
    array = np.asarray(raw, dtype=float)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        return None

    if array.shape[0] == 1 and n_classes == 2:
        # A two-class booster emits a single number: a probability when the
        # objective applied the logistic, a raw margin when it did not. Index 1
        # is the positive class.
        value = float(array[0])
        if not 0.0 <= value <= 1.0:
            value = _logistic(value)
        return _finite([1.0 - value, value])

    if array.shape[0] != n_classes:
        return None
    return _finite([float(x) for x in array])


def _logistic(margin: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-_MARGIN_CLAMP, min(_MARGIN_CLAMP, margin))))


def _finite(values: list[float]) -> list[float] | None:
    """NaN sorts unpredictably, so a non-finite score invalidates the whole row."""
    return values if all(math.isfinite(v) for v in values) else None
