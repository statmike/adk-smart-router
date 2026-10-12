"""The end-to-end check's scoring, which is the last thing that can catch a bad artifact.

`scripts/check_router.py` is the only place the whole chain runs together, so
its verdict has to be trustworthy in both directions. A check that passes a
mislabelled model is worthless. A check that fails a good one gets ignored, and
then it is also worthless.

The BigQuery and GCS halves are not tested here. They are thin, and the script
exercises them every time it runs.
"""

from __future__ import annotations

import re
from pathlib import Path

from check_router import EVAL_PCT, OPENING_KEY, Row, Score, report, score

CLASSES = ("get_invoice", "cancel_order", "track_shipment", "update_email", "ABSTAIN")

TRAIN_SQL = (
    Path(__file__).resolve().parents[1] / "src" / "adk_smart_router" / "sql" / "train_router.sql"
).read_text()


# --- the split, which has to be the same split ---------------------------------------
#
# This script's whole job is to disagree with the run record when something is wrong, so a
# split that has quietly stopped matching the procedure's turns it into a false alarm
# generator. It did: `opening_key` landed in the SQL and this script went on hashing
# session ids. These two tests are why that cannot happen again quietly.


def test_the_script_holds_out_sessions_the_same_way_the_procedure_does() -> None:
    # The SQL lives inside a triple-quoted FORMAT block, where `\s` is written `\\s`.
    in_sql = OPENING_KEY.replace(r"\s", r"\\s")
    assert in_sql in TRAIN_SQL, (
        "check_router.py's split key is no longer the one train_router.sql uses. A coverage "
        "gap measured against the run record would be this drift, not the artifact."
    )


def test_the_held_out_share_is_the_procedure_s_default() -> None:
    declared = re.search(r"DECLARE eval_pct INT64 DEFAULT (\d+)", TRAIN_SQL)
    assert declared is not None
    assert int(declared.group(1)) == EVAL_PCT


class FakePredictor:
    """Ranks by a per-row script rather than by a booster.

    Each vector is a one-element list holding the index of the ranking to return,
    which keeps the tests about the counting rather than about xgboost.
    """

    def __init__(self, rankings: list[tuple[str, ...] | tuple[()]]) -> None:
        self._rankings = rankings

    def predict(self, vector):
        from adk_smart_router.predictor import Prediction

        ranking = self._rankings[int(vector[0])]
        return tuple(Prediction(label, 1.0 - i / 100) for i, label in enumerate(ranking))


def rows(*tools: str) -> list[Row]:
    return [Row(prompt=f"p{i}", vector=[i], tool_name=tool) for i, tool in enumerate(tools)]


def vectors(rs: list[Row]) -> list[list[float] | None]:
    return [r.vector for r in rs]


# --- Counting -------------------------------------------------------------------


def test_counts_top1_and_coverage_separately():
    rs = rows("get_invoice", "cancel_order")
    predictor = FakePredictor(
        [
            ("get_invoice", "cancel_order", "ABSTAIN"),  # top-1 hit
            ("get_invoice", "cancel_order", "ABSTAIN"),  # in the top 3, not top 1
        ]
    )
    got = score("stored", predictor, rs, vectors(rs), top_n=3)
    assert (got.scored, got.top1, got.covered) == (2, 1, 2)
    assert got.accuracy == 0.5
    assert got.coverage == 1.0


def test_top_n_actually_truncates():
    rs = rows("update_email")
    predictor = FakePredictor([("get_invoice", "cancel_order", "update_email")])
    assert score("stored", predictor, rs, vectors(rs), top_n=2).coverage == 0.0
    assert score("stored", predictor, rs, vectors(rs), top_n=3).coverage == 1.0


def test_an_unranked_row_is_not_scored_as_covered():
    """Passthrough keeps every tool, so counting it as a hit would flatter the model."""
    rs = rows("get_invoice", "cancel_order")
    predictor = FakePredictor([("get_invoice",), ()])
    got = score("stored", predictor, rs, vectors(rs), top_n=3)
    assert (got.scored, got.covered, got.empty) == (1, 1, 1)
    assert got.coverage == 1.0


def test_a_missing_embedding_is_not_scored_either():
    rs = rows("get_invoice", "cancel_order")
    predictor = FakePredictor([("get_invoice",), ("cancel_order",)])
    got = score("served", predictor, rs, [rs[0].vector, None], top_n=3)
    assert (got.scored, got.empty) == (1, 1)


def test_an_empty_score_reports_zero_rather_than_dividing_by_zero():
    empty = Score("stored", scored=0, top1=0, covered=0, empty=4)
    assert empty.accuracy == 0.0
    assert empty.coverage == 0.0


# --- The verdict ------------------------------------------------------------------


def good(**overrides) -> Score:
    fields = {"label": "stored", "scored": 400, "top1": 250, "covered": 392, "empty": 0}
    fields.update(overrides)
    return Score(**fields)


def test_passes_a_healthy_artifact():
    assert report([good()], recorded_coverage=0.9852, top_n=7) is True


def test_fails_when_nothing_could_be_scored():
    assert report([good(scored=0, top1=0, covered=0)], None, 7) is False


def test_fails_near_chance_top1_even_when_coverage_looks_fine():
    """A reversed label order is the case this exists for.

    Coverage can stay respectable while the mapping is wrong, because a wide
    enough top-N catches the right tool by accident. Top-1 does not.
    """
    reversed_mapping = good(top1=12, covered=300)
    assert report([reversed_mapping], recorded_coverage=None, top_n=7) is False


def test_fails_when_it_disagrees_with_the_run_record():
    assert report([good()], recorded_coverage=0.75, top_n=7) is False


def test_tolerates_sampling_noise_against_the_run_record():
    assert report([good(covered=380)], recorded_coverage=0.9852, top_n=7) is True


def test_fails_when_re_embedding_changes_the_ranking():
    served = Score("re-embedded", scored=400, top1=250, covered=320, empty=0)
    assert report([good(), served], recorded_coverage=0.9852, top_n=7) is False


def test_passes_when_re_embedding_agrees():
    served = Score("re-embedded", scored=400, top1=249, covered=391, empty=0)
    assert report([good(), served], recorded_coverage=0.9852, top_n=7) is True
