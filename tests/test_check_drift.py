"""The third way of noticing a stale model, and the one that reads the request rather than the bill.

Two things are worth guarding here. The queries have to prune, because one of them reads the
column BQAA fills with the whole serialized tool list. And every line of the readout has to
degrade to a stated absence, because the common first run is against an artifact published
before any of this existed.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import check_drift

from adk_smart_router import drift

CLASSES = ("get_invoice", "cancel_order", "ABSTAIN")


class FakeJob:
    def __init__(self, rows):
        self._rows = rows
        self.total_bytes_processed = 1024**3

    def result(self):
        return self._rows


class FakeClient:
    def __init__(self, rows=()):
        self.sql = []
        self.params = []
        self._rows = rows

    def query(self, sql, job_config=None):
        self.sql.append(sql)
        self.params.append({p.name: p.value for p in (job_config.query_parameters or [])})
        return FakeJob(self._rows)


def args(**overrides):
    fields = {
        "project": "proj",
        "dataset": "ds",
        "agent": "support",
        "surface_days": 1,
        "days": 7,
        "max_sessions": 20_000,
        "backfill": 30,
        "estimate": False,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def model(baseline=None):
    return SimpleNamespace(
        classes=CLASSES, metadata=SimpleNamespace(drift_baseline=baseline, version="v1")
    )


# --- The queries have to stay cheap ---------------------------------------------------


def test_the_surface_read_prunes_partition_event_type_and_agent():
    """This column holds the whole serialized tool list, 94 KB a row on a wide surface."""
    for clause in ("event_type = 'LLM_REQUEST'", "agent = @agent", "TIMESTAMP_SUB"):
        assert clause in check_drift.SURFACE_SQL


def test_the_surface_is_a_union_and_not_the_newest_request():
    """A routed request carries the menu it was pruned to. Only the union sees the surface."""
    assert "DISTINCT" in check_drift.SURFACE_SQL


def test_the_confidence_read_is_pinned_to_one_model_version():
    """A quantile taken across two generations describes neither of them."""
    assert "model_version = @version" in check_drift.CONFIDENCE_SQL


def test_the_confidence_read_is_bounded_in_rows_as_well_as_days():
    assert "LIMIT @limit" in check_drift.CONFIDENCE_SQL


def test_the_agent_name_is_bound_and_never_pasted():
    client = FakeClient([{"tool_name": "get_invoice"}])
    check_drift.read_surface(client, args(agent="support"))
    assert "support" not in client.sql[0]
    assert client.params[0]["agent"] == "support"


def test_the_surface_comes_back_as_the_set_of_declared_names():
    client = FakeClient([{"tool_name": "a"}, {"tool_name": "b"}])
    surface, scanned = check_drift.read_surface(client, args())
    assert surface == {"a", "b"}
    assert scanned == 1024**3


def test_what_a_read_cost_is_reported_in_units_a_human_bills_in():
    assert check_drift.gigabytes(1024**3) == "1.00 GB"


# --- Every line degrades to a stated absence -------------------------------------------


def test_an_artifact_with_no_baseline_says_so_instead_of_printing_a_zero(capsys):
    now = drift.signals(["a", "b"], ["a"])
    check_drift.report(now, None, drift.verdict(now, None))
    out = capsys.readouterr().out
    assert "not recorded" in out
    assert "no retrain due" in out


def test_an_agent_that_has_routed_nothing_says_so_instead_of_printing_zero_confidence(capsys):
    now = drift.signals(["a"], ["a"], confidences=[])
    check_drift.report(now, None, drift.verdict(now, None))
    assert "nothing to take a quantile of" in capsys.readouterr().out


def test_the_reading_names_the_move_against_the_baseline(capsys):
    baseline = drift.Signals(unseen_share=0.261, declared=100, unclassed=26)
    now = drift.Signals(unseen_share=0.773, declared=100, unclassed=77)
    check_drift.report(now, baseline, drift.verdict(now, baseline))
    out = capsys.readouterr().out
    assert "RETRAIN DUE" in out
    assert "26.1%" in out and "77.3%" in out
    assert "+51.2%" in out


# --- --backfill, which is how the two-point fit gets argued with ------------------------


def test_a_backfill_without_a_baseline_explains_itself_rather_than_guessing_one(capsys):
    client = FakeClient([{"day": "2026-09-04", "tools": ["a"]}])
    assert check_drift.backfill(client, args(), model()) == 1
    assert "no baseline" in capsys.readouterr().out


def test_a_backfill_marks_the_days_each_candidate_margin_would_have_fired_on(capsys):
    """The surface widening from two known tools to ten is the shape of the 19.63 event."""
    client = FakeClient(
        [
            {"day": "2026-09-03", "tools": list(CLASSES[:2])},
            {"day": "2026-09-04", "tools": [*CLASSES[:2], *(f"new_{i}" for i in range(8))]},
        ]
    )
    baseline = drift.Signals(unseen_share=0.0, declared=2, unclassed=0)
    assert check_drift.backfill(client, args(), model(baseline)) == 0
    out = capsys.readouterr().out
    assert "DUE" in out
    assert "the shipped default" in out


def test_a_backfill_over_a_quiet_month_flags_nothing(capsys):
    client = FakeClient([{"day": f"2026-09-0{i}", "tools": list(CLASSES[:2])} for i in range(1, 5)])
    baseline = drift.Signals(unseen_share=0.0, declared=2, unclassed=0)
    check_drift.backfill(client, args(), model(baseline))
    out = capsys.readouterr().out
    assert "DUE" not in out
    assert "0 of 4" in out


def test_a_backfill_with_no_traffic_is_not_an_error(capsys):
    assert check_drift.backfill(FakeClient([]), args(), model()) == 0
    assert "no LLM_REQUEST rows" in capsys.readouterr().out


def test_the_candidate_margins_include_the_one_that_ships():
    assert drift.DEFAULT_UNSEEN_SHARE_MARGIN in check_drift.BACKFILL_MARGINS
