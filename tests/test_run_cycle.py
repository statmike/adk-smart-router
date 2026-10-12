"""The trigger: the piece between a drift verdict and a retrain that used to be a person.

`run_cycle.sh` is shell, so most of what can go wrong here is a contract between two files
drifting apart without anything failing. The exit code `check_drift.py` uses for "due" is
written down in two places, the ledger shape `run_cycle.sh` writes on a held-off cycle has to
be the shape `v_router_cycles` reads back, and the interval query has to exclude exactly the
outcomes that did not pay for a training run. Each of those is a test here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import check_drift
import pytest

from adk_smart_router import drift

REPO = Path(__file__).resolve().parent.parent
RUN_CYCLE = REPO / "scripts" / "run_cycle.sh"
RETRAIN = REPO / "scripts" / "retrain.sh"
DASHBOARD = REPO / "src" / "adk_smart_router" / "sql" / "dashboard.sql"

CLASSES = ("get_invoice", "cancel_order", "ABSTAIN")


# --- The exit code, which is the whole interface between the two scripts ---------------


class FakeJob:
    def __init__(self, rows):
        self._rows = rows
        self.total_bytes_processed = 1024**2

    def result(self):
        return self._rows


class FakeClient:
    def __init__(self, rows):
        self._rows = rows

    def query(self, sql, job_config=None):
        return FakeJob(self._rows)


def run_main(monkeypatch, *, baseline, age_days, argv):
    """`check_drift.main()` over a fake warehouse, returning its exit code."""
    rows = [{"tool_name": "get_invoice", "confidence": 0.8}]
    model = SimpleNamespace(
        classes=CLASSES,
        age_days=age_days,
        metadata=SimpleNamespace(drift_baseline=baseline, version="v1"),
    )
    monkeypatch.setattr(
        check_drift, "ModelStore", lambda cfg: SimpleNamespace(load=lambda a: model)
    )
    monkeypatch.setattr(check_drift.bigquery, "Client", lambda project: FakeClient(rows))
    monkeypatch.setattr(check_drift.lifecycle, "record", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["check_drift.py", "--project", "proj", "--bucket", "b", *argv])
    return check_drift.main()


def settled() -> drift.Signals:
    """A baseline the one-tool fake surface sits exactly on, so nothing drifts."""
    return drift.Signals(unseen_share=0.0, declared=1, unclassed=0)


def test_a_healthy_model_exits_zero_even_under_exit_code(monkeypatch):
    assert run_main(monkeypatch, baseline=settled(), age_days=1, argv=["--exit-code"]) == 0


def test_a_due_model_exits_with_the_due_code(monkeypatch):
    code = run_main(monkeypatch, baseline=settled(), age_days=999, argv=["--exit-code"])
    assert code == check_drift.DUE_EXIT_CODE


def test_the_due_code_is_not_one(monkeypatch):
    """1 is what this exits with when it breaks, and a timer that cannot tell the two apart
    retrains on a broken query rather than on a drifted model."""
    assert check_drift.DUE_EXIT_CODE not in (0, 1)


def test_due_is_not_a_failure_without_the_flag(monkeypatch):
    """The flag is opt-in, because every other caller reads the verdict from the ledger and a
    non-zero exit would turn a healthy verdict into a failed step."""
    assert run_main(monkeypatch, baseline=settled(), age_days=999, argv=[]) == 0


def test_no_record_still_reports_the_verdict(monkeypatch):
    code = run_main(
        monkeypatch, baseline=settled(), age_days=999, argv=["--exit-code", "--no-record"]
    )
    assert code == check_drift.DUE_EXIT_CODE


def test_the_shell_script_agrees_on_the_code():
    """The one number both files have to hold, and neither imports from the other."""
    assert f"DUE={check_drift.DUE_EXIT_CODE}" in RUN_CYCLE.read_text()


# --- What run_cycle.sh does with it ----------------------------------------------------


@pytest.fixture(scope="module")
def script() -> str:
    return RUN_CYCLE.read_text()


def test_the_drift_check_and_the_retrain_share_a_cycle_id(script):
    """Otherwise the verdict and what it caused are two rows in `v_router_cycles`, and the
    question "why did this retrain run" stops being a lookup."""
    assert "PYTHONPATH=src uv run python -c" in script
    for step in ("scripts/check_drift.py", '"$SCRIPT_DIR/retrain.sh"'):
        after = script.split(step, 1)[1][:400]
        assert "--cycle-id" in after and '"$CYCLE_ID"' in after


def test_retrain_accepts_the_cycle_id_it_is_handed():
    assert "--cycle-id) CYCLE_ID=" in RETRAIN.read_text()


def test_the_interval_comes_from_the_ledger_not_from_the_runner(script):
    """A container that will not exist tomorrow cannot keep a timestamp in a file."""
    assert "router_lifecycle" in script
    assert "MIN_INTERVAL_DAYS" in script


def test_the_interval_query_ignores_runs_that_never_trained(script):
    """`--skip-train` and a held-off cycle both write a skipped train row. Counting either as
    a training run would push the next real cycle a week out for free."""
    assert "step = 'train' AND outcome != 'skipped'" in script


def test_the_interval_query_prunes(script):
    """`recorded_at` is the partitioning column, and this runs on a timer."""
    assert "recorded_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)" in script


def test_a_missing_ledger_stops_the_cycle(script):
    """No ledger means no interval, and a timer with no interval retrains on every tick."""
    assert "--ensure-table" in script


def test_holding_off_leaves_a_row(script):
    assert "--step train --outcome skipped" in script


def test_the_held_off_row_reads_back_as_held_off():
    """The shape `run_cycle.sh` writes has to be the shape the view recognises, or the minimum
    interval working looks exactly like a runner that died."""
    sql = DASHBOARD.read_text()
    assert "THEN 'held off'" in sql
    assert "COUNTIF(step = 'train' AND outcome = 'skipped') > 0" in sql
    assert "COUNTIF(step IN ('drift', 'train')) = COUNT(*)" in sql


def test_held_off_is_tested_before_incomplete():
    sql = DASHBOARD.read_text()
    assert sql.index("THEN 'held off'") < sql.index("ELSE 'incomplete'")


def test_the_watch_is_a_separate_job(script):
    """Folding the rollback into the retrain trigger would tie two cadences together, and the
    rollback is the one that has to keep running when a retrain is failing."""
    assert "watch_generations.py" in script
    assert "uv run scripts/watch_generations.py" not in script


# --- It refuses to guess whose deployment it is ----------------------------------------


def test_it_will_not_run_without_a_project():
    """No fallback project and no fallback bucket. This one trains and publishes."""
    done = subprocess.run(
        [str(RUN_CYCLE), "--confirm"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert done.returncode == 2
    assert "SMART_ROUTER_BQ_PROJECT" in done.stderr


def test_help_answers_without_credentials():
    done = subprocess.run(
        [str(RUN_CYCLE), "--help"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}
    )
    assert done.returncode == 0
    assert "run_cycle.sh" in done.stdout
    assert "--min-interval-days" in done.stdout
