"""The shell's way into the ledger, which has to be impossible to fail a cycle with.

`train` runs inside BigQuery and `deploy` is the adopter's own hook, so neither can call
`lifecycle.record` directly. This script is how they say what they did, which means it is
called from the middle of a retrain and must always exit 0.
"""

from __future__ import annotations

import pytest
import record_step

from adk_smart_router import lifecycle


def run(monkeypatch, argv):
    """Call main() with an argv, capturing the step it would write."""
    written = []
    monkeypatch.setattr("sys.argv", ["record_step.py", *argv])
    monkeypatch.setattr(
        record_step.lifecycle,
        "record",
        lambda project, dataset, step, client=None: (
            written.append((project, dataset, step)) or True
        ),
    )
    code = record_step.main()
    return code, written


BASE = ["--project", "proj", "--dataset", "ds", "--agent", "support"]


def test_it_writes_the_step_it_was_asked_for(monkeypatch):
    code, written = run(
        monkeypatch,
        [*BASE, "--step", "train", "--outcome", "ok", "--run-id", "r-7", "--cycle-id", "cyc1"],
    )
    project, dataset, step = written[0]
    assert code == 0
    assert (project, dataset) == ("proj", "ds")
    assert (step.step, step.outcome, step.run_id, step.cycle_id) == ("train", "ok", "r-7", "cyc1")


def test_a_step_run_outside_a_cycle_mints_its_own_id(monkeypatch):
    """A deploy hook firing on its own is a single-step cycle, which is what happened."""
    _, written = run(monkeypatch, [*BASE, "--step", "deploy", "--outcome", "ok"])
    assert len(written[0][2].cycle_id) == 12


def test_metrics_arrive_as_a_map(monkeypatch):
    _, written = run(
        monkeypatch,
        [*BASE, "--step", "train", "--outcome", "ok", "--metrics", '{"rows": 5373}'],
    )
    assert written[0][2].metrics == {"rows": 5373}


def test_a_quoting_mistake_in_metrics_loses_the_metrics_and_not_the_row(monkeypatch):
    """The caller is a shell script partway through a cycle. The outcome is the part to keep."""
    code, written = run(
        monkeypatch, [*BASE, "--step", "train", "--outcome", "failed", "--metrics", "{rows: 5373"]
    )
    assert code == 0
    assert written[0][2].metrics is None
    assert written[0][2].outcome == "failed"


def test_metrics_that_are_json_but_not_an_object_are_dropped(monkeypatch):
    _, written = run(
        monkeypatch, [*BASE, "--step", "train", "--outcome", "ok", "--metrics", "[1, 2]"]
    )
    assert written[0][2].metrics is None


def test_no_metrics_is_not_an_empty_map(monkeypatch):
    assert record_step.parse_metrics(None) is None
    assert record_step.parse_metrics("") is None


def test_a_row_that_does_not_land_still_exits_zero(monkeypatch):
    """This is called mid-cycle. A ledger that can stop a deploy is the wrong trade."""
    monkeypatch.setattr(
        "sys.argv", ["record_step.py", *BASE, "--step", "deploy", "--outcome", "ok"]
    )
    monkeypatch.setattr(
        record_step.lifecycle, "record", lambda project, dataset, step, client=None: False
    )
    assert record_step.main() == 0


def test_it_only_accepts_words_the_ledger_knows():
    """A typo becomes a row nothing queries for, so argparse refuses it up front."""
    parser_steps = lifecycle.STEPS
    assert "deploy" in parser_steps and "train" in parser_steps


def test_the_row_says_what_wrote_it(monkeypatch):
    _, written = run(
        monkeypatch, [*BASE, "--step", "train", "--outcome", "ok", "--source", "retrain.sh"]
    )
    assert written[0][2].source == "retrain.sh"


# --- Creating the table without writing to it -------------------------------------


def test_ensure_table_creates_the_ledger_and_writes_no_row(monkeypatch, capsys):
    """`dashboard.sql` cannot deploy while the table is missing, and a fresh agent has no row.

    Every other path to the table goes through `record`, which creates it on the way to an
    insert. An adopter who has never retrained would have to invent a step to get a table, and
    an invented step is a lie in a chart that is meant to be a history.
    """
    made = []
    monkeypatch.setattr("sys.argv", ["record_step.py", "--project", "proj", "--ensure-table"])
    monkeypatch.setattr(
        record_step.lifecycle,
        "ensure_table",
        lambda project, dataset: made.append((project, dataset)),
    )
    monkeypatch.setattr(
        record_step.lifecycle,
        "record",
        lambda *a, **k: pytest.fail("--ensure-table must not write a row"),
    )
    assert record_step.main() == 0
    assert made == [("proj", record_step.DEFAULT_DATASET)]
    assert lifecycle.TABLE in capsys.readouterr().out


def test_recording_still_needs_a_step_and_an_outcome(monkeypatch, capsys):
    """Dropping `required=True` to make room for `--ensure-table` must not make them optional."""
    monkeypatch.setattr("sys.argv", ["record_step.py", "--project", "proj", "--step", "train"])
    assert record_step.main() == 2
    assert "--outcome" in capsys.readouterr().err
