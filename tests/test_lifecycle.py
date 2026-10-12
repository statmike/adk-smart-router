"""The ledger's one promise is that it cannot break the thing it describes.

`record` is called from inside a retrain cycle, after the step it records has already happened.
Every test here that looks like paranoia is guarding that: a warehouse that is missing, refusing,
misconfigured or on fire has to leave the cycle running. The rest guard the field list, which is
the same boundary `test_provenance.py` guards for the same reason — a row that quietly grows a
column is how operational metadata turns into something else.
"""

from __future__ import annotations

import json

import pytest

from adk_smart_router import lifecycle

# --- Fakes ------------------------------------------------------------------------------


class FakeJob:
    def __init__(self, rows=(), raises=None):
        self._rows = rows
        self._raises = raises

    def result(self):
        if self._raises is not None:
            raise self._raises
        return self._rows


class FakeClient:
    """Records what it was asked. `errors` is what each insert answers, in order."""

    def __init__(self, errors=(), rows=(), query_raises=None):
        self.inserts: list[tuple[str, list[dict]]] = []
        self.queries: list[str] = []
        self._errors = list(errors)
        self._rows = rows
        self._query_raises = query_raises

    def insert_rows_json(self, table, rows):
        self.inserts.append((table, rows))
        return self._errors.pop(0) if self._errors else []

    def query(self, sql, job_config=None):
        self.queries.append(sql)
        self.job_config = job_config
        return FakeJob(self._rows, self._query_raises)


def a_step(**kw):
    fields = {"cycle_id": "abc123", "step": "publish", "agent": "support", "outcome": "ok"}
    return lifecycle.Step(**{**fields, **kw})


# --- record cannot break the cycle --------------------------------------------------------


def test_a_client_that_raises_on_every_call_does_not_raise_out_of_record():
    """The whole reason this module returns a bool instead of nothing."""

    class Exploding:
        def insert_rows_json(self, table, rows):
            raise RuntimeError("BigQuery is having a day")

        def query(self, sql, job_config=None):
            raise RuntimeError("BigQuery is still having a day")

    assert lifecycle.record("p", "d", a_step(), client=Exploding()) is False


def test_a_client_that_returns_errors_forever_does_not_raise_either():
    client = FakeClient(errors=[[{"index": 0}], [{"index": 0}]])
    assert lifecycle.record("p", "d", a_step(), client=client) is False


def test_a_clean_insert_reports_that_it_landed():
    client = FakeClient()
    assert lifecycle.record("p", "d", a_step(), client=client) is True
    assert len(client.inserts) == 1


def test_the_row_goes_to_the_fully_qualified_ledger_table():
    client = FakeClient()
    lifecycle.record("proj", "ds", a_step(), client=client)
    table, _ = client.inserts[0]
    assert table == f"proj.ds.{lifecycle.TABLE}"


# --- the missing-table retry is exactly one ----------------------------------------------


def test_a_first_failure_creates_the_table_and_retries_once():
    """The ledger appears wherever the scripts run, without a provisioning step of its own."""
    client = FakeClient(errors=[[{"index": 0}]])
    assert lifecycle.record("p", "d", a_step(), client=client) is True
    assert len(client.inserts) == 2
    assert any("CREATE TABLE IF NOT EXISTS" in q for q in client.queries)


def test_a_second_failure_gives_up_rather_than_looping():
    """A schema or permission problem is not fixed by trying again, and this runs inside a cycle."""
    client = FakeClient(errors=[[{"index": 0}], [{"index": 0}]])
    lifecycle.record("p", "d", a_step(), client=client)
    assert len(client.inserts) == 2


def test_a_clean_first_insert_never_touches_ddl():
    client = FakeClient()
    lifecycle.record("p", "d", a_step(), client=client)
    assert client.queries == []


# --- the field list is a boundary ---------------------------------------------------------

EXPECTED_FIELDS = {
    "cycle_id",
    "step",
    "recorded_at",
    "agent",
    "environment",
    "outcome",
    "reason",
    "run_id",
    "version",
    "settings_before",
    "settings_after",
    "metrics",
    "source",
}


def test_the_row_carries_exactly_the_declared_fields_and_no_others():
    assert set(a_step().as_row()) == EXPECTED_FIELDS


def test_every_declared_field_has_a_column_in_the_ddl():
    for field in EXPECTED_FIELDS:
        assert f"  {field} " in lifecycle.DDL, field


def test_the_read_query_selects_every_column_the_writer_writes():
    """A column written and never read is a column nothing will notice going wrong."""
    for field in EXPECTED_FIELDS:
        assert field in lifecycle.READ_SQL, field


# --- what goes into the JSON columns -------------------------------------------------------


def test_settings_and_metrics_are_json_text_with_sorted_keys():
    """Sorted so two identical settings diff as identical rather than by dict ordering."""
    row = a_step(settings_after={"top_n": 11, "menu_count": 24}).as_row()
    assert row["settings_after"] == '{"menu_count": 24, "top_n": 11}'
    assert json.loads(row["settings_after"])["top_n"] == 11


def test_an_unset_json_column_is_null_and_not_the_string_null():
    row = a_step().as_row()
    assert row["metrics"] is None
    assert row["settings_before"] is None


def test_a_value_json_cannot_serialize_does_not_take_the_write_down_with_it():
    """`default=str` rather than a raise: a metric nobody thought about must not lose the row."""
    row = a_step(metrics={"when": object()}).as_row()
    assert "when" in json.loads(row["metrics"])


def test_the_timestamp_is_utc_and_offset_aware():
    at = a_step().as_row()["recorded_at"]
    assert at.endswith("+00:00")


# --- the vocabulary -------------------------------------------------------------------------


def test_declining_and_rejecting_are_different_outcomes():
    """The tuner declines, which is normal. A gate rejects, which is a model found wanting."""
    assert "declined" in lifecycle.OUTCOMES
    assert "rejected" in lifecycle.OUTCOMES


def test_every_step_of_a_cycle_has_a_name():
    for step in ("train", "tune", "publish", "drift"):
        assert step in lifecycle.STEPS


def test_two_cycle_ids_differ_and_are_short_enough_to_read():
    first, second = lifecycle.new_cycle_id(), lifecycle.new_cycle_id()
    assert first != second
    assert len(first) == 12


# --- reading it back --------------------------------------------------------------------------


def test_a_ledger_that_does_not_exist_yet_reads_as_empty_rather_than_as_an_error():
    """The normal state on an agent that has never retrained, and `status` runs there first."""
    from google.api_core import exceptions as api

    client = FakeClient(query_raises=api.NotFound("no such table"))
    assert lifecycle.read("p", "d", client=client) == []


def test_any_other_failure_reading_is_not_swallowed():
    """`record` is best-effort because a cycle depends on it. A report is allowed to fail."""
    from google.api_core import exceptions as api

    client = FakeClient(query_raises=api.Forbidden("no"))
    with pytest.raises(api.Forbidden):
        lifecycle.read("p", "d", client=client)


def test_the_agent_name_is_bound_and_not_pasted():
    client = FakeClient(rows=[])
    lifecycle.read("p", "d", agent="support", client=client)
    assert "support" not in client.queries[0]
    assert "@agent" in client.queries[0]


def test_the_read_is_bounded_in_both_time_and_rows():
    """The 90-day floor every view in dashboard.sql carries, for the same reason."""
    assert "TIMESTAMP_SUB" in lifecycle.READ_SQL
    assert "LIMIT @limit" in lifecycle.READ_SQL


# --- the table itself ---------------------------------------------------------------------------


def test_the_ledger_is_partitioned_and_clustered_like_every_other_table_here():
    assert "PARTITION BY DATE(recorded_at)" in lifecycle.DDL
    assert "CLUSTER BY agent, step" in lifecycle.DDL


def test_creating_it_twice_is_not_an_error():
    assert "CREATE TABLE IF NOT EXISTS" in lifecycle.DDL


def test_no_column_is_required():
    """A failed step has to be able to leave a partial row behind."""
    assert "NOT NULL" not in lifecycle.DDL
