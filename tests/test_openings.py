"""The one query in the package, and the two ways it could be quietly wrong.

It cannot be run here — that takes a project, credentials and a few thousand logged turns —
so this is a lint over its text plus a walk through the client it drives. Both of the failures
worth catching are silent ones. A cache-hit predicate written in the base table's spelling
matches nothing and lets the router vouch for its own predictions, which is the same trap
`test_sql_exclusions.py` guards the committed SQL against, and this query lives in a Python
string where that lint cannot see it. An agent name pasted into the SQL instead of bound as a
parameter is the other.
"""

from __future__ import annotations

from test_sql_exclusions import BASE_SPELLING, GARBLED, VIEW_SPELLING

from adk_smart_router import openings

# --- The text of the query ------------------------------------------------------------


def test_the_cache_hit_exclusion_is_written_in_the_view_s_own_spelling():
    """Against `v_tool_completed` the marker is `tool_result`. The base spelling never matches."""
    assert VIEW_SPELLING in openings.FIRST_CALL_SQL
    assert BASE_SPELLING not in openings.FIRST_CALL_SQL


def test_neither_half_of_the_predicate_is_garbled():
    for wrong in GARBLED:
        assert wrong not in openings.FIRST_CALL_SQL


def test_the_control_plane_tool_is_excluded_like_everywhere_else():
    assert "transfer_to_agent" in openings.FIRST_CALL_SQL


def test_failed_calls_are_read_so_that_a_session_can_be_disqualified_by_one():
    """Without the error half, a value an error message leaked would look like a demonstration."""
    assert "v_tool_error" in openings.FIRST_CALL_SQL
    assert "WHERE first_call.ok" in openings.FIRST_CALL_SQL


def test_the_caller_s_strings_are_bound_and_not_pasted():
    assert "@agent" in openings.FIRST_CALL_SQL and "@since" in openings.FIRST_CALL_SQL
    assert "{project}" in openings.FIRST_CALL_SQL and "{dataset}" in openings.FIRST_CALL_SQL


# --- Driving a client -------------------------------------------------------------------


class FakeJob:
    def __init__(self, rows):
        self._rows = rows

    def result(self):
        return self._rows


class FakeClient:
    """Records what it was asked, and answers with rows shaped like BigQuery's."""

    def __init__(self, rows):
        self.rows = rows
        self.sql = ""
        self.config = None

    def query(self, sql, job_config=None):
        self.sql = sql
        self.config = job_config
        return FakeJob(self.rows)


def test_the_rows_come_back_as_a_tool_to_session_count():
    client = FakeClient([{"tool_name": "get_order", "sessions": 12}])
    found = openings.first_calls("p", "d", "support", "2026-01-01", client=client)

    assert found == {"get_order": 12}


def test_the_table_is_the_one_the_caller_named():
    client = FakeClient([])
    openings.first_calls("proj", "ds", "support", "2026-01-01", client=client)

    assert "`proj.ds.v_tool_completed`" in client.sql
    assert "`proj.ds.v_tool_error`" in client.sql


def test_the_agent_and_the_day_travel_as_parameters():
    client = FakeClient([])
    openings.first_calls("p", "d", "support", "2026-03-04", client=client)

    bound = {p.name: p.value for p in client.config.query_parameters}
    assert bound == {"agent": "support", "since": "2026-03-04"}
    assert "support" not in client.sql


def test_an_agent_with_no_logged_call_comes_back_empty_rather_than_guessing():
    assert openings.first_calls("p", "d", "quiet", "2026-01-01", client=FakeClient([])) == {}
