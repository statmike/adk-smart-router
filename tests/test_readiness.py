"""The readiness read, and the four ways its answer could mislead someone into spending money.

The queries cannot be run here, so their text gets the same lint `test_openings.py` applies to
the other one: a cache-hit predicate in the wrong spelling matches nothing and fails open, and
an agent name pasted in rather than bound is the other silent bug.

The verdict logic gets more attention than the SQL, because it is the part that tells a stranger
whether to provision a warehouse. Two mistakes matter and both are one-directional. Calling a
warehouse ready when it is not sends someone off to build a pipeline for a corpus that cannot
train it. Calling it unready when it is loses an adopter. So every boundary here is asserted on
both sides, and "thin" is asserted to be neither.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_sql_exclusions import BASE_SPELLING, GARBLED, VIEW_SPELLING

from adk_smart_router import readiness

REPO = Path(__file__).resolve().parent.parent

BOTH = (readiness.CORPUS_SQL, readiness.SPREAD_SQL)


# --- The text of the queries ------------------------------------------------------------


def test_the_cache_hit_exclusion_is_written_in_the_view_s_own_spelling():
    """Counting the router's own memo as a tool call would let it vouch for its own prediction."""
    assert VIEW_SPELLING in readiness.CORPUS_SQL
    assert BASE_SPELLING not in readiness.CORPUS_SQL


def test_neither_half_of_the_predicate_is_garbled():
    for wrong in GARBLED:
        assert wrong not in readiness.CORPUS_SQL


@pytest.mark.parametrize("sql", BOTH)
def test_the_control_plane_tool_is_excluded_like_everywhere_else(sql):
    assert "transfer_to_agent" in sql


@pytest.mark.parametrize("sql", BOTH)
def test_the_caller_s_strings_are_bound_and_not_pasted(sql):
    assert "@agent" in sql and "@days" in sql
    assert "{project}" in sql and "{dataset}" in sql


@pytest.mark.parametrize("sql", BOTH)
def test_every_read_prunes_the_partition_and_the_cluster(sql):
    """`agent_events` is day-partitioned and clustered. A read without these scans everything."""
    assert "timestamp >= TIMESTAMP_SUB" in sql
    assert "agent = @agent" in sql


def test_the_base_table_read_filters_on_the_leading_cluster_column():
    assert "event_type = 'USER_MESSAGE_RECEIVED'" in readiness.CORPUS_SQL


def test_the_prompt_text_comes_off_the_base_table_where_it_actually_lives():
    """`v_user_message_received` carries no prompt text, so the view would report zero forever."""
    assert "$.text_summary" in readiness.CORPUS_SQL
    # Named in a comment, which is the explanation. Never read from, which is the point.
    assert "dataset}.v_user_message_received`" not in readiness.CORPUS_SQL
    assert "dataset}.agent_events`" in readiness.CORPUS_SQL


def test_the_first_tool_of_a_session_is_joined_and_not_correlated():
    """A regression, and one only a live run could find.

    The first draft read `calls` from inside the `labelled` SELECT, keyed on the outer
    session_id. BigQuery rejects that outright: "Correlated subqueries that reference other
    tables are not supported unless they can be de-correlated." Every unit test passed, because
    a fake client never parses the SQL. It is a join now, and this asserts the shape.
    """
    assert "LEFT JOIN first_call f USING (session_id)" in readiness.CORPUS_SQL
    assert "FROM calls c WHERE" not in readiness.CORPUS_SQL


# --- Agreement with the training run ------------------------------------------------------


def test_the_class_floor_is_the_one_the_training_run_uses():
    """Reported readiness against a floor the trainer does not use is a number about nothing."""
    sql = (REPO / "src/adk_smart_router/sql/train_router.sql").read_text()
    assert f"DECLARE min_class_rows INT64 DEFAULT {readiness.MIN_CLASS_ROWS}" in sql


def test_the_default_window_is_the_one_the_retrain_actually_asks_for():
    script = (REPO / "scripts/retrain.sh").read_text()
    assert f"LOOKBACK_DAYS={readiness.TRAIN_LOOKBACK_DAYS}" in script


# --- Deriving the shares ------------------------------------------------------------------


def test_an_empty_corpus_has_no_shares_rather_than_zero_ones():
    """Zero would read as a stop. Nothing measured is not the same as measured and bad."""
    empty = readiness.Corpus()
    assert empty.text_share is None
    assert empty.static_coverage is None


def test_the_shares_are_over_the_denominators_they_claim():
    c = readiness.Corpus(prompts=200, prompts_with_text=150, tool_opening=80, covered_by_top_n=60)
    assert c.text_share == 0.75
    assert c.static_coverage == 0.75


# --- The verdict --------------------------------------------------------------------------


def _ready(corpus: readiness.Corpus | None, spread: readiness.Spread | None = None, top_n=11):
    return readiness.Readiness(
        agent="support", days=35, top_n=top_n, floor=10, corpus=corpus, spread=spread
    )


def _standing(findings, fragment):
    """The standing of the one finding whose title contains `fragment`."""
    found = [f for f in findings if fragment in f.title]
    assert len(found) == 1, f"{fragment} matched {len(found)} findings"
    return found[0].standing


def _healthy_readiness(corpus_overrides=None, spread=None):
    """A warehouse where every gate passes, so one override at a time isolates one gate."""
    corpus = readiness.Corpus(
        **{
            "prompts": 5000,
            "prompts_with_text": 5000,
            "sessions": 3000,
            "trainable": 3000,
            "tool_opening": 2000,
            "tool_calls": 4000,
            "tools": 30,
            "tools_at_floor": 25,
            "covered_by_top_n": 1200,
            **(corpus_overrides or {}),
        }
    )
    return _ready(corpus, spread or readiness.Spread(sessions=2000, p50=1, p90=3, p99=6, widest=9))


def test_a_dataset_with_no_analytics_table_stops_and_says_which_dataset_it_means():
    findings = readiness.verdict(_ready(None))

    assert [f.standing for f in findings] == ["stop"]
    assert "not one this project creates" in findings[0].detail


def test_a_healthy_warehouse_is_all_go():
    findings = readiness.verdict(_healthy_readiness())

    assert {f.standing for f in findings} == {"go"}


def test_a_window_with_no_prompts_at_all_stops_before_blaming_the_traffic():
    findings = readiness.verdict(_ready(readiness.Corpus()))

    assert _standing(findings, "carry text") == "stop"
    assert "--agent" in findings[0].detail and "--days" in findings[0].detail


def test_prompts_logged_without_text_stop_and_name_the_setting_that_causes_it():
    findings = readiness.verdict(_ready(readiness.Corpus(prompts=5000, prompts_with_text=0)))

    assert _standing(findings, "carry text") == "stop"
    assert "log message content" in findings[0].detail


def test_a_partial_text_share_is_thin_rather_than_a_stop():
    """Some prompts are empty or out of the length band in any real log. That is not a blocker."""
    findings = readiness.verdict(_healthy_readiness({"prompts": 5000, "prompts_with_text": 4000}))

    assert _standing(findings, "carry text") == "thin"


def test_a_text_share_a_hair_under_one_is_still_go():
    findings = readiness.verdict(_healthy_readiness({"prompts": 10000, "prompts_with_text": 9950}))

    assert _standing(findings, "carry text") == "go"


def test_no_tool_reaching_the_floor_stops_because_there_is_no_class_to_learn():
    findings = readiness.verdict(
        _healthy_readiness({"tools": 4, "tools_at_floor": 0, "tool_opening": 12})
    )

    assert _standing(findings, "labelled sessions") == "stop"


def test_a_corpus_under_the_gate_s_resolution_is_thin_and_still_trainable():
    findings = readiness.verdict(
        _healthy_readiness({"tool_opening": readiness.ROWS_FOR_A_STABLE_GATE - 1})
    )

    assert _standing(findings, "labelled sessions") == "thin"
    assert not [f for f in findings if f.standing == "stop"]


def test_a_corpus_at_the_gate_s_resolution_is_go():
    findings = readiness.verdict(
        _healthy_readiness({"tool_opening": readiness.ROWS_FOR_A_STABLE_GATE})
    )

    assert _standing(findings, "labelled sessions") == "go"


def test_sessions_wider_than_the_menu_stop_the_whole_idea():
    """Session-stable pruning chooses once. Past the kept set is a miss, not a repair."""
    findings = readiness.verdict(
        _healthy_readiness(spread=readiness.Spread(sessions=900, p50=4, p90=12, p99=20, widest=25))
    )

    assert _standing(findings, "inside a menu") == "stop"


def test_a_p90_exactly_at_the_menu_size_still_fits():
    findings = readiness.verdict(
        _healthy_readiness(spread=readiness.Spread(sessions=900, p50=2, p90=11, p99=14, widest=18))
    )

    assert _standing(findings, "inside a menu") == "go"


def test_a_frequency_table_that_already_covers_everything_stops_the_project():
    """DESIGN 3.5: at or above 0.95 the correct product is a configuration file."""
    findings = readiness.verdict(
        _healthy_readiness({"tool_opening": 2000, "covered_by_top_n": 1900})
    )

    assert _standing(findings, "frequency table") == "stop"
    assert "correct product" in [f for f in findings if "frequency" in f.title][0].detail


def test_just_under_the_static_ceiling_leaves_room_and_says_what_settles_it():
    findings = readiness.verdict(
        _healthy_readiness({"tool_opening": 2000, "covered_by_top_n": 1890})
    )

    assert _standing(findings, "frequency table") == "go"
    assert "P3" in [f for f in findings if "frequency" in f.title][0].detail


def test_nothing_opening_with_a_tool_is_thin_rather_than_a_static_stop():
    """Zero of zero is not 95%, and a stop sign there would be arithmetic rather than evidence."""
    findings = readiness.verdict(
        _healthy_readiness({"tool_opening": 0, "covered_by_top_n": 0, "tools_at_floor": 25})
    )

    assert _standing(findings, "frequency table") == "thin"


def test_the_spread_finding_is_skipped_rather_than_faked_when_the_view_is_missing():
    findings = readiness.verdict(_ready(readiness.Corpus(prompts=10, prompts_with_text=10)))

    assert not [f for f in findings if "inside a menu" in f.title]


# --- Driving a client ---------------------------------------------------------------------


class FakeJob:
    def __init__(self, rows, scanned=0, missing=False):
        self._rows = rows
        self._missing = missing
        self.total_bytes_processed = scanned

    def result(self):
        if self._missing:
            from google.api_core import exceptions as api

            raise api.NotFound("no such table")
        return self._rows


class FakeClient:
    """Answers each query in order, recording what it was asked."""

    def __init__(self, *jobs):
        self._jobs = list(jobs)
        self.sql: list[str] = []
        self.configs: list[object] = []

    def query(self, sql, job_config=None):
        self.sql.append(sql)
        self.configs.append(job_config)
        return self._jobs.pop(0)


CORPUS_ROW = {
    "prompts": 500,
    "prompts_with_text": 500,
    "sessions": 300,
    "trainable": 300,
    "tool_opening": 200,
    "tool_calls": 400,
    "tools": 12,
    "tools_at_floor": 9,
    "covered_by_top_n": 150,
}
SPREAD_ROW = {"sessions": 200, "p50": 1, "p90": 3, "p99": 5, "widest": 7}


def test_the_rows_land_in_the_dataclasses_the_verdict_reads():
    client = FakeClient(FakeJob([CORPUS_ROW], 2_000_000), FakeJob([SPREAD_ROW], 1_000_000))
    found = readiness.read("p", "d", "support", days=35, top_n=11, floor=10, client=client)

    assert found.corpus == readiness.Corpus(**CORPUS_ROW)
    assert found.spread == readiness.Spread(**SPREAD_ROW)
    assert found.scanned == 3_000_000


def test_a_null_from_bigquery_reads_as_zero_rather_than_crashing():
    """Every aggregate here returns NULL on an empty window, and none of them means unknown."""
    client = FakeClient(FakeJob([dict.fromkeys(CORPUS_ROW)]), FakeJob([dict.fromkeys(SPREAD_ROW)]))
    found = readiness.read("p", "d", "support", days=35, top_n=11, floor=10, client=client)

    assert found.corpus == readiness.Corpus()
    assert found.spread == readiness.Spread()


def test_a_missing_table_is_a_diagnosis_and_not_an_exception():
    client = FakeClient(FakeJob([], missing=True))
    found = readiness.read("p", "d", "support", days=35, top_n=11, floor=10, client=client)

    assert found.corpus is None
    assert not found.found_the_log


def test_the_second_read_is_skipped_when_the_first_found_no_log():
    """Two stack traces where one sentence will do, and a second billed scan for nothing."""
    client = FakeClient(FakeJob([], missing=True))
    readiness.read("p", "d", "support", days=35, top_n=11, floor=10, client=client)

    assert len(client.sql) == 1


def test_the_tables_are_the_ones_the_caller_named():
    client = FakeClient(FakeJob([CORPUS_ROW]), FakeJob([SPREAD_ROW]))
    readiness.read("proj", "ds", "support", days=35, top_n=11, floor=10, client=client)

    assert "`proj.ds.agent_events`" in client.sql[0]
    assert "`proj.ds.v_tool_completed`" in client.sql[0]
    assert "`proj.ds.v_tool_completed`" in client.sql[1]


def test_every_caller_supplied_value_travels_as_a_parameter():
    client = FakeClient(FakeJob([CORPUS_ROW]), FakeJob([SPREAD_ROW]))
    readiness.read("p", "d", "support", days=14, top_n=7, floor=25, client=client)

    bound = {p.name: p.value for p in client.configs[0].query_parameters}
    assert bound == {"agent": "support", "days": 14, "top_n": 7, "floor": 25}
    assert "support" not in client.sql[0]
