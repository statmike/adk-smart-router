"""Four things about `dashboard.sql` that nothing else can catch until it is too late.

A view is not like the rest of the SQL in this project. A stored procedure fails at CALL time,
loudly, in front of whoever ran it. A view fails on a dashboard refresh, in front of nobody, and
the chart either goes blank or quietly costs money. So the checks worth automating here are the
ones whose symptom appears somewhere other than a terminal.

None of these needs BigQuery, credentials or data. They read the committed SQL as text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "src" / "adk_smart_router" / "sql" / "dashboard.sql"
DOC = ROOT / "docs" / "dashboard.md"

# The tables a careless view can scan in full. `agent_events` is the largest thing in the dataset.
BIG = ("agent_events", "v_llm_response", "v_tool_completed")

FLOOR = "TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)"

# `deploy_sql.sh` substitutes exactly these. Anything else that looks like a parameter is one.
PLACEHOLDERS = ("@project", "@dataset", "@location", "@connection")

_VIEW = re.compile(r"^CREATE OR REPLACE VIEW `[^`]*\.(?P<name>\w+)` AS$", re.MULTILINE)


def statements() -> list[tuple[str, str]]:
    """Each view's name and its body, split the way `deploy_sql.sh --check` splits them."""
    text = SQL.read_text(encoding="utf-8")
    found = list(_VIEW.finditer(text))
    assert found, "no CREATE OR REPLACE VIEW blocks in dashboard.sql"
    bodies = []
    for i, match in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        bodies.append((match.group("name"), text[match.end() : end]))
    return bodies


def names() -> list[str]:
    return [name for name, _ in statements()]


# --- The checks -------------------------------------------------------------------


@pytest.mark.parametrize("name,body", statements(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_read_of_a_big_table_is_floored(name: str, body: str) -> None:
    """A view with no date floor scans all history, on whatever timer the dashboard refreshes.

    The floor is on `timestamp`, which is the partitioning column, so it is a partition prune
    rather than a filter. Counting reads against floors rather than just looking for one keeps a
    second, unfloored read from hiding behind the first one's predicate.
    """
    reads = sum(len(re.findall(rf"FROM `[^`]*\.{table}`", body)) for table in BIG)
    if not reads:
        return
    assert body.count(FLOOR) >= reads, (
        f"{name} reads {reads} large tables and carries {body.count(FLOOR)} date floors"
    )


def test_no_view_asks_for_a_query_parameter() -> None:
    """A view cannot take one, and `deploy_sql.sh` only knows to substitute four names.

    An `@agent` left in here renders unchanged, survives the deploy script's leftover check,
    and fails at CREATE time with an error about an undeclared parameter. Cheap to catch, and
    the temptation is real: every hand-run query in `sql/` is parameterised this way.
    """
    text = SQL.read_text(encoding="utf-8")
    code = "\n".join(line.split("--", 1)[0] for line in text.splitlines())
    stray = {p for p in re.findall(r"@\w+", code) if p not in PLACEHOLDERS}
    assert not stray, f"dashboard.sql references {sorted(stray)}, which nothing substitutes"


def test_every_view_is_namespaced_away_from_the_ones_bqaa_owns() -> None:
    """BQAA creates the `v_*` views in this dataset and may add more of them.

    A collision would not error. `CREATE OR REPLACE VIEW` would take BQAA's name, every query in
    the training pipeline would start reading a router view, and the next BQAA write would take
    it back.
    """
    for name in names():
        assert name.startswith("v_router_"), f"{name} is not in the v_router_ namespace"


def test_every_view_is_documented() -> None:
    """The file is only usable if a person can find out what a column means without reading SQL."""
    doc = DOC.read_text(encoding="utf-8")
    missing = [name for name in names() if name not in doc]
    assert not missing, f"docs/dashboard.md does not mention {missing}"


def test_the_parser_finds_the_views_it_claims_to_check() -> None:
    """A regex that matched nothing would make every parametrized test above vacuously pass.

    A floor rather than an exact count, so adding a view does not fail the suite while deleting
    the parser does.
    """
    assert len(names()) >= 8
    assert len(set(names())) == len(names()), "two views in one file share a name"


# --- The two views that read the retrain ledger -----------------------------------

LEDGER_VIEWS = ("v_router_generations", "v_router_cycles")
LEDGER_FLOOR = "recorded_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)"


@pytest.mark.parametrize("name", LEDGER_VIEWS)
def test_every_read_of_the_ledger_is_floored(name: str) -> None:
    """The same 90-day floor every other view carries, on the ledger's partitioning column.

    The ledger is small enough that the floor buys almost nothing today. It is here so that the
    two views agree with the other eight about what "recent" means, because a generations chart
    that reaches back further than the cost chart beside it invites a comparison neither one
    supports.
    """
    body = dict(statements())[name]
    reads = len(re.findall(r"FROM `[^`]*\.router_lifecycle`", body))
    assert reads, f"{name} does not read the ledger"
    assert body.count(LEDGER_FLOOR) >= reads, (
        f"{name} reads the ledger {reads} times and floors it {body.count(LEDGER_FLOOR)}"
    )


def test_the_cycles_view_pivots_every_step_the_ledger_can_record() -> None:
    """A step added to `lifecycle.STEPS` and not here is one the history silently drops.

    The failure is invisible: the row is written, the view still returns, and the new step is
    simply not on it. This is the only place the two lists can be compared.
    """
    from adk_smart_router import lifecycle

    body = dict(statements())["v_router_cycles"]
    missing = [step for step in lifecycle.STEPS if f"step = '{step}'" not in body]
    assert not missing, f"v_router_cycles does not pivot {missing}"


# --- Turn position, which is the thing the cost views are easiest to be wrong about -----


def row_number_windows(text: str) -> list[str]:
    """The text inside each `ROW_NUMBER() OVER (...)`, counting parentheses to find the end.

    A non-greedy regex stops at the first `)`, which in this file is the one closing a
    `JSON_VALUE(...)` in a PARTITION BY, and the truncated window then looks like it has no
    ORDER BY at all.
    """
    windows = []
    for match in re.finditer(r"ROW_NUMBER\(\) OVER \(", text):
        depth, i = 1, match.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        windows.append(text[match.end() : i - 1])
    return windows


def test_numbering_a_session_has_a_tiebreaker() -> None:
    """`ORDER BY timestamp` alone is not a total order, and two calls can share a millisecond.

    The symptom is the invisible kind. The view still returns, every count is still right, and
    only the numbering of the tied pair moves between refreshes, which is enough to move a
    session's first call from one to the other and shift `later_billed_input_tokens`. `event_id`
    is unique per row, so adding it makes the order total.
    """
    windows = row_number_windows(SQL.read_text(encoding="utf-8"))
    assert windows, "no ROW_NUMBER in dashboard.sql; delete this test or fix the reader"
    for window in windows:
        head, sep, tail = window.partition("ORDER BY")
        assert sep, f"ROW_NUMBER with no ORDER BY: {window!r}"
        keys = [key.strip() for key in tail.split(",")]
        assert len(keys) >= 2, f"ROW_NUMBER ordered only by {keys}, so ties are arbitrary"


def test_the_generations_view_reports_the_cost_both_ways() -> None:
    """`g1` and `g1_later` have to come from different arithmetic or the second one is decoration.

    Adding the later-turn columns and then computing the second ratio from the pooled ones would
    produce two identical columns, a dashboard showing them side by side, and a reader concluding
    that turn mix does not matter here. It does: the two differ by 19 points on this project's own
    warehouse (DESIGN.md 21.5).
    """
    body = dict(statements())["v_router_generations"]
    for column in ("g1", "g1_later", "routed_later_turns", "holdout_later_turns"):
        assert f"AS {column}" in body, f"v_router_generations does not expose {column}"

    ratio = re.search(r"SAFE_DIVIDE\(\s*(.*?)\s*\) AS g1_later", body, re.DOTALL)
    assert ratio, "g1_later is not a SAFE_DIVIDE, so its denominator can be zero"
    assert "later" in ratio.group(1), "g1_later is computed from the pooled columns"


def test_the_ledger_column_is_not_the_reserved_word() -> None:
    """`AT` is a keyword in GoogleSQL, so a column named `at` only parses inside backticks.

    The ledger shipped with one for a day. Nothing caught it, because every test of the ledger
    fakes the BigQuery client and a fake client parses nothing.
    """
    text = SQL.read_text(encoding="utf-8")
    assert not re.search(r"\bAS at\b|\bat TIMESTAMP\b|\.at\b", text)
