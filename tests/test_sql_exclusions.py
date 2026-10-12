"""The exclusion-predicate item from the adversarial tier in DESIGN.md 13.

Every query that reads tool-completion rows has to filter out the router's own cache hits, or
the next model trains on its predecessor's output. That much is easy to remember. The part that
is not is that the predicate has **two spellings**, and the wrong one is not a syntax error.

`v_tool_completed` renames the base table's `content.tool` and `content.result` to `tool_name`
and `tool_result`, so:

* against the view:       `JSON_VALUE(tool_result, '$._smart_router.cache_hit')`
* against `agent_events`: `JSON_VALUE(content, '$.result._smart_router.cache_hit')`

Written the wrong way round the predicate matches nothing, BigQuery raises nothing, and the
query silently admits every cache hit. It fails open, in the one place where failing open
corrupts the training corpus (DESIGN.md 9.1, 11.5).

So this is a lint over the committed SQL rather than a behavioural test. It reads the four files
as text, finds every read of a tool-completion source, and checks the predicate against that
read's source. It needs no BigQuery, no credentials and no data, which is the point: the check
that matters here is one a human reviewer performs by eye today, and eyes skip it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SQL_DIR = Path(__file__).resolve().parents[1] / "src" / "adk_smart_router" / "sql"

VIEW_SPELLING = "JSON_VALUE(tool_result, '$._smart_router.cache_hit')"
BASE_SPELLING = "JSON_VALUE(content, '$.result._smart_router.cache_hit')"

# The two ways to garble it. Neither is a syntax error and neither ever matches a row.
GARBLED = (
    "JSON_VALUE(tool_result, '$.result._smart_router",
    "JSON_VALUE(content, '$._smart_router",
)

# Both `@project.@dataset.table` (the hand-run preflight) and `%s.%s.table` (the procedures).
_READ = re.compile(r"^\s*(?:FROM|JOIN)\s+`[^`]*?\.(?P<table>\w+)`")
# `UNNEST` is here because a read can flatten a repeated field between its FROM and its
# WHERE, and cutting the block off at that line would hide the predicate behind it.
_PREDICATE_LINE = re.compile(r"^\s*(?:WHERE|AND|OR|--|UNNEST)")


def sql_files() -> list[Path]:
    found = sorted(SQL_DIR.glob("*.sql"))
    assert found, f"no SQL found under {SQL_DIR}"
    return found


class Read:
    """One `FROM <table>` and the SQL that decides which of its rows survive.

    The block runs from the `SELECT` that opens the branch to the end of its `WHERE` clause,
    because the marker is sometimes projected in the select list (P1 counts cache hits rather
    than dropping them) and sometimes filtered in the predicate.
    """

    def __init__(self, path: Path, lines: list[str], at: int, table: str) -> None:
        self.path = path
        self.table = table
        self.line = at + 1

        start = at
        for i in range(at - 1, max(at - 20, -1), -1):
            if lines[i].strip().startswith("SELECT"):
                start = i
                break
        end = at + 1
        while end < len(lines) and _PREDICATE_LINE.match(lines[end]):
            end += 1
        # Comments are stripped. Several of these blocks quote the wrong spelling on purpose,
        # to explain to the next reader why it is wrong, and a lint that could not tell a
        # warning from the bug it warns about would be useless here.
        self.code = "\n".join(line.split("--", 1)[0] for line in lines[start:end])

    @property
    def is_user_message_read(self) -> bool:
        """A prompt read. It never touches tool results, so no exclusion applies."""
        return "USER_MESSAGE_RECEIVED" in self.code

    @property
    def is_tool_starting_read(self) -> bool:
        """The args half of a tool call, which carries no result to exclude.

        BQAA writes one tool call as two rows: TOOL_STARTING holds `content.$.args` and
        TOOL_COMPLETED holds `content.$.result`. The marker only ever appears inside the
        result, so on a start row the predicate is a no-op. Requiring it here would put a
        line that can never match next to a line that must, which is how the next reader
        learns to trust the wrong one.
        """
        return "TOOL_STARTING" in self.code

    @property
    def is_provenance_read(self) -> bool:
        """The router's own decision records, which are metadata rather than tool results.

        `train_router.sql` reads these to drop sessions the router pruned, whose tool sets
        are censored by that prune (DESIGN.md 11.4). The rows carry no `content.result`, so
        a cache-hit predicate here could never match and would only teach the next reader
        to write one where it does nothing.
        """
        return "custom_metadata.smart_router" in self.code

    def __repr__(self) -> str:
        return f"{self.path.name}:{self.line} FROM {self.table}"


def reads_of(path: Path, *tables: str) -> list[Read]:
    lines = path.read_text(encoding="utf-8").splitlines()
    found = []
    for i, line in enumerate(lines):
        match = _READ.match(line)
        if match and match.group("table") in tables:
            found.append(Read(path, lines, i, match.group("table")))
    return found


# --- The lint itself -------------------------------------------------------------


@pytest.mark.parametrize("path", sql_files(), ids=lambda p: p.name)
def test_every_view_read_uses_the_view_spelling(path: Path) -> None:
    """A read of `v_tool_completed` must exclude cache hits, in the view's own column names."""
    for read in reads_of(path, "v_tool_completed"):
        assert VIEW_SPELLING in read.code, f"{read} has no cache-hit exclusion"
        assert BASE_SPELLING not in read.code, f"{read} uses the base spelling against the view"


@pytest.mark.parametrize("path", sql_files(), ids=lambda p: p.name)
def test_every_base_table_read_of_tool_rows_uses_the_base_spelling(path: Path) -> None:
    """A read of `agent_events` that reads tool results must exclude, in the base spelling.

    Four kinds of read, and only one of them carries a result: prompt rows, the args half of
    a tool call, the router's own provenance records, and the result half.
    """
    for read in reads_of(path, "agent_events"):
        if read.is_user_message_read or read.is_tool_starting_read or read.is_provenance_read:
            assert VIEW_SPELLING not in read.code and BASE_SPELLING not in read.code, (
                f"{read} reads no tool result, so a cache-hit predicate here can never match"
            )
            continue
        assert BASE_SPELLING in read.code, f"{read} has no cache-hit exclusion"
        assert VIEW_SPELLING not in read.code, f"{read} uses the view spelling against the base"


@pytest.mark.parametrize("path", sql_files(), ids=lambda p: p.name)
def test_a_file_that_reads_tool_starting_also_reads_the_result_half(path: Path) -> None:
    """The exemption above is only safe if the args half is paired with an excluded result half.

    A start row on its own says a tool was invoked and nothing about what it returned, so a
    query that read only TOOL_STARTING would count the router's own cache hits as tool calls
    and be exempted from the lint while doing it.
    """
    reads = reads_of(path, "agent_events")
    if not any(read.is_tool_starting_read for read in reads):
        return
    paired = [
        r
        for r in reads
        if not r.is_user_message_read and not r.is_tool_starting_read and not r.is_provenance_read
    ]
    assert paired, f"{path.name} reads TOOL_STARTING with no TOOL_COMPLETED read to pair it to"
    assert all(BASE_SPELLING in r.code for r in paired)


@pytest.mark.parametrize("path", sql_files(), ids=lambda p: p.name)
def test_neither_spelling_is_garbled_anywhere(path: Path) -> None:
    """The two halves of the predicate must not be mixed, in SQL or in a comment quoting it."""
    text = path.read_text(encoding="utf-8")
    for wrong in GARBLED:
        assert wrong not in text, f"{path.name} contains a spelling that can never match a row"


def test_the_lint_actually_finds_the_reads_it_claims_to_check() -> None:
    """A regex that matched nothing would make every test above vacuously pass.

    These counts are asserted as floors rather than exact numbers so that adding a query does
    not fail the suite, while deleting the parser does.
    """
    view_reads = [r for p in sql_files() for r in reads_of(p, "v_tool_completed")]
    base_reads = [r for p in sql_files() for r in reads_of(p, "agent_events")]

    assert len(view_reads) >= 6
    assert len(base_reads) >= 6
    assert sum(1 for r in base_reads if not r.is_user_message_read) >= 3
    assert {r.path.name for r in view_reads} >= {
        "preflight.sql",
        "train_router.sql",
        "backtest.sql",
    }
    assert "tool_ttl_report.sql" in {r.path.name for r in base_reads}


def test_a_swapped_spelling_would_be_caught(tmp_path: Path) -> None:
    """The negative control: feed the lint a file with the predicate written the wrong way round.

    Without this, a parser that silently stopped finding predicates would keep passing.
    """
    broken = tmp_path / "broken.sql"
    broken.write_text(
        "SELECT invocation_id\n"
        "FROM `p.d.v_tool_completed`\n"
        "WHERE agent = @agent\n"
        f"  AND {BASE_SPELLING} IS NULL\n",
        encoding="utf-8",
    )
    read = reads_of(broken, "v_tool_completed")[0]
    assert VIEW_SPELLING not in read.code
    assert BASE_SPELLING in read.code

    missing = tmp_path / "missing.sql"
    missing.write_text(
        "SELECT invocation_id\nFROM `p.d.v_tool_completed`\nWHERE agent = @agent\n",
        encoding="utf-8",
    )
    assert VIEW_SPELLING not in reads_of(missing, "v_tool_completed")[0].code
