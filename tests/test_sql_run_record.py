"""A lint over the run record's INSERT in `train_router.sql`.

The statement writes a column list by name, supplies the same number of expressions
positionally, and binds their parameters in a `USING` clause a hundred lines further down.
Adding a column means editing all three, and getting it wrong is not caught by anything: the
procedure is one long `EXECUTE IMMEDIATE FORMAT(...)` string, so BigQuery only finds out at run
time, at the end of a training job that has already paid for its embeddings and its Vertex fit.

That has now happened four times on purpose (`miss_reduction`, `n_sessions`, then
`keep_probability_mass` and `coverage_mass`) and will happen again. This file reads the
committed SQL as text and checks the three lists agree, which is the check a reviewer does by
eye and skips.
"""

from __future__ import annotations

import re
from pathlib import Path

SQL = (
    Path(__file__).resolve().parents[1] / "src" / "adk_smart_router" / "sql" / "train_router.sql"
).read_text()


def _insert_statement() -> str:
    start = SQL.index("INSERT INTO `%s` (")
    return SQL[start : SQL.index('"""', start)]


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside parentheses, so `FORMAT(a, b)` stays one item."""
    items, depth, current = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(current.strip())
            current = ""
        else:
            current += ch
    items.append(current.strip())
    return [i for i in items if i]


def _strip_comments(text: str) -> str:
    return re.sub(r"--[^\n]*", "", text)


def test_the_run_record_names_as_many_columns_as_it_supplies_values() -> None:
    statement = _strip_comments(_insert_statement())
    columns = _split_top_level(statement[statement.index("(") + 1 : statement.index(")\n")])
    values_body = statement[statement.index("VALUES (") + len("VALUES (") :]
    values = _split_top_level(values_body.rstrip().rstrip(")"))

    assert len(columns) == len(values), (
        f"{len(columns)} columns against {len(values)} values.\n"
        f"columns: {columns}\nvalues:  {values}"
    )
    assert columns[0] == "run_id" and values[0].startswith("GENERATE_UUID")


def test_every_parameter_the_insert_uses_is_bound_in_the_using_clause() -> None:
    statement = _insert_statement()
    used = set(re.findall(r"@(\w+)", statement))

    tail = SQL[SQL.index(statement) :]
    using = tail[tail.index("USING agent AS agent") + len("USING ") : tail.index("export_uri AS")]
    bound = set(re.findall(r"^\s*\w+ AS (\w+),", using, re.MULTILINE)) | {"export_uri"}

    assert used <= bound, f"unbound in USING: {sorted(used - bound)}"
    assert bound <= used, f"bound but never read: {sorted(bound - used)}"


def test_the_upgrade_path_covers_every_column_added_after_the_first_release() -> None:
    """`CREATE TABLE IF NOT EXISTS` leaves an existing table on its old schema.

    So each column added later needs an `ADD COLUMN IF NOT EXISTS` beside it, or the INSERT
    fails against every warehouse that trained before the column existed.
    """
    added = set(re.findall(r"ADD COLUMN IF NOT EXISTS (\w+)", _strip_comments(SQL)))
    assert added == {
        "baseline_coverage_corpus",
        "miss_reduction",
        "n_sessions",
        "keep_probability_mass",
        "coverage_mass",
    }, sorted(added)

    statement = _strip_comments(_insert_statement())
    columns = set(_split_top_level(statement[statement.index("(") + 1 : statement.index(")\n")]))
    assert added <= columns, f"altered in but never written: {sorted(added - columns)}"
