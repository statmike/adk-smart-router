"""Every `%` inside a `FORMAT` string, checked before BigQuery gets to reject it.

`train_router.sql` builds most of its statements with `EXECUTE IMMEDIATE FORMAT` over a
triple-quoted block, so every character of those blocks is a format string, comments
included. A bare
`%` in a comment is not a comment to `FORMAT`: it reads the next character as a specifier,
and `(21.5%)` raises *Invalid format specifier character ")"*.

The failure mode is what makes this worth a lint. `CREATE OR REPLACE PROCEDURE` accepts the
body without evaluating any of it, so `deploy_sql.sh` succeeds and `--dry-run` succeeds. The
error appears only when the procedure runs, which on this project is a training job that has
already paid for its embeddings. Between 2026-09-05 and the commit that added this file, a
`(21.5%)` written into a comment left the deployed procedure a full commit behind the
repository and nobody found out, because nothing between the edit and a retrain executes it.

BigQuery's specifiers are documented under FORMAT. The rule enforced here is the narrow one:
after a `%`, either another `%`, or optional flags, width and precision, then one of the
conversion characters. Anything else fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SQL_DIR = Path(__file__).resolve().parents[1] / "src" / "adk_smart_router" / "sql"

# `FORMAT(` followed by a triple-quoted block, which is how every generated statement in
# this repository is written. Single-quoted FORMAT calls are one line and fit on the eye.
BLOCK = re.compile(r'FORMAT\(\s*"""(.*?)"""', re.DOTALL)

# %%, or % then flags, width, precision, then a conversion character. BigQuery's set, plus
# the `*` width that takes its value from an argument.
SPECIFIER = re.compile(r"%(?:%|[-+ #0']*[0-9*]*(?:\.[0-9*]+)?[diouxXfFeEgGaAsTtp])")


def sql_files() -> list[Path]:
    return sorted(SQL_DIR.glob("*.sql"))


def bad_specifiers(block: str) -> list[str]:
    """Every `%` in the block that FORMAT would refuse, with a little context each."""
    found = []
    for match in re.finditer("%", block):
        at = match.start()
        if SPECIFIER.match(block, at):
            continue
        if at and SPECIFIER.match(block, at - 1) and block[at - 1] == "%":
            continue  # the second half of an escaped %%
        found.append(block[max(0, at - 40) : at + 10].replace("\n", " "))
    return found


@pytest.mark.parametrize("path", sql_files(), ids=lambda p: p.name)
def test_no_bare_percent_survives_in_a_generated_statement(path: Path) -> None:
    text = path.read_text()
    offenders = [bad for block in BLOCK.findall(text) for bad in bad_specifiers(block)]
    assert not offenders, (
        f"{path.name} has {len(offenders)} bare % inside a FORMAT block, which deploys "
        f"cleanly and then fails at run time: {offenders}"
    )


def test_the_lint_finds_the_percent_that_actually_broke_a_training_run() -> None:
    """The literal text from `train_router.sql`, so the check cannot rot into a no-op."""
    assert bad_specifiers("-- 88 of 409 eval sessions (21.5%) had a duplicate")
    assert not bad_specifiers("-- 88 of 409 eval sessions (21.5%%) had a duplicate")


def test_the_specifiers_the_pipeline_really_uses_are_accepted() -> None:
    assert not bad_specifiers("CREATE TABLE `%s` AS SELECT %s FROM `%s`")
    assert not bad_specifiers("JSON '{\"outputDimensionality\": %d}'")
    assert not bad_specifiers("returned %d vectors that are not %d wide (widest %d)")
    assert not bad_specifiers("FORMAT('v[OFFSET(%d)] AS f%d', i, i)")


def test_a_format_block_is_actually_being_read() -> None:
    """A regex that matched nothing would pass every file above and check nothing."""
    blocks = BLOCK.findall((SQL_DIR / "train_router.sql").read_text())
    assert len(blocks) > 10
    assert any("CREATE OR REPLACE MODEL" in block for block in blocks)
