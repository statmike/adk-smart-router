"""`scripts/README.md` is an index of `scripts/`, and this is what keeps it one.

The table drifted to 37 rows against 53 files before anyone noticed, because nothing failed when a
script landed without one. A missing row is not a broken build, so it needs a test or it comes
back. Two contracts:

  * every runnable file in the directory has exactly one capability row, in one of the four tables
  * every runnable file appears somewhere in the cost table, alone or in a grouped row

`deployment.py` and `deployment.sh` are exempt. They are sourced and imported rather than run, they
cost nothing because they make no calls, and the opening prose names them for that reason. The
exemption is a named set rather than a heuristic, so adding a third library is a deliberate edit
here.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
README = SCRIPTS / "README.md"

# Sourced or imported, never run. See the module docstring.
LIBRARIES = frozenset({"deployment.py", "deployment.sh"})

COST_HEADING = "## Cost"
# A table row, and the first cell of it. Grouped rows name several scripts in that one cell.
ROW = re.compile(r"^\| (`[^|]*`)\s*\|")
NAME = re.compile(r"`([a-z0-9_]+\.(?:py|sh))`")


def runnable() -> set[str]:
    """Every file in scripts/ that is meant to be run."""
    return {
        p.name for p in SCRIPTS.iterdir() if p.suffix in {".py", ".sh"} and p.name not in LIBRARIES
    }


def named_in_rows(lines: list[str]) -> list[str]:
    """Script names in the first cell of every table row, in order, with duplicates kept."""
    found = []
    for line in lines:
        cell = ROW.match(line)
        if cell:
            found.extend(NAME.findall(cell.group(1)))
    return found


def split_at_cost() -> tuple[list[str], list[str]]:
    lines = README.read_text().splitlines()
    cut = next(i for i, line in enumerate(lines) if line.startswith(COST_HEADING))
    return lines[:cut], lines[cut:]


def test_every_script_has_a_capability_row():
    capability, _ = split_at_cost()
    assert runnable() - set(named_in_rows(capability)) == set()


def test_no_script_has_two_capability_rows():
    capability, _ = split_at_cost()
    names = named_in_rows(capability)
    assert sorted(names) == sorted(set(names))


def test_every_script_has_a_cost_line():
    _, cost = split_at_cost()
    assert runnable() - set(named_in_rows(cost)) == set()


def test_no_row_names_a_script_that_does_not_exist():
    capability, cost = split_at_cost()
    named = set(named_in_rows(capability)) | set(named_in_rows(cost))
    assert {n for n in named if not (SCRIPTS / n).exists()} == set()


def test_the_libraries_are_described_even_though_they_have_no_row():
    prose = README.read_text()
    for name in LIBRARIES:
        assert f"`{name}`" in prose


def test_the_libraries_really_are_libraries():
    """The exemption is only honest while neither one is executable."""
    for name in LIBRARIES:
        assert not (SCRIPTS / name).stat().st_mode & 0o111
