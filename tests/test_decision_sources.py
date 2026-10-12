"""One vocabulary, a dozen copies of it, and the lint that keeps them saying the same thing.

`decision_source` is the field every question about the router is eventually asked through.
Which sessions were routed, which arm a comparison is measuring, which sessions the training
corpus has to throw away because the router already censored them. Fourteen values, and the
list of them is written out in a dozen places: the CASE in `sql/dashboard.sql`, the censoring
filter in `sql/train_router.sql` and `scripts/score_sets.py`, and the arm labels in eight
reporting scripts.

Those copies are deliberate. `scripts/replay_promotion.py` says it plainly: a promoter that
scored on a different definition than the operator reads would be answering a different
question. Each script stating its own definition is what stops an edit somewhere else
silently changing what a gate means. The cost of that choice is drift, and drift here is
silent in the worst way. A report that labels the arms with a stale list does not fail; it
prints a number that is wrong by however many sessions the stale value covers.

So this is a lint over the committed text rather than a behavioural test. It reads the
scripts and the SQL, finds every place the vocabulary is written down, and checks each one
against `provenance.DECISION_SOURCES` and `provenance.ROUTED_SOURCES`. No BigQuery, no
credentials, no data.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from adk_smart_router import plugin
from adk_smart_router.embedding import BREAKER_OPEN, CALL_ERROR, TIMEOUT
from adk_smart_router.provenance import DECISION_SOURCES, ROUTED_SOURCES
from adk_smart_router.pruner import PASSTHROUGH_SOURCES, SessionDecision

REPO = Path(__file__).resolve().parents[1]
SQL_DIR = REPO / "src" / "adk_smart_router" / "sql"
SCRIPTS = REPO / "scripts"

# The router never ran on these sessions, so no rule of its own produced them: one is the
# permanent control arm and the other is a process holding no artifact.
NEVER_RAN = frozenset({"holdout", "disabled"})

# The three ways the embedding call can fail. The session runs unrouted and the router never
# saw a vector to have an opinion about (DESIGN.md 19.23).
EMBED_FAILED = frozenset({TIMEOUT, CALL_ERROR, BREAKER_OPEN})

_QUOTED = re.compile(r"""['"]([a-z_]+)['"]""")

# `ROUTED_SOURCES = ("fresh", "reused")` and `ROUTED = {"fresh", "reused"}`.
_PY_COPY = re.compile(r"^(?:ROUTED_SOURCES|ROUTED)\s*=\s*[({]([^)}]*)[)}]", re.MULTILINE)

# `JSON_VALUE(record, '$.decision_source') IN (...)` and bare `decision_source IN (...)`.
# `IN UNNEST(@routed_sources)` is deliberately not matched: it binds to a Python constant,
# which the check above already covers.
_SQL_FILTER = re.compile(r"decision_source(?:'\))?\s+IN\s+\(([^)]*)\)")

# The dashboard's CASE over the same column. It is the only CASE in the SQL with literal arms.
_CASE_ARM = re.compile(r"WHEN '([a-z_]+)' THEN")

# Where a decision source is chosen. Restricted to these two files on purpose: `source=` is
# also the name of a `ttl_source` argument in `cache_hooks.py`, and a lint that could not tell
# the two apart would be reporting on the wrong field.
DECIDERS = ("plugin.py", "pruner.py")
_ASSIGNED = re.compile(r'\bsource="([a-z_]+)"')


def sources_in(text: str) -> frozenset[str]:
    """The quoted lowercase words in one list literal."""
    return frozenset(_QUOTED.findall(text))


def scripts() -> list[Path]:
    found = sorted(SCRIPTS.glob("*.py"))
    assert found, f"no scripts under {SCRIPTS}"
    return found


def sql_files() -> list[Path]:
    found = sorted(SQL_DIR.glob("*.sql"))
    assert found, f"no SQL under {SQL_DIR}"
    return found


# --- The vocabulary is complete and it divides cleanly ----------------------------


def test_every_source_belongs_to_exactly_one_kind() -> None:
    """The fourteen values are four kinds of thing, and reading a report needs the difference.

    Two mean the router acted. Seven mean the routing rule ran and declined, which is the
    router working. Two mean it never ran at all. Three mean it could not run because the
    embedding never came back, which is the fail-open path and an incident rather than a
    decision. Pooling the last two into the third is how an outage reads as a quiet agent.
    """
    kinds = (ROUTED_SOURCES, PASSTHROUGH_SOURCES, NEVER_RAN, EMBED_FAILED)
    assert set().union(*kinds) == DECISION_SOURCES
    assert sum(len(kind) for kind in kinds) == len(DECISION_SOURCES), "a value is in two kinds"


def test_the_code_that_chooses_a_source_only_chooses_named_ones() -> None:
    """A source written by the plugin but absent from the vocabulary reaches nothing.

    It would land in the warehouse, fall through every CASE arm and every filter, and be
    counted as a declined session by every report. Nothing would fail.
    """
    for name in DECIDERS:
        text = (REPO / "src" / "adk_smart_router" / name).read_text(encoding="utf-8")
        written = {value for value in _ASSIGNED.findall(text) if value}
        assert written, f"{name}: the lint stopped finding assignments"
        assert written <= DECISION_SOURCES, f"{name} writes {sorted(written - DECISION_SOURCES)}"


def test_the_sources_the_plugin_holds_in_constants_are_named_too() -> None:
    """The same claim for the sources no `source="..."` literal spells out.

    Three of the plugin's are held in module constants or returned by `_held`, so the text
    lint above cannot see them, and they are exactly the interesting ones: the decline rule
    and the two cache guards.
    """
    decided = SessionDecision(kept=frozenset({"refund"}))
    produced = {plugin._TOO_LATE.source, plugin._TOO_SMALL.source, plugin._NOT_WORTH}
    produced |= {failed.source for failed in plugin._EMBED_FAILED.values()}
    produced |= {plugin._held(decided, floor_ok=ok) for ok in (True, False)}

    assert produced <= DECISION_SOURCES, (
        f"the plugin can write {sorted(produced - DECISION_SOURCES)}"
    )
    assert {"cheap", "floor", "watch"} <= produced, "the lint stopped reaching the guards"


# --- Every copy of the list says the same thing ----------------------------------


@pytest.mark.parametrize("path", scripts(), ids=lambda p: p.name)
def test_every_script_copy_of_the_routed_list_matches(path: Path) -> None:
    """Eight scripts define this for themselves. All eight have to mean the same eight."""
    for literal in _PY_COPY.findall(path.read_text(encoding="utf-8")):
        assert sources_in(literal) == ROUTED_SOURCES, f"{path.name} labels a different arm"


@pytest.mark.parametrize("path", [*sql_files(), *scripts()], ids=lambda p: p.name)
def test_every_sql_filter_on_the_source_column_is_one_of_the_known_meanings(path: Path) -> None:
    """A query filtering on this column is asking one of two questions, and it must ask it whole.

    Either "was this session routed", or "did the embedding fail". A list that is neither is
    a hand-written subset, and a hand-written subset of a vocabulary that grows is a filter
    that quietly stops matching the rows it was written for.
    """
    for literal in _SQL_FILTER.findall(path.read_text(encoding="utf-8")):
        found = sources_in(literal)
        assert found in (ROUTED_SOURCES, EMBED_FAILED), (
            f"{path.name} filters on {sorted(found)}, which is neither arm nor fail-open"
        )


def test_the_dashboard_buckets_only_sources_that_exist() -> None:
    """A CASE arm for a retired value is dead, and a misspelled one is worse than dead.

    It reads as handled. The value it was meant to catch falls through to the ELSE and is
    labelled `declined`, which is the bucket that means the router worked.
    """
    arms = set(_CASE_ARM.findall((SQL_DIR / "dashboard.sql").read_text(encoding="utf-8")))
    assert arms, "the lint stopped finding the CASE"
    assert arms <= DECISION_SOURCES, f"dashboard.sql buckets {sorted(arms - DECISION_SOURCES)}"


# --- The lint finds what it claims to find ---------------------------------------


def test_the_lint_actually_matches_the_copies_it_checks() -> None:
    """Every test above passes vacuously against a regex that matches nothing.

    Floors rather than exact counts, so that adding a script does not fail the suite while
    deleting the parser does.
    """
    py_copies = [c for p in scripts() for c in _PY_COPY.findall(p.read_text(encoding="utf-8"))]
    sql_filters = [
        c
        for p in (*sql_files(), *scripts())
        for c in _SQL_FILTER.findall(p.read_text(encoding="utf-8"))
    ]

    assert len(py_copies) >= 8
    assert len(sql_filters) >= 3
    assert any(sources_in(c) == ROUTED_SOURCES for c in sql_filters)
    assert any(sources_in(c) == EMBED_FAILED for c in sql_filters)


def test_a_drifted_copy_would_be_caught(tmp_path: Path) -> None:
    """The negative control: the shapes this lint reads, written wrong."""
    assert sources_in('ROUTED_SOURCES = ("fresh", "reused", "grown")') != ROUTED_SOURCES
    assert sources_in("decision_source') IN ('fresh')") != ROUTED_SOURCES

    drifted = tmp_path / "drifted.py"
    drifted.write_text('ROUTED_SOURCES = ("fresh", "grown")\n', encoding="utf-8")
    (literal,) = _PY_COPY.findall(drifted.read_text(encoding="utf-8"))
    assert sources_in(literal) == {"fresh", "grown"}
