"""The line between the modules a request touches and the modules a retrain cycle touches.

The README says the serving path is "exactly the import closure of `plugin.py`", and until this
file that was a sentence rather than a fact. The claim matters for two reasons that have nothing
to do with tidiness. It is what lets someone extract the plugin without carrying BigQuery, the
CLI and the retrain ledger with it, which is the shape any upstream contribution would take. And
it is what keeps a hot-path module from quietly acquiring a dependency on something that reads
the warehouse, which is a latency and a credentials problem long before it is a design problem.

The closure is read from the source rather than from `sys.modules`, so it catches an import that
only fires under `TYPE_CHECKING` or inside a function body. Both still have to come along in an
extraction, and an import that is invisible at runtime is exactly the kind that drifts.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = "adk_smart_router"
SOURCE = pathlib.Path(__file__).resolve().parent.parent / "src" / PACKAGE

# Never run on a request. A retrain cycle writes these down, or reads them back, or prints them.
OFFLINE = frozenset(
    {
        "cli",
        "generation_watch",
        "lifecycle",
        "openings",
        "qualify",
        "readiness",
        "reachability",
        "status",
    }
)


def module_names() -> frozenset[str]:
    """Every module in the package, `__init__` excluded: it is the front door, not a capability."""
    return frozenset(p.stem for p in SOURCE.glob("*.py") if p.stem != "__init__")


def imports_of(module: str) -> frozenset[str]:
    """The package modules this one names, wherever it names them."""
    tree = ast.parse((SOURCE / f"{module}.py").read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # `from adk_smart_router.pruner import decide`, and `from adk_smart_router import x`.
            head, _, tail = node.module.partition(".")
            if head == PACKAGE:
                found.add(tail) if tail else found.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                head, _, tail = alias.name.partition(".")
                if head == PACKAGE and tail:
                    found.add(tail)
    return frozenset(found) & module_names()


def closure(start: str) -> frozenset[str]:
    """Everything reachable from `start` by imports, `start` included."""
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        module = frontier.pop()
        if module in seen:
            continue
        seen.add(module)
        frontier.extend(imports_of(module))
    return frozenset(seen)


def test_the_serving_path_is_exactly_the_import_closure_of_the_plugin():
    """Neither half may reach into the other by accident, in either direction."""
    assert closure("plugin") == module_names() - OFFLINE


def test_nothing_on_the_serving_path_imports_the_offline_half():
    """The same claim said the other way round, because the failure message is the point.

    The test above tells you the closure moved. This one tells you which module did it and what
    it reached for, which is the difference between a puzzle and a fix.
    """
    reached = {m: sorted(imports_of(m) & OFFLINE) for m in module_names() - OFFLINE}
    assert {m: names for m, names in reached.items() if names} == {}


def test_the_offline_half_is_allowed_to_import_the_serving_half():
    """Stated so that nobody enforces the symmetry by mistake.

    `qualify` prices a prompt with `prompt_size`, `status` reads what `model_store` fetched. The
    direction that matters is one way: a request must not pay for the warehouse.
    """
    borrowed = {name for m in OFFLINE for name in imports_of(m)} - OFFLINE
    assert borrowed, "if the offline half stopped sharing serving code, something was duplicated"
