#!/usr/bin/env python
"""E1 — build an agent whose tools and traffic both came from somebody else.

Every live number this project publishes was measured on the reference agent, whose 100
tools we wrote and whose conversations we wrote. `score_xlam.py` already answered the
offline half of R12: the routing rule holds on `xlam-function-calling-60k`, a corpus of
60,000 queries over 3,179 mostly real RapidAPI endpoints. What it could not answer is
whether the *plugin* holds there, because scoring a rule offline never sends a request.

This script builds the agent that lets us send them. It reads xLAM's own schemas, takes
the 100 most-called tools, and writes a Python module of stubs whose signatures and
docstrings are xLAM's text rather than ours. ADK turns those into function declarations,
so the tool surface the model sees is a third party's. `queries.jsonl` beside it is every
xLAM row whose called tools all lie on that surface: 15,786 queries, verbatim.

Why 100 and not 25 or 400
-------------------------
To match the reference agent exactly (DESIGN.md 19.15). The point of E1 is to change the
data and nothing else, so the surface width, the model and the router settings all stay
where they were and the two runs can be read against each other. `score_xlam.py` scores
the same width offline, so there is an offline number to compare the live one to.

A row belongs to the corpus when every tool it calls is on the surface. A row calling
something off-surface is not a hard case, it is a different agent, and keeping it would
grade the router on tools the agent does not have.

What is faithful here and what is not
-------------------------------------
Faithful: the tool names, the parameter names, the parameter types, the descriptions, and
the queries. Those are the bytes the model reasons over and the ones the router prunes.

Not faithful: the tool *bodies*. xLAM has no backends, so every stub returns a small
deterministic acknowledgement. That is the same shape as the reference agent's light mode,
which means the tool-response channel is out of scope here exactly as it is there. E2
covers that channel separately.

Also not faithful: xLAM rows are single queries, not conversations. A session here is one
query, and the turns after it are the model calling a tool and reading the result. So the
sessions are short and the router's menu is locked from a real first message, which is the
mechanism under test, but nobody should read a multi-turn saving off this run.

Deterministic, offline, free. No credentials and no network beyond the one-time corpus
download that `score_xlam.py --fetch` already did.

Usage:
    uv run python scripts/build_xlam_agent.py
    uv run python scripts/build_xlam_agent.py --surface 400 --out /tmp/wider
"""

from __future__ import annotations

import argparse
import collections
import json
import keyword
import subprocess
import textwrap
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SOURCE_URL = (
    "https://huggingface.co/datasets/lockon/xlam-function-calling-60k"
    "/resolve/main/xlam_function_calling_60k.json"
)
CORPUS_PATH = Path(".xlam/xlam.json")
AGENT_DIR = Path("examples/xlam_agent")
SURFACE = 100

# xLAM writes types as free text, so this maps what is actually in the top surfaces onto
# something ADK can build a declaration from. Anything unlisted becomes `str`, which is
# lossy on purpose: a wrong-but-simple annotation still declares the parameter, and a
# nested generic that Gemini rejects would take the whole tool out of the surface.
TYPES = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "list": "list[str]",
    "List": "list[str]",
    "List[str]": "list[str]",
    "List[int]": "list[int]",
    "List[float]": "list[float]",
    "List[Union[int, float]]": "list[float]",
}

EMPTY = {"str": '""', "int": "0", "float": "0.0", "bool": "False"}


@dataclass(frozen=True)
class Param:
    """One tool parameter, already reduced to something renderable."""

    name: str
    annotation: str
    description: str
    default: str | None  # rendered Python source, or None when the parameter is required


@dataclass(frozen=True)
class Tool:
    """One xLAM tool, ready to be written out as a stub."""

    name: str
    description: str
    params: tuple[Param, ...]


def fetch(path: Path) -> None:
    """Download xlam_function_calling_60k.json on demand from Hugging Face if absent."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"downloading {SOURCE_URL} -> {path} ...")
    urllib.request.urlretrieve(SOURCE_URL, tmp)  # noqa: S310
    tmp.replace(path)


def load(path: Path, *, auto_fetch: bool = False) -> list[dict[str, object]]:
    if not path.exists():
        if auto_fetch:
            fetch(path)
        else:
            raise SystemExit(
                f"{path} is missing. Run `build_xlam_agent.py --fetch` "
                "or `score_xlam.py --fetch` first."
            )
    rows: list[dict[str, object]] = json.loads(path.read_text())
    return rows


def called(row: dict[str, object]) -> frozenset[str]:
    return frozenset(call["name"] for call in json.loads(str(row["answers"])))


def surface(rows: Sequence[dict[str, object]], width: int) -> list[str]:
    """The `width` most-called tool names, most popular first."""
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        counts.update(called(row))
    return [name for name, _ in counts.most_common(width)]


def schemas(rows: Sequence[dict[str, object]], names: Sequence[str]) -> dict[str, dict]:
    """The commonest schema for each named tool.

    Eleven of the top hundred are described more than one way across the corpus, usually a
    reworded sentence. Taking the modal definition is a choice, and a defensible one: the
    agent needs exactly one declaration per tool, and the version most rows agree on is the
    one most of the traffic was written against.
    """
    seen: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    wanted = set(names)
    for row in rows:
        for tool in json.loads(str(row["tools"])):
            if tool["name"] in wanted:
                seen[tool["name"]][json.dumps(tool, sort_keys=True)] += 1
    return {name: json.loads(seen[name].most_common(1)[0][0]) for name in names if seen[name]}


def literal(text: str, annotation: str) -> str | None:
    """xLAM's default value as Python source, or None if it will not coerce cleanly."""
    text = str(text).strip()
    if annotation == "str":
        return repr(text)
    if not text:
        return None
    if annotation == "bool":
        return {"true": "True", "false": "False"}.get(text.lower())
    try:
        return repr(int(text)) if annotation == "int" else repr(float(text))
    except ValueError:
        return None


def parameter(name: str, spec: dict) -> Param | None:
    """One parameter, or None if it cannot be declared safely."""
    if not name.isidentifier() or keyword.iskeyword(name):
        return None
    declared = str(spec.get("type", "str"))
    optional = "optional" in declared
    annotation = TYPES.get(declared.split(",")[0].strip(), "str")
    default = None
    if optional:
        default = literal(spec.get("default", ""), annotation) or EMPTY.get(annotation, "None")
    return Param(name, annotation, " ".join(str(spec.get("description", "")).split()), default)


def tool_of(schema: dict) -> Tool:
    """One xLAM schema as a renderable tool, required parameters first.

    Reordering is forced rather than chosen: Python will not accept a parameter without a
    default after one with a default, and xLAM's dictionaries are in neither order. Order
    is not part of the declaration Gemini reads, so nothing about the surface changes.
    """
    params = [parameter(n, s) for n, s in schema.get("parameters", {}).items()]
    kept = [p for p in params if p is not None]
    ordered = [p for p in kept if p.default is None] + [p for p in kept if p.default is not None]
    description = " ".join(str(schema.get("description", "")).split())
    return Tool(schema["name"], description, tuple(ordered))


def render(tool: Tool) -> str:
    """One stub, as source. Wrapped by hand because the docstring is the declaration."""
    signature = ", ".join(
        f"{p.name}: {p.annotation}" + (f" = {p.default}" if p.default else "") for p in tool.params
    )
    head = f"def {tool.name}({signature}) -> dict[str, str]:"
    # 92 not 100: four spaces of indent and three quote characters come first.
    lines = textwrap.wrap(tool.description or f"Call {tool.name}.", width=92)
    body = ['    """' + (lines[0] if lines else "")]
    body += [f"    {line}" for line in lines[1:]]
    if tool.params:
        body.append("")
        body.append("    Args:")
        for p in tool.params:
            text = p.description or f"The {p.name.replace('_', ' ')}."
            wrapped = textwrap.wrap(f"{p.name}: {text}", width=84)
            body.append(f"        {wrapped[0]}")
            body += [f"            {line}" for line in wrapped[1:]]
    body.append('    """')
    arguments = ", ".join(f"{p.name}={p.name}" for p in tool.params)
    body.append(f"    return _acknowledge({tool.name.__repr__()}, {arguments})")
    return "\n".join([head, *body])


HEADER = '''"""The xLAM agent's __COUNT__ stub tools, generated from somebody else's schemas.

Do not edit. `scripts/build_xlam_agent.py` writes this file from the __COUNT__ most-called
tools in `xlam-function-calling-60k` (Salesforce, CC-BY-4.0, arXiv:2406.18518). Every
name, parameter and sentence below is that dataset's, which is the whole point: the
declarations the model reads here were written by people who had never heard of this
project. ROADMAP's E1, and DESIGN.md 19.15 for the reference agent this mirrors.

The bodies are ours and they are deliberately empty of behaviour. xLAM ships no backends,
so a stub acknowledges its arguments and returns. That keeps the tool-response channel out
of the measurement, exactly as the reference agent's light mode does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _acknowledge(name: str, **arguments: Any) -> dict[str, str]:  # noqa: ANN401
    """A small deterministic result, so a call has an outcome without having a backend.

    ANN401 is the honest annotation here: this takes whatever the hundred signatures above
    it take. Their declared types are on the functions themselves, which is where ADK reads
    them from.
    """
    given = ", ".join(f"{key}={value!r}" for key, value in sorted(arguments.items()))
    return {"tool": name, "status": "ok", "arguments": given or "none"}


'''

FOOTER = """

ALL_TOOLS: list[Callable[..., Any]] = [
__NAMES__
]
"""


def module(tools: Sequence[Tool]) -> str:
    stubs = "\n\n\n".join(render(tool) for tool in tools)
    names = "\n".join(f"    {tool.name}," for tool in tools)
    return HEADER.replace("__COUNT__", str(len(tools))) + stubs + FOOTER.replace("__NAMES__", names)


def corpus(rows: Sequence[dict[str, object]], names: Sequence[str]) -> list[dict[str, object]]:
    """Every row whose called tools all lie on the surface, in corpus order."""
    on_surface = set(names)
    kept = []
    for row in rows:
        tools = called(row)
        if tools and tools <= on_surface:
            kept.append(
                {"id": str(row["id"]), "query": str(row["query"]).strip(), "tools": sorted(tools)}
            )
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--surface", type=int, default=SURFACE, help="how many tools to declare")
    parser.add_argument("--out", type=Path, default=AGENT_DIR)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download the xLAM corpus from Hugging Face on demand if missing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load(args.corpus, auto_fetch=args.fetch)
    names = surface(rows, args.surface)
    tools = [tool_of(schema) for schema in schemas(rows, names).values()]
    queries = corpus(rows, names)

    args.out.mkdir(parents=True, exist_ok=True)
    written = args.out / "tools.py"
    written.write_text(module(tools))
    # Formatted by the formatter that checks it, rather than by hand-wrapping every
    # signature here. Without this a regeneration would leave the repo failing its own
    # `ruff format --check`, which is a trap for whoever regenerates it next.
    subprocess.run(["ruff", "format", "--quiet", str(written)], check=True)  # noqa: S603, S607
    lines = "".join(json.dumps(row) + "\n" for row in queries)
    (args.out / "queries.jsonl").write_text(lines)

    declared = sum(len(t.params) for t in tools)
    print(f"{len(rows):,} xLAM rows read from {args.corpus}")
    print(f"  surface: {len(tools)} tools, {declared} declared parameters")
    print(f"  corpus:  {len(queries):,} queries whose tools all lie on it")
    print(f"  wrote {args.out / 'tools.py'} and {args.out / 'queries.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
