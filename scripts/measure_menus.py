"""What a fixed tool menu costs in coverage, and what it buys in shared prefix.

DESIGN.md 18.16 measured G1 at +7.5% and found the cause: the router shipped 40 distinct
tool menus across 65 sessions, so every pruned prefix was cold at the start of a session
while the control's single prefix stayed permanently hot. The prune was worth -28% held at
the control's cache hit rate and it gave 13.1 points of that hit rate away.

The fix has to make the pruned prefix uniform again without giving back the coverage the
router exists to provide. Two mechanisms could do that and they fail differently:

    canonical menus  choose M fixed menus of `top_n` tools and snap each session to its
                     nearest. M prefixes instead of one per session. Costs coverage: a
                     snapped menu is by definition not the best tools for this user.

    stable core      always declare the same K most popular tools, in the same position,
                     and let the router choose only the remaining `top_n - K` slots. Every
                     session then shares a literal prefix of the system instruction plus K
                     declarations, which is what implicit caching matches on. Costs token
                     saving rather than coverage, and only if K is large.

This script measures both against the real corpus, offline, with no live traffic. Coverage
here is the gate's own definition: the fraction of tool-call rows whose tool was in the menu
shipped for that session (DESIGN.md 10, G2).

Both schemes are scored with **oracle** assignment: each session gets the best menu or the
best tail available to it. That is an upper bound, not an achievable number, because in
production the router has to predict the menu from the first message before any tool is
called. An upper bound is the right thing to measure first: if the ceiling is below the
static baseline the mechanism is dead and no amount of modelling saves it.

Canonical menus went on to ship, so the choosing now lives in
`adk_smart_router.menus` and this script imports it. Re-running this measures the code
that is actually published, which is the only version of it worth a number.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
from collections.abc import Collection
from pathlib import Path

from google.cloud import bigquery

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
# `declaration_tokens` reads the reference agent's tools. Scripts that import this module
# already add it; adding it here is what makes the module runnable on its own.
sys.path.insert(0, str(REPO_ROOT / "examples"))

from adk_smart_router.menus import choose, pad, popularity  # noqa: E402

# Measured on the reference agent, 2026-09-01. Both are properties of the fixture rather
# than of any model, so they are printed alongside the results and not hidden in the maths.
SYSTEM_INSTRUCTION_TOKENS = 1777
CACHE_FLOOR_TOKENS = 2048

# Which model's tokenizer answers `count_tokens`. The reference agent's serving model, so
# the ratio measured here is the ratio that will be billed.
COUNT_TOKENS_MODEL = "gemini-2.5-flash"

CORPUS_SQL = """
SELECT session_id, STRING_AGG(tool_name, '|' ORDER BY timestamp) AS tools
FROM `{project}.{dataset}.v_tool_completed`
WHERE agent = @agent
  AND status = 'OK'
  AND tool_name != 'transfer_to_agent'
  AND STARTS_WITH(session_id, 'corpus')
  AND JSON_VALUE(tool_result, '$._smart_router.cache_hit') IS NULL
GROUP BY session_id
"""


def load_sessions(project: str, dataset: str, agent: str) -> list[list[str]]:
    """One list of tool calls per session, in the order they were made."""
    client = bigquery.Client(project=project)
    job = client.query(
        CORPUS_SQL.format(project=project, dataset=dataset),
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("agent", "STRING", agent)]
        ),
    )
    return [row["tools"].split("|") for row in job.result()]


def _tools_for_agent(agent: str) -> list[object]:
    if agent == "xlam":
        try:
            from xlam_agent.tools import ALL_TOOLS as XLAM_TOOLS
        except ImportError as exc:
            raise SystemExit(
                "examples/xlam_agent/tools.py is missing. "
                "Run `uv run python scripts/build_xlam_agent.py --fetch` first."
            ) from exc

        return list(XLAM_TOOLS)
    from reference_agent.tools import ALL_TOOLS

    return list(ALL_TOOLS)


def declaration_chars(
    only: Collection[str] | None = None, *, agent: str = "support"
) -> dict[str, int]:
    """Each tool's declaration size in characters, which is exact and free.

    `only` narrows the surface to a set of tool names. Pricing a smaller agent than the one
    this repository declares today is what it is for, and it has to be applied here rather
    than by filtering the result, because the chars-per-token ratio belongs to the
    declarations that were counted (DEV_NOTES.md, 2026-09-04).
    """
    from google.adk.tools.function_tool import FunctionTool
    from google.genai import types

    from adk_smart_router import prompt_size

    sizes = {}
    for fn in _tools_for_agent(agent):
        tool = fn if isinstance(fn, FunctionTool) else FunctionTool(func=fn)  # type: ignore[arg-type]
        declaration = tool._get_declaration()
        if declaration is None:
            continue
        if only is not None and declaration.name not in only:
            continue

        class Req:
            pass

        req = Req()
        req.config = types.GenerateContentConfig(  # type: ignore[attr-defined]
            tools=[types.Tool(function_declarations=[declaration])]
        )
        req.contents = []  # type: ignore[attr-defined]
        chars = prompt_size.count_chars(req, None)  # type: ignore[arg-type]
        sizes[declaration.name] = chars.full
    return sizes


async def declaration_tokens(
    only: Collection[str] | None = None, *, agent: str = "support"
) -> dict[str, int]:
    """Each tool's declaration cost, measured against the model rather than estimated.

    Characters per tool are exact and free. The characters-per-token ratio is neither, and
    it cannot be assumed: it is a property of these declarations and not of JSON, and the
    starting constant `prompt_size` ships is 26% off on this surface (DEV_NOTES.md,
    2026-09-04). One `count_tokens` call over the whole tool list fixes it, and each tool's
    share of the characters splits the total. That is exact in aggregate, which is how every
    caller uses it, and within a token or two per tool.

    Falls back to the estimate, loudly, if the call cannot be made. A wrong number that
    announces itself is usable; a wrong number that does not is what this replaces.

    `only` narrows the surface, and the ratio is then measured over that narrower surface.
    The original twenty-five declarations run at 8.30 chars per token and the whole hundred
    at 6.17, so a subset priced on the full list's ratio is wrong by a third.
    """
    from adk_smart_router import prompt_size

    chars = declaration_chars(only, agent=agent)
    total_chars = sum(chars.values())
    if not total_chars:
        return {}

    ratio = await _measured_schema_ratio(only, agent=agent)
    if ratio is None:
        print(
            f"  count_tokens unavailable; estimating declarations at "
            f"{prompt_size.SCHEMA_CHARS_PER_TOKEN} chars/token, which is a guess",
            file=sys.stderr,
        )
        ratio = prompt_size.SCHEMA_CHARS_PER_TOKEN
    return {name: round(size / ratio) for name, size in chars.items()}


async def _measured_schema_ratio(
    only: Collection[str] | None = None, *, agent: str = "support"
) -> float | None:
    """Characters per token for this agent's declarations, from one `count_tokens` call.

    ADC only, `vertexai=True`, the same contract `embedding.py` holds to.

    Pinning the client to Vertex has a second effect worth knowing about, because it is not
    obvious and it is load-bearing. ADK builds a declaration with a `response_json_schema` on
    Vertex and without one on the Gemini Developer API, 69 characters a tool, so
    `declaration_chars` reads 88,431 on this surface in one shell and 81,531 in another
    (DESIGN.md 19.46). That difference cancels out here. The ratio is measured over the same
    declarations it will divide, against a count that is always Vertex's, so `chars / ratio`
    comes back as Vertex's token count either way: 14,324 against 14,327 on the hundred-tool
    agent, which is rounding.

    That is the behaviour we want, since the traffic these numbers are compared against was
    logged by a Vertex deployment. It is worth stating because `smart-router qualify`, which
    estimates rather than measures, is not immune and has to warn about it.
    """
    from google import genai
    from google.adk.tools.function_tool import FunctionTool
    from google.genai import types

    from adk_smart_router.config import load_config

    cfg = load_config()
    declarations = []
    for fn in _tools_for_agent(agent):
        tool = fn if isinstance(fn, FunctionTool) else FunctionTool(func=fn)  # type: ignore[arg-type]
        declaration = tool._get_declaration()
        if declaration is not None and (only is None or declaration.name in only):
            declarations.append(declaration)

    try:
        client = genai.Client(
            vertexai=True,
            project=cfg.bq_project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=cfg.vertex_location,
        )
        # An empty prompt is not accepted, so a one-character one is sent and subtracted.
        with_tools = await client.aio.models.count_tokens(
            model=COUNT_TOKENS_MODEL,
            contents=".",
            config=types.CountTokensConfig(tools=[types.Tool(function_declarations=declarations)]),
        )
        bare = await client.aio.models.count_tokens(model=COUNT_TOKENS_MODEL, contents=".")
    except Exception as exc:  # noqa: BLE001 - any failure means fall back and say so
        print(f"  count_tokens failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    tokens = (with_tools.total_tokens or 0) - (bare.total_tokens or 0)
    if tokens <= 0:
        return None
    return sum(declaration_chars(only).values()) / tokens


# --- coverage -----------------------------------------------------------------


def coverage(sessions: list[list[str]], menu_for: collections.abc.Callable) -> float:
    """Fraction of tool-call rows whose tool was in the menu that session shipped."""
    hits = total = 0
    for calls in sessions:
        menu = menu_for(calls)
        hits += sum(1 for call in calls if call in menu)
        total += len(calls)
    return hits / total if total else 0.0


def best_menu(calls: list[str], menus: collections.abc.Sequence[frozenset[str]]) -> frozenset[str]:
    """The menu in the pool that covers most of this session's calls."""
    return max(menus, key=lambda menu: sum(1 for call in calls if call in menu))


# --- the two schemes ----------------------------------------------------------


def report_canonical(sessions: list[list[str]], top_n: int, counts: list[int]) -> None:
    ranked = popularity(sessions)
    n_sessions = len(sessions)
    print(f"\ncanonical menus, {top_n} tools each, oracle assignment")
    print(f"  {'menus':>6} {'coverage':>9} {'sessions/prefix':>16}")
    for count in counts:
        menus = choose(sessions, size=top_n, count=count)
        got = coverage(sessions, lambda calls, menus=menus: best_menu(calls, menus))
        print(f"  {len(menus):>6} {got:>9.4f} {n_sessions / max(len(menus), 1):>16.1f}")
    bespoke = coverage(sessions, lambda calls: set(pad(dict.fromkeys(calls), ranked, top_n)))
    print(f"  {'bespoke':>6} {bespoke:>9.4f} {1.0:>16.1f}   (today: one menu per session)")


def report_stable_core(sessions: list[list[str]], top_n: int, sizes: dict[str, int]) -> None:
    ranked = popularity(sessions)
    print(f"\nstable core, {top_n} tools declared, K of them fixed")
    print(f"  {'K':>3} {'coverage':>9} {'shared prefix':>14} {'cacheable':>10} {'pruned away':>12}")
    for k in range(top_n + 1):
        core = ranked[:k]
        core_tokens = SYSTEM_INSTRUCTION_TOKENS + sum(sizes.get(t, 0) for t in core)
        free = top_n - k

        def menu_for(calls: list[str], core: list[str] = core, free: int = free) -> set[str]:
            tail = [c for c in dict.fromkeys(calls) if c not in core][:free]
            return set(pad([*core, *tail], ranked, top_n))

        got = coverage(sessions, menu_for)
        dropped = sum(sizes.values()) - sum(sorted(sizes.values(), reverse=True)[:top_n])
        cacheable = "yes" if core_tokens >= CACHE_FLOOR_TOKENS else "no"
        print(f"  {k:>3} {got:>9.4f} {core_tokens:>14} {cacheable:>10} {dropped:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--dataset", default="smart_router")
    parser.add_argument("--agent", default="support")
    parser.add_argument("--top-n", type=int, default=7)
    parser.add_argument(
        "--dump-tokens",
        default=None,
        help="write each tool's declaration cost to this JSON file and stop. "
        "`publish_router.py --declaration-tokens` reads it, which is how the publish "
        "gate gets the token counts the decline rule needs (DESIGN.md 19.13).",
    )
    args = parser.parse_args()

    if args.dump_tokens:
        sizes = asyncio.run(declaration_tokens())
        Path(args.dump_tokens).write_text(json.dumps(sizes, indent=2, sort_keys=True) + "\n")
        print(f"wrote {len(sizes)} declaration sizes to {args.dump_tokens}")
        return

    sessions = load_sessions(args.project, args.dataset, args.agent)
    sizes = asyncio.run(declaration_tokens())
    ranked = popularity(sessions)
    rows = sum(len(calls) for calls in sessions)

    print(f"{len(sessions)} corpus sessions, {rows} tool-call rows, {len(sizes)} declarations")
    print(
        f"system instruction {SYSTEM_INSTRUCTION_TOKENS} tokens, cache floor {CACHE_FLOOR_TOKENS}"
    )
    static = set(ranked[: args.top_n])
    got = coverage(sessions, lambda _calls: static)
    print(f"\nstatic top-{args.top_n} (one menu for everyone): {got:.4f}")

    report_canonical(sessions, args.top_n, [2, 4, 8, 12, 16, 24])
    report_stable_core(sessions, args.top_n, sizes)


if __name__ == "__main__":
    main()
