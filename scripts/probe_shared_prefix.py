"""Does Gemini's implicit cache match a shared prefix of the tool list, or the whole block?

`measure_menus.py` shows that a stable core of five declarations keeps oracle coverage at
1.0 while giving every session an identical leading 2081 tokens, which is above the 2048
floor. That only helps if the implicit cache will actually serve a partial match: the same
system instruction and the same first five declarations, followed by a different tail.

If the cache matches prefixes, requests with different tails still report roughly the core's
worth of cached tokens and the fix in `ROADMAP.md` is worth building. If it treats the tool
list as one atomic block, different tails report nothing cached, and canonical menus are the
only mechanism left.

Three arms, in order, sharing one system instruction:

    warm     identical core and identical tail, sent twice. Establishes that anything
             caches at all in this session, and how much.
    prefix   identical core, a different tail every time. The question.
    control  a different core every time. Nothing should cache, which is what makes a
             positive result in the `prefix` arm mean something.

Cheap: a dozen short calls of about 2.3k tokens each.
"""

from __future__ import annotations

import argparse
import os
import sys

from google import genai
from google.genai import types

CORE_SIZE = 5
TAIL_SIZE = 2

_DESCRIPTION = (
    "Look up one record for the current customer and return it as JSON. "
    "Read-only. Safe to call more than once."
)


def system_instruction() -> str:
    """The reference agent's own instruction, so the prefix is the size we measured."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from reference_agent.instruction import INSTRUCTION

    return INSTRUCTION


def declaration(name: str) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=name,
        description=_DESCRIPTION,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "record_id": types.Schema(
                    type=types.Type.STRING, description="Identifier of the record to fetch."
                )
            },
            required=["record_id"],
        ),
    )


def tools(core_id: int, tail_id: int) -> list[types.Tool]:
    """Declaration order is the experiment: the core is always first, in the same order."""
    core = [declaration(f"core_{core_id}_{i}") for i in range(CORE_SIZE)]
    tail = [declaration(f"tail_{tail_id}_{i}") for i in range(TAIL_SIZE)]
    return [types.Tool(function_declarations=[*core, *tail])]


def ask(
    client: genai.Client, model: str, instruction: str, core: int, tail: int
) -> tuple[int, int]:
    """One call. Returns (prompt tokens, cached tokens)."""
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text="Say ok.")])],
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            tools=tools(core, tail),
            temperature=0.0,
        ),
    )
    usage = response.usage_metadata
    return int(usage.prompt_token_count or 0), int(usage.cached_content_token_count or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument(
        "--location", default=os.environ.get("SMART_ROUTER_VERTEX_LOCATION", "global")
    )
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    instruction = system_instruction()

    print(f"{args.model}, core {CORE_SIZE} declarations, tail {TAIL_SIZE}\n")
    print(f"  {'arm':>8} {'call':>5} {'prompt':>7} {'cached':>7}")

    plan = [
        *[("warm", 0, 0)] * (args.repeats + 1),
        *[("prefix", 0, tail) for tail in range(1, args.repeats + 1)],
        *[("control", core, 0) for core in range(1, args.repeats + 1)],
    ]
    results: dict[str, list[int]] = {"warm": [], "prefix": [], "control": []}
    for index, (arm, core, tail) in enumerate(plan, start=1):
        prompt, cached = ask(client, args.model, instruction, core, tail)
        print(f"  {arm:>8} {index:>5} {prompt:>7} {cached:>7}")
        results[arm].append(cached)

    # The first `warm` call is the one that populates the cache, so it is excluded from the
    # arm's own summary. Every other call in every arm had a chance to hit.
    print()
    for arm, cached in results.items():
        served = cached[1:] if arm == "warm" else cached
        hits = sum(1 for value in served if value > 0)
        best = max(served, default=0)
        print(f"  {arm:>8}: {hits}/{len(served)} cached, most cached in one call {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
