"""Does the steering directive cost the session its cache from turn two onward?

The A/B in `DESIGN.md` 18.18 put the routed arm 24% over the control, and splitting the
result by turn put almost all of the gap in turns one and two. Turn one is a cold-start
story and is understood. Turn two is not: it caches against turn one of its own session, so
it should not care how many distinct menus the router publishes, and yet the routed arm hit
0.694 against the control's 0.994.

One thing differs between the two arms at that point. On the turn the router makes a
decision it inserts a one-line advisory through ADK's `_insert_transient_user_content`, and
that helper places the content *before* the latest user message rather than after it. So a
routed turn one sends

    [system] [tools] [directive] [user 1]

and turn two, where the decision is reused and no directive is inserted, sends

    [system] [tools] [user 1] [model 1] [user 2]

The two agree on the system instruction and the declarations and then diverge, at the third
element, for the rest of the session. If implicit caching matches a literal prefix, turn two
can reuse the system instruction and the tool list and nothing else, and every later turn
inherits the same break.

Three arms, each a genuine two-turn exchange, each with its own nonce in the system
instruction so the arms cannot warm one another:

    plain       turn one sends no directive at all. The control.
    before      the directive goes ahead of the user message. Today's behaviour.
    after       the directive goes behind the user message. The cheap repair, if it works.

What is being read is the second call's `cached_content_token_count`. `plain` should cache
turn one whole. If `before` caches roughly the system instruction and the declarations and
stops, the directive is the defect. If `after` matches `plain`, moving it is enough; if
`after` only recovers part of the gap, the directive has to go.

Cheap: twelve short calls of about 2.5k tokens each.
"""

from __future__ import annotations

import argparse
import os
import sys

from google import genai
from google.genai import types

DECLARATIONS = 11

_DESCRIPTION = (
    "Look up one record for the current customer and return it as JSON. "
    "Read-only. Safe to call more than once."
)

DIRECTIVE = (
    "Based on this request, these tools are most likely relevant: tool_0, tool_1, tool_2. "
    "Use another available tool if one fits better."
)

FIRST = "My last order has not arrived. What can you tell me about it?"
SECOND = "Thanks. Can you also tell me whether I can still cancel it?"


def system_instruction(nonce: str) -> str:
    """The reference agent's own instruction, plus a nonce that isolates one arm's pair.

    Without the nonce the second pair of calls would hit the first pair's cache and every
    arm after the first would look like it cached perfectly.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from reference_agent.instruction import INSTRUCTION

    return f"{INSTRUCTION}\n\nSession reference: {nonce}."


def tools() -> list[types.Tool]:
    """A fixed declaration list, the size the routed arm actually sent."""
    declarations = [
        types.FunctionDeclaration(
            name=f"tool_{index}",
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
        for index in range(DECLARATIONS)
    ]
    return [types.Tool(function_declarations=declarations)]


def user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def ask(
    client: genai.Client, model: str, instruction: str, contents: list[types.Content]
) -> tuple[int, int, types.Content]:
    """One call. Returns (prompt tokens, cached tokens, the model's reply as history)."""
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            tools=tools(),
            temperature=0.0,
        ),
    )
    usage = response.usage_metadata
    reply = response.candidates[0].content
    return int(usage.prompt_token_count or 0), int(usage.cached_content_token_count or 0), reply


def turn_one(arm: str) -> list[types.Content]:
    """What the first call sends, which is the only thing the three arms disagree about."""
    if arm == "before":
        return [user(DIRECTIVE), user(FIRST)]
    if arm == "after":
        return [user(FIRST), user(DIRECTIVE)]
    return [user(FIRST)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument(
        "--location", default=os.environ.get("SMART_ROUTER_VERTEX_LOCATION", "global")
    )
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    print(f"{args.model}, {DECLARATIONS} declarations, two-turn exchanges\n")
    print(f"  {'arm':>8} {'pair':>5} {'t1 prompt':>10} {'t2 prompt':>10} {'t2 cached':>10}")

    served: dict[str, list[float]] = {"plain": [], "before": [], "after": []}
    for arm in ("plain", "before", "after"):
        for pair in range(1, args.repeats + 1):
            instruction = system_instruction(f"{arm}-{pair}")
            first, _, reply = ask(client, args.model, instruction, turn_one(arm))
            history = [user(FIRST), reply, user(SECOND)]
            second, cached, _ = ask(client, args.model, instruction, history)
            print(f"  {arm:>8} {pair:>5} {first:>10} {second:>10} {cached:>10}")
            served[arm].append(cached / second if second else 0.0)

    print()
    for arm, rates in served.items():
        mean = sum(rates) / len(rates) if rates else 0.0
        print(f"  {arm:>8}: turn two cached {mean:.1%} of its prompt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
