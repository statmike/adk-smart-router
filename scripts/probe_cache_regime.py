#!/usr/bin/env python
"""Preflight P5 (DESIGN.md 3.4) — does a tool declaration change break the cache?

This is build-order task 0: the cheapest possible test of the most expensive
assumption in the project. Everything in DESIGN.md 7 rests on one claim, that
tool declarations sit inside the model's cacheable prefix, so a router that
changes the tool set mid-session destroys the cache and costs more than it saves.

That claim is verified for ADK's *explicit* cache: `_generate_cache_fingerprint`
SHA-256s `config.tools`, and the REST `CachedContent` resource carries `tools[]`
as "Input only. Immutable." (DESIGN.md 18.1). But explicit caching is off by
default — `App.context_cache_config` is `None` and `@experimental` — so most
deployments run on *implicit* caching, where tool participation in the matched
prefix is undocumented. That gap is risk R1.

If tools turn out to be absent from the implicit prefix, DESIGN.md 3.5 says the
design changes shape: DESIGN.md 7's session locking becomes conditional on
`context_cache_config`, and per-turn repruning is allowed when it is absent,
where the uncached saving is 10x larger anyway.

What the probe does, per model:

  * implicit arm — one growing conversation. A warm-up turn and `--control-turns`
    stable turns hold one tool set, so a cache has time to form; ADK creates a
    cache only after two consecutive matching fingerprints, and provider-side
    implicit caching also needs a repeated prefix, so one call per cell would
    prove nothing. The next turn swaps exactly one declaration and changes
    nothing else, the turn after repeats the swapped set, and the last two
    restore the original. Raise `--control-turns` if the control turns read no
    cache: without a cache the swap has nothing to break, and the arm reports
    INCONCLUSIVE for a reason unrelated to tools.
  * explicit arm — build a `CachedContent` carrying the tools, serve two turns
    off it, then try to override `tools` on a request that references it, then
    build a second cache for the second tool set. The override attempt is the
    direct test of "Input only. Immutable."

The two tool sets are the same size and differ by exactly one declaration whose
description is byte-identical, so `prompt_token_count` should barely move. That
column is printed so a reader can confirm the swap did not smuggle in a length
change.

Auth is ADC. Run `gcloud auth application-default login` first. No key files.

The default model ids are `gemini-2.5-flash` and `gemini-3-flash`. If the 3.x id
has moved, the cell fails with a 404 rather than a wrong answer; pass the current
id with `--models` or `SMART_ROUTER_PROBE_MODELS` and re-run. Verified 2026-09-01:
plain `gemini-3-flash` does not exist, and every 3.x
Flash id that does (`gemini-3-flash-preview`, `gemini-3.5-flash`,
`gemini-3.7-flash`) is 404 in a regional endpoint and resolves only under
`--location global`. `gemini-2.5-flash` serves from both.

Usage:

    uv run python scripts/probe_cache_regime.py             # cost estimate only
    uv run python scripts/probe_cache_regime.py --confirm   # actually spend it
    uv run python scripts/probe_cache_regime.py --confirm --location global \
        --models gemini-2.5-flash,gemini-3.7-flash --pad-tokens 32000 --control-turns 6
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from deployment import project_arg  # noqa: E402
from google import genai
from google.genai import errors, types

# --- Defaults, every one of them overridable by environment ------------------

DEFAULT_LOCATION = "us-central1"
DEFAULT_MODELS = "gemini-2.5-flash,gemini-3-flash"
DEFAULT_PAD_TOKENS = 6000  # clears the 4,096-token Gemini 3 explicit-cache floor
DEFAULT_TTL_SECONDS = 1800  # ADK's ContextCacheConfig default

# Published input rates, USD per 1M tokens (DESIGN.md 18.1). Used only to print
# an estimate before spending anything.
INPUT_USD_PER_MTOK: dict[str, float] = {
    "gemini-2.5-flash": 0.30,
    "gemini-3-flash": 0.50,
    "gemini-2.5-pro": 1.25,
    "gemini-3.1-pro": 2.00,
}
FALLBACK_INPUT_USD_PER_MTOK = 0.50

# The eleven declarations both sets share, plus the one that differs. Two tool
# sets, same size, one member swapped: the only variable in the experiment.
SHARED_TOOL_NAMES: tuple[str, ...] = (
    "get_order",
    "get_invoice",
    "list_charges",
    "get_account",
    "search_policy_docs",
    "get_exchange_rate",
    "list_subscriptions",
    "get_ticket",
    "list_addresses",
    "get_payment_method",
    "get_credit_note",
)
TOOL_ONLY_IN_SET_A = "get_shipment"
TOOL_ONLY_IN_SET_B = "get_warranty"

# Byte-identical across every declaration, so swapping a member moves the token
# count by roughly the difference between two short names.
_DECLARATION_DESCRIPTION = (
    "Look up one record for the current customer and return it as JSON. "
    "Read-only. Safe to call more than once."
)


def env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


# --- The experiment plan ------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One turn of the implicit-cache conversation."""

    index: int
    label: str
    tool_set: str  # "A" or "B"
    user_text: str


DEFAULT_CONTROL_TURNS = 2


def build_implicit_plan(control_turns: int = DEFAULT_CONTROL_TURNS) -> tuple[Turn, ...]:
    """One warm-up, N stable control turns, the swap, then hold and restore.

    `control_turns` is a parameter and not a constant because provider-side implicit caching
    warms up on its own schedule. If no control turn reads the cache there is nothing for the
    swap to break, and the whole arm reports INCONCLUSIVE for a reason that has nothing to do
    with tools. Lengthening the stable phase is the only honest way to fix that: shortening it,
    or reading a verdict off the restore turns, would be reading the answer off a cache whose
    contents were never established.
    """
    turns: list[Turn] = [
        Turn(1, "warm", "A", "In one short sentence, what is this policy document about?")
    ]
    for n in range(control_turns):
        turns.append(
            Turn(
                len(turns) + 1,
                "control",
                "A",
                f"In one short sentence, what does clause {(n * 2) + 2:04d} require?",
            )
        )
    tail = (
        ("SWAP", "B", 30),
        ("swap-hold", "B", 32),
        ("restore", "A", 34),
        ("restore-hold", "A", 36),
    )
    for label, which, clause in tail:
        turns.append(
            Turn(
                len(turns) + 1,
                label,
                which,
                f"In one short sentence, what does clause {clause:04d} require?",
            )
        )
    return tuple(turns)


IMPLICIT_PLAN: tuple[Turn, ...] = build_implicit_plan()

SYSTEM_INSTRUCTION = (
    "You are a support assistant for a fictional retailer. Answer in one short "
    "sentence. Do not call a tool unless the user explicitly asks you to act."
)


def declaration(name: str) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=name,
        description=_DECLARATION_DESCRIPTION,
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


def tool_set(which: str) -> list[types.Tool]:
    """Set A or set B. Same size, sorted by name, one member different."""
    extra = TOOL_ONLY_IN_SET_A if which == "A" else TOOL_ONLY_IN_SET_B
    names = sorted([*SHARED_TOOL_NAMES, extra])
    return [types.Tool(function_declarations=[declaration(name) for name in names])]


def padding_text(approx_tokens: int) -> str:
    """Deterministic filler long enough to clear the minimum cacheable prefix.

    Pruning below the model floor turns caching off entirely (DESIGN.md 7.3),
    which would make every row of this probe read as "no cache" for the wrong
    reason. Roughly four characters per token.
    """
    target_chars = approx_tokens * 4
    lines: list[str] = ["RETAILER SUPPORT POLICY HANDBOOK (synthetic, for cache probing only)."]
    size = len(lines[0])
    clause = 0
    while size < target_chars:
        line = (
            f"Clause {clause:04d}: support agents must confirm the customer identity before "
            f"disclosing order, payment or shipment detail, and must record the confirmation "
            f"method in the ticket within one business day of the interaction."
        )
        lines.append(line)
        size += len(line) + 1
        clause += 1
    return "\n".join(lines)


# --- Observations -------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    model: str
    cache_mode: str
    turn: int
    label: str
    tool_set: str
    tool_count: int
    prompt_tokens: int
    cached_tokens: int
    verdict: str


def _usage(response: types.GenerateContentResponse) -> tuple[int, int]:
    usage = response.usage_metadata
    if usage is None:
        return 0, 0
    return int(usage.prompt_token_count or 0), int(usage.cached_content_token_count or 0)


def _advance(contents: list[types.Content], response: types.GenerateContentResponse) -> None:
    """Append the model's reply, and a stub result if it decided to call a tool.

    The conversation has to stay well-formed across seven turns. A dangling
    function call would make the next request invalid and the cache reading
    meaningless.
    """
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or candidate.content is None:
        contents.append(types.Content(role="model", parts=[types.Part(text="(no answer)")]))
        return
    contents.append(candidate.content)
    calls = [part.function_call for part in (candidate.content.parts or []) if part.function_call]
    if calls:
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=call.name or "unknown", response={"result": "ok"}
                    )
                    for call in calls
                ],
            )
        )


# --- Implicit arm -------------------------------------------------------------


def run_implicit(
    client: genai.Client,
    model: str,
    pad_tokens: int,
    plan: tuple[Turn, ...] = IMPLICIT_PLAN,
) -> list[Observation]:
    """One growing conversation, swapping the tool set once the control turns are done."""
    contents: list[types.Content] = []
    observations: list[Observation] = []
    padding = padding_text(pad_tokens)
    baseline = 0

    for turn in plan:
        text = f"{padding}\n\n{turn.user_text}" if turn.index == 1 else turn.user_text
        contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
        tools = tool_set(turn.tool_set)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=list(tools),
            temperature=0.0,
            max_output_tokens=64,
        )
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as exc:
            observations.append(
                Observation(
                    model,
                    "implicit",
                    turn.index,
                    turn.label,
                    turn.tool_set,
                    12,
                    0,
                    0,
                    f"API error: {type(exc).__name__} {exc.code}",
                )
            )
            return observations

        prompt_tokens, cached_tokens = _usage(response)
        _advance(contents, response)

        if turn.label == "control" and cached_tokens > baseline:
            baseline = cached_tokens
        observations.append(
            Observation(
                model,
                "implicit",
                turn.index,
                turn.label,
                turn.tool_set,
                len(tools[0].function_declarations or []),
                prompt_tokens,
                cached_tokens,
                _implicit_verdict(turn, cached_tokens, baseline),
            )
        )
    return observations


def _implicit_verdict(turn: Turn, cached: int, baseline: int) -> str:
    if turn.index == 1:
        return "warm-up, no cache expected yet"
    if turn.label in ("control", "restore-hold"):
        if cached > 0:
            return "cache is reading, so the control arm is valid"
        return "no cache read, so this cell cannot answer the question"
    if turn.label == "SWAP":
        if baseline == 0:
            return "no baseline cache to break, inconclusive"
        if cached == 0:
            return "cache LOST on a tool-only change, so tools ARE in the prefix"
        if cached >= int(baseline * 0.9):
            return "cache HELD across a tool-only change, so tools are NOT in the prefix"
        return f"cache partly lost ({cached} of >={baseline}), report verbatim"
    if turn.label == "swap-hold":
        return "does the swapped set form its own cache" if cached > 0 else "swapped set uncached"
    return "original set restored"


# --- Explicit arm -------------------------------------------------------------


def run_explicit(
    client: genai.Client, model: str, pad_tokens: int, ttl_seconds: int
) -> list[Observation]:
    """Build a CachedContent per tool set and try to override `tools` on a request.

    This arm answers a different question from the implicit arm. Tool membership
    is frozen at cache-creation time, so a router that changes the set mid-session
    forces a delete plus a full re-ingest. The override attempt below is the
    cheapest direct evidence for that.
    """
    observations: list[Observation] = []
    padding = padding_text(pad_tokens)
    prefix = [types.Content(role="user", parts=[types.Part(text=padding)])]
    created: list[str] = []

    def observe(turn: int, label: str, which: str, prompt: int, cached: int, verdict: str) -> None:
        observations.append(
            Observation(model, "explicit", turn, label, which, 12, prompt, cached, verdict)
        )

    try:
        cache_a = _create_cache(client, model, prefix, "A", ttl_seconds)
        created.append(cache_a.name or "")
        tokens_a = _cache_tokens(cache_a)
        observe(1, "create-A", "A", tokens_a, 0, f"cache built over {tokens_a} tokens incl. tools")

        for index, question in enumerate(("Summarise clause 0002.", "Summarise clause 0003."), 2):
            prompt, cached = _serve(client, model, cache_a.name or "", question)
            observe(
                index,
                "serve-A",
                "A",
                prompt,
                cached,
                "served from the explicit cache" if cached else "explicit cache did not read",
            )

        observe(4, "OVERRIDE", "B", 0, 0, _override_verdict(client, model, cache_a.name or ""))

        cache_b = _create_cache(client, model, prefix, "B", ttl_seconds)
        created.append(cache_b.name or "")
        tokens_b = _cache_tokens(cache_b)
        observe(
            5,
            "create-B",
            "B",
            tokens_b,
            0,
            f"a one-declaration change forced a full {tokens_b}-token re-ingest",
        )
        prompt, cached = _serve(client, model, cache_b.name or "", "Summarise clause 0004.")
        observe(
            6,
            "serve-B",
            "B",
            prompt,
            cached,
            "the new set serves, from a new cache" if cached else "new cache did not read",
        )
    except errors.APIError as exc:
        observe(0, "error", "-", 0, 0, f"API error: {type(exc).__name__} {exc.code} {exc.message}")
    finally:
        for name in created:
            _delete_cache(client, name)

    return observations


def _create_cache(
    client: genai.Client,
    model: str,
    prefix: Sequence[types.Content],
    which: str,
    ttl_seconds: int,
) -> types.CachedContent:
    return client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            display_name=f"p5-probe-{which.lower()}",
            system_instruction=SYSTEM_INSTRUCTION,
            contents=list(prefix),
            tools=tool_set(which),
            ttl=f"{ttl_seconds}s",
        ),
    )


def _cache_tokens(cache: types.CachedContent) -> int:
    usage = cache.usage_metadata
    return int(usage.total_token_count or 0) if usage else 0


def _serve(client: genai.Client, model: str, cache_name: str, question: str) -> tuple[int, int]:
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part(text=question)])],
        config=types.GenerateContentConfig(
            cached_content=cache_name, temperature=0.0, max_output_tokens=64
        ),
    )
    return _usage(response)


def _override_verdict(client: genai.Client, model: str, cache_name: str) -> str:
    """Reference cache A but ask for tool set B. Record whatever happens."""
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(role="user", parts=[types.Part(text="Summarise clause 0006.")])
            ],
            config=types.GenerateContentConfig(
                cached_content=cache_name,
                tools=list(tool_set("B")),
                temperature=0.0,
                max_output_tokens=64,
            ),
        )
    except errors.APIError as exc:
        return f"REFUSED ({type(exc).__name__} {exc.code}), tools are immutable once cached"
    _, cached = _usage(response)
    served = "served from cache" if cached else "did not read the cache"
    return f"ACCEPTED and {served}, so check whether set B was honoured or silently dropped"


def _delete_cache(client: genai.Client, name: str) -> None:
    if not name:
        return
    try:
        client.caches.delete(name=name)
    except errors.APIError as exc:
        print(f"warning: could not delete {name}: {exc}", file=sys.stderr)


# --- Cost estimate ------------------------------------------------------------


def estimate_usd(
    models: Sequence[str], pad_tokens: int, turns: int = len(IMPLICIT_PLAN)
) -> tuple[float, list[str]]:
    """Rough, deliberately pessimistic. Printed before anything is spent."""
    lines: list[str] = []
    total = 0.0
    overhead = 3000  # system instruction, twelve declarations, accumulated history
    for model in models:
        rate = INPUT_USD_PER_MTOK.get(model, FALLBACK_INPUT_USD_PER_MTOK)
        implicit_tokens = turns * (pad_tokens + overhead)
        explicit_tokens = 2 * (pad_tokens + overhead) + 3 * (pad_tokens + overhead)
        cost = (implicit_tokens + explicit_tokens) * rate / 1_000_000
        total += cost
        lines.append(
            f"  {model:<20} {rate:.2f}/1M in   "
            f"implicit ~{implicit_tokens:,} tok, explicit ~{explicit_tokens:,} tok   "
            f"${cost:.4f}"
        )
    lines.append(f"  {'storage':<20} two short-lived caches per model, under a cent")
    return total, lines


# --- Reporting ----------------------------------------------------------------

_HEADERS = ("model", "cache", "turn", "phase", "set", "tools", "prompt tok", "cached tok")


def print_table(observations: Sequence[Observation]) -> None:
    widths = [20, 8, 4, 12, 3, 5, 10, 10]
    header = "  ".join(h.ljust(w) for h, w in zip(_HEADERS, widths, strict=True))
    print(header)
    print("-" * len(header))
    for obs in observations:
        cells = (
            obs.model,
            obs.cache_mode,
            str(obs.turn),
            obs.label,
            obs.tool_set,
            str(obs.tool_count),
            f"{obs.prompt_tokens:,}",
            f"{obs.cached_tokens:,}",
        )
        print("  ".join(str(c).ljust(w) for c, w in zip(cells, widths, strict=True)))
        print(f"    -> {obs.verdict}")


def implicit_conclusion(model: str, observations: Sequence[Observation]) -> str:
    """One of DESIGN.md 3.5's outcomes, or an honest refusal to conclude."""
    rows = [o for o in observations if o.model == model and o.cache_mode == "implicit"]
    controls = [o for o in rows if o.label == "control"]
    swaps = [o for o in rows if o.label == "SWAP"]
    if not controls or not swaps:
        return "INCONCLUSIVE — the arm did not complete."
    baseline = max((o.cached_tokens for o in controls), default=0)
    if baseline == 0:
        return (
            "INCONCLUSIVE — no implicit cache read on the stable control turns, so there was "
            "nothing for the tool swap to break. Nothing can be concluded about R1 from this "
            "model. Re-run with a larger --pad-tokens before believing it."
        )
    swapped = swaps[0].cached_tokens
    if swapped == 0:
        return (
            f"TOOLS PARTICIPATE — the control turns cached {baseline:,} tokens and the "
            "tool-only swap cached 0. Changing tool membership destroys the implicit cache."
        )
    if swapped >= int(baseline * 0.9):
        return (
            f"TOOLS ABSENT — the control turns cached {baseline:,} tokens and the tool-only "
            f"swap still cached {swapped:,}. The implicit prefix survived a membership change."
        )
    return (
        f"PARTIAL — control cached {baseline:,}, swap cached {swapped:,}. Neither outcome "
        "cleanly. Report the raw numbers and decide by hand."
    )


def print_conclusion(models: Sequence[str], observations: Sequence[Observation]) -> None:
    print()
    print("=" * 100)
    print("CONCLUSION — Preflight P5, DESIGN.md 3.4 / 3.5, risk R1")
    print("=" * 100)
    verdicts = {model: implicit_conclusion(model, observations) for model in models}
    for model, verdict in verdicts.items():
        print(f"\n  {model}\n    {verdict}")

    print("\n  Explicit arm, for contrast:")
    for row in observations:
        if row.cache_mode == "explicit" and row.label in ("OVERRIDE", "create-B"):
            print(f"    {row.model:<20} {row.label:<10} {row.verdict}")

    absent = [m for m, v in verdicts.items() if v.startswith("TOOLS ABSENT")]
    present = [m for m, v in verdicts.items() if v.startswith("TOOLS PARTICIPATE")]
    print("\n  Which DESIGN.md 3.5 outcome was observed:")
    if absent:
        print(
            "\n    OUTCOME: tools are absent from the implicit-cache prefix on "
            f"{', '.join(absent)}.\n"
            "    DESIGN.md 3.5's P5 row fires. Session-stability is only required when\n"
            "    explicit caching is on. Make DESIGN.md 7's locking conditional on\n"
            "    `context_cache_config`, and allow per-turn repruning when it is absent,\n"
            "    where the uncached saving is 10x larger anyway. R1 resolves to the branch\n"
            "    that changes the design."
        )
    elif present:
        print(
            "\n    OUTCOME: tools participate in the implicit-cache prefix on "
            f"{', '.join(present)}.\n"
            "    DESIGN.md 7 stands as written. Session-stable locking is required in both\n"
            "    cache regimes, not only under `context_cache_config`. R1 resolves to the\n"
            "    branch that leaves the design alone."
        )
    else:
        print(
            "\n    OUTCOME: none. Neither branch of DESIGN.md 3.5 was observed, so R1 is\n"
            "    still open and no DESIGN.md 7 code should be written yet."
        )
    print()


# --- Entry point --------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight P5: do tool declarations sit inside the cacheable prefix?"
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Actually call the API. Without it, estimate only."
    )
    parser.add_argument("--project", **project_arg())
    parser.add_argument("--location", default=env("SMART_ROUTER_VERTEX_LOCATION", DEFAULT_LOCATION))
    parser.add_argument(
        "--models",
        default=env("SMART_ROUTER_PROBE_MODELS", DEFAULT_MODELS),
        help="Comma-separated model ids.",
    )
    parser.add_argument(
        "--pad-tokens",
        type=int,
        default=int(env("SMART_ROUTER_PROBE_PAD_TOKENS", str(DEFAULT_PAD_TOKENS))),
        help="Filler tokens, to clear the minimum cacheable prefix (DESIGN.md 7.3).",
    )
    parser.add_argument(
        "--ttl-seconds",
        type=int,
        default=int(env("SMART_ROUTER_PROBE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS))),
    )
    parser.add_argument(
        "--control-turns",
        type=int,
        default=int(env("SMART_ROUTER_PROBE_CONTROL_TURNS", str(DEFAULT_CONTROL_TURNS))),
        help="Stable turns before the swap. Raise this if the control turns read no cache.",
    )
    parser.add_argument("--skip-explicit", action="store_true", help="Run the implicit arm only.")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    plan = build_implicit_plan(args.control_turns)
    total, lines = estimate_usd(models, args.pad_tokens, len(plan))

    print("Preflight P5 — cache-regime probe (DESIGN.md 3.4, build-order task 0)")
    print()
    settings = (
        ("project", args.project, "SMART_ROUTER_PROJECT"),
        ("location", args.location, "SMART_ROUTER_VERTEX_LOCATION"),
        ("models", ", ".join(models), "SMART_ROUTER_PROBE_MODELS"),
        ("pad tokens", f"{args.pad_tokens:,}", "SMART_ROUTER_PROBE_PAD_TOKENS"),
        (
            "turns",
            f"{len(plan)} implicit ({args.control_turns} control)",
            "SMART_ROUTER_PROBE_CONTROL_TURNS",
        ),
        ("cache ttl", f"{args.ttl_seconds}s", "SMART_ROUTER_PROBE_TTL_SECONDS"),
        ("auth", "Application Default Credentials, Vertex backend", "-"),
    )
    for label, value, var in settings:
        print(f"  {label:<12} {value:<44} ({var})")
    print()
    print("Estimated cost:")
    for line in lines:
        print(line)
    print(f"  {'TOTAL':<20} ~${total:.3f}")
    print()

    if not args.confirm:
        print("Nothing was called. Re-run with --confirm to spend the amount above.")
        return 2

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    observations: list[Observation] = []
    for model in models:
        print(f"running {model} ...", file=sys.stderr)
        observations.extend(run_implicit(client, model, args.pad_tokens, plan))
        if not args.skip_explicit:
            observations.extend(run_explicit(client, model, args.pad_tokens, args.ttl_seconds))

    print()
    print_table(observations)
    print_conclusion(models, observations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
