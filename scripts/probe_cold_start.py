#!/usr/bin/env python
"""What a prune actually costs an agent whose shared prefix falls under the cache floor.

`qualify` caps such an agent at `thin` rather than refusing it, and the reason it gives is an
argument, not a measurement: "turns two onward still cache and still save; turn one never
will" (DESIGN.md 19.31). Nobody has run an agent in that state. The evidence behind the claim
is one arm of the residency probe, which sent single fresh calls and so could only ever see
turn one.

An agent has two cacheable prefixes and they are different sizes. Within a session, turn N+1's
prompt begins with turn N's whole prompt, history included, so that is what has to clear the
floor. Across sessions, all two sessions share is the instruction and the declarations. Pruning
takes tokens out of the second one only, which is why a prune can leave the first comfortable
and the second short.

Follow that through a session and the claim looks wrong by one turn. If the pruned prefix is
1,900 tokens against a 2,048 floor, turn one cannot cache, and neither can turn two, because
turn two's cacheable prefix *is* turn one's prompt and turn one's prompt is the short one. Only
from turn three, once history has carried the whole prompt over the floor, does caching start.
That is the thing to measure.

The arms
--------
Three, run at once, each with a private head and a private declaration namespace so no two can
serve each other. All three are the same agent under three treatments.

    full        1,600-token head + 6,000 tokens of declarations = 7,600. What an unrouted
                agent sends. Every prefix it ever caches is over the floor.
    menu        the same head + 300 tokens of declarations = 1,900. What the router sends
                after pruning to a menu, and 133 tokens short of the floor.
    padded      a 2,000-token head + the same 300 tokens of declarations = 2,315. The remedy
                `qualify` recommends: lengthen the instruction until the pruned prefix clears
                the floor by itself. It buys back the floor for 400 tokens of filler, having
                just saved 5,670 tokens of declarations.

Each arm runs sessions of four turns. The first user message is short, the way a real opening
is, and every turn after it adds about 800 tokens. That detail is the whole experiment: if the
opening is long, turn one's prompt clears the floor by itself and turn two has an over-floor
prefix to match against, so the question never gets asked. A first run of this probe made
exactly that mistake. With a 20-token opening the menu arm's turn one lands at about 1,935
tokens, 113 short of the floor, and turn two has nothing over the floor to match.

Sessions start ten seconds apart. That is faster than real traffic on purpose. 19.30 measured
residency as a short TTL — near certain at five seconds, a coin flip at sixty, gone by ten
minutes — so a realistic cadence would leave every cold turn one ambiguous between "the floor
blocked it" and "the entry expired". Running close together holds residency near certain and
leaves the floor as the only thing that can differ between the arms.

The predictions, written before the run
---------------------------------------
    turn 1   full and padded cache from the second session onward; menu never does.
    turn 2   full and padded cache; menu does not, because its prefix is turn one's prompt.
    turn 3+  all three cache.
    overall  menu still bills far less than full, because 5,700 fewer tokens at full rate
             beats four cached turns at a tenth rate. Padded bills least of the three.

If turn two caches on the menu arm the second floor is not doing what 19.31 says it does. If
padded does not recover turn one *while the full arm does*, the one remedy `qualify` offers
does not work end to end, and 19.33's finding that filler caches like real content did not
survive contact with a session. The full arm is the control for that second question: it is
over the floor by 5,500 tokens, so if it does not open warm either, the schedule and not the
floor is what is being measured, and the remedy cannot be judged at all.

Cost and credentials
--------------------
About 745,000 input tokens, worst case $0.22, and three minutes of wall clock. Authentication
is Application Default Credentials. No key files.

    uv run python scripts/probe_cold_start.py              # plan and price it, call nothing
    uv run python scripts/probe_cold_start.py --confirm    # spend the amount it printed
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from google.genai import errors, types

# The package is not installed into the venv; pytest reaches it through `pythonpath` in
# pyproject.toml and scripts have to do the same by hand.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from deployment import project_arg  # noqa: E402
from probe_residency import (  # noqa: E402
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    RETRY_ATTEMPTS,
    RETRY_BACKOFF_S,
    TOKENS_PER_DECLARATION,
    declaration,
    head_body,
    input_usd_per_mtok,
)

from adk_smart_router.config import load_config, min_cache_tokens  # noqa: E402

# The three sizes the arms are built from, in prefix tokens. `MENU_PREFIX` is deliberately a
# little under the 2,048-token floor rather than far under it: the interesting agent is the one
# a prune just barely breaks, because that is the one where the remedy is cheap and the one
# `qualify` has to get right. Far under the floor is the `uncached` band and a different story.
FULL_DECLARATION_TOKENS = 6_000
MENU_DECLARATION_TOKENS = 300
HEAD_TOKENS = 1_600
PADDED_HEAD_TOKENS = 2_000

SESSIONS = 12
TURNS = 4
SESSION_GAP_S = 10.0

# The opening has to be short or the experiment does not happen: a long first message carries
# the menu arm's turn one over the floor by itself, and then turn two has an over-floor prefix
# to match and caches for a reason that has nothing to do with pruning. Twenty tokens is about
# what a person types to start a conversation.
FIRST_TURN_TOKENS = 20
TURN_TOKENS = 800  # user text added per turn after the first

MAX_OUTPUT_TOKENS = 8
CHARS_PER_TOKEN = 5.0  # the filler text below, matched to probe_residency's head

_TURN_TEXT = (
    "Turn {turn} of session {session}. Here is the background you asked for. "
    "The account has been active since 2019 and the last four interactions were about "
    "delivery windows, none of which were escalated. Reply with the single word ok. "
)

# Unique per session on purpose, so no two sessions share their opening. That makes the shared
# cross-session prefix exactly the instruction plus the declarations, which is the thing whose
# size the floor is being tested against.
_FIRST_TURN_TEXT = "Hi, question {session} about my delivery. Reply with the single word ok."


# --- The arms -------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One treatment of the same agent, run as `SESSIONS` sessions of `TURNS` turns."""

    name: str
    namespace: str  # two characters, so every declaration name is the same width
    head_tokens: int
    declaration_tokens: int
    why: str

    @property
    def declarations(self) -> int:
        return max(1, round(self.declaration_tokens / TOKENS_PER_DECLARATION))

    @property
    def prefix_tokens(self) -> int:
        """The cross-session prefix: instruction and declarations, no history."""
        return self.head_tokens + self.declarations * round(TOKENS_PER_DECLARATION)

    @property
    def calls(self) -> int:
        return SESSIONS * TURNS

    @property
    def tokens(self) -> int:
        """Worst case, nothing cached: every turn pays its whole prompt."""
        grown = [FIRST_TURN_TOKENS + turn * TURN_TOKENS for turn in range(TURNS)]
        return SESSIONS * sum(self.prefix_tokens + added for added in grown)


def arms() -> list[Arm]:
    return [
        Arm(
            "full",
            "fu",
            HEAD_TOKENS,
            FULL_DECLARATION_TOKENS,
            "unrouted: every prefix it caches is over the floor",
        ),
        Arm(
            "menu",
            "me",
            HEAD_TOKENS,
            MENU_DECLARATION_TOKENS,
            "pruned: the shared prefix falls under the floor",
        ),
        Arm(
            "padded",
            "pa",
            PADDED_HEAD_TOKENS,
            MENU_DECLARATION_TOKENS,
            "pruned, then the head lengthened until the prefix clears the floor",
        ),
    ]


def head(marker: str, tokens: int) -> str:
    """A synthetic system instruction of about `tokens`, private to `marker`.

    The marker is the first line, so two heads with different markers diverge at byte zero and
    no part of one can serve the other. That is what makes the three arms independent.
    """
    return f"COLD START ARM {marker}. Answer in one word. Do not call a tool.\n{head_body(tokens)}"


def turn_text(session: int, turn: int) -> str:
    """A user message, unique so nothing is a literal repeat. Turn one is short, the rest long."""
    if turn == 1:
        return _FIRST_TURN_TEXT.format(session=session)
    body = _TURN_TEXT.format(turn=turn, session=session)
    return (body * (int(TURN_TOKENS * CHARS_PER_TOKEN) // len(body) + 1))[
        : int(TURN_TOKENS * CHARS_PER_TOKEN)
    ]


# --- Running it -----------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One turn, and what the usage metadata said about it."""

    arm: str
    session: int
    turn: int  # from 1
    prefix_tokens: int  # the arm's target cross-session prefix
    prompt_tokens: int
    cached_tokens: int
    error: str = ""

    @property
    def share(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass
class Sink:
    """Where every arm's calls land. Threads write to it, so it holds a lock."""

    calls: list[Call] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, call: Call) -> None:
        with self.lock:
            self.calls.append(call)


def ask(
    client: genai.Client,
    model: str,
    instruction: str,
    tool_list: list[types.Tool],
    contents: list[types.Content],
) -> tuple[int, int, str, str]:
    """One request in a running conversation. Returns (prompt, cached, reply text, error).

    The reply is returned because it has to go back into `contents` for the next turn. That is
    the only difference from `probe_residency.ask`, and it is the difference between a probe
    that measures a fresh prefix and one that measures a session.
    """
    config = types.GenerateContentConfig(
        system_instruction=instruction,
        tools=tool_list,
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as exc:
            if exc.code not in (429, 500, 503) or attempt == RETRY_ATTEMPTS - 1:
                return 0, 0, "", f"{type(exc).__name__} {exc.code}"
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue
        usage = response.usage_metadata
        if usage is None:
            return 0, 0, "", "no usage metadata"
        return (
            int(usage.prompt_token_count or 0),
            int(usage.cached_content_token_count or 0),
            response.text or "ok",
            "",
        )
    return 0, 0, "", "retries exhausted"


def run_arm(client: genai.Client, model: str, arm: Arm, run_id: str, sink: Sink) -> None:
    """Run this arm's sessions one after another, `SESSION_GAP_S` apart.

    Sequential on purpose. Sessions of one agent arrive one after another in real traffic, and
    whether session N+1 opens warm is a question about how long ago session N ran. Running them
    at once would answer a different question.
    """
    instruction = head(f"{run_id} {arm.name}", arm.head_tokens)
    tool_list = [
        types.Tool(
            function_declarations=[declaration(arm.namespace, i) for i in range(arm.declarations)]
        )
    ]

    for session in range(1, SESSIONS + 1):
        started = time.monotonic()
        contents: list[types.Content] = []
        for turn in range(1, TURNS + 1):
            contents.append(
                types.Content(role="user", parts=[types.Part(text=turn_text(session, turn))])
            )
            prompt, cached, reply, error = ask(client, model, instruction, tool_list, contents)
            contents.append(types.Content(role="model", parts=[types.Part(text=reply)]))
            sink.add(Call(arm.name, session, turn, arm.prefix_tokens, prompt, cached, error))
        if session < SESSIONS:
            time.sleep(max(0.0, SESSION_GAP_S - (time.monotonic() - started)))


def run(client: genai.Client, model: str, run_id: str) -> list[Call]:
    sink = Sink()
    chosen = arms()
    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        for arm in chosen:
            pool.submit(run_arm, client, model, arm, run_id, sink)
    return sink.calls


# --- Reading the answer ----------------------------------------------------------------


def by_turn(calls: Sequence[Call], arm: str, turn: int, skip_first_session: bool = True) -> float:
    """Mean cached share of one arm's turn-`turn` calls, or 0.0 if it has none.

    The first session is dropped by default because nothing preceded it, so its turn one is
    cold for a reason that has nothing to do with the floor. Every other session's turn one had
    a chance.
    """
    rows = [
        c
        for c in calls
        if c.arm == arm
        and c.turn == turn
        and not c.error
        and (c.session > 1 or not skip_first_session)
    ]
    return statistics.mean(c.share for c in rows) if rows else 0.0


def mean_prompt(calls: Sequence[Call], arm: str, turn: int) -> float:
    """Mean prompt size of one arm's turn-`turn` calls, to check the arms came out planned size."""
    rows = [c for c in calls if c.arm == arm and c.turn == turn and not c.error]
    return statistics.mean(c.prompt_tokens for c in rows) if rows else 0.0


def opened_warm(calls: Sequence[Call], arm: str, threshold: float = 0.05) -> tuple[int, int]:
    """How many of this arm's sessions opened with a warm turn one, out of how many could have.

    Session one is excluded from both counts: nothing preceded it, so it is cold for a reason
    that has nothing to do with the floor. This is the measurement the remedy is judged on, and
    it is a count rather than a mean because a turn one either opened warm or it did not.
    """
    rows = [c for c in calls if c.arm == arm and c.turn == 1 and c.session > 1 and not c.error]
    return sum(1 for c in rows if c.share > threshold), len(rows)


def billed(calls: Sequence[Call], arm: str, multiplier: float) -> float:
    """Mean billed-equivalent tokens per turn, cached tokens discounted at `multiplier`."""
    rows = [c for c in calls if c.arm == arm and not c.error]
    if not rows:
        return 0.0
    return statistics.mean(
        (c.prompt_tokens - c.cached_tokens) + multiplier * c.cached_tokens for c in rows
    )


def verdict_second_floor(calls: Sequence[Call], floor: int) -> str:
    """Does the pruned arm lose turn two as well as turn one?

    The first thing this has to establish is that the question was asked at all. The hypothesis
    is about turn two matching against an under-floor stored prefix, so turn one's whole prompt
    has to be under the floor. If the opening ran long it is not, and every later turn caches
    for an ordinary reason that says nothing about pruning.
    """
    menu_one, menu_two, menu_three = (by_turn(calls, "menu", t) for t in (1, 2, 3))
    full_two = by_turn(calls, "full", 2)
    menu_one_prompt = mean_prompt(calls, "menu", 1)
    detail = (
        f"menu t1={menu_one:.3f} t2={menu_two:.3f} t3={menu_three:.3f};"
        f" full t2={full_two:.3f}; menu turn-one prompt {menu_one_prompt:,.0f} tokens"
        f" against a {floor:,}-token floor"
    )

    if menu_one_prompt >= floor:
        return (
            f"UNTESTED — {detail}. Turn one's whole prompt cleared the floor by itself, so turn"
            " two had an over-floor prefix to match against and whatever it did says nothing"
            " about pruning. Shorten the opening and run it again."
        )
    if full_two < 0.05:
        return (
            f"UNTESTED — {detail}. The over-floor control did not cache at turn two either, so"
            " within-session caching was not happening on this run for any arm and the menu"
            " arm's turn two has nothing to be compared against."
        )
    if menu_two < 0.05 and menu_three > 0.05:
        return (
            f"TWO TURNS, NOT ONE — {detail}. Pruning under the cross-session floor costs turn"
            " one and turn two, because turn two's cacheable prefix is turn one's prompt and"
            " that is the short one. Caching starts at turn three, once history has carried the"
            " whole prompt over the floor. DESIGN 19.31 says turns two onward still cache; it is"
            " wrong by a turn, and `qualify`'s message should say so."
        )
    if menu_two > 0.05:
        return (
            f"ONE TURN — {detail}. Turn two caches even though the prefix it matches against is"
            " under the floor, so the floor is applied to the request and not to the stored"
            " prefix. 19.31's message stands as written."
        )
    return f"UNCLEAR — {detail}. Read the per-turn table by hand."


def verdict_remedy(calls: Sequence[Call]) -> str:
    """Does lengthening the head buy turn one back?

    Judged against the full arm rather than against zero. The full arm is over the floor by
    5,500 tokens, so how often *it* opens warm is the ceiling any remedy could reach on this
    schedule. Padded failing to open warm only means something if full managed it.
    """
    warm = {name: opened_warm(calls, name) for name in ("full", "menu", "padded")}
    detail = ", ".join(f"{name} {hit}/{n}" for name, (hit, n) in warm.items())
    detail = f"sessions opening warm: {detail}"
    full_hit, full_n = warm["full"]
    padded_hit, _ = warm["padded"]
    menu_hit, _ = warm["menu"]

    if full_n == 0:
        return f"INCONCLUSIVE — {detail}. No session after the first completed its turn one."
    if full_hit == 0:
        return (
            f"INCONCLUSIVE — {detail}. Not even the over-floor control opened warm, so this"
            " schedule did not produce cross-session residency at all and the remedy cannot be"
            " judged. Shorten the gap between sessions and run it again."
        )
    if menu_hit > 0:
        return (
            f"THE PREMISE IS WRONG — {detail}. The under-floor arm opened warm across sessions,"
            " which it should not be able to do. Check the arm sizes before reading anything"
            " else here."
        )
    if padded_hit >= full_hit:
        return (
            f"THE REMEDY WORKS — {detail}. Four hundred tokens of filler in the system"
            " instruction bought back turn one as often as the unpruned agent gets it, having"
            " just saved 5,700 tokens of declarations. `qualify`'s advice holds end to end, not"
            " just at the prefix level 19.33 measured it at."
        )
    if padded_hit == 0:
        return (
            f"THE REMEDY DOES NOT WORK — {detail}. Clearing the floor with filler never restored"
            f" turn one, on a schedule where the unpruned agent managed it {full_hit} times."
            " Clearing the floor is necessary and not sufficient."
        )
    return (
        f"PARTIAL — {detail}. The remedy restored turn one some of the time, less often than the"
        " unpruned agent gets it. Clearing the floor helps and does not settle it."
    )


def verdict_economics(calls: Sequence[Call], multiplier: float) -> str:
    """After all of that, is the pruned agent cheaper?"""
    bills = {arm.name: billed(calls, arm.name, multiplier) for arm in arms()}
    if not bills["full"]:
        return "INCONCLUSIVE — the full arm did not complete."
    detail = ", ".join(f"{name} {value:,.0f}" for name, value in bills.items())
    saving = 1 - bills["menu"] / bills["full"]
    padded_saving = 1 - bills["padded"] / bills["full"]
    gap = padded_saving - saving
    # The gap is the remedy's price, and it is signed. Positive means the warmth the padding
    # bought back was worth more than the padding cost; negative means it was not, and the
    # cheapest thing to do is prune and accept the cold turns.
    if gap >= 0:
        cost = (
            f" Padding is worth {gap:.1%} on top of that: the warmth it buys back is worth more"
            " than the 400 tokens it costs on every turn."
        )
    else:
        cost = (
            f" Padding costs {-gap:.1%} rather than earning anything: 400 tokens on every turn"
            " is more than the warmth it buys back is worth, so the cheapest agent here is the"
            " pruned one that never pads and accepts the cold turns."
        )
    return (
        f"billed-equivalent tokens per turn: {detail}. Pruning saves {saving:.1%} even after"
        f" losing the cold turns.{cost}"
    )


# --- Reporting --------------------------------------------------------------------------


def print_table(calls: Sequence[Call]) -> None:
    print("\ncached share by turn, first session excluded — it had nothing to open warm against")
    print(
        f"  {'arm':<8} {'prefix':>8} "
        + " ".join(f"{'turn ' + str(t):>9}" for t in range(1, TURNS + 1))
    )
    for arm in arms():
        shares = " ".join(f"{by_turn(calls, arm.name, t):>9.3f}" for t in range(1, TURNS + 1))
        print(f"  {arm.name:<8} {arm.prefix_tokens:>8,} {shares}")

    print("\nprompt tokens by turn — the achieved sizes, against the plan")
    print(
        f"  {'arm':<8} {'prefix':>8} "
        + " ".join(f"{'turn ' + str(t):>9}" for t in range(1, TURNS + 1))
    )
    for arm in arms():
        row = []
        for turn in range(1, TURNS + 1):
            rows = [c for c in calls if c.arm == arm.name and c.turn == turn and not c.error]
            row.append(
                f"{statistics.mean(c.prompt_tokens for c in rows):>9,.0f}" if rows else f"{'-':>9}"
            )
        print(f"  {arm.name:<8} {arm.prefix_tokens:>8,} " + " ".join(row))

    print("\nevery session's turn one — where the cross-session floor shows up")
    print(f"  {'arm':<8} " + " ".join(f"{'s' + str(s):>7}" for s in range(1, SESSIONS + 1)))
    for arm in arms():
        row = []
        for session in range(1, SESSIONS + 1):
            found = [c for c in calls if c.arm == arm.name and c.session == session and c.turn == 1]
            row.append(f"{found[0].share:>7.3f}" if found and not found[0].error else f"{'-':>7}")
        print(f"  {arm.name:<8} " + " ".join(row))


def print_verdicts(calls: Sequence[Call], multiplier: float, floor: int) -> None:
    print()
    print("=" * 100)
    print("CONCLUSION — what a prune under the cross-session floor actually costs")
    print("=" * 100)
    for label, verdict in (
        ("the second floor", verdict_second_floor(calls, floor)),
        ("the remedy", verdict_remedy(calls)),
        ("the economics", verdict_economics(calls, multiplier)),
    ):
        print(f"\n  {label}\n    {verdict}")
    print()


# --- Entry point ------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    cfg = load_config()
    floor = min_cache_tokens(args.model) or 0
    chosen = arms()
    tokens = sum(arm.tokens for arm in chosen)
    usd = tokens / 1e6 * input_usd_per_mtok(args.model)
    minutes = SESSIONS * SESSION_GAP_S / 60.0

    print("Cold-start probe — what pruning under the cross-session cache floor costs\n")
    for label, value in (
        ("project", args.project),
        ("location", args.location),
        ("model", args.model),
        ("cache floor", f"{floor:,} tokens"),
        ("run id", args.run_id),
        ("auth", "Application Default Credentials, Vertex backend"),
    ):
        print(f"  {label:<12} {value}")

    print(f"\nPlan: {len(chosen)} arms x {SESSIONS} sessions x {TURNS} turns\n")
    print(f"  {'arm':<8} {'prefix':>8} {'decls':>7} {'floor':>8} {'turn 1':>8} {'floor':>8}  why")
    for arm in chosen:
        # Both have to be shown. The prefix is what the floor blocks across sessions; turn one's
        # whole prompt is what it blocks within one, and the second only tests the hypothesis if
        # it is under the floor too.
        opening = arm.prefix_tokens + FIRST_TURN_TOKENS
        print(
            f"  {arm.name:<8} {arm.prefix_tokens:>8,} {arm.declarations:>7,}"
            f" {'over' if arm.prefix_tokens >= floor else 'UNDER':>8}"
            f" {opening:>8,} {'over' if opening >= floor else 'UNDER':>8}  {arm.why}"
        )
    print(
        f"\n  {'TOTAL':<8} {tokens:>8,} input tokens, worst case ${usd:.2f}, about"
        f" {minutes:.0f} minutes"
    )
    print(
        "\n  Worst case assumes nothing caches anywhere, which is the outcome the probe exists"
        f"\n  to rule out. Output is negligible at max_output_tokens={MAX_OUTPUT_TOKENS}."
    )

    if not args.confirm:
        print("\nNothing was called. Re-run with --confirm to spend the amount above.")
        return 2

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    print("\nrunning ...", file=sys.stderr)
    observed = run(client, args.model, args.run_id)

    print_table(observed)
    print_verdicts(observed, cfg.cached_token_multiplier, floor)

    failed = [call for call in observed if call.error]
    if failed:
        print(f"  {len(failed)} of {len(observed)} calls failed; verdicts ignore them.\n")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"run_id": args.run_id, "model": args.model, "calls": [vars(c) for c in observed]},
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {len(observed)} calls to {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirm", action="store_true", help="required: this spends real money")
    parser.add_argument("--project", **project_arg())
    parser.add_argument(
        "--location", default=os.environ.get("SMART_ROUTER_VERTEX_LOCATION", DEFAULT_LOCATION)
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--run-id",
        default=dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"),
        help="stamped into every head marker. A fresh id is what stops a re-run from"
        " measuring the caches its own last run left behind.",
    )
    parser.add_argument("--out", default=None, help="write every call to this JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
