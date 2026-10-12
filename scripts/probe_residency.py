#!/usr/bin/env python
"""What actually governs how long a pruned prefix stays in the implicit cache.

The routed arm of `ab-v2-100tools` opened 58.4% cached against a holdout's 88.4%, and that
gap is worth more G1 than the prune itself. The obvious explanation is idle time: menus are
sent less often than one full list, so they go cold. The warehouse says that is wrong, and
says it three ways. Per-menu hit rate has no trend against cumulative sends (24 menus, 16 to
140 sends, no order), none against send rate (an 8x span in sends/minute, no order), and none
against prefix age (a brand-new 16k prefix reached 0.89 within 60 sends, while menus given 140
sends over 56 minutes reached 0.53). Worse, splitting the same run by prefix shows the deficit
is already there at *zero* idle: at a gap under two seconds the full list caches 0.893 and a
menu caches 0.725. Seventeen of the thirty-four points exist before any decay can start, and
no volume, rate or recency model can produce that.

Two variables do order every observation the project has logged, and neither has ever been
varied deliberately:

    size            2.7k-token menus cache ~15%, ~7.9k caches 50%, ~8.1k caches 78%, the
                    16.2k full list caches 89-99%. Monotone across four runs.
    head collision  the 24 menus and the full list all begin with the same ~1,600-token
                    system instruction, and 6 routed turn-ones cached exactly 1,586-1,602
                    tokens, which is that head matching on its own. So the prefixes provably
                    share a cacheable head. Whether they compete for it is untested.

They are confounded in every run, because the small prefixes are also the ones sharing a head
with a busier big prefix. Only a probe can separate them, which is what this is.

Two rules make this a residency probe rather than another A/B read sideways:

    every call is FRESH.  One system instruction, N synthetic declarations, one short unique
                          user message, no conversation history, never a growing chat.
                          Within-session caching already works at 96% and is not the question;
                          what is being measured is whether a prefix survives *between*
                          sessions.
    every arm is PRIVATE. Each arm's declarations are named into its own namespace and its
                          head opens with its own marker line, so no two arms and nothing
                          this project has already sent share a cached prefix. The marker sits
                          at byte zero because implicit caching matches a literal prefix: one
                          differing line at the front makes the entire head private, which is
                          also the one-line fix if block C fires.

    Block A, size.   Five arms at 2,048 / 4,096 / 8,192 / 16,384 / 32,768 prefix tokens, each
                     with a private head, each loaded with 24 sends at one per 2s, then probed
                     twice after each of 5s, 60s and 600s idle.
    Block B, volume  The negative results from the warehouse, put on the record deliberately.
    and rate.        At 4,096 and 16,384 tokens: 4, 24 and 96 sends at one per 2s, plus 24
                     sends at one per 20s. Each probed twice after 60s.
    Block C, head    Two 8,192-token arms whose ~1,600-token head is byte-identical. C1 runs
    collision.       alone. C2 runs the same schedule while a 16,384-token prefix carrying
                     that same head sends 120 requests across the same window.

What each outcome decides:

    2k under 0.2 and 16k/32k over 0.85 at identical load   size governs. Residency needs a
                                                           size term, `MEASURED_CURVE` needs
                                                           refitting per band, and menu width
                                                           becomes a priced tunable that
                                                           `tune_menus.py` can already sweep.
    all five sizes within a few points                     size is dead.
    share rises with sends in block B                      volume governs; the fix is a
                                                           warm-up sender at publish time.
    share flat in sends but falls with cadence             rate governs; the fix is fewer,
                                                           busier menus.
    C2 materially colder than C1                           prefixes sharing a head compete,
                                                           and a per-menu marker at the front
                                                           of the system instruction is a
                                                           one-line fix to try before any
                                                           CachedContent work.
    nothing moves anything                                 implicit residency is not
                                                           controllable from the request side
                                                           and explicit caching is the only
                                                           lever left.

Raw cached-token counts are printed next to the shares, because the shares hide the thing
worth seeing. In the A/B, cached-token values clustered at 0, then 1,586-1,602, then around
3,177-3,997, then 7,168-8,016. The 1,586 cluster is the shared head matching by itself, and
it is invisible in a percentage.

    uv run python scripts/probe_residency.py                 # plan and price it, call nothing
    uv run python scripts/probe_residency.py --confirm       # spend the amount it printed
    uv run python scripts/probe_residency.py --confirm --blocks A --out probe.json

Auth is ADC. Run `gcloud auth application-default login` first. No key files.
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

from cost_model import MODELS  # noqa: E402
from deployment import project_arg  # noqa: E402

DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-2.5-flash"

# The head every arm carries: a synthetic system instruction sized to match the ~1,600-token
# one the reference agent's menus share, because that is the head the A/B showed matching on
# its own and block C is about what happens when several prefixes want it at once.
HEAD_TOKENS = 1_600

# How a request's prompt tokens are built, all three measured on 2026-09-05 against this
# script's own text on gemini-2.5-flash. Sending 32, 96, 256 and 400 declarations billed
# 45.34, 45.11, 45.04 and 45.03 tokens each: linear to three figures, but only because the
# declaration names are fixed width. With a long namespace and a growing index the cost per
# declaration drifted from 47 to 60 and an arm asking for 16,384 tokens sent 21,752.
#
# `count_tokens` is deliberately not used for this. It is free and it is wrong: it priced the
# same declaration at 43.8 tokens, 27% under what `generate_content` then billed. Block A's
# whole claim is that the arms are the sizes they say they are, so the sizing has to come from
# the thing that bills. Every arm's achieved prompt size is printed next to its target, which
# is how a reader checks these three numbers have not drifted.
TOKENS_PER_DECLARATION = 45.0
REQUEST_OVERHEAD_TOKENS = 1_630  # the head, the user message and the request framing together
TOKENS_PER_PAD_CHAR = 0.225  # filler inside a description, 4.44 characters to the token

# Vertex refuses more than 512 function declarations in one request: 512 was accepted and 692
# came back 400. That caps an unpadded prefix at about 24.7k tokens, so the 32,768-token arm
# is the one arm that reaches its size by widening its declarations rather than adding more.
MAX_DECLARATIONS = 512

EST_CHARS_PER_TOKEN = 5.0  # the clause text below; at 4.0 the head came out at 1,277 tokens

# Two probes after each idle gap. The first is the analogue of a session's turn one, which is
# the quantity the whole project prices; the second says whether the first one re-warmed it.
PROBES_PER_GAP = 2

# Every arm in a wave has to run at the same time for block A's "at identical load" to mean
# anything, so the pool is sized to hold the widest wave rather than to be polite. Thirteen
# arms at one send per two seconds is under 7 QPS.
DEFAULT_MAX_WORKERS = 16

# 429s are expected when thirteen arms share a quota. A retried call is a *late* call, so the
# idle gap recorded against it is the one measured on the clock and not the one planned.
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_S = 5.0

MAX_OUTPUT_TOKENS = 8


# --- The plan -------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One prefix, loaded on a schedule and then probed after it has been left alone.

    `head_marker` is the identity of the head. Two arms with the same marker carry a
    byte-identical system instruction and so share a cacheable head; two arms with different
    markers differ at byte zero and share nothing. That single field is the whole of block C.

    `namespace` names the arm's declarations. It is two characters for every arm, because the
    per-declaration token cost is only constant if the names are all the same length, and an
    arm whose declarations cost more than the plan assumed sends a bigger prefix than the size
    it is labelled with — which in block A is the measurement itself.
    """

    block: str
    name: str
    namespace: str
    prefix_tokens: int
    sends: int
    interval_s: float
    probe_gaps: tuple[float, ...]
    head_marker: str
    probes_per_gap: int = PROBES_PER_GAP

    @property
    def calls(self) -> int:
        return self.sends + len(self.probe_gaps) * self.probes_per_gap

    @property
    def tokens(self) -> int:
        return self.calls * self.prefix_tokens

    @property
    def seconds(self) -> float:
        """Wall clock, ignoring per-call latency, which the caller cannot predict."""
        return self.sends * self.interval_s + sum(self.probe_gaps)


SIZES_A: tuple[int, ...] = (2_048, 4_096, 8_192, 16_384, 32_768)
SIZES_B: tuple[int, ...] = (4_096, 16_384)
GAPS_A: tuple[float, ...] = (5.0, 60.0, 600.0)
GAP_B: tuple[float, ...] = (60.0,)
LOAD_SENDS = 24
LOAD_INTERVAL_S = 2.0
SIZE_C = 8_192
NOISE_SIZE_C = 16_384
NOISE_SENDS_C = 120


def block_a() -> list[Arm]:
    """Size, held against an identical schedule. The question."""
    return [
        Arm("A", f"A-{size}", f"a{i}", size, LOAD_SENDS, LOAD_INTERVAL_S, GAPS_A, f"A-{size}")
        for i, size in enumerate(SIZES_A)
    ]


def block_b() -> list[Arm]:
    """Volume and rate. Controls, so that a negative result is on the record deliberately.

    Three send counts at one cadence isolate volume; one send count at a tenth of the cadence
    isolates rate. Both at two sizes, because if size governs then a control run at one size
    says nothing about the other.
    """
    arms: list[Arm] = []
    for size in SIZES_B:
        for sends in (4, 24, 96):
            name = f"B-{size}-x{sends}"
            arms.append(Arm("B", name, f"b{len(arms)}", size, sends, LOAD_INTERVAL_S, GAP_B, name))
        name = f"B-{size}-slow"
        arms.append(Arm("B", name, f"b{len(arms)}", size, LOAD_SENDS, 20.0, GAP_B, name))
    return arms


def block_c() -> list[Arm]:
    """Head collision. C1 and C2 are the same arm; only what else is on their head differs.

    The noise arm probes nothing. It exists to occupy the shared head while C2 is trying to
    stay resident on it, and it is twice C2's size so that if the head is a contended slot,
    the bigger claimant is the one holding it.
    """
    return [
        Arm("C", "C1-alone", "c0", SIZE_C, LOAD_SENDS, LOAD_INTERVAL_S, GAP_B, "C-shared"),
        Arm("C", "C2-contended", "c1", SIZE_C, LOAD_SENDS, LOAD_INTERVAL_S, GAP_B, "C-shared"),
        Arm("C", "C2-noise", "c2", NOISE_SIZE_C, NOISE_SENDS_C, 0.9, (), "C-shared"),
    ]


@dataclass(frozen=True)
class Wave:
    """Arms that run at the same time, and why they are allowed to."""

    name: str
    arms: tuple[Arm, ...]
    why: str

    @property
    def seconds(self) -> float:
        return max((arm.seconds for arm in self.arms), default=0.0)


def build_waves(blocks: Sequence[str] = ("A", "B", "C")) -> list[Wave]:
    """The whole experiment, as waves run one after another.

    A and B share a wave because every arm in them has a private head and a private namespace,
    so they cannot touch each other except through whatever global capacity the service has —
    and running them serially would take an hour of 600-second idles to find out the same
    thing. C runs afterwards, alone, because "C1 runs with nothing else on its head" is the
    control C2 is measured against and thirteen other arms in flight would spoil it.
    """
    chosen = set(blocks)
    waves: list[Wave] = []
    parallel = [arm for arm in (*block_a(), *block_b()) if arm.block in chosen]
    if parallel:
        waves.append(
            Wave("A+B", tuple(parallel), "private heads and namespaces, so they cannot collide")
        )
    if "C" in chosen:
        arms = block_c()
        waves.append(Wave("C1", (arms[0],), "the shared head, with nothing else on it"))
        waves.append(Wave("C2", (arms[1], arms[2]), "the shared head, with a bigger claimant"))
    return waves


# --- What gets sent -------------------------------------------------------------------


_CLAUSE = (
    "Clause {n:05d}: support agents must confirm the customer identity before disclosing "
    "order, payment or shipment detail, and must record the confirmation method in the "
    "ticket within one business day of the interaction."
)

_DECLARATION_DESCRIPTION = (
    "Look up one record for the current customer and return it as JSON. "
    "Read-only. Safe to call more than once."
)


def head_body(tokens: int = HEAD_TOKENS) -> str:
    """Filler clauses, deterministic and identical for every arm."""
    lines: list[str] = []
    size = 0
    n = 0
    while size < int(tokens * EST_CHARS_PER_TOKEN):
        line = _CLAUSE.format(n=n)
        lines.append(line)
        size += len(line) + 1
        n += 1
    return "\n".join(lines)


def head(marker: str) -> str:
    """A synthetic system instruction of about `HEAD_TOKENS`, private to `marker`.

    The marker is the first line, so two heads with different markers diverge at byte zero
    and no part of one can serve the other. Everything after it is byte-identical, which is
    what makes two arms sharing a marker share a head exactly rather than approximately, and
    is why the body is sized without reference to the marker.
    """
    return f"PROBE ARM {marker}. Answer in one word. Do not call a tool.\n{head_body()}"


def declaration(namespace: str, index: int, pad_chars: int = 0) -> types.FunctionDeclaration:
    """One synthetic tool. The index is fixed width because the name is billed by the byte."""
    filler = ("Retention policy note. " * (pad_chars // 23 + 1))[:pad_chars]
    return types.FunctionDeclaration(
        name=f"{namespace}_{index:03d}",
        description=f"{_DECLARATION_DESCRIPTION} {filler}"
        if pad_chars
        else _DECLARATION_DESCRIPTION,
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


def shape(prefix_tokens: int) -> tuple[int, int]:
    """How many declarations, and how much filler in each, add up to `prefix_tokens`.

    Count is the honest knob, because a bigger tool menu is more tools and that is what the
    router prunes. Filler exists only because Vertex stops at 512 declarations, which caps an
    unpadded prefix at about 24.7k tokens; the 32,768-token arm is the only one that needs it.
    """
    declared = max(0, prefix_tokens - REQUEST_OVERHEAD_TOKENS)
    count = max(1, round(declared / TOKENS_PER_DECLARATION))
    if count <= MAX_DECLARATIONS:
        return count, 0
    short = declared - MAX_DECLARATIONS * TOKENS_PER_DECLARATION
    return MAX_DECLARATIONS, max(0, round(short / (MAX_DECLARATIONS * TOKENS_PER_PAD_CHAR)))


def built_tokens(count: int, pad_chars: int) -> int:
    """What `shape`'s answer will actually bill, so a target can be printed beside it."""
    per = TOKENS_PER_DECLARATION + pad_chars * TOKENS_PER_PAD_CHAR
    return round(REQUEST_OVERHEAD_TOKENS + count * per)


def tools(namespace: str, count: int, pad_chars: int = 0) -> list[types.Tool]:
    return [
        types.Tool(
            function_declarations=[declaration(namespace, i, pad_chars) for i in range(count)]
        )
    ]


def user_text(run_id: str, arm: str, index: int) -> str:
    """Unique per call, so no request is ever a literal repeat of an earlier one.

    It sits after the prefix, so it cannot break the prefix match. Making it unique is what
    stops the service from answering out of a response cache and reporting a residency this
    probe did not measure.
    """
    return f"Reply with the single word ok. ({run_id} {arm} call {index})"


# --- Running it -----------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One request, and what the usage metadata said about it."""

    block: str
    arm: str
    prefix_tokens: int  # the target
    role: str  # "load" or "probe"
    index: int  # within the role, from 1
    planned_gap_s: float  # idle this call was scheduled to follow
    actual_gap_s: float  # idle it actually followed, on the clock
    prompt_tokens: int
    cached_tokens: int
    opens_gap: bool = False  # the turn-one analogue: first request after the prefix idled
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
    client: genai.Client, model: str, instruction: str, tool_list: list[types.Tool], text: str
) -> tuple[int, int, str]:
    """One fresh single-turn request. Returns (prompt tokens, cached tokens, error)."""
    config = types.GenerateContentConfig(
        system_instruction=instruction,
        tools=tool_list,
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    contents = [types.Content(role="user", parts=[types.Part(text=text)])]
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as exc:
            retryable = exc.code in (429, 500, 503)
            if not retryable or attempt == RETRY_ATTEMPTS - 1:
                return 0, 0, f"{type(exc).__name__} {exc.code}"
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue
        usage = response.usage_metadata
        if usage is None:
            return 0, 0, "no usage metadata"
        return int(usage.prompt_token_count or 0), int(usage.cached_content_token_count or 0), ""
    return 0, 0, "retries exhausted"


def run_arm(client: genai.Client, model: str, arm: Arm, run_id: str, sink: Sink) -> None:
    """Load the arm on its schedule, then probe it after each idle gap in turn.

    Gaps compound: the second gap is timed from the last probe of the first, because that
    probe is itself a send and refreshes the prefix. That is the same thing a real session
    does to a menu, which is why it is not corrected for.
    """
    instruction = head(f"{run_id} {arm.head_marker}")
    count, pad_chars = shape(arm.prefix_tokens)
    tool_list = tools(arm.namespace, count, pad_chars)
    call_index = 0
    last_at = 0.0

    def send(role: str, index: int, planned_gap: float, opens_gap: bool = False) -> None:
        nonlocal call_index, last_at
        call_index += 1
        started = time.monotonic()
        prompt, cached, error = ask(
            client, model, instruction, tool_list, user_text(run_id, arm.name, call_index)
        )
        sink.add(
            Call(
                block=arm.block,
                arm=arm.name,
                prefix_tokens=arm.prefix_tokens,
                role=role,
                index=index,
                planned_gap_s=planned_gap,
                actual_gap_s=started - last_at if last_at else 0.0,
                prompt_tokens=prompt,
                cached_tokens=cached,
                opens_gap=opens_gap,
                error=error,
            )
        )
        last_at = started

    for i in range(arm.sends):
        due = time.monotonic() + arm.interval_s
        send("load", i + 1, arm.interval_s if i else 0.0)
        if i + 1 < arm.sends:
            time.sleep(max(0.0, due - time.monotonic()))

    probe_index = 0
    for gap in arm.probe_gaps:
        time.sleep(max(0.0, gap - (time.monotonic() - last_at)))
        for within in range(arm.probes_per_gap):
            probe_index += 1
            send("probe", probe_index, gap, opens_gap=within == 0)


def run_waves(
    client: genai.Client,
    model: str,
    waves: Sequence[Wave],
    run_id: str,
    max_workers: int,
) -> list[Call]:
    sink = Sink()
    for wave in waves:
        print(
            f"  wave {wave.name}: {len(wave.arms)} arms, ~{wave.seconds / 60:.0f} min ({wave.why})",
            file=sys.stderr,
        )
        with ThreadPoolExecutor(max_workers=min(max_workers, len(wave.arms))) as pool:
            for arm in wave.arms:
                pool.submit(run_arm, client, model, arm, run_id, sink)
    return sink.calls


# --- Reading the answer ----------------------------------------------------------------


def first_after_gap(calls: Sequence[Call], arm: str, gap: float | None = None) -> list[Call]:
    """The turn-one analogues for one arm, optionally at one planned gap."""
    return [
        call
        for call in calls
        if call.arm == arm
        and call.opens_gap
        and not call.error
        and (gap is None or call.planned_gap_s == gap)
    ]


def arm_share(calls: Sequence[Call], arm: str, gap: float | None = None) -> float | None:
    """Mean cached share of an arm's first-after-gap probes, or None if it has none."""
    probes = first_after_gap(calls, arm, gap)
    return statistics.mean(call.share for call in probes) if probes else None


def clusters(calls: Sequence[Call], tolerance: int = 32) -> list[tuple[int, int, int]]:
    """Raw cached-token counts grouped into runs of near-equal values.

    This is the reading the shares throw away. In the A/B the values piled up at 0, then at
    1,586-1,602 — the shared head matching on its own, with none of the declarations behind
    it — then at higher plateaux. A percentage cannot show that, and it is the difference
    between "the cache missed" and "the cache served the part every prefix has in common".
    """
    values = sorted(call.cached_tokens for call in calls if not call.error)
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [(group[0], group[-1], len(group)) for group in groups]


def verdict_size(calls: Sequence[Call]) -> str:
    shares = {size: arm_share(calls, f"A-{size}") for size in SIZES_A}
    measured = {size: share for size, share in shares.items() if share is not None}
    if len(measured) < len(SIZES_A):
        return "INCONCLUSIVE — block A did not complete, so size was not varied cleanly."
    small = measured[SIZES_A[0]]
    big = min(measured[SIZES_A[-1]], measured[SIZES_A[-2]])
    spread = max(measured.values()) - min(measured.values())
    detail = ", ".join(f"{size:,}={share:.3f}" for size, share in measured.items())
    if small < 0.2 and big > 0.85:
        return (
            f"SIZE GOVERNS — {detail}. Residency needs a size term. Refit MEASURED_CURVE per"
            " size band, and treat menu width as a priced tunable that tune_menus.py can"
            " already sweep."
        )
    if spread < 0.05:
        return (
            f"SIZE IS DEAD — {detail}, a spread of {spread:.3f} across sixteen-fold in size."
            " Look to head collision, or to explicit caching."
        )
    return (
        f"PARTIAL — {detail}. Size moves it by {spread:.3f} without clearing either"
        " threshold. Report the raw numbers and decide by hand."
    )


def verdict_volume(calls: Sequence[Call]) -> str:
    rows = {
        size: [(sends, arm_share(calls, f"B-{size}-x{sends}")) for sends in (4, 24, 96)]
        for size in SIZES_B
    }
    detail: list[str] = []
    rising = []
    for size, points in rows.items():
        known = [(sends, share) for sends, share in points if share is not None]
        if len(known) < 3:
            return "INCONCLUSIVE — block B's volume arms did not complete."
        detail.append(f"{size:,}: " + " ".join(f"{sends}x={share:.3f}" for sends, share in known))
        rising.append(known[-1][1] - known[0][1] > 0.15)
    joined = "; ".join(detail)
    if all(rising):
        return (
            f"VOLUME GOVERNS — {joined}. The warehouse's null result was wrong, DESIGN 18.21's"
            " warmth table is right, and the fix is a cheap warm-up sender at publish time."
        )
    if any(rising):
        return f"PARTIAL — {joined}. Volume moved one size and not the other."
    return f"VOLUME IS DEAD — {joined}. Twenty-four times the sends buys nothing."


def verdict_rate(calls: Sequence[Call]) -> str:
    detail: list[str] = []
    colder = []
    for size in SIZES_B:
        fast = arm_share(calls, f"B-{size}-x{LOAD_SENDS}")
        slow = arm_share(calls, f"B-{size}-slow")
        if fast is None or slow is None:
            return "INCONCLUSIVE — block B's rate arms did not complete."
        detail.append(f"{size:,}: 1/2s={fast:.3f} 1/20s={slow:.3f}")
        colder.append(fast - slow > 0.15)
    joined = "; ".join(detail)
    if all(colder):
        return (
            f"RATE GOVERNS — {joined}. Same sends, a tenth of the cadence, materially colder."
            " The fix is fewer, busier menus."
        )
    if any(colder):
        return f"PARTIAL — {joined}. Cadence moved one size and not the other."
    return f"RATE IS DEAD — {joined}. A tenth of the cadence at the same sends costs nothing."


def verdict_head(calls: Sequence[Call]) -> str:
    alone = arm_share(calls, "C1-alone")
    contended = arm_share(calls, "C2-contended")
    if alone is None or contended is None:
        return "INCONCLUSIVE — block C did not complete."
    if alone - contended > 0.15:
        return (
            f"HEADS COMPETE — alone {alone:.3f}, contended {contended:.3f}. Prefixes sharing a"
            " system-instruction head evict each other. Give every menu unique leading bytes"
            " — a per-menu marker at the very front of the instruction — before costing any"
            " CachedContent work."
        )
    return (
        f"HEADS DO NOT COMPETE — alone {alone:.3f}, contended {contended:.3f}. A busier"
        " prefix on the same head does not cool this one, so head collision is not the"
        " mechanism and unique leading bytes would buy nothing."
    )


def verdict_overall(calls: Sequence[Call]) -> str:
    findings = (verdict_size(calls), verdict_volume(calls), verdict_rate(calls))
    if all(" IS DEAD" in v for v in findings) and verdict_head(calls).startswith(
        "HEADS DO NOT COMPETE"
    ):
        return (
            "NOTHING MOVES ANYTHING — implicit residency is not controllable from the request"
            " side. Explicit caching is the only lever left, and it has to clear the storage"
            " break-even before it is worth building."
        )
    return "At least one variable moved. Read the block verdicts above in order."


# --- Reporting --------------------------------------------------------------------------

_HEADERS = (
    "block",
    "arm",
    "target",
    "role",
    "n",
    "gap plan",
    "gap real",
    "prompt",
    "cached",
    "share",
)
_WIDTHS = (5, 16, 7, 5, 4, 8, 8, 8, 8, 6)


def print_table(calls: Sequence[Call]) -> None:
    """Every probe, and the last load before it, because a load is the warm-up it followed."""
    header = "  ".join(h.ljust(w) for h, w in zip(_HEADERS, _WIDTHS, strict=True))
    print(header)
    print("-" * len(header))
    for call in calls:
        if call.role != "probe":
            continue
        cells = (
            call.block,
            call.arm,
            f"{call.prefix_tokens:,}",
            call.role,
            str(call.index),
            f"{call.planned_gap_s:,.0f}",
            f"{call.actual_gap_s:,.0f}",
            f"{call.prompt_tokens:,}",
            f"{call.cached_tokens:,}",
            f"{call.share:.3f}",
        )
        print("  ".join(c.ljust(w) for c, w in zip(cells, _WIDTHS, strict=True)))
        if call.error:
            print(f"    -> {call.error}")


def print_summary(calls: Sequence[Call]) -> None:
    print("\nfirst probe after each idle gap, by arm — the turn-one analogue")
    header = f"{'arm':<16} {'target':>8} {'prompt':>8} {'gap':>7} {'n':>3}"
    print(f"  {header} {'cached tok':>11} {'share':>7}")
    seen: list[tuple[str, float]] = []
    for call in calls:
        key = (call.arm, call.planned_gap_s)
        if not call.opens_gap or call.error or key in seen:
            continue
        seen.append(key)
        probes = first_after_gap(calls, call.arm, call.planned_gap_s)
        # The prompt column is the achieved size. It should sit within a percent of the
        # target; if it does not, the sizing constants have drifted and block A is unbalanced.
        prompt = statistics.mean(p.prompt_tokens for p in probes)
        cached = statistics.mean(p.cached_tokens for p in probes)
        share = statistics.mean(p.share for p in probes)
        print(
            f"  {call.arm:<16} {call.prefix_tokens:>8,} {prompt:>8,.0f}"
            f" {call.planned_gap_s:>7,.0f} {len(probes):>3} {cached:>11,.0f} {share:>7.3f}"
        )

    print("\nraw cached-token counts, clustered — where the head shows up on its own")
    for low, high, count in clusters([c for c in calls if c.role == "probe"]):
        span = f"{low:,}" if low == high else f"{low:,}-{high:,}"
        print(f"  {span:>17}   {count} calls")


def print_verdicts(calls: Sequence[Call]) -> None:
    print()
    print("=" * 100)
    print("CONCLUSION — what governs residency for a pruned prefix")
    print("=" * 100)
    for label, verdict in (
        ("A  size", verdict_size(calls)),
        ("B  volume", verdict_volume(calls)),
        ("B  rate", verdict_rate(calls)),
        ("C  head", verdict_head(calls)),
    ):
        print(f"\n  {label}\n    {verdict}")
    print(f"\n  overall\n    {verdict_overall(calls)}\n")


# --- Cost -----------------------------------------------------------------------------


def input_usd_per_mtok(model: str) -> float:
    """The published input rate, from the one price table this project keeps."""
    key = model.replace("-", " ").lower()
    for rates in MODELS:
        if rates.name.lower() == key:
            return rates.input_usd
    return 0.30


def estimate(waves: Sequence[Wave], model: str) -> tuple[int, float, float, list[str]]:
    """Worst-case tokens, USD and minutes. Worst case means nothing caches at all."""
    rate = input_usd_per_mtok(model)
    lines: list[str] = []
    tokens = 0
    for wave in waves:
        for arm in wave.arms:
            tokens += arm.tokens
            lines.append(
                f"  {arm.name:<16} {arm.prefix_tokens:>7,} tok x {arm.calls:>4} calls"
                f" = {arm.tokens:>10,} tok"
            )
    minutes = sum(wave.seconds for wave in waves) / 60.0
    return tokens, tokens / 1e6 * rate, minutes, lines


# --- Entry point ------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    waves = build_waves(args.blocks)
    if not waves:
        print("no blocks selected.", file=sys.stderr)
        return 1

    tokens, usd, minutes, lines = estimate(waves, args.model)
    calls = sum(arm.calls for wave in waves for arm in wave.arms)

    print("Residency probe — what governs how long a pruned prefix stays cached\n")
    for label, value in (
        ("project", args.project),
        ("location", args.location),
        ("model", args.model),
        ("run id", args.run_id),
        ("blocks", ", ".join(args.blocks)),
        ("auth", "Application Default Credentials, Vertex backend"),
    ):
        print(f"  {label:<10} {value}")
    print(f"\nPlan: {calls:,} fresh single-turn calls across {len(waves)} waves")
    for line in lines:
        print(line)
    print(
        f"\n  {'TOTAL':<16} {tokens:>10,} input tokens, worst case ${usd:.2f} at"
        f" ${input_usd_per_mtok(args.model):.2f}/Mtok"
    )
    print(f"  {'':<16} about {minutes:.0f} minutes of wall clock, plus per-call latency")
    print(
        "\n  Worst case assumes nothing caches, which is the outcome the probe exists to rule"
        "\n  out; a run where anything caches costs less. Output is negligible at"
        f"\n  max_output_tokens={MAX_OUTPUT_TOKENS} and temperature 0."
    )

    print("\nHow each target size is built, from the measured constants:")
    for size in sorted(dict.fromkeys(arm.prefix_tokens for wave in waves for arm in wave.arms)):
        count, pad_chars = shape(size)
        padding = f", {pad_chars} filler chars each" if pad_chars else ""
        print(
            f"  target {size:>7,} -> {count:>3} declarations{padding}"
            f" = {built_tokens(count, pad_chars):>7,} tokens"
        )
    print(
        "  Check these against the prompt column of the run. If they have drifted the model"
        "\n  has been repriced and the three constants at the top need remeasuring."
    )

    if not args.confirm:
        print("\nNothing was called. Re-run with --confirm to spend the amount above.")
        return 2

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    print("\nrunning ...", file=sys.stderr)
    observed = run_waves(client, args.model, waves, args.run_id, args.max_workers)

    print()
    print_table(observed)
    print_summary(observed)
    print_verdicts(observed)

    failed = [call for call in observed if call.error]
    if failed:
        print(f"  {len(failed)} of {len(observed)} calls failed; verdicts ignore them.\n")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "model": args.model,
                    "calls": [vars(call) for call in observed],
                },
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
        "--blocks",
        nargs="+",
        default=["A", "B", "C"],
        choices=["A", "B", "C"],
        help="A size, B volume and rate, C head collision.",
    )
    parser.add_argument(
        "--run-id",
        default=dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ"),
        help="stamped into every declaration name and head marker. A fresh id is what stops"
        " a re-run from measuring the caches its own last run left behind.",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--out", default=None, help="write every call to this JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
