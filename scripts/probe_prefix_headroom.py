#!/usr/bin/env python
"""How far over the cache floor does a shared prefix have to be before it is actually kept?

`qualify` tests `prefix >= floor` and treats the cross-session case as answered. DESIGN.md
19.43 says it is not. A padded agent cleared the floor by 267 tokens and opened warm in one
session out of eleven; the control it ran against had 5,500 tokens of headroom and opened warm
in seven. Clearing the threshold and being retained are two different things, and the module
tests the first while claiming the second.

The likely mechanism is block granularity rounding a barely-over prefix back under, but that is
one inference from one margin, and one margin cannot calibrate a second one. So this sweeps the
margin. Eight prefixes, from half a thousand tokens under the floor to four thousand over it,
each loaded until it should be cached and then probed after an idle gap. The output is a hit
rate per headroom, and the number it exists to produce is the smallest headroom that is warm.

    -512   under the floor. The control that says where the floor is on the day of the run
           rather than where the documentation puts it. It has to be cold or nothing else here
           can be read.
       0   at the floor, which is exactly what `qualify` accepts today.
     128   through 4,096: the sweep. Fine near the floor, because that is where 19.43 saw the
           failure and where a block boundary would sit if block granularity is the mechanism,
           and coarse above it, because the only question up there is whether it has died out.

Every arm gets a private marker on the first line of its system instruction and a private
declaration namespace, so no arm's prefix is a prefix of another's and none can serve any part
of another. Size is reached with declarations rather than padding, because `probe_padding`
found padding is at best second best and because declarations are what an agent actually has.

Two calls before the sweep measure what a declaration costs on the day, and the arms are placed
against that measurement rather than against a constant. At 128 tokens of resolution a three
percent error in the per-declaration estimate is most of a step, so an uncalibrated sweep would
be labelling its own arms wrong.

An arm is *live* if it caches meaningfully more often than the under-floor control. Live is a
weaker word than warm on purpose: both measured runs found the arms over the edge caching about
a quarter of the time, which is not something to rely on but is unmistakably not nothing, and
where that starts is the edge. What each outcome decides:

    control live                   the floor is not where this project thinks it is, and every
                                   band in `qualify.py` is drawn against the wrong number.
                                   Nothing else in the run is readable until that is settled.
    live from headroom 0 up        19.43's failure was not the margin. `qualify` is right as
                                   written, and why the padded arm lost turn one becomes the
                                   open question instead.
    dead up to some headroom,      the number this probe exists for. `qualify` becomes
    live from there up             `prefix >= floor + headroom` and the cross-session test
                                   means what it claims.
    no clean split                 one reading cannot calibrate a margin. Record the shape and
                                   leave `qualify` alone.

The schedule is `probe_padding`'s, for its reason: DESIGN.md 19.30 established that residency is
a short TTL, so a five-second gap is where a cacheable prefix is close to certain to be warm and
a non-cacheable one is still cold.

    uv run python scripts/probe_prefix_headroom.py             # plan and price it, call nothing
    uv run python scripts/probe_prefix_headroom.py --confirm   # spend the amount it printed

Auth is ADC. Run `gcloud auth application-default login` first. No key files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from google import genai

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from deployment import project_arg  # noqa: E402
from probe_residency import (  # noqa: E402
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    TOKENS_PER_DECLARATION,
    Call,
    Sink,
    ask,
    head_body,
    input_usd_per_mtok,
    tools,
    user_text,
)

# The floor this run is drawn against. gemini-2.5-flash publishes 2,048. The under-floor arm is
# what checks that on the day, so this constant is the hypothesis and not an assumption.
FLOOR_TOKENS = 2_048

# The system instruction is the same small size in every arm, so every token of the sweep comes
# from declarations and the arms differ in one thing only.
HEAD_TOKENS = 300

# What the user message and the request framing add on top of the shared prefix. An error here
# shifts every arm by the same amount, so it moves the labels and not the transition between
# them, and the transition is what the run is actually reading.
FRAMING_TOKENS = 30

# The sweep, in tokens over the floor.
HEADROOMS: tuple[int, ...] = (-512, 0, 128, 256, 512, 1_024, 2_048, 4_096)

# Where calibration reads the cost of a declaration. Far enough apart that the difference is
# thousands of tokens and a token or two of noise cannot bend the line.
CALIBRATION_COUNTS: tuple[int, int] = (10, 110)

LOAD_SENDS = 12
LOAD_INTERVAL_S = 2.0
GAP_S = 5.0

# Twenty probes per arm, for `probe_padding`'s reason: implicit caching is best effort, and an
# arm read six times cannot tell a genuinely rare cache from a coin that landed once.
PROBES = 20

WARM = 0.50  # at or above this an arm caches reliably
COLD = 0.15  # a control at or under this is doing nothing, which is what a control is for

# An arm is live if it caches at least this often and at least this many times as often as the
# under-floor control. Live is not warm. Both measured runs put every arm over the edge between
# a quarter and a half, which is not something to rely on but is unmistakably not nothing, and
# where that starts is the edge this probe is looking for.
LIVE = 0.10
CONTROL_MULTIPLE = 3.0

# How many arms in a row have to be live before the smallest of them is called a threshold. Two,
# so the largest arm can never be a threshold on its own: "live from here up" needs an up.
MIN_LIVE_ARMS = 2


@dataclass(frozen=True)
class Arm:
    """One shared prefix, identified by how far over the floor it was aimed."""

    headroom: int
    declarations: int
    prefix_tokens: int

    @property
    def name(self) -> str:
        return f"{self.headroom:+d}"

    @property
    def namespace(self) -> str:
        """Private per arm, so no arm's declaration list is a prefix of another's.

        Fixed width, and the same width calibration uses. A declaration is billed by the byte,
        so an arm with a longer namespace pays more for every one of its declarations and lands
        somewhere other than where it was aimed. The first run of this probe had `hp0` against
        `hp2048` and put two arms 150 tokens apart from their targets in opposite directions.
        """
        return namespace(self.headroom)

    @property
    def prompt_tokens(self) -> int:
        return self.prefix_tokens + FRAMING_TOKENS


# --- Placing the arms -------------------------------------------------------------------

# Every namespace is this many characters, arms and calibration alike. The offset keeps the
# under-floor arms positive so the width never changes with the sign.
NAMESPACE_OFFSET = 8_192
NAMESPACE_WIDTH = 6


def namespace(headroom: int) -> str:
    return f"h{headroom + NAMESPACE_OFFSET:0{NAMESPACE_WIDTH - 1}d}"


def calls_per_arm(probes: int) -> int:
    return LOAD_SENDS + probes


def fit(counts: tuple[int, int], prompts: tuple[int, int]) -> tuple[float, float]:
    """Turn two measured requests into (prompt tokens at zero declarations, tokens each).

    Two points and a straight line, because the request is a straight line: a fixed head and
    framing, plus a fixed cost for every declaration after that.
    """
    low_n, high_n = counts
    low, high = prompts
    if high_n == low_n:
        raise ValueError("calibration needs two different declaration counts")
    per = (high - low) / (high_n - low_n)
    return low - per * low_n, per


def declarations_for(prefix_target: int, base: float, per: float) -> int:
    """How many declarations put the shared prefix at `prefix_target`."""
    return max(1, round((prefix_target - (base - FRAMING_TOKENS)) / per))


def build_arms(
    base: float = HEAD_TOKENS + FRAMING_TOKENS,
    per: float = TOKENS_PER_DECLARATION,
    headrooms: Sequence[int] = HEADROOMS,
) -> list[Arm]:
    """The sweep, placed against whatever a declaration is currently believed to cost."""
    arms = []
    for headroom in sorted(headrooms):
        count = declarations_for(FLOOR_TOKENS + headroom, base, per)
        arms.append(Arm(headroom, count, round(base - FRAMING_TOKENS + count * per)))
    return arms


def head(marker: str) -> str:
    """A private system instruction of about `HEAD_TOKENS`.

    The marker is the first line, so two arms diverge at byte zero and neither can serve any
    part of the other. It carries the run id as well as the arm, so a re-run cannot read the
    caches its own last run left behind.
    """
    return f"PROBE ARM {marker}. Answer in one word. Do not call a tool.\n{head_body(HEAD_TOKENS)}"


def measure(client: genai.Client, model: str, run_id: str) -> tuple[float, float] | None:
    """Two calls that say what a declaration costs today. None if either one failed.

    Both use a namespace the same width as the arms', or the line they fit would be the cost of
    a declaration plus the cost of the digits in the count.
    """
    prompts: list[int] = []
    for count in CALIBRATION_COUNTS:
        prompt, _, error = ask(
            client,
            model,
            head(f"{run_id} calibrate {count}"),
            tools(namespace(count), count),
            user_text(run_id, "calibrate", count),
        )
        if error or not prompt:
            return None
        prompts.append(prompt)
    return fit(CALIBRATION_COUNTS, (prompts[0], prompts[1]))


# --- Running it -----------------------------------------------------------------------


def run_arm(
    client: genai.Client, model: str, arm: Arm, run_id: str, sink: Sink, probes: int = PROBES
) -> None:
    """Load the arm until it should be cached, idle for the gap, then probe it."""
    instruction = head(f"{run_id} {arm.name}")
    tool_list = tools(arm.namespace, arm.declarations)
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
                block="H",
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

    for i in range(LOAD_SENDS):
        due = time.monotonic() + LOAD_INTERVAL_S
        send("load", i + 1, LOAD_INTERVAL_S if i else 0.0)
        if i + 1 < LOAD_SENDS:
            time.sleep(max(0.0, due - time.monotonic()))

    time.sleep(max(0.0, GAP_S - (time.monotonic() - last_at)))
    for index in range(1, probes + 1):
        send("probe", index, GAP_S, opens_gap=True)


# --- Reading it -----------------------------------------------------------------------


def probes(calls: Sequence[Call], arm: str) -> list[Call]:
    return [c for c in calls if c.arm == arm and c.role == "probe" and not c.error]


def hit_rate(calls: Sequence[Call], arm: str) -> float | None:
    """The share of probes that cached anything at all.

    How much of a prefix caches is a size question and DESIGN.md 19.30 answered it. Whether a
    prefix opens warm at all is what this probe is about, so a rate over calls is the statistic
    that matches the claim `qualify` makes.
    """
    found = probes(calls, arm)
    return sum(1 for call in found if call.cached_tokens) / len(found) if found else None


def headroom_of(arm: Arm) -> int:
    return arm.headroom


def is_live(calls: Sequence[Call], arm: Arm, control: float) -> bool:
    """Did this arm cache more than a prefix known to be under the floor?

    Live is a weaker claim than warm and it is the one the run can support. An arm that opens
    warm a quarter of the time is not something to rely on, but it is unambiguously doing
    something the control never does, and where that starts is the edge this probe is looking
    for. Both tests have to pass: an absolute floor so a run where everything caches a little
    does not read as a threshold, and a multiple of the control so a noisy control cannot
    manufacture one.
    """
    rate = hit_rate(calls, arm.name)
    return rate is not None and rate >= LIVE and rate >= CONTROL_MULTIPLE * control


def smallest_live(calls: Sequence[Call], arms: Sequence[Arm], control: float) -> Arm | None:
    """The smallest headroom at or over the floor that is live and stays live above itself.

    Staying live is the part that matters. A lone live arm, whether it has dead arms above it or
    is simply the largest in the sweep, is noise or a coincidence of load, and shipping a
    threshold on it would be calibrating a margin from a single reading, which is the mistake
    this probe exists to avoid.
    """
    over = [arm for arm in arms if arm.headroom >= 0]
    for index, arm in enumerate(over):
        band = over[index:]
        if len(band) < MIN_LIVE_ARMS:
            return None
        if all(is_live(calls, later, control) for later in band):
            return arm
    return None


def reliability(calls: Sequence[Call], arms: Sequence[Arm], winner: Arm) -> str:
    """One sentence on whether clearing the edge is sufficient or only necessary."""
    band = [arm for arm in arms if arm.headroom >= winner.headroom]
    worst = min(hit_rate(calls, arm.name) or 0.0 for arm in band)
    if worst >= WARM:
        return f"Over it, every arm opened warm on at least {worst:.0%} of its probes."
    return (
        f"Clearing it is necessary and not sufficient: over it, an arm still opened cold on as"
        f" many as {1 - worst:.0%} of its probes, so the margin buys the chance of a cache and"
        f" not the cache."
    )


def verdict(calls: Sequence[Call], arms: Sequence[Arm]) -> str:
    under = next((arm for arm in arms if arm.headroom < 0), None)
    if under is None:
        return "INCONCLUSIVE - the sweep had no under-floor control to read the rest against."
    control = hit_rate(calls, under.name)
    if control is None:
        return "INCONCLUSIVE - the under-floor control returned no usable probe."
    if control > COLD:
        return (
            f"THE FLOOR IS NOT WHERE WE THINK - a prefix of {under.prefix_tokens:,} tokens cached"
            f" on {control:.0%} of its probes against a published floor of {FLOOR_TOKENS:,}. Read"
            f" nothing else here until that is explained, because every band in qualify.py is"
            f" drawn against that number."
        )

    rates = [hit_rate(calls, arm.name) for arm in arms if arm.headroom >= 0]
    if not rates or any(rate is None for rate in rates):
        return "INCONCLUSIVE - an arm over the floor returned no usable probe."
    best = max(rate for rate in rates if rate is not None)

    winner = smallest_live(calls, arms, control)
    if winner is None:
        if best < LIVE:
            return (
                f"NOTHING OVER THE FLOOR CACHES - the best arm cached on {best:.0%} of its"
                f" probes, no better than the control. probe_padding found a 4,096-token prefix"
                f" caching, so a run this cold is more likely broken than that finding is wrong."
                f" Check the error count and the prompt column before changing anything."
            )
        return (
            f"NO CLEAN THRESHOLD - the arms over the floor peak at {best:.0%} but do not sort"
            f" into a dead band under {MIN_LIVE_ARMS} or more live ones. One reading cannot"
            f" calibrate a margin, so qualify.py keeps `prefix >= floor` and the question stays"
            f" open. If the only live arm was the largest, extend the sweep past it."
        )

    if winner.headroom == 0:
        return (
            f"NO HEADROOM IS NEEDED - a prefix at the floor cached on"
            f" {hit_rate(calls, winner.name) or 0:.0%} of its probes and every larger one did"
            f" too. qualify.py's `prefix >= floor` is right as written, and 19.43's padded arm"
            f" lost turn one for some reason other than its 267-token margin. That is now the"
            f" open question."
        )
    below = [arm for arm in arms if 0 <= arm.headroom < winner.headroom]
    if below:
        last_dead = max(below, key=headroom_of)
        bracket = (
            f" {last_dead.prefix_tokens:,} tokens cached on"
            f" {hit_rate(calls, last_dead.name) or 0:.0%} of its probes, the control's rate, and"
            f" {winner.prefix_tokens:,} tokens on {hit_rate(calls, winner.name) or 0:.0%}."
            f" The edge is somewhere in between, and"
        )
    else:
        # Nothing was run between the floor and the winner, so the edge could be lower. That is
        # a narrower sweep's answer, not a wrong one: it still rules the floor itself out.
        bracket = (
            f" {winner.prefix_tokens:,} tokens cached on"
            f" {hit_rate(calls, winner.name) or 0:.0%} of its probes. This sweep ran nothing"
            f" between the floor and it, so the edge may be lower, and"
        )
    return (
        f"HEADROOM IS {winner.headroom:,} TOKENS - the floor on its own is not enough."
        f"{bracket} {winner.headroom:,} is the smallest margin this run can defend."
        f" {reliability(calls, arms, winner)}"
        f" qualify.py's cross-session test becomes `prefix >= floor + {winner.headroom:,}`."
    )


def print_table(calls: Sequence[Call], arms: Sequence[Arm]) -> None:
    print(
        f"{'headroom':>9} {'decls':>6} {'aimed':>8} {'prompt':>8} {'n':>4}"
        f" {'hit':>6} {'cached':>8} {'share':>7}"
    )
    print("-" * 62)
    for arm in arms:
        found = probes(calls, arm.name)
        if not found:
            print(f"{arm.name:>9} {arm.declarations:>6} {arm.prefix_tokens:>8,}   no usable probe")
            continue
        warm = [call for call in found if call.cached_tokens]
        print(
            f"{arm.name:>9} {arm.declarations:>6} {arm.prefix_tokens:>8,}"
            f" {statistics.mean(c.prompt_tokens for c in found):>8,.0f} {len(found):>4}"
            f" {hit_rate(calls, arm.name) or 0:>6.2f}"
            f" {statistics.median(c.cached_tokens for c in warm) if warm else 0:>8,.0f}"
            f" {statistics.mean(c.share for c in found):>7.3f}"
        )
    print()


def estimate(arms: Sequence[Arm], model: str, probes: int = PROBES) -> tuple[int, float, float]:
    """Input tokens, dollars and seconds. Every arm runs at once, so the clock is one arm's."""
    calibration = round(sum(CALIBRATION_COUNTS) * TOKENS_PER_DECLARATION)
    tokens = sum(arm.prompt_tokens * calls_per_arm(probes) for arm in arms) + calibration
    seconds = LOAD_SENDS * LOAD_INTERVAL_S + GAP_S + probes
    return tokens, tokens / 1e6 * input_usd_per_mtok(model), seconds


def print_plan(arms: Sequence[Arm], model: str, probes: int = PROBES) -> None:
    total = len(arms) * calls_per_arm(probes)
    print(f"\nPlan: {total:,} fresh single-turn calls, all arms at once\n")
    print(f"  {'headroom':>8}  {'declarations':>12}  {'prefix':>7}  {'prompt':>7}")
    for arm in arms:
        print(
            f"  {arm.name:>8}  {arm.declarations:>12}"
            f"  {arm.prefix_tokens:>7,}  {arm.prompt_tokens:>7,}"
        )
    tokens, usd, seconds = estimate(arms, model, probes)
    print(
        f"\n  {tokens:,} input tokens, worst case ${usd:.2f};"
        f" about {seconds:.0f} seconds plus latency"
    )


def main() -> int:
    args = parse_args()
    arms = build_arms(headrooms=args.headrooms)

    print("Prefix headroom probe - how far over the floor a shared prefix has to be\n")
    for label, value in (
        ("project", args.project),
        ("location", args.location),
        ("model", args.model),
        ("run id", args.run_id),
        ("floor", f"{FLOOR_TOKENS:,} tokens"),
        ("auth", "Application Default Credentials, Vertex backend"),
    ):
        print(f"  {label:<10} {value}")

    print_plan(arms, args.model, args.probes)
    print(
        "\n  Those sizes are estimates. Two calls before the sweep measure what a declaration"
        "\n  really costs, and the arms are rebuilt against that before anything is loaded."
    )

    if not args.confirm:
        print("\nNothing was called. Re-run with --confirm to spend the amount above.")
        return 2

    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    print("\ncalibrating ...", file=sys.stderr)
    measured = measure(client, args.model, args.run_id)
    if measured is None:
        print("\n  Calibration failed. Falling back to the estimated sizes above.")
    else:
        base, per = measured
        print(f"\n  Measured: {per:.1f} tokens a declaration, {base:.0f} tokens without any.")
        arms = build_arms(base, per, args.headrooms)
        print_plan(arms, args.model, args.probes)

    print("\nrunning ...", file=sys.stderr)
    sink = Sink()
    with ThreadPoolExecutor(max_workers=len(arms)) as pool:
        for arm in arms:
            pool.submit(run_arm, client, args.model, arm, args.run_id, sink, args.probes)
    observed = sink.calls

    print()
    print_table(observed, arms)
    print("=" * 78)
    print(f"CONCLUSION\n\n  {verdict(observed, arms)}\n")

    failed = [call for call in observed if call.error]
    if failed:
        print(f"  {len(failed)} of {len(observed)} calls failed; the verdict ignores them.\n")

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
        help="stamped into every head marker, so a re-run cannot read its own old caches",
    )
    parser.add_argument("--out", default=None, help="write every call to this JSON file")
    parser.add_argument(
        "--probes",
        type=int,
        default=PROBES,
        help="probes per arm after the gap. Raise it when two arms are close enough that"
        " twenty reads cannot tell them apart.",
    )
    parser.add_argument(
        "--headrooms",
        type=headroom_list,
        default=HEADROOMS,
        help="comma-separated tokens over the floor, to narrow the sweep. Every arm runs at"
        " once, so fewer arms is also less contention for whatever cache slots exist.",
    )
    return parser.parse_args()


def headroom_list(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


if __name__ == "__main__":
    raise SystemExit(main())
