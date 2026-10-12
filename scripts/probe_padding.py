#!/usr/bin/env python
"""Does padding a prompt over the cache floor actually make it cache?

`qualify` gives the same advice twice — to an agent whose pruned prompt would fall under the
model's cache floor, and to an agent whose pruned *cross-session prefix* would (DESIGN.md
19.31) — and the advice is "lengthen the system instruction, or keep more tools, until it
clears the floor by itself". Nothing has ever tested the first half of that. DESIGN.md 18.13
says outright that it is not safe to assume padded content caches like content that does
work, and the roadmap has carried the question since.

It is a cheap question with an expensive consequence, because the advice is the only remedy
the tool offers. Three arms, all fresh single-turn calls, all with private heads and private
declaration namespaces so nothing shares a prefix with anything:

    under      a prefix deliberately under the floor. The control that says where the floor
               is on the day of the run, rather than where the documentation puts it.
    padded     the same declarations, with the system instruction padded until the prefix
               clears the floor. This is the remedy, exactly as `qualify` words it.
    declared   no padding, and enough extra declarations to reach the same prefix size as
               `padded`. This is what the remedy is being compared against: bulk that is
               doing a job.

What each outcome decides:

    under cold, padded and declared both warm     the remedy works and `qualify` may keep
                                                  recommending it. Composition does not
                                                  matter; only size does.
    padded materially colder than declared        padding is second best. `qualify` should
                                                  recommend keeping tools first and say that
                                                  padding is the weaker of the two.
    padded as cold as under                       the remedy does not work and both messages
                                                  in `qualify` are wrong. Say so, and the
                                                  only honest advice left is to keep tools.
    under warm                                    the floor is not where this project thinks
                                                  it is, and every band in `qualify.py` is
                                                  drawn against the wrong number.

The schedule is short on purpose. DESIGN.md 19.30 established that residency is a short TTL,
so a five-second gap is where a cacheable prefix is close to certain to be warm and a
non-cacheable one is still cold. That makes "does it cache at all" answerable in two minutes
for a few cents, and the sixty-second gap is a free second reading.

    uv run python scripts/probe_padding.py             # plan and price it, call nothing
    uv run python scripts/probe_padding.py --confirm    # spend the amount it printed

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
from google.genai import types

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
    declaration,
    head_body,
    input_usd_per_mtok,
    user_text,
)

# The floor this run is drawn against. gemini-2.5-flash publishes 2,048; the arms are placed
# either side of it with enough room that a few tokens of drift cannot move an arm across.
FLOOR_TOKENS = 2_048

# What a request costs before any declarations: the user message and the request framing,
# with the system instruction counted separately because here it is the variable.
FRAMING_TOKENS = 30

# The arms' shape. `KEPT` is the declaration count every arm carries, and stands for what a
# menu leaves behind. `declared` carries enough more to match `padded`'s size.
KEPT_DECLARATIONS = 30
UNDER_HEAD_TOKENS = 300  # 300 + 30 + 30*45 = 1,680, comfortably under the floor
TARGET_OVER_TOKENS = 4_096  # where DESIGN.md 19.30 measured caching working

LOAD_SENDS = 12
LOAD_INTERVAL_S = 2.0
GAPS: tuple[float, ...] = (5.0, 60.0)

# Twenty probes per gap per arm. The first pilot ran six and could not tell anything from
# anything: one hit in six reads as 17%, which is the same number a genuinely rare cache and
# a coin that landed once both produce. Implicit caching is best-effort and DESIGN.md 19.30
# measured identical requests one second apart disagreeing four times in twenty-five, so any
# arm here needs enough calls that a couple of stray hits cannot carry it.
PROBES_PER_GAP = 20

# Probes inside a gap run back to back, so after the first they are measuring the steady
# state under about one send a second rather than the gap. That is deliberate: whether a
# prefix caches *at all* is the floor question, and residency over time is 19.30's question.
WARM = 0.50  # above this an arm is caching
COLD = 0.15  # below this it is not; between the two, say so rather than pick a side


@dataclass(frozen=True)
class Arm:
    """One prefix, built from a head size and a declaration count rather than a total.

    `probe_residency` varies the total and reaches it however it must. This varies *where the
    tokens come from*, which is the whole question, so the two knobs are separate here.
    """

    name: str
    namespace: str
    head_tokens: int
    declarations: int
    why: str

    @property
    def prefix_tokens(self) -> int:
        """The instruction and declarations: what two sessions of this agent would share."""
        return round(self.head_tokens + self.declarations * TOKENS_PER_DECLARATION)

    @property
    def prompt_tokens(self) -> int:
        return self.prefix_tokens + FRAMING_TOKENS

    @property
    def calls(self) -> int:
        return LOAD_SENDS + len(GAPS) * PROBES_PER_GAP

    @property
    def tokens(self) -> int:
        return self.calls * self.prompt_tokens


def extra_declarations() -> int:
    """How many declarations `declared` needs to match `padded`'s prefix without padding."""
    room = TARGET_OVER_TOKENS - UNDER_HEAD_TOKENS - FRAMING_TOKENS
    return max(KEPT_DECLARATIONS, round(room / TOKENS_PER_DECLARATION))


def padded_head_tokens() -> int:
    """How much instruction `padded` needs to reach the same size with the same declarations."""
    return round(TARGET_OVER_TOKENS - FRAMING_TOKENS - KEPT_DECLARATIONS * TOKENS_PER_DECLARATION)


def build_arms() -> list[Arm]:
    return [
        Arm(
            "under",
            "pu",
            UNDER_HEAD_TOKENS,
            KEPT_DECLARATIONS,
            "a pruned prefix under the floor: the control",
        ),
        Arm(
            "padded",
            "pp",
            padded_head_tokens(),
            KEPT_DECLARATIONS,
            "the same menu, instruction padded over the floor: the remedy",
        ),
        Arm(
            "declared",
            "pd",
            UNDER_HEAD_TOKENS,
            extra_declarations(),
            "the same size reached with declarations: the comparison",
        ),
    ]


def head(marker: str, tokens: int) -> str:
    """A private system instruction of about `tokens`.

    The marker is the first line, so two arms diverge at byte zero and neither can serve any
    part of the other. Everything after it is the same filler `probe_residency` uses, which is
    what makes this a padding probe: the padding is text that does nothing.
    """
    return f"PROBE ARM {marker}. Answer in one word. Do not call a tool.\n{head_body(tokens)}"


def tools(namespace: str, count: int) -> list[types.Tool]:
    return [types.Tool(function_declarations=[declaration(namespace, i) for i in range(count)])]


def run_arm(client: genai.Client, model: str, arm: Arm, run_id: str, sink: Sink) -> None:
    """Load the arm, then probe it after each gap. Gaps compound, as in `probe_residency`."""
    instruction = head(f"{run_id} {arm.name}", arm.head_tokens)
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
                block="P",
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

    probe_index = 0
    for gap in GAPS:
        time.sleep(max(0.0, gap - (time.monotonic() - last_at)))
        for _ in range(PROBES_PER_GAP):
            probe_index += 1
            send("probe", probe_index, gap, opens_gap=True)


def probes(calls: Sequence[Call], arm: str, gap: float | None = None) -> list[Call]:
    return [
        call
        for call in calls
        if call.arm == arm
        and call.role == "probe"
        and not call.error
        and (gap is None or call.planned_gap_s == gap)
    ]


def share(calls: Sequence[Call], arm: str, gap: float | None = None) -> float | None:
    found = probes(calls, arm, gap)
    return statistics.mean(call.share for call in found) if found else None


def hit_rate(calls: Sequence[Call], arm: str, gap: float | None = None) -> float | None:
    """The share of probes that cached *anything*, which is the question here.

    How much of a prefix cached is a size question and DESIGN.md 19.30 already answered it.
    Whether a prefix caches at all is what the floor decides, so a rate over calls is the
    statistic that matches the claim.
    """
    found = probes(calls, arm, gap)
    return sum(1 for call in found if call.cached_tokens) / len(found) if found else None


def print_table(calls: Sequence[Call], arms: Sequence[Arm]) -> None:
    print(f"{'arm':<10} {'target':>8} {'prompt':>8} {'gap':>6} {'n':>3} {'hit':>6} {'share':>7}")
    print("-" * 54)
    for arm in arms:
        for gap in GAPS:
            found = probes(calls, arm.name, gap)
            if not found:
                continue
            prompt = statistics.mean(call.prompt_tokens for call in found)
            print(
                f"{arm.name:<10} {arm.prompt_tokens:>8,} {prompt:>8,.0f} {gap:>6.0f}"
                f" {len(found):>3} {hit_rate(calls, arm.name, gap) or 0:>6.2f}"
                f" {share(calls, arm.name, gap) or 0:>7.3f}"
            )
    print()


def verdict(calls: Sequence[Call]) -> str:
    under = hit_rate(calls, "under")
    padded = hit_rate(calls, "padded")
    declared = hit_rate(calls, "declared")
    if under is None or padded is None or declared is None:
        return "INCONCLUSIVE — an arm returned no usable probe."

    if under > COLD:
        return (
            f"THE FLOOR IS NOT WHERE WE THINK — the under-floor arm cached on {under:.0%} of its"
            f" probes at a prefix of {build_arms()[0].prefix_tokens:,} tokens against a published"
            f" floor of {FLOOR_TOKENS:,}. Every band in qualify.py is drawn against that number."
        )
    if padded < COLD:
        return (
            f"PADDING DOES NOT WORK — padded cached on {padded:.0%} of its probes against"
            f" declared's {declared:.0%} at the same prefix size. qualify.py recommends padding"
            f" in two places and both are wrong; keeping tools is the only remedy left."
        )
    if declared > WARM and padded < declared / 2:
        return (
            f"PADDING IS SECOND BEST — padded {padded:.0%} against declared {declared:.0%} at the"
            f" same size. Recommend keeping tools first, and say padding is the weaker move."
        )
    return (
        f"PADDING WORKS — under {under:.0%}, padded {padded:.0%}, declared {declared:.0%}. Size is"
        f" what the floor reads and composition is not, so qualify.py's advice stands: an agent"
        f" whose prefix falls under the floor can pad its system instruction back over it."
    )


def estimate(arms: Sequence[Arm], model: str) -> tuple[int, float, float]:
    tokens = sum(arm.tokens for arm in arms)
    minutes = (LOAD_SENDS * LOAD_INTERVAL_S + sum(GAPS) + len(GAPS) * PROBES_PER_GAP) / 60.0
    return tokens, tokens / 1e6 * input_usd_per_mtok(model), minutes


def main() -> int:
    args = parse_args()
    arms = build_arms()
    tokens, usd, minutes = estimate(arms, args.model)

    print("Padding probe — does bulk that does nothing cache like bulk that does\n")
    for label, value in (
        ("project", args.project),
        ("location", args.location),
        ("model", args.model),
        ("run id", args.run_id),
        ("floor", f"{FLOOR_TOKENS:,} tokens"),
        ("auth", "Application Default Credentials, Vertex backend"),
    ):
        print(f"  {label:<10} {value}")

    print(f"\nPlan: {sum(arm.calls for arm in arms):,} fresh single-turn calls, one wave\n")
    for arm in arms:
        print(
            f"  {arm.name:<10} head {arm.head_tokens:>6,} + {arm.declarations:>3} declarations"
            f" = prefix {arm.prefix_tokens:>6,}, prompt {arm.prompt_tokens:>6,}"
            f"  ({arm.why})"
        )
    print(
        f"\n  {'TOTAL':<10} {tokens:>10,} input tokens, worst case ${usd:.2f};"
        f" about {minutes:.0f} minutes plus latency"
    )
    print(
        "\n  Read the prompt column of the run against the plan above. The arms only mean what"
        "\n  they say if padded and declared came out the same size."
    )

    if not args.confirm:
        print("\nNothing was called. Re-run with --confirm to spend the amount above.")
        return 2

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    print("\nrunning ...", file=sys.stderr)
    sink = Sink()
    with ThreadPoolExecutor(max_workers=len(arms)) as pool:
        for arm in arms:
            pool.submit(run_arm, client, args.model, arm, args.run_id, sink)
    observed = sink.calls

    print()
    print_table(observed, arms)
    print("=" * 78)
    print(f"CONCLUSION\n\n  {verdict(observed)}\n")

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
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
