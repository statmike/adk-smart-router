#!/usr/bin/env python
"""Generate DESIGN.md's cost tables (2.2, 2.3, 11.7) instead of asserting them.

v0.1 of the design carried a growth-event table whose increments were
arithmetically impossible. Nobody noticed, because the numbers were typed rather
than computed. This script exists so that cannot happen again: every figure in
DESIGN.md 2.2, 2.3 and 11.7 comes out of the model below, and `--check` fails if
the model and the document drift apart.

Pure arithmetic. No network, no credentials, no GCP.

Policy P3 below prices a router that widens a session's menu mid-conversation.
The plugin no longer does that: growth events were deleted in DESIGN.md 19.26
because they were unreachable. P3 stays because it is the *argument* for the
prefix lock rather than a description of shipped behaviour — it is what
instability would have cost, and DESIGN.md 2.2 quotes it as exactly that. Read it
as a counterfactual, not as a policy anyone can select.

P5 is what P3 used to be for: the thing the shipped router actually risks. It is
P2 plus a miss rate. A miss now means the session wanted a tool its menu lacks
and never gets one, so the session is spent and yields nothing and the user asks
again. See `expected_with_misses`.

Usage:

    uv run python scripts/cost_model.py            # markdown to stdout
    uv run python scripts/cost_model.py --check    # exit 1 if the numbers moved

The billing model in one paragraph. A turn's input is a *prefix*
(system instruction + tool declarations) plus *history*. ADK creates an explicit
cache only once it has seen two consecutive turns with a matching fingerprint,
and the fingerprint is a SHA-256 over `config.tools` (DESIGN.md 18.1), so the
cache first serves on turn 2 of a stable run. On a serving turn, everything the
previous turn already sent is billed at the cached multiplier r, and only that
turn's new history is billed at full rate. Any change of tool *membership* resets
the clock; ordering is free, because the fingerprint sorts declarations by name.
"""

from __future__ import annotations

import argparse
import math
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

# --- Assumptions --------------------------------------------------------------
# These are the DESIGN.md 2.2 scenario parameters. Changing one changes every
# table below, which is the entire point of committing this file.


@dataclass(frozen=True)
class Scenario:
    """The session shape DESIGN.md 2.2 prices."""

    turns: int = 10
    tools_full: int = 50
    tools_kept: int = 5
    tokens_per_declaration: int = 200
    system_tokens: int = 2000
    history_growth_per_turn: int = 500
    cached_multiplier: float = 0.10


@dataclass(frozen=True)
class ModelRates:
    """Published USD per 1M tokens. Verified in DESIGN.md 18.1."""

    name: str
    input_usd: float
    cached_usd: float


MODELS: tuple[ModelRates, ...] = (
    ModelRates("Gemini 2.5 Flash", 0.30, 0.03),
    ModelRates("Gemini 2.5 Pro", 1.25, 0.125),
    ModelRates("Gemini 3 Flash", 0.50, 0.05),
    ModelRates("Gemini 3.1 Pro", 2.00, 0.20),
)

# Explicit-cache storage, USD per 1M tokens per hour. Only the 2.5 Flash figure
# is published in DESIGN.md 11.7; the others are extrapolated at the same
# multiple of the input rate, and the table says so.
STORAGE_USD_2_5_FLASH: float = 1.00
STORAGE_MULTIPLE_OF_INPUT: float = STORAGE_USD_2_5_FLASH / 0.30
DEFAULT_TTL_SECONDS: int = 1800  # ADK's ContextCacheConfig default

# The saving that DESIGN.md 2.1 treats as the qualification bar: pruning 50 tools
# to 5 on Flash with caching on. 2.3's floors are the tool counts at which each
# other regime reaches the same per-turn saving.
REFERENCE_SAVING_PER_TURN_USD: float = 0.00027
REFERENCE_TURNS_PER_MONTH: int = 176_000
# Back-derived from 2.1's own two numbers, not quoted independently. It is what
# the document implies one always-on Cloud Run instance costs per month.
IMPLIED_MONTHLY_INFRA_USD: float = REFERENCE_SAVING_PER_TURN_USD * REFERENCE_TURNS_PER_MONTH

# Two miss rates worth pricing P5 at. The first is what the shipped model measured on
# held-out sessions through its own menu pool (DESIGN.md 19.11); the second is the bar
# `publish_router.py --min-menu-recall` refuses below, so it is the worst thing that can be
# published rather than a measurement.
MEASURED_MENU_RECALL: float = 0.9895
PUBLISHED_MENU_RECALL_BAR: float = 0.90

# --- Recorded values: the contract between this script and DESIGN.md ----------
# `--check` recomputes everything and diffs against this block. If you change the
# model, update these AND the document in the same commit.

RECORDED: dict[str, float] = {
    "p0_total": 29100.0,
    "p1_total": 52500.0,
    "p2_total": 12000.0,
    "p3_min": 18730.0,
    "p3_max": 24750.0,
    "p4_total": 23700.0,
    "p1_ratio": 1.804,
    "p2_ratio": 0.412,
    "p4_ratio": 0.814,
    "steady_pruning_gain": 1.750,
    "steady_stability_gain": 6.250,
    "stability_over_pruning": 3.571,
    "floor_flash_cached": 50.0,
    "floor_pro_cached": 16.0,
    "floor_uncached": 10.0,
    "breakeven_turns_2_5_flash": 1.852,
    "floor_trap_penalty": 7.692,
    "p5_breakeven_miss_rate": 0.5876,
    "p5_at_measured_recall": 12127.0,
    "p5_at_publish_bar": 13333.0,
}
TOLERANCE: float = 0.005  # relative

# Places where the committed DESIGN.md prose still disagrees with this model.
# Clear an entry only by editing the document, never by editing the model.
KNOWN_DIVERGENCES: tuple[tuple[str, str, str], ...] = (
    (
        "DESIGN.md 2.2, P3 range",
        "~19,650-23,650",
        "18,730-24,750 (enumerated over all 36 placements of 2 growth events in turns 2-10)",
    ),
    (
        "DESIGN.md 2.3, Pro + caching floor",
        "~20 tools",
        "16 tools at the same $0.00027/turn reference that puts Flash at 50",
    ),
    (
        "DEV_NOTES 2026-08-30, P1 regression vs P2",
        "4.0x",
        "4.38x on session totals (52,500/12,000), or 6.25x at the steady-state turn",
    ),
)


# --- The model ----------------------------------------------------------------


@dataclass(frozen=True)
class TurnSpec:
    """One turn's tool situation.

    `fingerprint` stands in for ADK's SHA-256 over `config.tools`. Two turns
    cache-match when their fingerprints are equal, so a policy that keeps five
    tools but swaps which five must emit a fresh fingerprint every turn.
    """

    tool_count: int
    fingerprint: str


@dataclass(frozen=True)
class TurnCost:
    turn: int
    tool_count: int
    served_from_cache: bool
    billed_tokens: float


def simulate(specs: Sequence[TurnSpec], scenario: Scenario) -> list[TurnCost]:
    """Billed-equivalent input tokens per turn."""
    costs: list[TurnCost] = []
    for index, spec in enumerate(specs):
        prefix = scenario.system_tokens + spec.tool_count * scenario.tokens_per_declaration
        history_now = index * scenario.history_growth_per_turn
        hit = index > 0 and spec.fingerprint == specs[index - 1].fingerprint
        if hit:
            history_cached = history_now - scenario.history_growth_per_turn
            billed = (
                scenario.cached_multiplier * (prefix + history_cached)
                + scenario.history_growth_per_turn
            )
        else:
            billed = float(prefix + history_now)
        costs.append(TurnCost(index + 1, spec.tool_count, hit, billed))
    return costs


def total(specs: Sequence[TurnSpec], scenario: Scenario) -> float:
    return sum(cost.billed_tokens for cost in simulate(specs, scenario))


# --- Policies -----------------------------------------------------------------


def policy_p0(scenario: Scenario) -> list[TurnSpec]:
    """No router. Every tool declared, every turn, so the cache is stable."""
    return [TurnSpec(scenario.tools_full, "full") for _ in range(scenario.turns)]


def policy_p1(scenario: Scenario) -> list[TurnSpec]:
    """Re-decide every turn. Same size, different membership, so no cache is ever created."""
    return [TurnSpec(scenario.tools_kept, f"vary-{i}") for i in range(scenario.turns)]


def policy_p2(scenario: Scenario) -> list[TurnSpec]:
    """Decide on turn 1 and hold for the session."""
    return [TurnSpec(scenario.tools_kept, "kept") for _ in range(scenario.turns)]


def policy_p3(scenario: Scenario, growth_turns: Sequence[int]) -> list[TurnSpec]:
    """Monotonic union: start at the kept set, add one tool at each growth turn.

    `growth_turns` are 1-indexed. Turn 1 cannot be a growth event; it is the
    original decision.
    """
    specs: list[TurnSpec] = []
    count = scenario.tools_kept
    for turn in range(1, scenario.turns + 1):
        if turn in growth_turns:
            count += 1
        specs.append(TurnSpec(count, f"union-{count}"))
    return specs


def policy_p4(scenario: Scenario) -> list[TurnSpec]:
    """Pass through turn 1 unrouted, lock the kept set from turn 2."""
    head = [TurnSpec(scenario.tools_full, "full")]
    tail = [TurnSpec(scenario.tools_kept, "kept") for _ in range(scenario.turns - 1)]
    return head + tail


def expected_with_misses(routed: float, miss_rate: float) -> float:
    """What a routed session costs once some share of them miss and have to be redone.

    A miss is no longer a growth event. Nothing widens a menu mid-conversation, so the
    tool the session wanted is absent for the whole of it: the session is spent, yields
    nothing, and the user asks again. Attempts are independent, so the expected number of
    them is 1 / (1 - m).

    The pessimistic alternative — the retry runs unrouted, at `unrouted + m * unrouted` —
    gives a different curve and the *same* break-even, because both models are equal to the
    unrouted total at exactly the point where the geometric series is. So the break-even
    below does not depend on which of the two you believe, which is most of why it is worth
    quoting at all.
    """
    if not 0.0 <= miss_rate < 1.0:
        raise ValueError(f"miss_rate must be in [0, 1), got {miss_rate}")
    return routed / (1.0 - miss_rate)


def breakeven_miss_rate(routed: float, unrouted: float) -> float:
    """The miss rate at which routing stops being cheaper than not routing at all."""
    return 1.0 - routed / unrouted


def p3_placements(scenario: Scenario, growth_events: int) -> dict[tuple[int, ...], float]:
    """Every legal placement of N growth events, priced.

    The design quotes P3 as a range rather than a number because the total
    depends on *when* the set grows, not just how often. This enumerates it
    instead of guessing at the endpoints.
    """
    turns = range(2, scenario.turns + 1)
    return {
        placement: total(policy_p3(scenario, placement), scenario)
        for placement in combinations(turns, growth_events)
    }


def evenly_spaced(scenario: Scenario, growth_events: int) -> tuple[int, ...]:
    """A representative placement: growth events spread across the session."""
    step = scenario.turns / (growth_events + 1)
    return tuple(sorted({max(2, round(step * (k + 1))) for k in range(growth_events)}))


# --- Steady-state decomposition (DESIGN.md 2.2, final paragraph) ---------------


def _last_turn_billed(tool_count: int, cached: bool, scenario: Scenario) -> float:
    """What the final turn of the session costs, which is the steady state."""
    index = scenario.turns - 1
    prefix = scenario.system_tokens + tool_count * scenario.tokens_per_declaration
    if not cached:
        return float(prefix + index * scenario.history_growth_per_turn)
    history_cached = (index - 1) * scenario.history_growth_per_turn
    return scenario.cached_multiplier * (prefix + history_cached) + scenario.history_growth_per_turn


@dataclass(frozen=True)
class Decomposition:
    full_cached: float
    kept_cached: float
    kept_uncached: float
    pruning_gain: float
    stability_gain: float
    stability_over_pruning: float


def decompose(scenario: Scenario) -> Decomposition:
    full_cached = _last_turn_billed(scenario.tools_full, True, scenario)
    kept_cached = _last_turn_billed(scenario.tools_kept, True, scenario)
    kept_uncached = _last_turn_billed(scenario.tools_kept, False, scenario)
    pruning = full_cached / kept_cached
    stability = kept_uncached / kept_cached
    return Decomposition(
        full_cached, kept_cached, kept_uncached, pruning, stability, stability / pruning
    )


# --- DESIGN.md 2.3: the qualification floor -----------------------------------


@dataclass(frozen=True)
class Regime:
    name: str
    rate_usd_per_mtok: float
    note: str


REGIMES: tuple[Regime, ...] = (
    Regime("Gemini Flash + context caching", 0.03, "Weakest regime: pruned tokens bill at 0.03."),
    Regime("Gemini Pro + context caching", 0.125, "4.2x the per-token saving."),
    Regime("No context caching in play", 0.30, "10x the saving, but 7's locking buys nothing."),
)


def saving_per_turn_usd(tools: int, regime: Regime, scenario: Scenario) -> float:
    """USD saved per turn by pruning `tools` down to the kept set, at this regime's rate."""
    removed = max(0, tools - scenario.tools_kept) * scenario.tokens_per_declaration
    return removed * regime.rate_usd_per_mtok / 1_000_000


def qualification_floor(regime: Regime, scenario: Scenario, target_usd: float) -> int:
    """Smallest tool count whose per-turn saving reaches `target_usd`.

    Closed form, then a ceiling: the saving is linear in the tool count.
    """
    per_tool = scenario.tokens_per_declaration * regime.rate_usd_per_mtok / 1_000_000
    return scenario.tools_kept + math.ceil(target_usd / per_tool)


# --- DESIGN.md 11.7: explicit-cache storage break-even ------------------------


def storage_rate(model: ModelRates) -> float:
    """USD per 1M cached tokens per hour."""
    return model.input_usd * STORAGE_MULTIPLE_OF_INPUT


def breakeven_cached_turns(model: ModelRates, ttl_seconds: int) -> float:
    """Cache reads needed inside one TTL before explicit caching pays for its storage.

    Storage bills for the whole TTL whether or not anyone reads the cache, so the
    break-even is (storage for one TTL) / (saving per read). The cached token
    count cancels out, which is why this is a pure turn count.
    """
    storage = storage_rate(model) * (ttl_seconds / 3600.0)
    saving_per_read = model.input_usd - model.cached_usd
    return storage / saving_per_read


# --- DESIGN.md 7.3 cross-check: the minimum-cacheable-prefix trap -------------


def floor_trap_penalty(
    system_tokens: int, tools_before: int, tools_after: int, floor_tokens: int, scenario: Scenario
) -> tuple[int, int, float]:
    """Price the case where pruning drops the prefix under the model's cache floor.

    Returns (unrouted prefix, pruned prefix, how much worse the router is).
    """
    before = system_tokens + tools_before * scenario.tokens_per_declaration
    after = system_tokens + tools_after * scenario.tokens_per_declaration
    billed_before = before * scenario.cached_multiplier if before >= floor_tokens else float(before)
    billed_after = after * scenario.cached_multiplier if after >= floor_tokens else float(after)
    return before, after, billed_after / billed_before


# --- Rendering ----------------------------------------------------------------


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def invocation_line() -> str:
    return "uv run python " + shlex.join(["scripts/cost_model.py", *sys.argv[1:]])


def render(scenario: Scenario) -> str:
    out: list[str] = []
    add = out.append

    add("# Cost model output")
    add("")
    add("Generated by `scripts/cost_model.py`. Do not hand-edit the tables below;")
    add("re-run the script and paste. `--check` fails if DESIGN.md and this model diverge.")
    add("")
    add("```")
    add(invocation_line())
    add("```")
    add("")

    add("## Assumptions")
    add("")
    add(
        md_table(
            ["Parameter", "Value", "Source"],
            [
                ["Turns per session", str(scenario.turns), "DESIGN.md 2.2"],
                ["Tools declared, unrouted", str(scenario.tools_full), "DESIGN.md 2.2"],
                ["Tools kept after pruning", str(scenario.tools_kept), "`max_tools_kept`, D3"],
                [
                    "Tokens per declaration",
                    str(scenario.tokens_per_declaration),
                    "DESIGN.md 2.2",
                ],
                ["System prompt tokens", f"{scenario.system_tokens:,}", "DESIGN.md 2.2"],
                [
                    "History growth per turn",
                    f"+{scenario.history_growth_per_turn}",
                    "DESIGN.md 2.2",
                ],
                [
                    "Cached-token multiplier r",
                    f"{scenario.cached_multiplier:.2f}",
                    "DESIGN.md 18.1, DEV_NOTES 2026-08-30",
                ],
                [
                    "First cache-serving turn",
                    "2",
                    "ADK creates a cache after two matching fingerprints (18.1)",
                ],
            ],
        )
    )
    add("")
    add("Three assumptions carry most of the weight, so state them plainly.")
    add("")
    add("1. The cached portion of a turn is everything the previous turn already sent:")
    add("   system prompt, tool declarations and history up to the previous turn. Only")
    add("   that turn's new history is billed at full rate.")
    add("2. A cache first serves on turn 2 of a stable run, not turn 3. ADK's probe in")
    add("   18.1 measured 7 of 8 turns served on a stable set, which is turns 2 through 8.")
    add("3. Only tool *membership* matters. Ordering is free, because the fingerprint")
    add("   sorts `function_declarations` by name.")
    add("")

    # --- 2.2 ---
    p0 = total(policy_p0(scenario), scenario)
    p1 = total(policy_p1(scenario), scenario)
    p2 = total(policy_p2(scenario), scenario)
    p4 = total(policy_p4(scenario), scenario)
    placements = p3_placements(scenario, growth_events=2)
    p3_min = min(placements.values())
    p3_max = max(placements.values())
    p3_min_at = min(placements, key=lambda k: placements[k])
    p3_max_at = max(placements, key=lambda k: placements[k])
    p3_even = evenly_spaced(scenario, 2)
    p3_typical = placements[p3_even]

    add("## DESIGN.md 2.2 — billed-equivalent input tokens, 10-turn session")
    add("")
    add(
        md_table(
            ["Policy", "Total", "vs. no router"],
            [
                [f"P0 - no router, {scenario.tools_full} tools", f"{p0:,.0f}", "1.00x"],
                [
                    f"P1 - re-decide every turn ({scenario.tools_kept} tools, set varies)",
                    f"{p1:,.0f}",
                    f"{p1 / p0:.2f}x - a regression",
                ],
                ["P2 - decide on turn 1, hold for the session", f"{p2:,.0f}", f"{p2 / p0:.2f}x"],
                [
                    "P3 - monotonic union, grows twice",
                    f"{p3_min:,.0f}-{p3_max:,.0f}",
                    f"{p3_min / p0:.2f}-{p3_max / p0:.2f}x",
                ],
                ["P4 - pass-through turn 1, lock on turn 2", f"{p4:,.0f}", f"{p4 / p0:.2f}x"],
            ],
        )
    )
    add("")
    add(
        f"P3 is a range because the total depends on *when* the set grows. Cheapest is"
        f" growth at turns {list(p3_min_at)} ({p3_min:,.0f}); dearest is turns"
        f" {list(p3_max_at)} ({p3_max:,.0f}); evenly spaced at turns {list(p3_even)} costs"
        f" {p3_typical:,.0f}. Late growth is worse than early growth, because a growth event"
        f" pays full rate on a history that has already grown."
    )
    add("")

    add("### P3 growth-event detail")
    add("")
    add(
        md_table(
            ["Growth turns", "Total", "vs. no router", "vs. P2"],
            [
                [
                    str(list(placement)),
                    f"{value:,.0f}",
                    f"{value / p0:.2f}x",
                    f"{value / p2:.2f}x",
                ]
                for placement, value in sorted(placements.items(), key=lambda kv: kv[1])[:6]
            ],
        )
    )
    add("")
    add("Six cheapest placements, sorted. The full enumeration is 36 placements.")
    add("")

    # --- P5 ---
    breakeven = breakeven_miss_rate(p2, p0)
    add("### P5 — P2 with misses, which is what the shipped router risks")
    add("")
    add(
        "A miss is a session that wanted a tool its menu lacks. Nothing widens a menu"
        " mid-conversation any more, so the session is spent, yields nothing, and the user"
        " asks again."
    )
    add("")
    miss_rows: list[list[str]] = []
    for label, rate in (
        ("perfect recall", 0.0),
        (f"measured recall {MEASURED_MENU_RECALL:.4f}", 1.0 - MEASURED_MENU_RECALL),
        (f"the publish bar {PUBLISHED_MENU_RECALL_BAR:.2f}", 1.0 - PUBLISHED_MENU_RECALL_BAR),
        ("break-even", breakeven),
    ):
        expected = expected_with_misses(p2, rate)
        miss_rows.append([label, f"{rate:.4f}", f"{expected:,.0f}", f"{expected / p0:.2f}x"])
    add(md_table(["Miss rate", "m", "Expected total", "vs. no router"], miss_rows))
    add("")
    add(
        f"**Routing survives a miss rate of {breakeven:.1%}.** That is the useful number and it"
        f" is uncomfortably large: priced in tokens alone, the shipped model could miss half its"
        f" sessions and still be cheaper than declaring everything. The measured miss rate is"
        f" {1 - MEASURED_MENU_RECALL:.2%}, fifty times inside the bound, and even the worst"
        f" artifact the publish gate will pass sits five times inside it."
    )
    add("")
    add(
        "So the reason to hold recall high is not this table. A miss is a user who did not get"
        " an answer, and no token model prices that. `pruner.worth_pruning` charges a miss"
        " `miss_cost_tokens` (1,600 by default) against a session's saving, which is far more"
        " than the arithmetic here would justify, and DESIGN.md 19.12 says so: the dial is set"
        " where the recall and saving criteria both clear, not where the tokens balance."
    )
    add("")

    add("### Per-turn trace, P2 against P0")
    add("")
    p0_trace = simulate(policy_p0(scenario), scenario)
    p2_trace = simulate(policy_p2(scenario), scenario)
    add(
        md_table(
            ["Turn", "P0 billed", "P0 cached?", "P2 billed", "P2 cached?"],
            [
                [
                    str(a.turn),
                    f"{a.billed_tokens:,.0f}",
                    "yes" if a.served_from_cache else "no",
                    f"{b.billed_tokens:,.0f}",
                    "yes" if b.served_from_cache else "no",
                ]
                for a, b in zip(p0_trace, p2_trace, strict=True)
            ],
        )
    )
    add("")

    add("### Steady-state decomposition")
    add("")
    dec = decompose(scenario)
    add(
        md_table(
            ["Final-turn condition", "Billed tokens"],
            [
                [f"{scenario.tools_full} tools, cache held", f"{dec.full_cached:,.0f}"],
                [f"{scenario.tools_kept} tools, cache held", f"{dec.kept_cached:,.0f}"],
                [f"{scenario.tools_kept} tools, no cache", f"{dec.kept_uncached:,.0f}"],
            ],
        )
    )
    add("")
    add(
        f"Pruning {scenario.tools_full} to {scenario.tools_kept} with the cache held is worth"
        f" **{dec.pruning_gain:.2f}x**. Holding the cache with the prune size fixed is worth"
        f" **{dec.stability_gain:.2f}x**. Cache stability is worth"
        f" **{dec.stability_over_pruning:.2f}x more than pruning itself.** The router's primary"
        f" job is not to prune small, it is to prune once."
    )
    add("")

    # --- 2.3 ---
    add("## DESIGN.md 2.3 — qualification floor")
    add("")
    add(
        f"A regime qualifies at the tool count where pruning to {scenario.tools_kept} saves at"
        f" least the reference **${REFERENCE_SAVING_PER_TURN_USD:.5f}/turn**, which is what"
        f" Flash-plus-caching saves at {scenario.tools_full} tools (DESIGN.md 2.1). At"
        f" {REFERENCE_TURNS_PER_MONTH:,} turns/month that pays for roughly one always-on Cloud"
        f" Run instance, an implied **${IMPLIED_MONTHLY_INFRA_USD:,.2f}/month**. That figure is"
        f" back-derived from 2.1's own two numbers, not quoted independently."
    )
    add("")
    floors = {
        regime.name: qualification_floor(regime, scenario, REFERENCE_SAVING_PER_TURN_USD)
        for regime in REGIMES
    }
    add(
        md_table(
            ["Regime", "Marginal rate $/1M", "Cost-positive above", "Saving/turn at that count"],
            [
                [
                    regime.name,
                    f"{regime.rate_usd_per_mtok:.3f}",
                    f"~{floors[regime.name]} tools",
                    f"${saving_per_turn_usd(floors[regime.name], regime, scenario):.5f}",
                ]
                for regime in REGIMES
            ],
        )
    )
    add("")
    add(f"### Monthly volume needed at {scenario.tools_full} tools")
    add("")
    volume_rows: list[list[str]] = []
    for regime in REGIMES:
        saving = saving_per_turn_usd(scenario.tools_full, regime, scenario)
        volume_rows.append(
            [regime.name, f"${saving:.5f}", f"{IMPLIED_MONTHLY_INFRA_USD / saving:,.0f}"]
        )
    add(md_table(["Regime", "Saving/turn", "Turns/month to cover infra"], volume_rows))
    add("")
    add("Below the floor, recommend the memo cache (DESIGN.md 8) alone, or nothing.")
    add("")

    # --- 11.7 ---
    add("## DESIGN.md 11.7 — explicit-cache storage break-even")
    add("")
    add(
        f"Explicit caching bills storage for the whole TTL whether or not the cache is read."
        f" Break-even is (storage for one TTL) / (saving per read). The cached token count"
        f" cancels, so the answer is a pure turn count. Storage on 2.5 Flash is"
        f" ${STORAGE_USD_2_5_FLASH:.2f}/1M/hour, **{STORAGE_MULTIPLE_OF_INPUT:.2f}x the input"
        f" rate**; the other models assume that same multiple, because only the 2.5 Flash"
        f" figure is published."
    )
    add("")
    add(
        md_table(
            [
                "Model",
                "Input $/1M",
                "Cached $/1M",
                "Storage $/1M/hr",
                f"Break-even reads @ {DEFAULT_TTL_SECONDS}s",
            ],
            [
                [
                    model.name,
                    f"{model.input_usd:.3f}",
                    f"{model.cached_usd:.3f}",
                    f"{storage_rate(model):.2f}",
                    f"{breakeven_cached_turns(model, DEFAULT_TTL_SECONDS):.2f}",
                ]
                for model in MODELS
            ],
        )
    )
    add("")
    add(
        "The break-even is identical across models because every one of them prices cached"
        " input at exactly 0.10x input, and the storage assumption is proportional too. If a"
        " model ever breaks that ratio, this table stops being flat and the assumption above"
        " needs replacing with a published number."
    )
    add("")
    add("### TTL sensitivity, Gemini 2.5 Flash")
    add("")
    flash = MODELS[0]
    add(
        md_table(
            ["TTL seconds", "Storage per 1M cached tokens", "Break-even cache reads"],
            [
                [
                    str(ttl),
                    f"${storage_rate(flash) * ttl / 3600.0:.4f}",
                    f"{breakeven_cached_turns(flash, ttl):.2f}",
                ]
                for ttl in (300, 600, 900, DEFAULT_TTL_SECONDS, 3600)
            ],
        )
    )
    add("")
    add(
        f"At the default {DEFAULT_TTL_SECONDS}s TTL a session needs"
        f" {breakeven_cached_turns(flash, DEFAULT_TTL_SECONDS):.2f} cache reads to break even,"
        " so 2 reads, so a 3-turn session: turn 1 creates the cache, turns 2 and 3 read it."
        " **An agent whose sessions run 2 turns then go idle pays more with explicit caching"
        " than without.** Do not enable explicit caching below the model's break-even turn"
        " count. The converse is an unclaimed benefit: pruning reduces cached tokens, so it"
        f" reduces storage spend at a {STORAGE_MULTIPLE_OF_INPUT:.2f}x larger per-token lever"
        " than the input rate."
    )
    add("")

    # --- 7.3 cross-check ---
    add("## DESIGN.md 7.3 cross-check — the minimum-cacheable-prefix trap")
    add("")
    add(
        "Pruning is not monotonically good. The same arithmetic prices the case where pruning"
        " pushes a cacheable prefix under the model's floor."
    )
    add("")
    rows: list[list[str]] = []
    for label, floor_tokens, before_tools, after_tools, sys_tokens in (
        ("Gemini 2.5 (floor 2,048)", 2048, 8, 5, 1000),
        ("Gemini 3 (floor 4,096)", 4096, 18, 5, 1000),
    ):
        before, after, penalty = floor_trap_penalty(
            sys_tokens, before_tools, after_tools, floor_tokens, scenario
        )
        rows.append(
            [
                label,
                f"{before_tools} -> {after_tools}",
                f"{before:,} -> {after:,}",
                f"{before * scenario.cached_multiplier:,.0f} -> {after:,}",
                f"{penalty:.2f}x worse",
            ]
        )
    add(
        md_table(
            ["Model", "Tools", "Prefix tokens", "Billed-equivalent", "Router penalty"],
            rows,
        )
    )
    add("")
    add(
        "On Gemini 3 the floor is twice as wide, so the trap catches agents up to about 20"
        " tools. This is why `pruner` passes through unrouted when the pruned prefix would"
        " fall below a floor the live prefix clears."
    )
    add("")

    if KNOWN_DIVERGENCES:
        add("## Still diverging from the committed DESIGN.md text")
        add("")
        add(
            "The model is the source of truth. These entries are cleared by editing the"
            " document, never by editing the model."
        )
        add("")
        add(
            md_table(
                ["Where", "Document says", "Model computes"],
                [list(row) for row in KNOWN_DIVERGENCES],
            )
        )
        add("")

    return "\n".join(out)


# --- --check ------------------------------------------------------------------


def computed_values(scenario: Scenario) -> dict[str, float]:
    p0 = total(policy_p0(scenario), scenario)
    p1 = total(policy_p1(scenario), scenario)
    p2 = total(policy_p2(scenario), scenario)
    p4 = total(policy_p4(scenario), scenario)
    placements = p3_placements(scenario, growth_events=2)
    dec = decompose(scenario)
    floors = {
        regime.name: qualification_floor(regime, scenario, REFERENCE_SAVING_PER_TURN_USD)
        for regime in REGIMES
    }
    _, _, penalty = floor_trap_penalty(1000, 8, 5, 2048, scenario)
    return {
        "p5_breakeven_miss_rate": breakeven_miss_rate(p2, p0),
        "p5_at_measured_recall": expected_with_misses(p2, 1.0 - MEASURED_MENU_RECALL),
        "p5_at_publish_bar": expected_with_misses(p2, 1.0 - PUBLISHED_MENU_RECALL_BAR),
        "p0_total": p0,
        "p1_total": p1,
        "p2_total": p2,
        "p3_min": min(placements.values()),
        "p3_max": max(placements.values()),
        "p4_total": p4,
        "p1_ratio": p1 / p0,
        "p2_ratio": p2 / p0,
        "p4_ratio": p4 / p0,
        "steady_pruning_gain": dec.pruning_gain,
        "steady_stability_gain": dec.stability_gain,
        "stability_over_pruning": dec.stability_over_pruning,
        "floor_flash_cached": float(floors["Gemini Flash + context caching"]),
        "floor_pro_cached": float(floors["Gemini Pro + context caching"]),
        "floor_uncached": float(floors["No context caching in play"]),
        "breakeven_turns_2_5_flash": breakeven_cached_turns(MODELS[0], DEFAULT_TTL_SECONDS),
        "floor_trap_penalty": penalty,
    }


def run_check(scenario: Scenario) -> int:
    """Recompute everything and diff against RECORDED. Returns a process exit code."""
    actual = computed_values(scenario)
    drift: list[str] = []
    print(f"cost_model --check   tolerance {TOLERANCE:.1%} relative")
    print()
    for key in sorted(RECORDED):
        expected = RECORDED[key]
        got = actual[key]
        scale = abs(expected) if expected else 1.0
        ok = abs(got - expected) / scale <= TOLERANCE
        status = "ok  " if ok else "DRIFT"
        print(f"  {status}  {key:<28} recorded={expected:<12,.4g} got={got:,.4g}")
        if not ok:
            drift.append(key)
    missing = sorted(set(actual) - set(RECORDED))
    for key in missing:
        print(f"  DRIFT  {key:<28} not recorded")
        drift.append(key)
    print()
    if drift:
        print(f"FAIL: {len(drift)} value(s) drifted. Update RECORDED and DESIGN.md together.")
        return 1
    print("PASS: the model and the recorded values agree.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate DESIGN.md's cost tables (2.2, 2.3, 11.7). Pure arithmetic."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Recompute and exit non-zero if the numbers drifted from the recorded block.",
    )
    parser.add_argument("--turns", type=int, default=Scenario.turns, help="Turns per session.")
    parser.add_argument(
        "--tools", type=int, default=Scenario.tools_full, help="Tools declared unrouted."
    )
    parser.add_argument(
        "--kept", type=int, default=Scenario.tools_kept, help="Tools kept after pruning."
    )
    args = parser.parse_args()

    scenario = Scenario(turns=args.turns, tools_full=args.tools, tools_kept=args.kept)
    if args.check:
        if scenario != Scenario():
            print("--check only validates the default DESIGN.md 2.2 scenario.", file=sys.stderr)
            return 2
        return run_check(scenario)
    print(render(scenario))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
