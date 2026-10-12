"""Whether pruning can pay for an agent, answered before anyone provisions anything.

Every other measurement in this project is retrospective. `cache_watch` needs both arms of
live traffic, `report_ab.py` needs a finished experiment, and `scripts/predict_saving.py`
needs the agent's own logged turns with their arrival times. All three are better answers
than this one, and all three arrive after the user has enabled four APIs, created a dataset
and a bucket, accumulated a thousand labelled rows and trained a model. This module is the
answer available in the minute after `uv add`: two integers about the agent's prompt, no
cloud, no credentials, no model.

The question is not how many tools the agent has. `min_tools_to_route = 20` reads as if it
were the qualification bar and is not: it descends from `tokens_per_declaration = 200`, a
constant this project went on to measure at 37 and at 106, and it is a cheap per-request
early-out rather than an economic test. What decides whether pruning pays is how the
agent's whole prompt sits against the model's context-cache floor, and a tool count cannot
see that (`prompt_size` opens with the same argument at the other end of the pipeline).

The statistic
-------------
For an agent whose prompt is comfortably above the floor, the steady-state saving is

    r * D_removed / billed(P)        billed(P) = (1 - c) * P + r * c * P

with r = `cached_token_multiplier` (0.10 on Vertex) and c = 0.95, the share of a busy
agent's prompt that is cache-resident in the steady state (DESIGN.md 19.17).

The numerator is `r * D_removed`, not the difference of two `billed` values, and the
distinction is the whole arithmetic. Declarations sit at the head of the prompt and never
change inside a session, so they are entirely inside the cached prefix: every token a menu
removes was already billing at r, and the saving is exactly r times it. The (1 - c) term is
this turn's new history in absolute tokens, which a prune does not touch. `billed(P)` is
only there to turn a prompt size into a plausible bill for the denominator, and it is
checked against both agents this project has measured: 16,681 tokens gives 2,419 against
2,323 billed, and 4,002 gives 580 against 705 billed. It runs low on the small agent, which
makes the fraction read high, and the `thin` band is drawn wide enough to absorb that.

Two floors, not one
-------------------
The floor applies to whatever is being cached, and an agent has two different things that
cache. Inside a session, turn N+1's prompt begins with turn N's, so the whole prompt is the
cacheable prefix and the floor is judged against it. Across sessions, all that two sessions
share is the system instruction and the declarations, so *that* is the cacheable prefix and
the floor is judged against it alone. History cannot help, because no other session has it.

The probe of DESIGN.md 19.30 shows the second floor biting: its 2,048-token arm built a
prefix of 2,035 tokens, thirteen short, and it never cached once in six attempts across
three idle gaps, while every larger arm did. Nothing about that arm was different except
that its shared prefix did not clear the floor by itself.

That is why `qualify` takes `history_tokens` separately rather than folded into the prompt.
An agent with a 10,000-token history and a 3,000-token prefix that prunes to 1,500 passes
the within-session test comfortably and can never have a warm first turn, which is the turn
that cost 22 points of G1 on the one agent measured end to end.

The floor is not where it is published
--------------------------------------
Both floors are judged against `floor + CACHE_FLOOR_HEADROOM` rather than against the number
the model card publishes, because a prefix that merely clears the published minimum does not
cache. `scripts/probe_prefix_headroom.py` swept the margin over 720 probes on gemini-2.5-flash
(DESIGN.md 19.50): 2,281 tokens, 233 clear of a 2,048-token floor, cached on 1% of its reads,
which is what a prefix 518 tokens *under* the floor did. 2,581 tokens cached on 13% to 28%
across three runs. The published number says when a prefix becomes eligible to be cached; the
headroom is where being cached starts happening.

Two things follow, and the second matters as much as the first. Clearing the published floor
is not enough, which is what widened these tests. And clearing the headroom is not sufficient
either: no prefix in the sweep, at any size up to 6,132 tokens, opened warm more than 28% of
the time. Every band below is about whether a cache is *possible*, and `predict_saving.py` is
still the only thing that can say how often one actually happens.

The bands
---------
`trap`      pruning would carry a cacheable prompt under the floor. The only refusal, and
            the only band with direct evidence of loss: DESIGN.md 18.12 measured +25.5%.
`uncached`  the prompt is under the floor already, so nothing caches and every removed
            token saves at full rate. Proceeds, with a caveat: the win per token is the
            best there is and the absolute number is small, and no agent in this band has
            ever been measured (see the note on the absolute floor below).
`thin`      above the floor, but the saving is under `PAYS_ABOVE`, or the cross-session
            prefix would fall under it. Warns, never refuses.
`pays`      above the floor, saving at or above `PAYS_ABOVE`.

An agent with fewer tools than `max_tools_kept` removes nothing, and that is answered
before any of the four, as `thin` with a reason that says so. Running it through the bands
instead produces a true sentence that misleads: `thin` means the win is small, and there is
no win here to be small. It is a common answer rather than an edge case, because it is what
the smallest agents get, and they are the ones most likely to be asking.

The `trap` message advises keeping more tools until the thing that caches clears the floor by
itself. The cross-session message used to advise padding the system instruction as well, and
no longer does. §19.43 ran twelve sessions of a padded agent and it opened warm once in eleven
tries against a control's seven, so clearing the floor by 267 tokens is not enough to be
retained across sessions, and the padding cost 16.6% of the saving on every turn to buy that.
§19.33 is not contradicted — filler does cache like real content at the prefix level — it just
does not survive being put in a session at a margin this thin. Keeping more tools stays in the
advice because it is different in kind: it clears the floor with content the agent uses.

There are two ways a shared prefix ends up too small, and they get different answers. If the
prune is what carried it under, the verdict caps at `thin` and the message says so, because the
router chose that. If it was already under before any prune, nothing caps: turn one opens cold
for the routed and unrouted arm alike, so none of that cost belongs to the prune and capping
would charge the router for the agent's own shape. The message still says it, because a reader
choosing a menu size needs to know their prefix will never serve another session.

The cross-session case caps the verdict at `thin` rather than refusing, and the asymmetry is
deliberate. Losing the within-session prefix is measured, total and immediate, so it refuses.
Losing the cross-session one costs turn-one warmth and nothing else, which §19.43 measured
end to end: the pruned agent's turn one cached zero tokens in all twelve sessions, and it
still billed 43.4% less per turn than the unpruned one. A warning, not a refusal.

It costs exactly one turn, not two. The obvious worry is that turn two must be lost as well,
since turn two's cacheable prefix is turn one's prompt and that is the short one. §19.43 shows
turn two served by a 1,696-token stored chunk against a 2,048-token floor, so the floor does
not gate what may be stored. It gates whether an incoming request is eligible to hit, and every
turn after the first is over it.

`PAYS_ABOVE = 0.25` is a two-point calibration and nothing more: 12.9% on the agent that
lost money (25 tools, DESIGN.md 18.24, +22.4% whole run) and 34.0% on the agent that won
(100 tools, 19.17, -23.4%). Both of those divide by what the agent was really billed. Run
through this module, which has only a prompt size to divide by, the same two agents read
15.7% and 32.6%: the denominator's 18% error on the small agent pushes the loser up nearly
three points and the two anchors close on each other. 0.25 still sits between them, and
anywhere from about 0.18 to 0.32 would, so the bar is a statement about how much risk of a
null result is acceptable before spending rather than a measurement. It is set nearer the
winner because the cost of a wrong `pays` is a provisioned project and a corpus, and the
cost of a wrong `thin` is a warning someone reads and overrides.

What it cannot see, and who supersedes it
-----------------------------------------
Turn-one menu residency. An unrouted agent has one prefix that all its traffic keeps warm;
a routed agent has one per menu, and whether a session opens warm depends on when other
sessions ran. §19.17 measured a routed first turn caching 54.8% against a holdout's 88.6%,
and that gap was worth 22 points of G1 — more than the prune itself. Nothing derivable from
one prompt can see it, because it is a question about arrival times.

§19.30 gives that blind spot a shape, which is the most useful thing this module can say
about it without seeing any traffic. Residency is a short TTL and nothing else: a prefix
sent seconds ago opens warm, one idle for a minute is a coin flip, one idle for ten minutes
retains nothing, and no amount of send volume or cadence changes that. So the quantity that
decides whether a routed agent keeps its menus warm is sends per menu per minute, which is
turns per minute divided by menu count. Below about one, every session opens cold and the
verdict below is an upper bound. The same arithmetic says a quiet agent wants fewer, wider
menus and a busy one can afford more, narrower ones.

So `scripts/predict_saving.py` supersedes this module the moment the agent has logged
traffic with timestamps, and this module should be read as the step before that one rather
than as a cheaper version of it. It is also blind to the absolute size of the win: a
fraction says nothing about whether the saving covers the BigQuery, embedding and retrain
bill, which is a question about the agent's traffic volume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from adk_smart_router.config import MIN_CACHE_TOKENS, RouterConfig, min_cache_tokens

__all__ = ["Band", "Verdict", "qualify"]

Band = Literal["trap", "uncached", "thin", "pays"]

# The share of a steady-state prompt that is already cache-resident. Measured on the busy
# single-prefix arm of DESIGN.md 19.17. Only the denominator depends on it.
CACHE_SHARE = 0.95

# Where `thin` becomes `pays`. Two measured agents, 0.129 and 0.340 against their real
# bills and 0.157 and 0.326 through the arithmetic below; see the module docstring for why
# the bar sits nearer the winner and how little the evidence pins it.
PAYS_ABOVE = 0.25

# What to assume when `min_cache_tokens` does not recognise the model. The largest published
# floor, because that is the conservative direction: raising the assumed floor can only move
# an agent out of `pays` and into `trap` or `uncached`, never the other way, and a wrong
# `pays` is the expensive mistake (DESIGN.md 18.12).
UNKNOWN_MODEL_FLOOR = max(MIN_CACHE_TOKENS.values())

# How far over the published floor a prefix has to be before it is actually kept and served.
# `scripts/probe_prefix_headroom.py` swept the margin on gemini-2.5-flash over 720 probes
# (DESIGN.md 19.50): 2,281 tokens, 233 clear of a 2,048-token floor, cached on 1% of its reads,
# the same rate as a prefix half a thousand tokens *under* the floor. 2,581 tokens cached on
# 13% to 28%. The published minimum is where a prefix becomes eligible; this is where it starts
# happening. Applied to every model, on the same argument as UNKNOWN_MODEL_FLOOR: it was
# measured on one model, and the direction it errs in is refusing a prune rather than selling
# one. Nothing over the edge cached more than 28% of the time either, so clearing it buys the
# chance of a warm prefix and not the prefix.
CACHE_FLOOR_HEADROOM = 512


@dataclass(frozen=True)
class Verdict:
    """What one agent's prompt arithmetic says about pruning it."""

    band: Band
    # Fraction of a steady-state turn's input bill that stops being paid. Negative in the
    # `trap` band, where it is a ratio of input-token bills and not a whole-run G1: the one
    # measured trap came out at +25.5% of total spend (DESIGN.md 18.12) because output
    # tokens and turns that never cached dilute it. Read the sign, not the magnitude.
    steady_saving: float
    saved_tokens: int  # the same quantity in billed tokens per steady-state turn
    floor: int  # the model's published context-cache floor
    retained: int  # the size the bands were judged against: `floor` plus CACHE_FLOOR_HEADROOM
    reason: str

    @property
    def proceeds(self) -> bool:
        """False only for `trap`. Three of the four bands are worth going ahead on."""
        return self.band != "trap"


def qualify(
    model: str,
    prompt_tokens: int,
    removed_tokens: int,
    cfg: RouterConfig,
    history_tokens: int = 0,
) -> Verdict:
    """Judge one agent from the size of its prompt and the size of the menu's cut.

    Args:
        model: the model the agent runs on, used only to look up the cache floor.
        prompt_tokens: the agent's whole steady-state prompt with every tool declared —
            system instruction, history and declarations together. Not turn one: history
            only grows, and a prompt measured at turn one understates the denominator and
            so overstates the saving.
        removed_tokens: how many of those tokens a typical menu would drop.
        cfg: for `cached_token_multiplier`. Nothing else is read, and the plugin is
            deliberately not imported: this has to run before anything is configured.
        history_tokens: how much of `prompt_tokens` is conversation rather than system
            instruction and declarations. Only used to find the cross-session prefix, which
            is everything else. Left at zero, the second floor test compares the whole
            prompt to the floor, which is what the within-session test already did, so it
            costs nothing and claims nothing.

    Raises:
        ValueError: on sizes that cannot describe a real request. This is an install-time
            tool answering one question once, so a nonsensical input should stop it rather
            than be absorbed the way the serving path absorbs bad input.
    """
    if prompt_tokens <= 0:
        raise ValueError(f"prompt_tokens must be positive, got {prompt_tokens}")
    if removed_tokens < 0:
        raise ValueError(f"removed_tokens must not be negative, got {removed_tokens}")
    if removed_tokens > prompt_tokens:
        raise ValueError(
            f"removed_tokens ({removed_tokens}) exceeds the whole prompt ({prompt_tokens})"
        )
    if not 0 <= history_tokens <= prompt_tokens:
        raise ValueError(
            f"history_tokens ({history_tokens}) is not inside the prompt ({prompt_tokens})"
        )

    floor = min_cache_tokens(model) or UNKNOWN_MODEL_FLOOR
    assumed = "" if min_cache_tokens(model) else f" (floor assumed: {model} is not a known model)"
    # What every band below is actually judged against. See CACHE_FLOOR_HEADROOM: a prefix that
    # merely clears the published floor caches at the same rate as one under it.
    retained = floor + CACHE_FLOOR_HEADROOM
    pruned = prompt_tokens - removed_tokens

    if removed_tokens == 0:
        # Every band below is a statement about what the prune costs or earns, and there is
        # no prune. Answering it as `thin` with a saving of 0.0% is technically true and
        # tells the reader the wrong thing: `thin` means the win is small, and here there is
        # no win to be small. This is the ordinary answer for a small agent, because a menu
        # of `max_tools_kept` cannot drop anything from an agent with fewer tools than that.
        return Verdict(
            band="thin",
            steady_saving=0.0,
            saved_tokens=0,
            floor=floor,
            retained=retained,
            reason=(
                "there is nothing to prune. A menu of this size keeps every tool the agent "
                "declares, so the pruned prompt is the prompt and no arithmetic below it "
                "applies. That is the expected answer for an agent with fewer tools than the "
                "menu keeps, and it does not change until the tool list grows past that. The "
                "tool result cache is a separate feature and is unaffected."
            ),
        )

    # The cross-session prefix: everything two sessions of this agent have in common. See
    # "Two floors, not one" above. `removed_tokens` all come out of it, because declarations
    # are the only thing a menu drops.
    prefix = prompt_tokens - history_tokens
    cold_turn_one = prefix - removed_tokens < retained <= prefix
    # And the case where the prune is not what did it: the prefix was already too small to be
    # kept, so turn one opens cold whether the agent routes or not. That is worth saying and
    # is not worth capping the verdict for, because both arms pay it equally.
    cold_by_shape = prefix < retained
    billed = (1.0 - CACHE_SHARE) * prompt_tokens + cfg.cached_token_multiplier * (
        CACHE_SHARE * prompt_tokens
    )

    if pruned < retained <= prompt_tokens:
        # The prompt caches today and would not after the prune, so every kept token goes
        # back to billing at full rate. This is the one band the project has watched happen.
        lost = round(billed - pruned)
        return Verdict(
            band="trap",
            steady_saving=lost / billed,
            saved_tokens=lost,
            floor=floor,
            retained=retained,
            reason=(
                f"the full prompt of {prompt_tokens:,} tokens caches, but pruning it to "
                f"{pruned:,} lands under {retained:,} tokens{assumed}, so the whole prompt "
                f"goes back to full rate. That is the {floor:,}-token cache floor plus the "
                f"{CACHE_FLOOR_HEADROOM:,} tokens of margin a prefix needs before it is "
                f"actually kept, measured over 720 probes in scripts/probe_prefix_headroom.py. "
                f"Do not route this agent: the one agent measured in this band billed 25.5% "
                f"more (docs/scenarios.md#the-trap). Lengthen the system instruction, or keep "
                f"more tools, until the pruned prompt clears {retained:,} by itself."
            ),
        )

    if prompt_tokens < retained:
        return Verdict(
            band="uncached",
            steady_saving=removed_tokens / prompt_tokens,
            saved_tokens=removed_tokens,
            floor=floor,
            retained=retained,
            reason=(
                f"the whole prompt of {prompt_tokens:,} tokens is under {retained:,}{assumed} "
                f"— the {floor:,}-token cache floor plus the {CACHE_FLOOR_HEADROOM:,} tokens "
                f"of margin a prefix needs before it is actually kept "
                f"(scripts/probe_prefix_headroom.py) — so "
                f"nothing caches and each of the {removed_tokens:,} removed tokens saves at "
                f"full rate. That is the best per-token win there is, "
                f"and the smallest absolute one: judge it as {removed_tokens:,} tokens a turn "
                f"against what BigQuery, embedding and retraining cost you. No agent in this "
                f"band has been measured."
            ),
        )

    saved = round(cfg.cached_token_multiplier * removed_tokens)
    saving = saved / billed
    cold = (
        f" Separately, the {prefix:,} tokens two of this agent's sessions share drop to "
        f"{prefix - removed_tokens:,} once pruned, under the {retained:,} tokens a prefix needs "
        f"to be kept ({floor:,} published floor plus {CACHE_FLOOR_HEADROOM:,} of margin, "
        f"scripts/probe_prefix_headroom.py), so no "
        f"session can ever open on another session's cache. That costs turn one and only turn "
        f"one: measured over twelve sessions, a pruned agent in this state cached nothing at "
        f"all on turn one and still billed 43.4% less per turn than the same agent unpruned "
        f"(scripts/probe_cold_start.py). Padding the system instruction back over the floor "
        f"does not fix it and costs more than it earns, so do not. Keeping more tools does "
        f"clear the floor with content the agent uses, if turn-one latency matters more than "
        f"the tokens."
        if cold_turn_one
        else ""
    )
    shape = (
        f" Separately, the {prefix:,} tokens two of this agent's sessions share are already "
        f"under the {retained:,} a prefix needs to be kept ({floor:,} published floor plus "
        f"{CACHE_FLOOR_HEADROOM:,} of margin, scripts/probe_prefix_headroom.py), before any "
        f"prune. So turn one opens cold whether this agent routes or not, and none of that "
        f"cost belongs to the prune: the saving below is what you get, with no turn-one "
        f"penalty on top. What would change it is a longer system instruction or more "
        f"declared tools, both of which are decisions about the agent rather than about the "
        f"router."
        if cold_by_shape
        else ""
    )
    if saving < PAYS_ABOVE:
        return Verdict(
            band="thin",
            steady_saving=saving,
            saved_tokens=saved,
            floor=floor,
            retained=retained,
            reason=(
                f"pruning {removed_tokens:,} of {prompt_tokens:,} tokens saves about "
                f"{saving:.1%} of a steady-state turn, under the {PAYS_ABOVE:.0%} bar. The "
                f"agent that reads 15.7% here went on to bill 22.4% more over the whole run "
                f"(docs/scenarios.md), because a cold menu on turn one cost more than the prune "
                f"saved. Not a refusal — declare more tools, or measure your own traffic with "
                f"scripts/predict_saving.py before spending." + cold + shape
            ),
        )
    if cold_turn_one:
        return Verdict(
            band="thin",
            steady_saving=saving,
            saved_tokens=saved,
            floor=floor,
            retained=retained,
            reason=(
                f"pruning {removed_tokens:,} of {prompt_tokens:,} tokens saves about "
                f"{saving:.1%} of a steady-state turn, which clears the {PAYS_ABOVE:.0%} bar "
                f"on its own." + cold
            ),
        )

    return Verdict(
        band="pays",
        steady_saving=saving,
        saved_tokens=saved,
        floor=floor,
        retained=retained,
        reason=(
            f"pruning {removed_tokens:,} of {prompt_tokens:,} tokens saves about {saving:.1%} "
            f"of a steady-state turn, at or above the {PAYS_ABOVE:.0%} bar, and the pruned "
            f"prompt of {pruned:,} tokens still clears the {retained:,} a prefix needs to be "
            f"kept{assumed}. The agent that reads 32.6% here billed 23.4% less end to end "
            f"(docs/evidence.md). Confirm against your own logged traffic with "
            f"scripts/predict_saving.py, which can see turn-one menu residency and this "
            f"cannot." + shape
        ),
    )
