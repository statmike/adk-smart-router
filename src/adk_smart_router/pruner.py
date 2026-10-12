"""The session-stable pruning rule. Pure, total, deterministic.

Zero I/O, zero logging, zero clock, zero randomness, zero ADK. Everything here is
a function of its arguments, which is what makes the rule provable in unit tests.

The one non-obvious rule: `kept=None` means "send the full live tool set this
turn", not "keep nothing". Uncertainty always widens the aperture, it never
narrows it (DESIGN.md 7.2, rule 2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from adk_smart_router.config import FRAMEWORK_TOOLS, RouterConfig
from adk_smart_router.menus import PredictionLike, rank

__all__ = [
    "ABSTAIN",
    "MASS_TOLERANCE",
    "PASSTHROUGH_SOURCES",
    "PredictionLike",
    "SessionDecision",
    "breaks_cache",
    "contained_mass",
    "decide",
    "too_few_tools",
    "top_k_by_mass",
    "worth_pruning",
]

# Sources that mean "the routing rule ran and did not prune this turn". The caller uses this
# to decide whether a decision is worth acting on. "floor" and "watch" are the two cache
# guards, kept apart because they mean different things to whoever is reading
# (DESIGN.md 18.21); neither is ever produced by `decide`, which cannot see either.
# "late" is the same shape: the plugin's prefix lock, refusing to prune a session
# that has already sent a full list (DESIGN.md 18.22).
#
# Not every unpruned session lands here, and the difference matters when reading the
# warehouse. A holdout session, a router with no artifact, and a session whose embedding
# never came back are all unpruned too, and none of them reached this rule to be recorded by
# it. `provenance.DECISION_SOURCES` is the whole vocabulary and
# `provenance.ROUTED_SOURCES` is the two that mean a prune happened, which is what every
# query filters on. This set is one of the four parts they divide into, and
# `tests/test_decision_sources.py` asserts the division.
#
# There was an eighth, "budget", the terminal pin-to-full once a session had spent its
# growth events. Growth is gone (see `decide`), so "budget" is gone with it. It never
# occurred in the warehouse, so no historical row carries it and nothing that reads
# provenance has to keep accepting it.
PASSTHROUGH_SOURCES: frozenset[str] = frozenset(
    {"lowconf", "floor", "watch", "late", "noop", "small", "cheap"}
)

# The classifier's "no opinion" label. It is a training label, never an action
# (D10): as an action it collapses to the empty set on a mature model.
ABSTAIN = "ABSTAIN"

# Slack on the cumulative-mass comparison below, and on the same comparison written out
# in `sql/train_router.sql`.
#
# The boundary is hit exactly more often than it looks, because probability vectors that
# sum to round numbers are common. Three different summations decide it: the running
# accumulation in `top_k_by_mass`, BigQuery's `SUM` in the training SQL, and Python's
# `sum` in the test that cross-checks the two. All three are the same arithmetic in a
# different order, so they disagree in the last bit. CPython 3.12 made that visible by
# giving `sum` compensated summation: a ranking whose exact cumulative mass is 0.9
# started keeping one more tool here than the SQL scored as kept, which is precisely the
# disagreement `coverage_mass` exists to rule out (DESIGN.md 19.74).
#
# 1e-9 is the tolerance `startup_checks.artifact_skew` already applies to this number.
# It is far above any accumulation error over a few hundred terms and far below any
# difference between two probabilities that means anything.
MASS_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SessionDecision:
    """One session's routing state. Stored per session, replaced on every turn."""

    kept: frozenset[str] | None  # None means "send the full live set" this turn
    low_conf_turns: int = 0
    terminal_passthrough: bool = False
    # fresh | reused | cheap | lowconf | floor | noop | small
    source: str = "fresh"


# --- Helpers ------------------------------------------------------------------


def too_few_tools(live: frozenset[str], cfg: RouterConfig) -> bool:
    """Below the qualification floor, where the router costs more than it saves.

    Lifted out of `decide` so the plugin can ask it before paying for an embedding, which is
    what `min_tools_to_route` is documented to avoid. `decide` still asks it too, because the
    rule belongs to the rule and not to the caller that happens to short-circuit on it.
    """
    return len(live) < cfg.min_tools_to_route


def top_k_by_mass(preds: Sequence[PredictionLike], mass: float, cap: int) -> tuple[str, ...]:
    """Labels in descending probability until cumulative mass is reached, capped at `cap`.

    Ties break by label ascending so the same prediction vector always yields the
    same set, which is what keeps the prefix cache stable across replicas. The mass
    comparison carries `MASS_TOLERANCE` for the same reason: so that this rule and the
    one the training SQL scores agree on a ranking that lands exactly on the threshold.
    """
    if not preds:
        return ()
    ranked = sorted(preds, key=lambda p: (-p.probability, p.label))
    # A cap below 1 would return the empty set, and an empty tool list is the one
    # thing the pruner must never produce (DESIGN.md 7.2, rule 3).
    limit = max(1, cap)

    chosen: list[str] = []
    cumulative = 0.0
    for pred in ranked[:limit]:
        chosen.append(pred.label)
        cumulative += pred.probability
        if cumulative >= mass - MASS_TOLERANCE:
            break
    return tuple(chosen)


def breaks_cache(full_tokens: int, pruned_tokens: int, floor: int | None) -> bool:
    """True when pruning would drop a cacheable prompt under the model's cache floor.

    Both counts are the **whole prompt**: system instruction, conversation history and
    declarations together. Feeding this a tool-prefix estimate instead is the mistake
    that made the router more expensive than doing nothing (DESIGN.md 18.12), and it is
    why the two numbers are passed in rather than derived from a config constant here.

    Deliberately not the whole story. Cache floors are published and then not obeyed, so
    a request above the floor may not cache and one below it may. `cache_watch.CacheWatch`
    is the half of the guard that learns what actually happens; this half is the prior
    that has to hold before there is anything to learn from.
    """
    if floor is None:
        return False  # unknown model: no floor to reason about, so no opinion
    if pruned_tokens >= floor:
        return False  # still large enough to cache after the prune
    return full_tokens >= floor


def contained_mass(preds: Sequence[PredictionLike], kept: frozenset[str]) -> float:
    """Probability the session only wants tools that survived the prune.

    ABSTAIN counts as contained, and that is the point rather than a special case. A
    session the model is confident needs no tool at all is a session where every removal
    is free and nothing can be missed.
    """
    return sum(p.probability for p in preds if p.label == ABSTAIN or p.label in kept)


def worth_pruning(saved_tokens: int, mass: float, cfg: RouterConfig) -> bool:
    """Whether pruning to a menu pays, at this session's odds (DESIGN.md 19.5).

        prune iff  cached_rate * tokens removed * turns  >  P(miss) * what a miss costs

    Both sides are billed tokens over the session. The left is certain and small: the
    declarations removed are already billed at the cached rate, so a prune only ever wins
    a tenth of what it removes, every turn. The right is a gamble: `1 - mass` is the
    chance the session reaches for something that is no longer declared, and a miss costs
    a whole cold prefix.

    The counts are passed in rather than derived here for the same reason `breaks_cache`
    takes them: this module owns the rule and the caller owns the measurement, because
    only the caller can see the request the tokens are actually in.
    """
    saving = cfg.cached_token_multiplier * saved_tokens * cfg.expected_session_turns
    return saving > (1.0 - mass) * cfg.miss_cost_tokens


def _passthrough(prior: SessionDecision | None, source: str) -> SessionDecision:
    """Send the full live set this turn, carrying the terminal flag forward.

    The caller stores exactly one SessionDecision per session, so a passthrough
    must not silently clear a session that has already been pinned for good.

    `low_conf_turns` does reset. Every path through here is a turn on which the
    model was confident, or one where confidence was never consulted, so it is
    not evidence that the session is off-distribution. Carrying the counter
    across such a turn would let two uncertain turns an hour apart pin the
    session to passthrough for good, and `terminal_passthrough` is irreversible.
    """
    if prior is None:
        return SessionDecision(kept=None, source=source)
    return SessionDecision(
        kept=None,
        low_conf_turns=0,
        terminal_passthrough=prior.terminal_passthrough,
        source=source,
    )


# --- The decision -------------------------------------------------------------


def decide(
    preds: Sequence[PredictionLike],
    live: frozenset[str],
    known: frozenset[str],
    prior: SessionDecision | None,
    cfg: RouterConfig,
    menus: Sequence[frozenset[str]] = (),
) -> SessionDecision:
    """Decide which tools to declare this turn. Total: every input yields a decision.

    `menus`, when the published artifact carries them, replaces the bespoke set with the
    closest of a few canonical ones. The tools chosen barely change; the number of distinct
    declaration lists the cache has to hold goes from one per session to one per menu, which
    is the difference between the router saving money and costing it (DESIGN.md 18.16).
    """
    if prior is not None and prior.terminal_passthrough:
        # Terminal and sticky. Two uncertain turns is strong evidence that this
        # session is off-distribution, so we stop recomputing for good.
        return replace(prior, source="lowconf")

    if too_few_tools(live, cfg):
        return _passthrough(prior, "small")

    # `transfer_to_agent` is a real member of tools_dict. Pruning it disables
    # multi-agent delegation, and the transfer then stops appearing in training
    # data, so the next model is even more certain (DESIGN.md 7.2, rule 1).
    protected = FRAMEWORK_TOOLS | cfg.never_prune
    known_labels = known - {ABSTAIN}
    # Tools the model has no class for: new MCP servers, freshly deployed tools, and tools no
    # corpus session ever called. They are passed through because a tool the model cannot name
    # is a tool it cannot ask for, so removing one on the model's silence would be removing it
    # on no evidence at all. `prunable_by_default` is the operator saying they have the
    # evidence the model lacks; it is subtracted here and nowhere else, so the only thing it
    # can do is stop a tool riding along in the passthrough.
    #
    # A tool in both `prunable_by_default` and `never_prune` still ships, because `protected`
    # is unioned back in below. That is the safe direction and it is deliberate rather than
    # incidental: the operator has said two contradictory things and the one that keeps the
    # session working wins. `startup_checks` reports the contradiction so it does not stand.
    unknown = live - known_labels - cfg.prunable_by_default

    top = preds[0] if preds else None
    if top is None:
        # No prediction at all, so there is nothing to prune towards. Widen the aperture.
        # Deliberately does NOT write `kept`. Under session-stability the naive reading
        # ("keep the frozen set") narrows the aperture on uncertainty, which is backwards
        # (DESIGN.md 7.2, rule 2).
        #
        # A *weak* prediction no longer lands here. `min_confidence_to_route` used to
        # veto this branch on `top.probability` and on a top-ranked ABSTAIN, and it is
        # gone: whether a doubtful menu is worth sending depends on what the menu saves,
        # which is a question this function cannot see and `worth_pruning` answers for
        # the caller (DESIGN.md 19.5, 19.8).
        #
        # NOTE(contract): `low_conf_turns >= 2` is unreachable from the current plugin
        # wiring, which pins a session the moment it sends a full list rather than after
        # two uncertain turns (DESIGN.md 18.22). The counter stays because the rule is
        # specified and proven here, and because re-prediction is the obvious v1.1 knob.
        low = (prior.low_conf_turns if prior else 0) + 1
        return SessionDecision(
            kept=None,
            low_conf_turns=low,
            terminal_passthrough=low >= 2,
            source="lowconf",
        )

    if prior is not None and prior.kept is not None:
        # A session that already holds a kept set keeps it, unconditionally. That is the
        # whole of session stability: the declared list is chosen once and never moves.
        #
        # There used to be a growth path here, widening the set (or hopping to a wider
        # menu) when the top prediction named a tool the session had pruned away, up to
        # `max_growth_events` times. It was deleted in v1 because it could not fire, not
        # because it stopped being wanted: the plugin only calls `decide` when the session
        # has no kept set yet, which is the exact negation of the condition this branch
        # needed. Zero of 13,474 provenance records across eleven A/B runs were `grown` or
        # `budget`. Menu recall is 0.9110 at the published setting, so roughly one session
        # in eleven really does want a tool its menu lacks; that miss is priced by
        # `worth_pruning` before the prune happens, and is not repaired afterwards.
        return replace(prior, source="reused")

    chosen = rank(preds, menus) if menus else None
    if chosen is None:
        chosen = frozenset(top_k_by_mass(preds, cfg.keep_probability_mass, cfg.max_tools_kept))
    kept = (chosen & live) | unknown | (protected & live)

    if not kept or kept >= live:
        # Never send an empty tool list, and never pay a cache break for nothing.
        return _passthrough(prior, "noop")

    # The cache-floor guard used to live here, deciding once per session from a config
    # constant. It now runs in the plugin, once, on the turn the decision is made, against
    # the size this agent's prompts settle at. What to keep is decided here; whether
    # applying it pays is decided there (DESIGN.md 7.3, 18.20).
    return SessionDecision(
        kept=kept,
        low_conf_turns=0,
        terminal_passthrough=False,
        source="fresh",
    )
