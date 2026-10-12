"""Watches what pruning actually costs this agent, and says when to stop.

The static floor guard in `pruner.breaks_cache` asks whether a prune would drop the
prompt under a documented threshold. That is a reasonable prior and it is not enough,
because the thresholds are not obeyed. Measured on Gemini 2.5 Flash, whose documented
implicit-cache minimum is 2048 tokens:

    a 1919-token request cached 1778 tokens
    a 2119-token request cached nothing, twice in a row

Implicit caching is best effort. It depends on prefix reuse, on recency, on load, and
on rules nobody outside Google can see. A router that decides whether to prune from a
published constant is guessing about behaviour it can simply observe instead.

So this observes it. Every response reports its prompt size and how much of it came from
cache. Group those by whether the router pruned the request that produced them, and the
comparison falls out. The control arm and the sessions the router declined supply the
unpruned side for free, which is a second reason the permanent holdout earns its keep
(DESIGN.md 11.4).

**Billed tokens, not cache hit rate.** The first version of this compared cache hit rates
and demoted on a 20-point gap. The G1 run in DESIGN.md 18.16 lost 13.1 points, stayed
politely under the margin, and cost money for 310 turns. Hit rate was the wrong quantity.
A cached input token bills at a tenth of an uncached one, so a prune that removes 28% of
the prompt is undone by a cache loss a third that size, and no threshold expressed in
percentage points makes that legible. Billed-equivalent tokens per call is the quantity
G1 is defined in, both inputs arrive on every response, and comparing it directly needs
no judgement about how many points of hit rate a token is worth.

This is the part that makes the router safe on an agent nobody has characterised. A
constant can be wrong by 5.4x, as `tokens_per_declaration` was. A proxy can be watching
the wrong thing, as hit rate was. A measurement of what the agent was actually billed
cannot be either.

**Only calls that are not a session's first count.** A pooled average over both arms is not
a measurement of pruning, because turn position dominates the price. A session's first model
call has no prefix of its own to hit and bills about three times a continuation. So the pooled
average of an arm is mostly a statement about what fraction of that arm was turn one, and the
two arms do not have the same fraction — the router's own rules decide which sessions prune,
and therefore decide the mix. Measured over the 2554 priced calls of DESIGN.md 18.22:

    unpruned  first calls   446   1941 billed      29.6% of the arm
    unpruned  later calls  1063    660 billed
    pruned    first calls   148   2443 billed      14.2% of the arm
    pruned    later calls   897    782 billed

Pruning was 25.9% worse on first calls and 18.4% worse on later ones, and 2.4% *cheaper*
pooled. That is Simpson's paradox, not noise and not a warm-up transient: no amount of extra
sampling fixes it, because the confound is the arm composition itself. The earlier version of
this module excluded pruned calls whose menu had not been sent often enough to be "warm", which
addressed a real asymmetry but not this one; replaying all five A/B runs, that rule reached the
wrong conclusion on three of them and this one reaches the right conclusion on all five.

Excluding turn one from both arms leaves a comparison where both sides are caching against a
prefix their own session established seconds earlier, which is the only regime where the prune
is the difference. It costs the guard the ability to see a loss that happens *only* on turn
one, and that is deliberate: a one-way per-process guard cannot tell "this deployment's menus
are young" from "this agent should never prune", and demoting on the coldest sample available
is the DESIGN.md 18.21 failure repeating. The economic question has better instruments that
run before any money is spent — `scripts/predict_saving.py` and `scripts/tune_settings.py`.
This is a safety net for the DESIGN.md 7.3 trap, where a prune turns caching off and the
signature is pruned calls with real history getting no cache at all.

**The single-call agent gets its own comparison, on its own terms.** The exclusion above leaves
a hole: an agent whose sessions are one model call each has every call excluded, so no verdict
ever arrives and pruning could be losing money quietly. A classify-and-return agent looks exactly
like that, and it is a real population rather than a corner case.

The confound the exclusion exists to remove is the *mix*: the two arms hold different fractions
of turn one, so pooling across positions compares composition instead of pruning. Comparing first
calls only to first calls removes that confound completely, because every call on both sides is
turn one. The comparison is sound. What it is not is cheap. A first call has no prefix of its own
and caches only against whatever other sessions have recently sent, so its price swings on prefix
reuse rather than on this session's history, and a young deployment's menu prefixes are genuinely
cold before they are genuinely warm. Demoting on that is the DESIGN.md 18.21 failure repeating.

So the first-call comparison is a fallback with two brakes and one gate. It needs ten times the
samples, it needs three times the margin, and it is only ever consulted while the ordinary
comparison has nothing to say. That gate is self-selecting rather than a heuristic: reaching 500
first calls in *both* arms before either later-call arm reaches 50 requires more than nine
sessions in ten to be a single call, which is the population the rule is for. A multi-turn agent
reaches its later-call verdict long before the fallback is eligible and never sees it.

The cool-off after such a demotion counts first calls, and it is stretched by the same factor as
the sample requirement, so both tracks put the same share of an agent's traffic at risk on a
retry: about 9% of calls, once. `coverage()` still reports the split so an operator can see which
comparison is carrying the agent, and `verdict_basis()` says which one earned a demotion.

**Demotion is bounded, and so is coming back.** Demotion is per agent and immediate. It is not
permanent, because the conditions that earn one are not permanent either — a model version
changes underneath a deployment, a menu set is republished — and an agent stuck in passthrough
until someone notices is a silent, indefinite loss. So a demoted agent is re-promoted after a
cool-off measured in its own unpruned calls, and then it must earn its demotion again from an
empty counter. What keeps that from oscillating forever is that the cool-off doubles each time
and the number of re-promotions is capped: after `max_repromotions` of them the demotion is
permanent for the life of the process. The agent flaps a bounded number of times and then
stops, which is the property a safety mechanism needs.
"""

from __future__ import annotations

from dataclasses import dataclass


def costs_more(
    pruned_billed: float,
    pruned_calls: int,
    whole_billed: float,
    whole_calls: int,
    margin: float,
) -> bool:
    """Whether pruning billed more per call than not pruning, by more than `margin`.

    The single statement of the rule. `CacheWatch` below asks it of counters it keeps in
    memory and `scripts/fleet_cache_watch.py` asks it of a warehouse aggregate over every
    replica. They have to be the same question: an offline gate that refused publishes the
    live guard would allow, or allowed what it would refuse, is a second opinion rather than
    a wider view of the same one (DESIGN.md 19.75).

    Cross-multiplied rather than comparing the two per-call averages, so the threshold does
    not shift with how the call counts happen to divide. An empty arm is never harmful,
    because there is nothing to compare it against.
    """
    if not pruned_calls or not whole_calls:
        return False
    return pruned_billed * whole_calls >= (1.0 + margin) * whole_billed * pruned_calls


@dataclass
class _Counts:
    """Billed-equivalent tokens for one arm of the comparison."""

    calls: int = 0
    billed: float = 0.0
    cached: int = 0  # kept for the operator log, never for the verdict

    @property
    def per_call(self) -> float:
        return self.billed / self.calls if self.calls else 0.0

    @property
    def hit_rate(self) -> float:
        return self.cached / self.calls if self.calls else 0.0


class CacheWatch:
    """Per-agent evidence about whether pruning is actually cheaper.

    Not thread-safe, for the same reason `TokenScale` is not: the arithmetic is a handful
    of increments and a lost update costs one sample.
    """

    def __init__(
        self,
        *,
        min_samples: int = 50,
        margin: float = 0.05,
        cached_multiplier: float = 0.10,
        cooloff_calls: int = 500,
        max_repromotions: int = 2,
        first_call_min_samples: int = 500,
        first_call_margin: float = 0.15,
    ) -> None:
        """
        Args:
            min_samples: how many calls each arm needs before a verdict is possible.
                Both arms, not the total. A verdict from 200 pruned calls and 3
                unpruned ones is a verdict about nothing.
            margin: how much more expensive pruning has to be, as a fraction of the
                unpruned cost, before it is called harmful. Small on purpose. The point
                of the router is to spend less, so "the same" is already a failure; the
                margin exists only to keep sampling noise from demoting an agent that is
                breaking even.
            cached_multiplier: what a cached input token bills at, relative to an
                uncached one. 0.10 on Vertex for the Gemini families this ships against.
            cooloff_calls: how many of this agent's own counted unpruned calls have to go
                by before a demotion is lifted. Calls rather than seconds so the module
                stays clock-free, and because it self-scales: a busy agent retries in
                minutes and a quiet one does not spend its retry on stale conditions.
            max_repromotions: how many times an agent may be brought back. Past this the
                demotion is permanent for the life of the process. This is the bound that
                makes the demote/re-promote pair terminate instead of flapping.
            first_call_min_samples: how many calls each arm of the first-call comparison
                needs before that fallback may speak. Ten times `min_samples`, both because
                a turn-one price is a noisier quantity and because the ratio is what keeps
                the fallback out of a multi-turn agent's way.
            first_call_margin: the margin the first-call comparison demotes on. Three times
                `margin`, because turn-one caching depends on other sessions' prefixes and a
                young deployment looks harmful before it looks anything.
        """
        self._min_samples = min_samples
        self._margin = margin
        self._cached_multiplier = cached_multiplier
        self._cooloff_calls = cooloff_calls
        self._max_repromotions = max_repromotions
        self._first_call_min_samples = first_call_min_samples
        self._first_call_margin = first_call_margin
        self._pruned: dict[str, _Counts] = {}
        self._whole: dict[str, _Counts] = {}
        # The same two arms again, over the calls that open a session. Kept apart rather
        # than pooled in: turn position, not pruning, is what dominates a call's price.
        self._pruned_first: dict[str, _Counts] = {}
        self._whole_first: dict[str, _Counts] = {}
        self._demoted: set[str] = set()
        self._demotions: dict[str, int] = {}  # lifetime count, per agent, never reset
        self._cooling: dict[str, int] = {}  # counted calls since the current demotion
        self._basis: dict[str, str] = {}  # which comparison earned the current demotion

    def observe(
        self,
        agent: str,
        *,
        pruned: bool,
        prompt_tokens: int,
        cached_tokens: int,
        first_call: bool,
    ) -> None:
        """Record one response: was the request pruned, and what did it bill?

        `first_call` says whether this was the first model call of its session. Those go into
        a second pair of arms rather than into the main comparison, because a first call has
        no prefix of its own to hit and its price says more about turn position than about
        pruning. Mixing them in would be worse than dropping them: it is the difference in
        the two arms' mixes that misleads. Keeping them apart costs nothing and is what lets
        a single-call agent be judged at all.
        """
        if prompt_tokens <= 0:
            return  # no usage reported, so nothing to price
        cached = min(max(cached_tokens, 0), prompt_tokens)
        if first_call:
            table = self._pruned_first if pruned else self._whole_first
        else:
            table = self._pruned if pruned else self._whole
        counts = table.setdefault(agent, _Counts())
        counts.calls += 1
        counts.billed += (prompt_tokens - cached) + cached * self._cached_multiplier
        if cached > 0:
            counts.cached += 1
        if agent in self._demoted:
            self._cool(agent, first_call)
            return
        basis = self._verdict(agent)
        if basis:
            self._demote(agent, basis)

    def suppressed(self, agent: str) -> bool:
        """True while pruning has been shown to cost this agent money."""
        return agent in self._demoted

    def billed(self, agent: str) -> tuple[float, float, int, int]:
        """(pruned per call, unpruned per call, pruned calls, unpruned calls). For logging."""
        pruned = self._pruned.get(agent, _Counts())
        whole = self._whole.get(agent, _Counts())
        return pruned.per_call, whole.per_call, pruned.calls, whole.calls

    def first_call_billed(self, agent: str) -> tuple[float, float, int, int]:
        """The same four numbers as `billed`, over the calls that open a session."""
        pruned = self._pruned_first.get(agent, _Counts())
        whole = self._whole_first.get(agent, _Counts())
        return pruned.per_call, whole.per_call, pruned.calls, whole.calls

    def verdict_basis(self, agent: str) -> str:
        """Which comparison earned this agent's most recent demotion: "later", "first", or "".

        Worth logging, because the two say different things. "later" means pruning is losing
        against a prefix the session itself established, which is the DESIGN.md 7.3 trap.
        "first" means pruning is losing against whatever prefixes other sessions leave warm,
        which can also mean the deployment is young.

        It outlives the demotion on purpose, so the re-promotion log can say how much traffic
        the retry is about to spend. The next demotion replaces it, and `reset` clears it,
        because a verdict about the previous artifact says nothing about the new one.
        """
        return self._basis.get(agent, "")

    def coverage(self, agent: str) -> tuple[int, int]:
        """(calls in the main comparison, calls in the first-call one).

        The first number at zero against a large second one means this agent's sessions are
        a single model call each. That used to be a hole; it is now the case the first-call
        fallback exists for, and the split is still reported so an operator can see which
        comparison is carrying the agent.
        """
        pruned = self._pruned.get(agent, _Counts())
        whole = self._whole.get(agent, _Counts())
        opening = self._pruned_first.get(agent, _Counts())
        opening_whole = self._whole_first.get(agent, _Counts())
        return pruned.calls + whole.calls, opening.calls + opening_whole.calls

    def hit_rates(self, agent: str, first_call: bool = False) -> tuple[float, float]:
        """(pruned, unpruned) cache hit rates. Context for the operator, never the verdict.

        Over the calls after a session's first by default, or over the ones that open a
        session when asked, so the log can quote the rates for whichever comparison spoke.
        """
        pruned_table = self._pruned_first if first_call else self._pruned
        whole_table = self._whole_first if first_call else self._whole
        return (
            pruned_table.get(agent, _Counts()).hit_rate,
            whole_table.get(agent, _Counts()).hit_rate,
        )

    def reset(self, agent: str) -> None:
        """Forget this agent's evidence, because the artifact it was evidence about is gone.

        Called when the plugin swaps in a new model version. Everything here was measured
        against the old one: its menus, its declared sets, its prefixes. A republished artifact
        is very often the *fix* for the demotion, so carrying the verdict across the swap
        leaves an agent in passthrough for the life of the process precisely when someone has
        just repaired it, which is the failure the cool-off exists to avoid.

        The lifetime demotion count survives on purpose. It is what doubles the cool-off and
        what eventually makes a demotion permanent, so keeping it means each new artifact gets
        one chance rather than a fresh budget of them, and an agent that loses money under
        every version it is given still converges to off.
        """
        self._forget(agent)
        self._basis.pop(agent, None)

    def demotions(self, agent: str) -> int:
        """How many times this agent has been demoted, over the life of the process."""
        return self._demotions.get(agent, 0)

    def cooloff(self, agent: str) -> int:
        """Counted calls this agent's current demotion has left to run, 0 if it is not
        demoted and -1 if the demotion is permanent.

        The exposure a re-promotion buys is `min_samples / (cooloff + min_samples)` of the
        agent's calls spent pruning badly, once, before the guard can demote it again. The
        plugin prints that arithmetic so an operator can see the price of the retry. A
        first-call demotion stretches both halves of that fraction by the same factor, so the
        share is the same and only the wait is longer.
        """
        if agent not in self._demoted:
            return 0
        limit = self._cooloff_for(agent)
        if limit is None:
            return -1
        return max(limit - self._cooling.get(agent, 0), 0)

    def _demote(self, agent: str, basis: str) -> None:
        self._demoted.add(agent)
        self._demotions[agent] = self._demotions.get(agent, 0) + 1
        self._cooling[agent] = 0
        self._basis[agent] = basis

    def _cooloff_for(self, agent: str) -> int | None:
        """How long this demotion lasts in counted calls, or None if it never ends.

        Doubling per demotion, so an agent that keeps earning demotions spends a share of
        its traffic on retries that halves every cycle, and the cap makes the total finite
        rather than merely small. Both halves matter: a cap alone would leave the agent
        flapping forever at a fixed low rate, which is not a mechanism that terminates.

        A demotion earned on first calls waits proportionally longer, because the evidence it
        will need to demote again is proportionally more expensive to gather. Scaling the
        cool-off by the same ratio keeps the retry's cost a fixed share of traffic instead of
        making it ten times dearer on the agents least able to afford it.
        """
        demotions = self._demotions.get(agent, 0)
        if demotions > self._max_repromotions:
            return None
        base = self._cooloff_calls
        if self._basis.get(agent) == "first" and self._min_samples:
            base = base * self._first_call_min_samples // self._min_samples
        return base << (demotions - 1)  # doubled once per demotion so far

    def _cool(self, agent: str, first_call: bool) -> None:
        """Count one call against the current demotion, and lift it when the cool-off is up.

        Only calls of the kind the demotion was based on count. A single-call agent has no
        later calls to wait for, and a multi-turn agent's first calls are not what its
        verdict was drawn from, so in both cases the cool-off is measured in the currency the
        guard will need to spend to reach a verdict again.
        """
        if first_call != (self._basis.get(agent) == "first"):
            return
        limit = self._cooloff_for(agent)
        if limit is None:
            return
        self._cooling[agent] = self._cooling.get(agent, 0) + 1
        if self._cooling[agent] < limit:
            return
        # Re-promoted. The old counters go with the old verdict: they are what the agent has
        # already been judged on, and keeping them would let a demotion that is no longer
        # true re-fire the instant the pruned arm reaches `min_samples` again. `_basis` is the
        # exception and stays, so the re-promotion log can say what the retry is about.
        self._forget(agent)

    def _forget(self, agent: str) -> None:
        """Lift the demotion and drop every counter it was drawn from."""
        self._demoted.discard(agent)
        self._cooling.pop(agent, None)
        self._pruned.pop(agent, None)
        self._whole.pop(agent, None)
        self._pruned_first.pop(agent, None)
        self._whole_first.pop(agent, None)

    def _verdict(self, agent: str) -> str:
        """Which comparison, if either, says pruning is harmful right now.

        The main one is asked first and always. The first-call fallback is asked only while
        the main one is short of samples, which on a multi-turn agent is a window that closes
        long before the fallback could be eligible anyway.
        """
        if self._harmful(
            self._pruned.get(agent, _Counts()),
            self._whole.get(agent, _Counts()),
            self._min_samples,
            self._margin,
        ):
            return "later"
        if self._counted(agent) >= self._min_samples:
            return ""
        if self._harmful(
            self._pruned_first.get(agent, _Counts()),
            self._whole_first.get(agent, _Counts()),
            self._first_call_min_samples,
            self._first_call_margin,
        ):
            return "first"
        return ""

    def _counted(self, agent: str) -> int:
        """The smaller of the main comparison's two arms: what it has to work with."""
        return min(
            self._pruned.get(agent, _Counts()).calls,
            self._whole.get(agent, _Counts()).calls,
        )

    def _harmful(self, pruned: _Counts, whole: _Counts, samples: int, margin: float) -> bool:
        if pruned.calls < samples or whole.calls < samples:
            return False
        return costs_more(pruned.billed, pruned.calls, whole.billed, whole.calls, margin)
