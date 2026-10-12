"""`SmartRouterPlugin`: the ADK wiring. Every rule it applies lives in another module.

This file decides when to ask a question, never what the answer is. The pruning rule is
`pruner.decide`, the request edits are `request_edits`, the rollout split is `rollout`, the
cache eligibility is `cache_hooks`, the vector is `embedding.Embedder`, the record is
`provenance`. A policy decision written here is in the wrong file.

The one non-obvious rule: ADK does not fail open. `PluginManager._run_callbacks` re-raises a
plugin exception as `RuntimeError` and aborts the whole run, so every callback carries
`@fail_open`. `None` is "no opinion" for every ADK plugin callback, which is why passing
through is always a safe answer (DESIGN.md 7.1, 11.2).
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar, cast

# run_config is the documented home of StreamingMode; it re-exports it from a private module,
# hence the ignore. Importing the private module instead would be the worse dependency.
from google.adk.agents.run_config import StreamingMode  # type: ignore[attr-defined]
from google.adk.plugins.base_plugin import BasePlugin

from adk_smart_router import judges, prompt_size, request_edits, rollout
from adk_smart_router.cache_hooks import CacheHooks
from adk_smart_router.cache_watch import CacheWatch
from adk_smart_router.config import Mode, RouterConfig, load_config, min_cache_tokens
from adk_smart_router.embedding import BREAKER_OPEN, CALL_ERROR, TIMEOUT, Embedder
from adk_smart_router.judges import LiveLlmJudge, PrototypeJudge, ToolJudge
from adk_smart_router.model_store import ModelStore, RouterModel
from adk_smart_router.predictor import Prediction, Predictor
from adk_smart_router.prefix_lock import PrefixLock
from adk_smart_router.provenance import RoutingRecord, carry_onto, emit, new_decision_id
from adk_smart_router.pruner import (
    SessionDecision,
    breaks_cache,
    contained_mass,
    decide,
    too_few_tools,
    worth_pruning,
)
from adk_smart_router.session_calls import SessionCalls
from adk_smart_router.startup_checks import StartupChecks
from adk_smart_router.steady_state import PromptSizes
from adk_smart_router.tool_cache import ToolResultCache

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

# One embedding failure pins the whole session to passthrough. A per-turn retry would produce
# mid-session cache breaks, which cost more than the routing saves (DESIGN.md 11.2).
#
# Keyed by why the embedding failed, because the three reasons are fixed by different people
# and used to arrive in BigQuery indistinguishable: ab-20260907 lost 47 sessions to Vertex
# quota latching the breaker and ab-20260911m8 lost 40 to plain timeouts, both recorded as
# `error`, and both times the cause had to be reconstructed from outside the warehouse
# (DESIGN.md 19.23).
_EMBED_FAILED = {
    reason: SessionDecision(kept=None, terminal_passthrough=True, source=reason)
    for reason in (TIMEOUT, CALL_ERROR, BREAKER_OPEN)
}


def _embed_failure(reason: str) -> SessionDecision:
    """The session decision for a failed embedding. Unknown reasons keep the old value.

    `error` has always meant "the router never got a vector", so it is the honest fallback
    for a reason this mapping has no name for.
    """
    return _EMBED_FAILED.get(reason, _EMBED_FAILED[CALL_ERROR])


# This session already sent a full tool list, so the prefix is set and pruning now would buy a
# smaller prompt at the price of a new one. Terminal for the same reason as above: the moment
# has passed and no later turn brings it back (DESIGN.md 7.2, 18.22).
_TOO_LATE = SessionDecision(kept=None, terminal_passthrough=True, source="late")

# This agent has fewer live tools than `min_tools_to_route`, so there is nothing here worth
# an embedding call. Not terminal: the live tool list is read from the request every turn and
# an agent that gains tools should start being routed without waiting for new sessions.
_TOO_SMALL = SessionDecision(kept=None, source="small")

# The decline rule said no: what this menu removes, over an expected session, is worth less
# than the chance of missing a tool it dropped (DESIGN.md 19.5). Not terminal, because the
# session may still settle into a decision on a later turn the same way any passthrough can.
_NOT_WORTH = "cheap"


def _held(decision: SessionDecision, *, floor_ok: bool) -> str:
    """The source to record when a decision was made but a cache guard withheld it.

    The two guards are named apart. They fail for different reasons, they are fixed by
    different people, and recording both as "floor" hid a live incident: DESIGN.md 18.21
    reads as 186 sessions tripping the cache floor and was in fact one watch demotion,
    after which the router never routed again. A source that cannot distinguish "this
    prompt is too small to prune" from "this agent has been switched off" is not
    instrumentation.

    A decision that was already a passthrough keeps its own reason, because "the guard
    stopped us pruning" is a misleading thing to write about a turn that was never going
    to prune.
    """
    if decision.kept is None:
        return decision.source
    return "floor" if not floor_ok else "watch"


# --- Fail open ----------------------------------------------------------------

_Callback = TypeVar("_Callback", bound=Callable[..., Awaitable[Any]])


def fail_open(fn: _Callback) -> _Callback:
    """Swallow any exception, count it, and return None. Applied to every callback.

    Mirrors BQAA's own `_safe_callback`. Without it a bad model artifact can 500 every request,
    because ADK turns a plugin exception into a `RuntimeError` that aborts the run.
    """

    @functools.wraps(fn)
    async def wrapper(self: SmartRouterPlugin, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        try:
            return await fn(self, *args, **kwargs)
        except Exception:
            logger.exception("smart_router: %s failed; passing through", fn.__name__)
            self._count_failopen(fn.__name__)
            return None  # None == "no opinion" for every ADK plugin callback

    # Marked so a reflective test can assert no callback was ever added unprotected.
    marked: Any = wrapper
    marked.__smart_router_failopen__ = True
    return cast(_Callback, wrapper)


@dataclass(frozen=True)
class _Router:
    """One agent's verified artifact and the booster loaded from it."""

    model: RouterModel
    predictor: Predictor


@dataclass(frozen=True)
class _Decided:
    """A session's decision, the id linking it to the cache serves under it, and the
    floor guard's answer.

    `floor_ok` is latched. The floor guard is asked once, on the turn the decision is
    made, and never asked again for that decision. Asking it every turn is what made the
    router change its mind between turn one and turn two, which costs the session its own
    prefix even though the kept set never moved (DESIGN.md 18.19). `applied` records what
    the session last actually did, so the one guard that *can* still change its answer —
    the watch, one-way and deliberate — shows up as a cache break rather than silently.
    """

    decision: SessionDecision
    decision_id: str
    floor_ok: bool = False
    applied: bool = False


@dataclass(frozen=True)
class _Sent:
    """One request as it went out, held until the response arrives to price it."""

    model: str | None
    agent_name: str
    base_chars: int
    schema_chars: int
    pruned: bool
    # Whether this was the first model call of its session, and so the one call with no
    # prefix of its own to hit. Read at send time, because the question is about the prefix
    # the previous sends left there. The watch excludes these from both arms (DESIGN.md
    # 18.22, and `session_calls` for why).
    first_call: bool = False


class SmartRouterPlugin(BasePlugin):
    """Prunes the tool menu once per session, and serves declared-cacheable tool results."""

    def __init__(
        self,
        cfg: RouterConfig | None = None,
        *,
        name: str = "smart_router",
        model_store: ModelStore | None = None,
        embedder: Embedder | None = None,
        cache: ToolResultCache | None = None,
        custom_judges: Sequence[ToolJudge] = (),
        clock: Callable[[], float] = time.monotonic,
        sampler: Callable[[], float] = random.random,
    ) -> None:
        super().__init__(name=name)
        self._cfg = cfg if cfg is not None else load_config()
        self._store = model_store if model_store is not None else ModelStore(self._cfg)
        self._embedder = embedder if embedder is not None else Embedder(self._cfg)
        self._startup = StartupChecks(self, self._cfg)
        self._hooks = CacheHooks(
            self._cfg, self._startup, cache=cache, clock=clock, sampler=sampler
        )
        self._prototype_judge = PrototypeJudge(self._cfg, self._embedder)
        self._llm_judge = LiveLlmJudge(self._cfg)
        self._custom_judges: tuple[ToolJudge, ...] = tuple(custom_judges)

        # Three maps, one meaning each. A loaded router is kept until a newer version is
        # published (DESIGN.md 21.4); a failed load is believed only until
        # `model_retry_after_s` is up (§19.73); `_checked_at` is when the re-poll last looked.
        self._routers: dict[str, _Router] = {}
        self._failed_at: dict[str, float] = {}
        self._checked_at: dict[str, float] = {}
        self._loading: dict[str, asyncio.Lock] = {}  # one loader per agent; see `_resolve`
        # Strong references to in-flight refresh tasks. Without this the event loop keeps only
        # a weak one and a refresh can be collected mid-flight (CPython asyncio contract).
        self._refreshing: set[asyncio.Task[None]] = set()
        self._swaps = 0
        self._clock = clock
        self._decisions: OrderedDict[str, _Decided] = OrderedDict()
        self._failopen: dict[str, int] = {}
        self._stale_warned: set[str] = set()

        # What the last request on each session looked like, waiting on the response that
        # prices it. Bounded the same way `_decisions` is: a run that never reaches
        # `after_model_callback` must not leak an entry.
        self._sent: OrderedDict[str, _Sent] = OrderedDict()
        self._scales: dict[str, prompt_size.TokenScale] = {}
        self._watch = CacheWatch(
            min_samples=self._cfg.cache_watch_min_samples,
            margin=self._cfg.cache_watch_margin,
            cached_multiplier=self._cfg.cached_token_multiplier,
            cooloff_calls=self._cfg.cache_watch_cooloff_calls,
            max_repromotions=self._cfg.cache_watch_max_repromotions,
            first_call_min_samples=self._cfg.cache_watch_first_call_min_samples,
            first_call_margin=self._cfg.cache_watch_first_call_margin,
        )
        self._sizes = PromptSizes(min_samples=self._cfg.steady_state_min_samples)
        self._calls = SessionCalls(capacity=self._cfg.session_decision_cache_size)
        self._lock = PrefixLock(capacity=self._cfg.session_decision_cache_size)
        self._warned: set[str] = set()  # agents whose current demotion has been logged

    def _active_aux_judges(self) -> tuple[ToolJudge, ...]:
        """Auxiliary zero-shot or custom judges configured alongside or in place of XGBoost."""
        active: list[ToolJudge] = []
        seen: set[str] = set()
        custom_by_name = {j.name: j for j in self._custom_judges}
        for name in self._cfg.judges:
            if name == "xgboost" or name in seen:
                continue
            if name == "prototype":
                active.append(self._prototype_judge)
                seen.add(name)
            elif name == "llm_live":
                active.append(self._llm_judge)
                seen.add(name)
            elif name in custom_by_name:
                active.append(custom_by_name[name])
                seen.add(name)
        for judge in self._custom_judges:
            if judge.name not in seen:
                active.append(judge)
                seen.add(judge.name)
        return tuple(active)

    # --- Observability --------------------------------------------------------

    @property
    def failopen_counts(self) -> Mapping[str, int]:
        """How many times each callback swallowed an exception. Read-only."""
        return dict(self._failopen)

    @property
    def cache(self) -> ToolResultCache:
        """The tool result cache, for tests and for reading `.stats`."""
        return self._hooks.cache

    @property
    def loaded_versions(self) -> Mapping[str, str]:
        """The model version each agent is serving right now, and how many times it moved.

        Worth exposing because the re-poll makes this a thing that changes without a deploy
        (DESIGN.md 21.4). Before it, "which version is this process on" was answerable from
        the deployment; now the process is the only thing that knows.
        """
        return {name: r.model.metadata.version for name, r in self._routers.items()}

    @property
    def model_swaps(self) -> int:
        """How many times a refresh has replaced a loaded artifact in this process."""
        return self._swaps

    @property
    def embedder_stats(self) -> Mapping[str, int]:
        """Embedding hits, misses and losses. Read-only.

        `(timeouts + failures + skipped) / misses` is the share of sessions pinned to
        passthrough before the router ever saw a prediction, not a retry rate: one embedding
        failure is terminal for its session (DESIGN.md 11.2). Exposed because it had no
        accessor at all, so a run that lost 17.9% of its sessions to timeouts finished without
        a single line saying so (DESIGN.md 19.23).
        """
        return self._embedder.stats

    async def warmup(self) -> bool:
        """Pay the embedding cold start before serving traffic. Optional but recommended.

        Skipping it costs the first session this process handles, not the first turn: the
        cold call overruns `embed_timeout_ms`, and one embedding failure pins a session to
        passthrough (DESIGN.md 11.2, 18.10). Everything still works, one session just goes
        unrouted, and on a replica that cycles often that is not a rounding error.

        Not automatic. There is no event loop at construction time, and a background task
        fired from the first callback would have to outlive the request that started it.
        An explicit await at startup is the honest version.
        """
        return await self._embedder.warmup()

    def set_mode(self, mode: Mode) -> None:
        """Flip the kill switch on a running plugin. One call disables prune and serve.

        The config is frozen, so both this object and the cache hooks hold a reference to the
        same instance and both have to be rebound together. Today the mode is read once from
        `SMART_ROUTER_MODE` at process start; this is the seam an operator-driven re-poll would
        use (DESIGN.md 11.1).
        """
        self._cfg = self._cfg.with_mode(mode)
        self._hooks.set_mode(mode)

    def _count_failopen(self, callback_name: str) -> None:
        self._failopen[callback_name] = self._failopen.get(callback_name, 0) + 1

    # --- before_model: prune ---------------------------------------------------

    @fail_open
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        """Prune `config.tools` in place. Always returns None: a value would skip the model."""
        pruned = await self._route(callback_context, llm_request)
        if pruned is not None:
            # Every request the plugin lets through is measured, pruned or not. The
            # unpruned ones are what `CacheWatch` compares against, and most of them come
            # from the holdout and from the sessions the router declined.
            self._track(callback_context, llm_request, pruned=pruned)
        return None

    async def _route(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> bool | None:
        """Apply the routing decision. None when the plugin is inert, else whether it pruned."""
        self._startup.run(callback_context)  # diagnostics only, never control flow

        cfg = self._cfg
        # 1. Live/bidi shares `config` by reference and preprocesses once, before the socket
        #    opens, so a prune there is permanent and unrecoverable (DESIGN.md 7.1).
        run_config = callback_context.run_config
        if run_config is not None and run_config.streaming_mode != StreamingMode.NONE:
            self._startup.live_streaming(getattr(run_config.streaming_mode, "name", "?"))
            return None
        # 2. Kill switch. One flip disables prune and cache-serve (DESIGN.md 11.1).
        if cfg.mode == "off":
            return None

        session = callback_context.session
        app_name = session.app_name
        user_id = callback_context.user_id
        session_id = session.id
        agent_name = callback_context.agent_name

        # 3. The permanent control arm. Recorded, not silent: an analyst has to be able to tell
        #    which turns are the control.
        if rollout.in_holdout(cfg, app_name, user_id, session_id):
            self._record(callback_context, source="holdout")
            return False
        # 4. The rollout ramp, independent of the holdout by salt.
        if not rollout.is_routed(cfg, app_name, user_id, session_id):
            return False
        # 5. Plugins are Runner-scoped, so the model is resolved per agent (D9).
        #    When cold-start judges (`prototype`, `llm_live`, or `custom_judges`) are active,
        #    an absent or expired GCS artifact falls back to cold-start routing rather than
        #    disabling the router.
        use_xgb = "xgboost" in cfg.judges
        router = await self._resolve(agent_name) if use_xgb else None
        aux_judges = self._active_aux_judges()
        if router is None and not aux_judges:
            # No artifact in hand: none published for this agent, or one that failed its
            # checks, or one that has aged past the cap. `disabled` was always the documented
            # name for this and was never written, which made it the one decline a reader
            # could not see. From the warehouse an agent with no artifact and an agent with no
            # plugin look identical, and they have nothing in common to fix. The reason is in
            # the log line `ModelStore.load` already writes; this is what makes a reader go
            # and look for it.
            self._record(callback_context, source="disabled")
            return False
        # The earliest point where the live tool list and the model's labels are both in hand,
        # which is what makes the classless set knowable. Deliberately above the prefix lock and
        # the embedding call: neither has anything to do with the inventory, and putting the
        # line below them would hide it from an agent whose sessions all arrive late or whose
        # embedder is down (DESIGN.md 7.2).
        live = frozenset(llm_request.tools_dict)
        if router is not None:
            known = frozenset(router.model.classes)
            self._startup.classless_tools(
                agent_name, live, known, lambda tools: self._declaration_tokens(llm_request, tools)
            )
            # Same place and the same reason: the first turn that has an artifact in hand is the
            # first turn that can compare this config against the one it was gated on (DESIGN.md
            # 19.72). Both report once per agent and neither changes what happens next.
            self._startup.artifact_skew(agent_name, router.model.metadata)
        else:
            known = live

        scope = rollout.scope_key(app_name, user_id, session_id, agent_name)
        prior = self._recall(scope)
        # A settled session is one that is pinned to passthrough or already holds a kept set.
        # An empty kept set counts as unsettled: the pruner never produces one, and re-deciding
        # is the safe reading if it ever did.
        if prior is not None and (prior.decision.terminal_passthrough or prior.decision.kept):
            return self._reuse(callback_context, llm_request, scope, prior)

        # An agent with too few tools to be worth routing, asked before anything is spent on
        # it. `min_tools_to_route` is documented as "too few tools to be worth the embedding
        # call" and `decide` is where the rule lives, but `decide` runs after the embedding, so
        # until this line the early-out saved nothing it was named for. Above the prefix lock
        # as well: for an agent that will never prune, "late" is a true statement about the
        # prefix and a misleading answer to "why did nothing happen here".
        if too_few_tools(live, cfg):
            self._remember(scope, _Decided(_TOO_SMALL, new_decision_id()))
            self._record(callback_context, source=_TOO_SMALL.source, router=router)
            return False

        # The prefix this session established on its first call is the thing being protected.
        # Once a full tool list has gone out, a pruned one is not a smaller prompt, it is a
        # different prefix at full price, and the measured cost of finding that out on turn two
        # is 6.6x the turn it replaces (DESIGN.md 18.22). Checked before the embedding call,
        # because a decision that can no longer be applied is not worth paying to make.
        if self._lock.locked(scope):
            self._remember(scope, _Decided(_TOO_LATE, new_decision_id()))
            self._record(callback_context, source=_TOO_LATE.source, router=router)
            return False

        prompt = request_edits.latest_user_text(llm_request)
        if prompt is None:
            return False  # a tool-call continuation turn carries no new intent

        needs_embedding = router is not None or any(j.name == "prototype" for j in aux_judges)
        vector: list[float] | None = None
        if needs_embedding:
            vector, why = await self._embedder.embed_with_reason(prompt)
            if vector is None:
                failed = _embed_failure(why)
                self._remember(scope, _Decided(failed, new_decision_id()))
                self._record(callback_context, source=failed.source, router=router)
                return False

        xgb_preds = (
            router.predictor.predict(vector)
            if (router is not None and vector is not None)
            else None
        )
        if aux_judges:
            specs = request_edits.tool_specs(llm_request)
            aux_outputs = await asyncio.gather(
                *(j.predict(prompt, vector, specs, agent_name=agent_name) for j in aux_judges)
            )
            aux_results = [(j.weight, out) for j, out in zip(aux_judges, aux_outputs, strict=True)]
            preds, effective_known, effective_cfg = judges.combine_predictions(
                xgb_preds=xgb_preds,
                xgb_known=known if router is not None else frozenset(),
                aux_results=aux_results,
                live=live,
                cfg=cfg,
            )
            if not preds:
                self._record(callback_context, source="disabled")
                return False
        else:
            preds = list(xgb_preds or ())
            effective_known = known
            effective_cfg = cfg

        if router is not None:
            menus = router.model.menus
        elif cfg.cold_start_menu_count > 0 and any(j.name == "prototype" for j in aux_judges):
            menus = self._prototype_judge.menus_for(
                agent_name,
                size=cfg.cold_start_max_tools_kept,
                count=cfg.cold_start_menu_count,
            )
        else:
            menus = ()
        decision = decide(
            preds,
            live,
            effective_known,
            prior.decision if prior else None,
            effective_cfg,
            menus,
        )

        # Two questions about one measurement, asked here and nowhere else. Both are
        # questions about the session rather than about this turn, and asking either again
        # later only ever means asking it of a longer prompt, which is how the router used
        # to talk itself into changing the tool list mid-session (DESIGN.md 7.3, 18.19).
        #
        # Does the prune pay comes first, because its answer does not depend on the other
        # and because a menu that saves too little to matter is not worth asking a cache
        # question about. A "no" here is the decision, not a guard on it: the session gets
        # a passthrough whose reason is that pruning was not worth the risk, and every
        # later turn reads that off the stored decision like any other passthrough.
        floor_ok = False
        if decision.kept is not None:
            full, pruned = self._prompt_tokens(llm_request, agent_name, decision.kept)
            if worth_pruning(full - pruned, contained_mass(preds, decision.kept), cfg):
                floor_ok = not breaks_cache(full, pruned, min_cache_tokens(llm_request.model))
            else:
                decision = replace(decision, kept=None, source=_NOT_WORTH)
        applied = floor_ok and self._watch_allows(agent_name)
        if applied:
            request_edits.apply_prune(llm_request, cast(frozenset[str], decision.kept))

        if prior is not None and decision == prior.decision:
            decision_id = prior.decision_id  # written only on change
        else:
            decision_id = new_decision_id()
        self._remember(scope, _Decided(decision, decision_id, floor_ok, applied))

        cold_version = (
            "cold_start:" + "+".join(j.name for j in aux_judges)
            if router is None and aux_judges
            else None
        )
        self._record(
            callback_context,
            source=decision.source if applied else _held(decision, floor_ok=floor_ok),
            decision_id=decision_id,
            kept=decision.kept if applied else None,
            router=router,
            model_version_override=cold_version,
            top=preds[0] if preds else None,
            # Turn one has no established prefix to break; a later change of membership does.
            cache_break=applied and prior is not None and prior.decision.kept != decision.kept,
        )
        return applied

    def _reuse(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        scope: str,
        prior: _Decided,
    ) -> bool:
        """Re-apply a decision this session already made. No embed, no predict, no floor guard.

        This is the point of session stability: the router prunes once and then stops paying,
        both in latency and in broken prefix caches (DESIGN.md 5.2, 11.7). The floor guard was
        the last thing here that could still change its mind, and it is now answered once at
        decide time and carried on the decision (DESIGN.md 18.19).

        The watch still runs every turn, because it is the emergency brake and it has to be
        able to stop an agent that is losing money. That transition does break the prefix,
        which is why it is recorded as a cache break rather than passed off as an ordinary
        held turn.

        The watch's verdict is *not* one-way any more — a demotion lifts after a cool-off —
        but the transition it can cause here still is, because of the prefix lock. A session
        that has already sent the full list cannot go back to pruning: the smaller list is
        not a smaller prompt, it is a new prefix at full price, and paying 6.6x for it
        mid-conversation is the DESIGN.md 18.22 mistake. So re-promotion reaches new sessions
        only, and the live ones ride out the conversation they are already in.

        This method is why the pruner has no growth path. A settled session never gets back to
        `pruner.decide`, so the branches that widened a kept set mid-conversation could never
        run, and they were deleted rather than left as an unreachable v1.1 knob. Re-introducing
        growth means re-predicting here, at one embedding call per turn against the one call per
        session per agent DESIGN.md 11.7 budgets.
        """
        kept = prior.decision.kept
        applied = (
            kept is not None
            and prior.floor_ok
            and self._watch_allows(callback_context.agent_name)
            and not self._lock.locked(scope)
        )
        if applied:
            request_edits.apply_prune(llm_request, cast(frozenset[str], kept))
        if applied != prior.applied:
            self._remember(scope, replace(prior, applied=applied))
        self._record(
            callback_context,
            source="reused" if applied else _held(prior.decision, floor_ok=prior.floor_ok),
            decision_id=prior.decision_id,
            kept=kept if applied else None,
            # The kept set is the same either way. Starting or stopping to apply it moves the
            # declared tool list just as far as changing it would (DESIGN.md 18.19).
            cache_break=applied != prior.applied,
        )
        return applied

    # --- The cache guard --------------------------------------------------------

    def _watch_allows(self, agent_name: str) -> bool:
        """Whether this agent is still allowed to prune at all. Asked on every turn.

        The watch compares what pruned and unpruned requests for this agent were actually
        billed, over the calls that are not a session's first (DESIGN.md 7.3, 18.13). Its
        verdict can flip back: a demotion lifts after a cool-off, a bounded number of times.
        What keeps that from moving a live conversation is the prefix lock, checked in
        `_reuse`; a re-promotion only reaches sessions that have not sent anything yet.
        """
        return not (self._cfg.cache_watch_enabled and self._watch.suppressed(agent_name))

    def _prompt_tokens(
        self, llm_request: LlmRequest, agent_name: str, kept: frozenset[str]
    ) -> tuple[int, int]:
        """This agent's whole prompt in tokens, with the full tool list and with `kept`.

        The two guards that run at decide time both want this pair. The cache floor wants to
        know whether the second number is still large enough to cache, and the decline rule
        wants the difference, which is what pruning removes.

        Measured once per decision, never per turn. The question is about the session, and the
        turn in front of us is the shortest the session will ever be: history only grows, so
        an answer taken from turn one flips to the opposite answer a turn or two later, and
        the flip costs the session its prefix (DESIGN.md 18.19).

        `steady_state.PromptSizes` supplies the size this agent's prompts settle at, once it
        has seen enough turns to say. The declarations the prune removes are measured from the
        request in front of us, because that part does not grow with the conversation. Before
        there is enough history it falls back to this turn, which is the old behaviour for
        exactly as long as there is nothing better to use.
        """
        scale = self._scale_for(llm_request.model)
        chars = prompt_size.count_chars(llm_request, kept)
        full = scale.full_tokens(chars)
        pruned = scale.pruned_tokens(chars)
        typical = self._sizes.typical(agent_name)
        if typical is not None and typical > full:
            full, pruned = typical, typical - (full - pruned)
        return full, pruned

    def _declaration_tokens(self, llm_request: LlmRequest, tools: frozenset[str]) -> int:
        """What `tools`' declarations cost on this request, in the tokens the model bills.

        Expressed as the difference between the request with and without them, because a
        difference is the only shape `TokenScale` offers and it is exactly the schema term:
        the prose half cancels. Diagnostics only, so an early-process estimate is fine.
        """
        chars = prompt_size.count_chars(llm_request, frozenset(llm_request.tools_dict) - tools)
        scale = self._scale_for(llm_request.model)
        return scale.full_tokens(chars) - scale.pruned_tokens(chars)

    def _track(
        self, callback_context: CallbackContext, llm_request: LlmRequest, *, pruned: bool
    ) -> None:
        """Remember what we are about to send, so `after_model_callback` can price it.

        Whether this is the session's first call is read here and not in
        `after_model_callback`, because it is a fact about the prefix this request is
        meeting. Read after recording the send, every call would look like a later one.
        """
        chars = prompt_size.count_chars(llm_request, None)  # the request exactly as it stands
        scope = rollout.scope_key(
            callback_context.session.app_name,
            callback_context.user_id,
            callback_context.session.id,
            callback_context.agent_name,
        )
        # An unpruned call commits this session to the full list. Recorded here rather than in
        # `_route` so that every path reaches it, including the ones that never make a decision
        # at all: a turn with no user text, a session below the rollout ramp that later crosses
        # it, or an agent whose artifact was not ready yet (DESIGN.md 18.22).
        if not pruned:
            self._lock.sent_unpruned(scope)

        first_call = self._calls.is_first(scope)
        self._calls.sent(scope)
        self._sent[scope] = _Sent(
            model=llm_request.model,
            agent_name=callback_context.agent_name,
            base_chars=chars.base,
            schema_chars=chars.kept,
            pruned=pruned,
            first_call=first_call,
        )
        self._sent.move_to_end(scope)
        while len(self._sent) > self._cfg.session_decision_cache_size:
            self._sent.popitem(last=False)

    def _scale_for(self, model_name: str | None) -> prompt_size.TokenScale:
        """One calibration per model. Tokenizers differ; a shared ratio would blur them."""
        key = model_name or ""
        scale = self._scales.get(key)
        if scale is None:
            scale = prompt_size.TokenScale()
            self._scales[key] = scale
        return scale

    # --- after_model: learn what it actually cost -------------------------------

    @fail_open
    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        """Fold this response's real token counts back into the guards.

        The model reports exactly how large the prompt was and exactly how much of it
        cached. v1 threw both away and guessed from config constants instead, which is
        how a 5.4x error survived to production (DESIGN.md 18.12). Always returns None:
        a returned response would replace the model's own.
        """
        usage = getattr(llm_response, "usage_metadata", None)
        if usage is None:
            return None
        scope = rollout.scope_key(
            callback_context.session.app_name,
            callback_context.user_id,
            callback_context.session.id,
            callback_context.agent_name,
        )
        # Popped, not read: a streaming response fires this callback repeatedly and only
        # the first carries usage worth learning from.
        sent = self._sent.pop(scope, None)
        if sent is None:
            return None

        self._scale_for(sent.model).observe(
            sent.base_chars, sent.schema_chars, usage.prompt_token_count or 0
        )
        self._sizes.observe(sent.agent_name, usage.prompt_token_count or 0)
        if self._cfg.cache_watch_enabled:
            self._watch.observe(
                sent.agent_name,
                pruned=sent.pruned,
                prompt_tokens=usage.prompt_token_count or 0,
                cached_tokens=usage.cached_content_token_count or 0,
                first_call=sent.first_call,
            )
            self._report_watch(sent.agent_name)
        return None

    def _report_watch(self, agent_name: str) -> None:
        """Log the watch's verdict when it changes, once per change.

        Both directions are worth a line. A demotion is the router switching itself off for
        an agent, and a re-promotion is it spending real money to find out whether the
        conditions that earned the demotion still hold, so an operator should be able to see
        the retry coming and read its price.
        """
        suppressed = self._watch.suppressed(agent_name)
        if suppressed == (agent_name in self._warned):
            return
        on_first = self._watch.verdict_basis(agent_name) == "first"
        if not suppressed:
            self._warned.discard(agent_name)
            logger.warning(
                "smart_router: %s is pruning again after its cool-off (demotion %d of at most "
                "%d); the counters are empty, so the next %d pruned %s are the experiment "
                "and are the whole cost of it if pruning is still losing",
                agent_name,
                self._watch.demotions(agent_name),
                self._cfg.cache_watch_max_repromotions + 1,
                self._cfg.cache_watch_first_call_min_samples
                if on_first
                else self._cfg.cache_watch_min_samples,
                "session openings" if on_first else "calls",
            )
            return
        self._warned.add(agent_name)
        if on_first:
            pruned_cost, whole_cost, n_pruned, n_whole = self._watch.first_call_billed(agent_name)
        else:
            pruned_cost, whole_cost, n_pruned, n_whole = self._watch.billed(agent_name)
        pruned_hits, whole_hits = self._watch.hit_rates(agent_name, first_call=on_first)
        later, first = self._watch.coverage(agent_name)
        cooloff = self._watch.cooloff(agent_name)
        ends = "permanently, its retries are used up" if cooloff < 0 else f"for {cooloff} calls"
        logger.warning(
            "smart_router: pruning is costing %s more than it saves, judged on %s "
            "(%.0f billed tokens per call over %d pruned, against %.0f over %d unpruned; "
            "cache hit rate %.0f%% against %.0f%%; this agent has sent %d calls after a "
            "session's first and %d that opened one); pruning is now off for this agent %s",
            agent_name,
            "the calls that open a session, because it has too few of any other kind"
            if on_first
            else "the calls after a session's first",
            pruned_cost,
            n_pruned,
            whole_cost,
            n_whole,
            pruned_hits * 100,
            whole_hits * 100,
            later,
            first,
            ends,
        )

    def _record(
        self,
        callback_context: CallbackContext,
        *,
        source: str,
        decision_id: str | None = None,
        kept: frozenset[str] | None = None,
        router: _Router | None = None,
        model_version_override: str | None = None,
        top: Prediction | None = None,
        cache_break: bool = False,
    ) -> None:
        """One provenance record for one decision, including the decisions not to route.

        Rolling deploys run several model versions at once, so "what was active then" is not
        recoverable after the fact and has to be stamped on the turn (DESIGN.md 11.3).
        """
        meta = router.model.metadata if router else None
        emit(
            callback_context,
            RoutingRecord(
                decision_id=decision_id or new_decision_id(),
                agent_name=callback_context.agent_name,
                session_id=callback_context.session.id,
                mode=self._cfg.mode,
                model_version=meta.version if meta else model_version_override,
                model_digest=meta.sha256 if meta else None,
                top_label=top.label if top else None,
                confidence=top.probability if top else None,
                kept_tools=tuple(sorted(kept or ())),
                decision_source=source,
                cache_break=cache_break,
            ),
        )

    # --- on_event: provenance reaches the warehouse ---------------------------

    @fail_open
    async def on_event_callback(self, *, invocation_context: Any, event: Any) -> None:  # noqa: ANN401
        """Move this turn's provenance onto the next event BQAA will log.

        `emit` can only reach the invocation's metadata bag, and ADK copies that bag onto
        nothing. BQAA reads `event.custom_metadata`, so without this hop the records live for
        the length of the invocation and land in `agent_events` never (DESIGN.md 11.3).

        ADK runs this for every event it yields, which is more events than BQAA logs, and
        `carry_onto` declines the ones it does not. Draining onto a function-call event
        instead of holding for the final response is what lost 32% of all decisions in
        production (DESIGN.md 19.23).

        Always returns None. A non-None return short-circuits every plugin registered behind
        this one, which by the required ordering (DESIGN.md 6.3) includes BQAA itself.
        """
        carry_onto(invocation_context, event)
        return None

    # --- before_tool / after_tool: the result cache ---------------------------

    @fail_open
    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        """Return a memoized result, or None to let the tool run."""
        self._startup.run(tool_context)
        return self._hooks.serve(
            tool, tool_args, tool_context, decision_id=self._decision_id_for(tool_context)
        )

    @fail_open
    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> None:
        """Always returns None. A returned dict would replace the tool's real result."""
        # Also run here: a tool can complete before this plugin has seen any other callback, and
        # the agent authorization scan has to run before the first cache write, not just the
        # first read.
        self._startup.run(tool_context)
        self._hooks.store(tool, tool_args, tool_context, result)
        return None

    def _decision_id_for(self, tool_context: ToolContext) -> str:
        """The routing decision this tool call sits under, or a fresh id if it was not routed."""
        scope = rollout.scope_key(
            tool_context.session.app_name,
            tool_context.user_id,
            tool_context.session.id,
            tool_context.agent_name,
        )
        known = self._decisions.get(scope)
        return known.decision_id if known else new_decision_id()

    # --- Session decision cache ------------------------------------------------

    def _recall(self, scope: str) -> _Decided | None:
        decided = self._decisions.get(scope)
        if decided is not None:
            self._decisions.move_to_end(scope)
        return decided

    def _remember(self, scope: str, decided: _Decided) -> None:
        # In-process, not ADK session state: a `frozenset` round-trips as a `frozenset` on
        # InMemorySessionService and as a `list` on DatabaseSessionService, and the decision is
        # worthless to a replica that did not make it (D8).
        self._decisions[scope] = decided
        self._decisions.move_to_end(scope)
        while len(self._decisions) > self._cfg.session_decision_cache_size:
            self._decisions.popitem(last=False)

    # --- Model resolution ------------------------------------------------------

    async def _resolve(self, agent_name: str) -> _Router | None:
        """The loaded router for this agent, or None. Loads once, and retries a failure."""
        router = self._routers.get(agent_name)
        if router is None:
            if self._load_failed_recently(agent_name):
                return None
            router = await self._load_once(agent_name)
            if router is None:
                return None
        # Before the age gate on purpose. A process whose artifact has aged out is exactly the
        # process that most needs to notice a fresher one, and gating the poll behind the age
        # check would mean the router stops looking at the moment it stops routing.
        self._maybe_refresh(agent_name, router)
        # Age is checked per use, not per load, so a long-lived process demotes itself when its
        # artifact goes stale (DESIGN.md 5.3).
        if router.model.age_days > self._cfg.max_model_age_days:
            if agent_name not in self._stale_warned:
                self._stale_warned.add(agent_name)
                logger.warning(
                    "smart_router: model for %s is %.1f days old, over the %d day cap; "
                    "passing through until a fresher artifact is published",
                    agent_name,
                    router.model.age_days,
                    self._cfg.max_model_age_days,
                )
            return None
        return router

    def _load_failed_recently(self, agent_name: str) -> bool:
        """Is the last failed load for this agent still believed?

        Expiring here rather than on a timer keeps the whole mechanism inside the one path
        that reads it, at the cost of a check that clears state. That trade is worth naming:
        the alternative is a background task with a lifecycle, which is what §11.1's artifact
        re-poll has been waiting on for months.
        """
        failed_at = self._failed_at.get(agent_name)
        if failed_at is None:
            return False
        if self._clock() - failed_at < self._cfg.model_retry_after_s:
            return True
        self._failed_at.pop(agent_name, None)
        return False

    async def _load_once(self, agent_name: str) -> _Router | None:
        """Load this agent's artifact under a deadline, with one loader at a time.

        Single-flight. Without the lock, every first turn that arrives before the first one
        finishes launches its own `_build`: a 2.2 s `ModelStore.load` and a 0.25 s `Predictor`
        init each, all but one of them thrown away, and the waste grows with concurrency.

        The deadline exists because `_build` blocks on GCS with no bound of its own, and a
        stuck connection would hold the first turn of the process open for as long as the
        socket lasted. `wait_for` stops the waiting, not the thread: the orphan runs to
        completion in the executor and its result is dropped. That is a leaked worker slot,
        once per agent, and it is also why a retry after a slow first attempt is usually cheap.
        `ModelStore._ensure_local` returns the cached file when the digest matches, so the
        orphan's download is not wasted, only its `Predictor` (DESIGN.md 19.73).
        """
        lock = self._loading.setdefault(agent_name, asyncio.Lock())
        async with lock:
            winner = self._routers.get(agent_name)
            if winner is not None:
                return winner  # someone loaded it while we waited
            if self._load_failed_recently(agent_name):
                return None  # and someone else's attempt failed while we waited
            try:
                router = await asyncio.wait_for(
                    asyncio.to_thread(self._build, agent_name), self._cfg.model_load_timeout_s
                )
            except TimeoutError:
                logger.warning(
                    "smart_router: loading the model for %s took longer than %gs; passing "
                    "through and retrying in %gs",
                    agent_name,
                    self._cfg.model_load_timeout_s,
                    self._cfg.model_retry_after_s,
                )
                router = None
            if router is None:
                self._failed_at[agent_name] = self._clock()
                return None
            self._routers[agent_name] = router
            # A router loaded this instant is the freshest thing the bucket had a moment ago,
            # so the first re-poll belongs one interval from now and not on the next turn.
            self._checked_at[agent_name] = self._clock()
            return router

    def _build(self, agent_name: str) -> _Router | None:
        """Discovery, download and booster load. Blocking, and never raises."""
        model = self._store.load(agent_name)
        if model is None:
            return None  # no artifact for this agent is a valid steady state, not an error
        try:
            return _Router(model=model, predictor=Predictor(model))
        except Exception as exc:
            # Caller negative-caches this for `model_retry_after_s`. A corrupt booster fails
            # identically every attempt, so the window is what keeps the cost of being wrong
            # about that down to one load per agent per window.
            logger.warning("smart_router: booster for %s did not load: %s", agent_name, exc)
            return None

    # --- Artifact re-poll (DESIGN.md 21.4) ---------------------------------------

    def _maybe_refresh(self, agent_name: str, router: _Router) -> None:
        """Schedule a version check if the interval has elapsed. Never blocks the turn.

        The hot path is one dict lookup and one subtraction. Everything else happens in a
        task nobody waits on, and the turn that scheduled it is served by the router already
        loaded. A request never waits on Cloud Storage for a version it is not going to use.

        `_checked_at` is stamped here rather than in the task, so the hundred turns that
        arrive while a refresh is in flight schedule nothing. The cost of that is one lost
        poll interval if the task dies before it runs, which is the same outcome as the poll
        finding nothing.
        """
        if self._cfg.refresh_interval_s <= 0 or self._cfg.pin_version:
            return
        last = self._checked_at.get(agent_name)
        now = self._clock()
        if last is not None and now - last < self._cfg.refresh_interval_s:
            return
        self._checked_at[agent_name] = now
        try:
            task = asyncio.create_task(self._refresh(agent_name, router.model.metadata.version))
        except RuntimeError:
            return  # no running loop: nothing to schedule onto, and nothing to report
        self._refreshing.add(task)
        task.add_done_callback(self._refreshing.discard)

    async def _refresh(self, agent_name: str, loaded: str) -> None:
        """Swap in a newer artifact, or leave the loaded one exactly as it is.

        Every failure here means "keep serving what we have", which is why nothing in it
        raises and nothing in it clears `_routers`. A bucket that cannot be listed, an
        artifact that fails its digest, a booster that will not load, a version built against
        a different embedding model: all of them are `_build` returning None or `_swap_in`
        never being reached, and all of them leave the process routing (§21.8).

        A refusal is deliberately not negative-cached. `_failed_at` exists to stop a cold
        process retrying a missing artifact every turn; here there is no turn being blocked
        and no passthrough being caused, so the next interval simply looks again.
        """
        lock = self._loading.setdefault(agent_name, asyncio.Lock())
        if lock.locked():
            return  # a cold load or another refresh is already doing this
        async with lock:
            try:
                router = await asyncio.wait_for(
                    asyncio.to_thread(self._build_if_newer, agent_name, loaded),
                    self._cfg.model_load_timeout_s,
                )
            except TimeoutError:
                logger.warning(
                    "smart_router: refreshing the model for %s took longer than %gs; "
                    "still serving %s",
                    agent_name,
                    self._cfg.model_load_timeout_s,
                    loaded,
                )
                return
            except Exception as exc:
                logger.warning("smart_router: refresh for %s failed: %s", agent_name, exc)
                return
            if router is not None:
                self._swap_in(agent_name, router, loaded)

    def _build_if_newer(self, agent_name: str, loaded: str) -> _Router | None:
        """One list call, and a full build only if the answer changed. Blocking.

        Both directions of "changed" count. A version higher than the loaded one is the new
        artifact this exists for; a version *lower* than it is the rollback in §21.5, where
        withdrawing a generation makes an older one current again. Comparing for equality
        rather than for order is what makes the second one work.
        """
        latest = self._store.latest_version(agent_name)
        if latest is None or latest == loaded:
            return None
        return self._build(agent_name)

    def _swap_in(self, agent_name: str, router: _Router, loaded: str) -> None:
        """Rebind the agent's router to one that is already built and verified.

        The swap is a single assignment of a fully constructed object, so there is no window
        in which an agent has half a router. Sessions already in flight keep the reference
        they resolved, and the routing decision is made once per session and cached under
        `_decisions`, so a swap cannot change a declared tool list mid-session and cannot
        break a prefix (§18.22).
        """
        self._routers[agent_name] = router
        self._failed_at.pop(agent_name, None)
        self._checked_at[agent_name] = self._clock()
        # A fresh artifact deserves a fresh warning if it too goes stale.
        self._stale_warned.discard(agent_name)
        # And a fresh verdict. Everything the cache watch has on this agent was measured
        # against the menus and declared sets of the version being replaced, and a republish
        # is usually the repair for a demotion rather than a continuation of it. The lifetime
        # demotion count is not cleared, so this is one more chance and not a new budget.
        self._watch.reset(agent_name)
        self._swaps += 1
        logger.info(
            "smart_router: agent %s moved from model version %s to %s",
            agent_name,
            loaded,
            router.model.metadata.version,
        )

    # NOTE(contract): the mode is still read once, at process start. §21.4 argues that
    # re-reading `SMART_ROUTER_MODE` in a running container is not useful, because nothing
    # can change it there, and that the durable kill switch is the artifact side: withdraw
    # every version and the next refresh finds nothing complete, `load` returns None, and
    # the plugin passes through. That marker is built: `model_store.WITHDRAWN_NAME`, written by
    # `scripts/watch_generations.py`. So there are two kill switches with different reaches.
    # `SMART_ROUTER_MODE` is per process and needs a restart; withdrawing every version reaches
    # the whole fleet on its next re-poll and needs nothing restarted.
