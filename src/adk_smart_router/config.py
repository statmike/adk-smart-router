"""Configuration for the smart router. Data only, no behaviour.

Every knob the router has lives here. Two rules govern this file:

1. It ships disabled. `mode` defaults to "off" and `route_fraction` to 0.0, so
   installing the plugin changes nothing until someone turns it on.
2. Fields marked "parity" must match the training SQL exactly. A train/serve
   embedding mismatch produces a model that passes the offline gate and predicts
   garbage in production (DESIGN.md 9.1), so those fields are single-sourced:
   one field used on both sides, never one field per side.

There is deliberately no `agent_name` field. Plugins install on the Runner, not
on an agent, so the agent is resolved per request from the callback context
(DESIGN.md D9).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

Mode = Literal["off", "active"]

# Published minimum promptable size for context caching, mirroring ADK's
# google.adk.models.gemini_context_cache_manager. Pruning a cacheable prompt
# below the floor turns caching off entirely, and every kept token then bills at
# ten times what it did (DESIGN.md 7.3).
#
# Treat these as a prior, not as truth. Measured on gemini-2.5-flash, whose
# published floor is 2048: a 1919-token request cached 1778 tokens, and a
# 2119-token one cached nothing twice running. Implicit caching is best effort
# and the published number is a lower bound on eligibility, not a guarantee in
# either direction. `cache_watch.CacheWatch` is what actually decides whether
# pruning is costing an agent its cache hits (DESIGN.md 18.13).
MIN_CACHE_TOKENS: Mapping[str, int] = {"gemini-2.5-": 2048, "gemini-3": 4096}

# ADK's control plane. Never prunable, never a training label: pruning it
# silently disables multi-agent delegation, and because the transfer then never
# happens it stops appearing in training data (DESIGN.md 7.2).
FRAMEWORK_TOOLS: frozenset[str] = frozenset({"transfer_to_agent"})

# Reserved envelope key stamped onto every cache-served payload so the warehouse
# can tell a hit from a real tool call (DESIGN.md 11.5).
CACHE_HIT_KEY = "_smart_router"


@dataclass(frozen=True)
class RouterConfig:
    """Frozen configuration. Construct once, pass everywhere."""

    # --- Where things live ----------------------------------------------------
    gcs_bucket: str = ""
    bq_project: str = ""
    bq_dataset: str = "smart_router"

    # --- Rollout --------------------------------------------------------------
    mode: Mode = "off"
    route_fraction: float = 0.0
    # A permanent control arm, not a rollout phase. It is the only unbiased
    # corpus and the only valid A/B baseline (DESIGN.md 11.4).
    holdout_fraction: float = 0.05

    # --- Embedding (parity: must match sql/train_router.sql) ------------------
    embedding_model: str = "gemini-embedding-2"
    embedding_dims: int = 768
    task_type: str | None = None
    # `global`, not a region. gemini-embedding-2 is not published to us-central1
    # or us-east4: those return 404, the embedder fails open, the breaker latches,
    # and the router passes every turn through while looking healthy. Measured, not
    # assumed. `us` also works and returns bit-identical vectors; `global` is the
    # faster of the two (DESIGN.md 18.7, scripts/check_embedding_parity.py).
    vertex_location: str = "global"
    bq_connection: str = ""
    # A hot-path budget, and the argument for it is entirely about asymmetry.
    #
    # Measured 2026-09-05 against the real endpoint, 616 warm calls over 19 fresh
    # processes at concurrency 1 and 4: p50 147, p95 214, p99 239, max 265 ms.
    # Concurrency does not degrade the body. 2.1% of those calls ran over the old
    # 250 ms bar and twelve of the thirteen were under 315 ms, so 250 was censoring
    # calls that were about to succeed.
    #
    # A censored call costs the whole session, not the turn: one embedding failure
    # pins the session to passthrough (DESIGN.md 11.2), which is ~2,500 billed
    # tokens at §19.17's 543 saved per turn over 4.67 turns. The extra budget is
    # only ever paid by a call already past 250 ms, once per session, on a turn that
    # averages 3,371 ms. 600 ms is 2x the observed non-pathological max; 1,000 ms
    # would also catch the rare correlated multi-second stall but puts the worst
    # case at 4x G3's bar for a further 0.8% of calls (DESIGN.md 19.23).
    #
    # G3 is unaffected: it is a p95 of realised added latency, measured at 214 ms,
    # not a worst case, and it stays green while the over-budget rate is under 5%.
    embed_timeout_ms: int = 600
    # Not a hot-path number. `Embedder.warmup()` pays the ~1.9 s cold start once,
    # before traffic, so this only has to be generous enough that a slow first
    # call still succeeds (DESIGN.md 18.10).
    embed_warmup_timeout_ms: int = 10_000
    embed_cache_size: int = 512
    embed_breaker_threshold: int = 5
    embed_breaker_cooldown_s: float = 30.0
    # How often a continuing embedding failure re-reports itself. One line per
    # failure buries the incident under its own noise; one line per process hides
    # how large it got, which is exactly what happened in ab-20260911m8: 40 lost
    # sessions and a single log line to show for them (DESIGN.md 19.23). Counted,
    # not timed, because this class holds no clock on purpose.
    embed_failure_log_every: int = 25

    # --- Pruning --------------------------------------------------------------
    keep_probability_mass: float = 0.90
    # Parity: must equal `top_n` in sql/train_router.sql, or the gate is calibrated
    # for a set size that is never served. Was 5, then 7 on the 25-tool surface: the
    # trained model covered 0.9649 at N=5 and 0.9852 at N=7. Raised to 11 for the
    # 100-tool surface of DESIGN.md 19.15, where a menu of 7 is 7% of the catalogue
    # rather than 28%. Measured on that surface, realised set recall is 0.8958 at 7
    # and 0.9187 at 11, and the saving falls from 31.1% to 29.3%: two points of recall
    # for under two points of tokens (DESIGN.md 19.16).
    max_tools_kept: int = 11
    # What a missing tool costs, in billed tokens, and how many model calls a session
    # makes. Together they are the whole decline rule: prune when the tokens saved over
    # the session beat the chance of a miss times its price (DESIGN.md 19.5).
    #
    # The price is a dial, not a measurement. §19.1 measured an actual miss at 2,278
    # billed tokens, and pricing it there declines so often that the saving falls to
    # −6.9%. 1,600 is the setting with margin on both gate criteria on the reference
    # corpus: recall 0.9896 against a 0.98 bar, and −10.7% predicted against −10%.
    # Raising it buys recall and gives back saving, monotonically; `score_sets.py
    # --sweep` prints the curve for any corpus.
    #
    # It replaced `min_confidence_to_route`, which asked whether the model was sure.
    # Being unsure only matters when the menu is also cheap, and that question declined
    # two sessions in three to buy 0.0015 of recall (DESIGN.md 19.8).
    miss_cost_tokens: int = 1600
    # Measured at 4.6 on the reference corpus. A session's real length is unknowable on
    # the turn the decision is made, and this is the one place the rule uses a prior.
    expected_session_turns: float = 4.6
    # A per-request early-out, not an economic test. It descends from an assumed 200 tokens
    # per declaration, later measured at 37 on the small surface and 106 on the large one,
    # so read it as "too few tools to be worth the embedding call" and nothing more. Whether
    # pruning pays for a given agent is a question about prompt size against the model's
    # cache floor, which is what `qualify` answers.
    min_tools_to_route: int = 20
    # Tools that always ship, whatever the model predicts. The safe knob of the pair below.
    never_prune: frozenset[str] = frozenset()
    # Tools that ship only when the model asks for them by name. The unsafe knob of the pair,
    # and the exact opposite of `never_prune`: that one adds a tool to every menu, this one
    # takes a tool out of every menu it was not predicted into.
    #
    # It exists because a tool no corpus session ever called has no class in the model, and a
    # tool with no class is passed through untouched on every routed turn (see `pruner.decide`).
    # On the reference agent that is 32 of 100 tools carrying 4,901 of 14,324 declaration
    # tokens: a third of the declarations, in every prompt, for tools the router has the most
    # evidence nobody wants. Nothing else can remove them, because the model cannot predict a
    # label it was never trained on.
    #
    # What a wrong name here costs, stated where the name is typed. A tool listed here is
    # absent from the prompt, so the model cannot call it. The session that wanted it does not
    # get a degraded answer, it gets a miss it cannot recover from within the turn, and it
    # cannot recover from it later either: the kept set is frozen for the session and there is
    # no growth path back (DESIGN.md 7.2, rule 1). Only that agent's own traffic can say
    # whether a tool is really dead, so this ships empty and stays empty until an operator has
    # looked. `startup_checks` prints the candidate list and what it costs, and `report_ab.py
    # --prunable` is the detector: a named tool the holdout arm calls and the routed arm never
    # calls is a name that was wrong.
    #
    # The two sets are resolved in the safe direction. A tool named in both still ships, and
    # `startup_checks` says so at boot rather than letting the contradiction pass silently.
    prunable_by_default: frozenset[str] = frozenset()

    # --- Cache protection -----------------------------------------------------
    # Two guards, in order of how much they know. The floor guard compares the
    # agent's typical prompt against `MIN_CACHE_TOKENS`, and is asked once per
    # session rather than once per turn. The watch compares what pruned and
    # unpruned requests were actually billed, runs every turn, and takes over
    # once there is enough of both to compare (DESIGN.md 18.13, 18.19).
    #
    # v1 replaced `tokens_per_declaration` and `system_prompt_tokens` with a
    # measurement of the real request. They were 200 and 1000 against a measured
    # 37 and 1003, and the first of those errors is what let the router raise
    # bills by 25% while believing it was cutting them (DESIGN.md 18.12).
    cache_watch_enabled: bool = True
    # Per arm, not in total. A verdict drawn from 200 pruned calls and 3 unpruned
    # ones is a verdict about nothing. 50 rather than 25: replaying the shipped
    # rule over the five logged A/B runs, 25 is small enough to false-fire on
    # ab-20260910a at 695 billed against 595, on a run that finished at 645
    # against 702. 50 is the smallest value that separates all five cleanly, and
    # it is reached about four and a half minutes into a run of that traffic.
    cache_watch_min_samples: int = 50
    # How much more expensive pruning has to be, as a fraction of the unpruned
    # cost, before it is called harmful. Small on purpose: the router exists to
    # spend less, so breaking even is already a failure, and the margin is here
    # only to stop sampling noise demoting an agent (DESIGN.md 18.16).
    cache_watch_margin: float = 0.05
    # What a cached input token bills at, relative to an uncached one. 0.10 on
    # Vertex for the Gemini families this ships against. It is the number that
    # decides whether pruning can ever pay, so it is configurable rather than
    # buried: at 0.10 a prune has to survive removing tokens that were already
    # 90% off.
    cached_token_multiplier: float = 0.10
    # How many turns an agent has to contribute before `steady_state.PromptSizes` will
    # state a typical prompt size. The floor guard uses that size instead of the size of
    # the turn in front of it, so that it answers once, for the session, rather than
    # answering "no" on a short first turn and "yes" on the second (DESIGN.md 18.19).
    steady_state_min_samples: int = 25
    # How many of an agent's counted calls have to go by before a demotion is lifted and
    # pruning is tried again. Calls rather than seconds, so the watch stays clock-free and
    # so a quiet agent does not spend its retry on stale conditions. 500 is an exposure
    # budget, not a measurement: it puts min_samples/(500 + min_samples) of the agent's
    # traffic into the retry, roughly 9% of calls once, at whatever per-call excess earned
    # the demotion.
    cache_watch_cooloff_calls: int = 500
    # How many times an agent may be brought back. The cool-off doubles each time, and past
    # this count the demotion is permanent for the life of the process, so an agent that
    # genuinely cannot be pruned stops flapping instead of retrying forever at a fixed rate.
    cache_watch_max_repromotions: int = 2
    # An agent whose sessions are one model call each has every call excluded from the
    # comparison above, so it never reaches a verdict. The fallback compares its first
    # calls to the holdout's first calls, which removes the turn-position confound
    # completely because both sides are turn one. These two numbers are the brakes on it.
    # Ten times `cache_watch_min_samples`: a turn-one price swings on whether other
    # sessions have left a prefix warm, and the ratio is also the gate, since reaching
    # 500 in both arms before either later-call arm reaches 50 takes a traffic mix that
    # is more than nine tenths single-call sessions.
    cache_watch_first_call_min_samples: int = 500
    # Three times `cache_watch_margin`, for the same reason: a deployment whose menu
    # prefixes are young looks harmful on turn one before it looks like anything, and
    # demoting on the coldest sample available is the DESIGN.md 18.21 failure repeating.
    cache_watch_first_call_margin: float = 0.15

    # --- Tool result cache ----------------------------------------------------
    cache_enabled: bool = True
    # Name to TTL, and the only way a tool becomes cacheable. There is deliberately no
    # default TTL to fall back on: a knob that caches a tool nobody named is the one
    # mistake in DESIGN.md 8 that cannot be undone by reading the config.
    cacheable_tools: Mapping[str, int] = field(default_factory=dict)
    cache_min_ttl_s: int = 30
    cache_max_ttl_s: int = 3600
    cache_max_entries: int = 512
    cache_probe_rate: float = 0.03
    cache_max_payload_bytes: int = 65536

    # --- Model artifacts ------------------------------------------------------
    pin_version: str | None = None
    max_model_age_days: int = 14
    max_artifact_bytes: int = 50_000_000
    local_cache_dir: str = "/tmp/adk_routers"
    environment: str = "prod"
    # How long the first turn of a process is willing to wait for the artifact, and how
    # long a failed load is believed before anyone tries again.
    #
    # Neither is a hot-path budget. The load happens once per agent per process and the
    # turn waiting on it passes through, so the cost of waiting is one unrouted turn and
    # the cost of not waiting is a process that never routes. 30 s is generous against a
    # measured 2.2 s discovery-plus-download, which is the point: it is a stuck-connection
    # bound, not a latency target.
    #
    # The retry window is what stops a transient failure from being permanent. A negative
    # entry is a decision made once with no evidence that it will still be true in an hour,
    # and the previous behaviour cached it for the life of the process. 300 s bounds the
    # damage of a bad minute at five minutes of passthrough, and bounds the cost of a
    # genuinely absent artifact at one listing call per agent per five minutes, which is
    # nothing (DESIGN.md 19.73).
    model_load_timeout_s: float = 30.0
    model_retry_after_s: float = 300.0
    # How long a process will serve a version after a newer one exists. 0 disables the
    # re-poll, and so does `pin_version`, which is the escape hatch for anyone who wants
    # versions to move only on deploys.
    #
    # Before this, a loaded router was kept for the life of the process and a published
    # artifact reached serving only through a redeploy. That is the one place the project
    # fell short of its own claim, because the whole point of publishing a better model is
    # that agents get it (DESIGN.md 21.4).
    #
    # 900 s is chosen against the cost of being wrong in each direction. Too long and a
    # process serves a slightly worse model for a while, which is bounded and small. Too
    # short and every replica lists the bucket constantly, which is a real bill on a large
    # fleet. At 900 a hundred replicas make four hundred list calls an hour, and the common
    # case stops at the list: a download happens only when the version actually changed.
    refresh_interval_s: float = 900.0

    # --- Composable Judge Layer (Cold-Start & Hybrid Ensembling) --------------
    # Which scoring judges participate on the first user turn of a session.
    # Defaults to `("xgboost",)` so existing deployments behave identically.
    #
    # Supported built-in names:
    # - `"xgboost"`: Local in-memory XGBoost booster exported from BQML (`model.ubj`).
    # - `"prototype"`: Zero-training-data semantic tool-schema & prerequisite-graph
    #   judge sharing the same Turn-1 `gemini-embedding-2` query vector (0 extra RPCs
    #   once tool schemas are embedded for the agent). Routes during Day-0 cold start
    #   or when the GCS artifact has aged out, and scores classless/newly-deployed
    #   tools when paired with `"xgboost"`.
    # - `"llm_live"`: Optional zero-training-data Gemini Flash-Lite router reusing the
    #   ADK agent's existing Gemini API connection when no trained artifact exists.
    #
    # Combining `("xgboost", "prototype")` runs both in memory on the single Turn-1
    # embedding vector: `xgboost` governs trained tools and empirical menu snapping,
    # while `prototype` scores classless tools so uncalled ones can be safely pruned
    # instead of riding on every prompt, and newly deployed tools are rescued when
    # semantically relevant.
    judges: tuple[str, ...] = ("xgboost",)
    prototype_weight: float = 0.15
    # Cap on tools kept when routing solely via cold-start judges (`prototype` or
    # `llm_live`) with no trained co-occurrence menus to snap to. Measured on the
    # 100-tool reference corpus: k=15 reaches 0.9248 set recall and 89.2% token
    # efficiency with zero training rows.
    cold_start_max_tools_kept: int = 15
    # Optional count of deterministic canonical menus synthesized from the live tool
    # catalog (`CatalogIndex`) when no trained artifact menu pool is loaded. 0 keeps
    # bespoke per-session cold-start pruning (`source="pruned"`).
    cold_start_menu_count: int = 0
    llm_judge_model: str = "gemini-2.5-flash-lite"
    llm_judge_timeout_ms: int = 800

    # --- Bounded caches -------------------------------------------------------
    session_decision_cache_size: int = 4096

    def with_mode(self, mode: Mode) -> RouterConfig:
        """Return a copy with a different mode. Used by the kill switch."""
        return replace(self, mode=mode)


def min_cache_tokens(model_name: str | None) -> int | None:
    """The explicit-cache token floor for a Gemini model, or None if unknown.

    Mirrors ADK's `_minimum_cache_tokens`. Unknown models return None, which the
    caller must read as "no guard possible", not as "no floor".
    """
    if not model_name:
        return None
    bare = model_name.rsplit("/", maxsplit=1)[-1]
    for prefix, floor in MIN_CACHE_TOKENS.items():
        if bare.startswith(prefix):
            return floor
    return None


def load_config(**overrides: object) -> RouterConfig:
    """Build a RouterConfig from environment variables, then apply overrides.

    Environment wins over the dataclass defaults; explicit keyword overrides win
    over the environment. Only the operational knobs are exposed as env vars, so
    an operator can flip the router off or pin a version without a code change
    (DESIGN.md 11.1). Anything unset or unparseable falls back to the default
    rather than raising, because a config error must not take down serving.
    """
    env: dict[str, object] = {}
    _put_str(env, "gcs_bucket", "SMART_ROUTER_GCS_BUCKET")
    _put_str(env, "bq_project", "SMART_ROUTER_BQ_PROJECT")
    _put_str(env, "bq_dataset", "SMART_ROUTER_BQ_DATASET")
    _put_str(env, "bq_connection", "SMART_ROUTER_BQ_CONNECTION")
    _put_str(env, "vertex_location", "SMART_ROUTER_VERTEX_LOCATION")
    _put_str(env, "environment", "SMART_ROUTER_ENVIRONMENT")
    _put_str(env, "local_cache_dir", "SMART_ROUTER_LOCAL_CACHE_DIR")
    _put_float(env, "route_fraction", "SMART_ROUTER_ROUTE_FRACTION")
    _put_float(env, "holdout_fraction", "SMART_ROUTER_HOLDOUT_FRACTION")
    _put_float(env, "refresh_interval_s", "SMART_ROUTER_REFRESH_INTERVAL_S")
    _put_float(env, "prototype_weight", "SMART_ROUTER_PROTOTYPE_WEIGHT")
    _put_int(env, "cold_start_menu_count", "SMART_ROUTER_COLD_START_MENU_COUNT")

    raw_mode = os.environ.get("SMART_ROUTER_MODE")
    if raw_mode in ("off", "active"):
        env["mode"] = raw_mode

    raw_judges = os.environ.get("SMART_ROUTER_JUDGES")
    if raw_judges:
        parsed = tuple(j.strip() for j in raw_judges.split(",") if j.strip())
        if parsed:
            env["judges"] = parsed

    pin = os.environ.get("SMART_ROUTER_PIN_VERSION")
    if pin:
        env["pin_version"] = pin

    return RouterConfig(**{**env, **overrides})  # type: ignore[arg-type]


def _put_str(target: dict[str, object], key: str, var: str) -> None:
    value = os.environ.get(var)
    if value:
        target[key] = value


def _put_float(target: dict[str, object], key: str, var: str) -> None:
    raw = os.environ.get(var)
    if not raw:
        return
    try:
        target[key] = float(raw)
    except ValueError:
        return


def _put_int(target: dict[str, object], key: str, var: str) -> None:
    raw = os.environ.get(var)
    if not raw:
        return
    try:
        target[key] = int(raw)
    except ValueError:
        return
