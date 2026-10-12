"""What the router noticed about the App it was installed into. Diagnostics, run once.

Most of these are pure log lines aimed at whoever wired the App up. They catch silent
misconfigurations, which are the worst kind: the router keeps running and looks fine.

* Plugin order. If BQAA is registered before the router, every cache hit is logged as a real
  tool call with `total_ms=0`, so measured TTLs inflate without bound and the next model is
  trained on the router's own echo (DESIGN.md 6.3, 11.5).
* Context cache config. Without one the prefix is never cached, so session-stable pruning saves
  only the declaration tokens and none of the cache-hit discount it was designed around (6.3).
* `prunable_by_default`, twice. Once for the tools it contradicts `never_prune` about, and once
  for a holdout too small to notice that a name in it was wrong.
* Tools with no class, per agent. They ride through every prune untouched and nothing in the
  model can ever remove them, so the only way an operator learns what they cost is to be told.
* Version skew, per agent. An artifact measured at one set size and served at another still
  predicts correctly, and its published recall and saving stop describing what the process
  does. Two settings decide that size, `max_tools_kept` and `keep_probability_mass`, and
  either one can drift on its own (DESIGN.md 19.72, 19.74).
* Live/bidi streaming. The router cannot prune a live run and passes it through, which from
  outside is indistinguishable from a router that is working and finding nothing to remove
  (DESIGN.md 7.1, 19.53).

One check does change behaviour. An agent that declares its own `before_tool_callback` is
recorded here, and the tool result cache refuses that agent, because a plugin-level cache hit
short-circuits before the agent's own callback chain runs and that chain is where most ADK
users put authorization (DESIGN.md 8.3).

Nothing here raises. A failed diagnostic must never be the reason routing stops.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import math
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from adk_smart_router.config import FRAMEWORK_TOOLS, RouterConfig

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.tools.tool_context import ToolContext

    from adk_smart_router.model_store import ArtifactMetadata

logger = logging.getLogger(__name__)


def judge_wiring_issues(
    cfg: RouterConfig,
    *,
    has_custom_judges: bool = False,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
) -> list[str]:
    """Configuration or missing-package problems for the configured `judges`.

    Shared between `StartupChecks` (which logs once on the first turn) and
    `smart-router doctor` (which fails pre-deploy). Accepting `find_spec` makes missing
    optional dependencies (`xgboost`, `google-cloud-storage`) testable without mutating
    `sys.modules`.
    """
    issues: list[str] = []
    if not cfg.judges and not has_custom_judges:
        issues.append(
            "RouterConfig.judges is empty and no custom_judges are registered, so no routing "
            "judge can run."
        )
        return issues

    if "xgboost" in cfg.judges:
        missing = [
            pkg
            for mod, pkg in (
                ("xgboost", "xgboost"),
                ("google.cloud.storage", "google-cloud-storage"),
            )
            if find_spec(mod) is None
        ]
        if missing:
            joined = ", ".join(missing)
            issues.append(
                f"RouterConfig.judges={cfg.judges!r} includes 'xgboost', which requires "
                f"{joined} (not installed). Install {joined} (or google-adk[smart-router]), "
                f"or set SMART_ROUTER_JUDGES=prototype for zero-dependency cold-start routing."
            )
        has_aux = any(j in ("prototype", "llm_live") for j in cfg.judges) or has_custom_judges
        if cfg.mode == "active" and not cfg.gcs_bucket and not has_aux:
            issues.append(
                "mode='active' with judges=('xgboost',) and no gcs_bucket configured has no "
                "artifact source and no cold-start fallback judge, so every session will pass "
                "through as 'disabled'. Set SMART_ROUTER_GCS_BUCKET, or include 'prototype' "
                "in SMART_ROUTER_JUDGES."
            )
    return issues


# The smallest holdout share at which `prunable_by_default` can still correct itself.
#
# A tool named there is gone from every routed session, so the only traffic that can still
# call it is the holdout, and `sql/train_router.sql` makes the holdout the sole source of
# training rows anyway. Demand for a named tool is therefore discoverable at exactly the
# holdout's rate and nowhere else. Measured on the ab-v2 run: 3 of 289 holdout sessions, 1.0%,
# called a tool the model had no class for. At the shipped `holdout_fraction` of 0.05 that is
# one informative session in about two thousand, which no retrain cadence can act on; at 0.10
# it is one in a thousand, which a weekly retrain can.
#
# It is a round number and not an optimum. Nobody has worked out how much holdout a given
# list size needs, and the honest statement is that 0.05 is demonstrably too little rather
# than that 0.10 is enough.
MIN_HOLDOUT_FOR_PRUNABLE = 0.10


def _is_bq_analytics(plugin: object) -> bool:
    """Duck-typed so the router never has to import BQAA, which is an optional extra."""
    cls = type(plugin)
    fingerprint = f"{cls.__module__}.{cls.__name__}".lower()
    return "bigquery" in fingerprint and "analytics" in fingerprint


def _gates_tools(agent: Any) -> bool:  # noqa: ANN401 - any BaseAgent subclass, duck-typed
    """True if the agent declares its own `before_tool_callback`."""
    canonical = getattr(agent, "canonical_before_tool_callbacks", None)
    if canonical is not None:
        return bool(canonical)
    return bool(getattr(agent, "before_tool_callback", None))


def agent_tree(agent: Any) -> list[Any]:  # noqa: ANN401 - any BaseAgent subclass, duck-typed
    """Every agent reachable from `agent`: up to the root, then down through `sub_agents`.

    The plugin can be handed any node in the tree, and the cache decision has to be made for
    all of them before the first tool call, not lazily as each agent is first seen.
    """
    root = agent
    while getattr(root, "parent_agent", None) is not None:
        root = root.parent_agent
    found: list[Any] = []
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        found.append(node)
        stack.extend(getattr(node, "sub_agents", None) or [])
    return found


class StartupChecks:
    """Runs the checks at most once each, and remembers which agents gate their own tools."""

    def __init__(self, owner: object, cfg: RouterConfig) -> None:
        # `owner` is the plugin instance, needed only to find its own position in the plugin
        # list. Holding it here keeps the ordering check out of the plugin itself.
        self._owner = owner
        # Only the fields the kill switch cannot move are read from this, so holding the
        # instance the plugin was built with is safe even after `set_mode` rebinds the plugin's.
        self._cfg = cfg
        self._done = False
        # Weak, not a set of `id()` values. An app that builds its agent tree per request frees
        # the old tree, CPython recycles the address, and a plain `set[int]` would then report a
        # brand new agent as already scanned. If that new agent is the one with the
        # authorization callback, the only DESIGN.md 8.3 control is silently off. A weak
        # mapping drops its entry when the agent dies, so an address cannot be inherited.
        self._scanned: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
        self._gated: set[str] = set()
        self._classless_reported: set[str] = set()
        self._skew_reported: set[str] = set()
        self._live_reported = False

    def _seen(self, agent: Any) -> bool:  # noqa: ANN401 - ADK BaseAgent, duck-typed
        return self._scanned.get(id(agent)) is agent

    def _mark_seen(self, agent: Any) -> None:  # noqa: ANN401 - ADK BaseAgent, duck-typed
        # A TypeError means the agent is not weak-referenceable. Rescanning it every turn costs
        # a few attribute reads and the scan is idempotent, so the skip is an optimisation and
        # never a control.
        with contextlib.suppress(TypeError):
            self._scanned[id(agent)] = agent

    def gates_own_tools(self, agent_name: str) -> bool:
        """True if this agent has its own `before_tool_callback` and must not be cached."""
        return agent_name in self._gated

    def run(self, ctx: CallbackContext | ToolContext) -> None:
        """Run whatever has not run yet. Swallows everything: diagnostics never gate."""
        try:
            ictx = ctx.get_invocation_context()
            if not self._done:
                self._done = True
                self._run_once_only(ictx)
            self._scan_agents(getattr(ictx, "agent", None))
        except Exception:
            logger.exception("smart_router: startup checks failed")

    def _run_once_only(self, ictx: Any) -> None:  # noqa: ANN401 - ADK InvocationContext
        """The three first-turn checks, each isolated from the other two.

        `_done` is already set, so a check that raises does not get a second chance. That is
        deliberate: a diagnostic that fails on turn one fails the same way on every turn, and
        logging it once beats logging it forever. What is not deliberate is one of them taking
        the others down with it. They are three independent statements about three independent
        misconfigurations, and the one most likely to raise is the one that walks ADK's plugin
        list, which is exactly the structure most likely to change under a version bump.

        Isolating them protects `_scan_agents` as well, which is not a diagnostic: it is what
        finds the agents that gate their own tools, and an agent it never scanned is one whose
        `before_tool_callback` the cache does not know to respect (DESIGN.md 8.3).
        """
        checks: tuple[Callable[[], None], ...] = (
            lambda: self._check_plugin_order(ictx),
            lambda: self._check_context_cache(ictx),
            self._check_prunable_by_default,
            self._check_judge_wiring,
        )
        for check in checks:
            try:
                check()
            except Exception:
                logger.exception("smart_router: a startup check failed")

    def _check_judge_wiring(self) -> None:
        has_custom = bool(getattr(self._owner, "_custom_judges", ()))
        for issue in judge_wiring_issues(self._cfg, has_custom_judges=has_custom):
            logger.warning("smart_router: %s", issue)

    def _check_plugin_order(self, ictx: Any) -> None:  # noqa: ANN401 - ADK InvocationContext
        plugins = list(getattr(ictx.plugin_manager, "plugins", []))
        if self._owner not in plugins:
            return
        for other in plugins[: plugins.index(self._owner)]:
            if _is_bq_analytics(other):
                logger.warning(
                    "smart_router: %s is registered BEFORE SmartRouterPlugin. Every cache hit "
                    "will be logged as a real tool call with total_ms=0 and the cache payload as "
                    "tool_result, so measured TTLs inflate without bound and the router is "
                    "trained on its own echo. Move SmartRouterPlugin first in App(plugins=[...]) "
                    "(docs/operations.md#plugin-order).",
                    type(other).__name__,
                )
                return

    def _check_context_cache(self, ictx: Any) -> None:  # noqa: ANN401 - ADK InvocationContext
        if getattr(ictx, "context_cache_config", None) is None:
            logger.warning(
                "smart_router: this App has no context_cache_config, so the prefix is never "
                "cached and session-stable pruning saves nothing beyond the declaration tokens "
                "themselves (docs/install.md)"
            )

    def _check_prunable_by_default(self) -> None:
        """Two ways the unsafe knob can be set wrong, neither of which shows up in serving.

        A contradiction between the two lists is resolved in `pruner.decide` by keeping the
        tool, which is the right answer and an invisible one: the operator gets the behaviour
        of `never_prune` while reading a config that says otherwise. Said out loud here.

        A holdout too small to notice a wrong name is the other one, and it is worse, because
        nothing at all goes wrong until a customer asks for a tool that is not there.
        """
        cfg = self._cfg
        if not cfg.prunable_by_default:
            return

        both = sorted(cfg.prunable_by_default & (cfg.never_prune | FRAMEWORK_TOOLS))
        if both:
            logger.warning(
                "smart_router: %s named in both prunable_by_default and the protected set "
                "(never_prune plus %s). They will keep shipping, because the safe reading of a "
                "contradiction wins: %s. Remove them from one list so the config says what the "
                "router does.",
                len(both),
                sorted(FRAMEWORK_TOOLS),
                ", ".join(both),
            )

        if cfg.holdout_fraction < MIN_HOLDOUT_FOR_PRUNABLE:
            logger.warning(
                "smart_router: prunable_by_default names %s tools while holdout_fraction is "
                "%.2f. A named tool is absent from every routed prompt, so the holdout is the "
                "only traffic left that can call it and the only traffic that trains the next "
                "model. Below %.2f a session that wants one arrives too rarely to be seen, and "
                "a wrong name stays wrong. Raising the holdout is the price of this knob "
                "(docs/operations.md).",
                len(cfg.prunable_by_default),
                cfg.holdout_fraction,
                MIN_HOLDOUT_FOR_PRUNABLE,
            )

    def live_streaming(self, mode: str) -> None:
        """Say once that a live run is passing through untouched, and why it has to.

        The router's pass-through for `streaming_mode != NONE` is correct and silent, and
        silent is the problem. An operator who installs the plugin into a voice or bidi app
        sees no error, no warning and no saving, which looks exactly like a router that ran
        and found nothing worth pruning. The two cases want opposite responses, so the
        distinction has to be said out loud.
        """
        if self._live_reported:
            return
        self._live_reported = True
        logger.warning(
            "smart_router: this run uses streaming_mode=%s, so the router is passing it "
            "through untouched and will save nothing on it. A live run sends its tool "
            "declarations once, when the socket opens, and the callback the router prunes in "
            "does not fire until after that; worse, the request it is handed shares its "
            "config with the open connection, so a prune there would be permanent rather "
            "than merely late. This is a limitation of the injection point and not of the "
            "prediction (docs/architecture.md#one-turn-start-to-finish).",
            mode,
        )

    def classless_tools(
        self,
        agent_name: str,
        live: frozenset[str],
        known: frozenset[str],
        cost: Callable[[frozenset[str]], int],
    ) -> None:
        """Report, once per agent, the live tools the model has no class for and their price.

        These are the tools `pruner.decide` passes through untouched, forever: the model was
        never trained on a label for them, so it can never predict one, so no prune can ever
        reach them. On the reference agent they are a third of the declarations. An operator
        cannot act on that without the names, and nothing else in the system will ever say
        them, which is the whole reason this line exists.

        `cost` is a callable and not an int because measuring it walks the request, and this
        is called from the hot path. The measurement is only paid on the one turn per agent
        that reports.
        """
        if agent_name in self._classless_reported:
            return
        self._classless_reported.add(agent_name)

        classless = live - known - self._cfg.prunable_by_default
        if not classless:
            return
        logger.info(
            "smart_router: agent %s declares %s of %s tools the router has no class for, about "
            "%s tokens of every prompt. Either they are new, or no training session ever called "
            "them; either way no prediction can remove them and only prunable_by_default can, "
            "and only if you already know they are dead. They are: %s",
            agent_name,
            len(classless),
            len(live),
            cost(classless),
            ", ".join(sorted(classless)),
        )

    def artifact_skew(self, agent_name: str, meta: ArtifactMetadata) -> None:
        """Say once, per agent, that this config was not the one the artifact was gated on.

        Everything about an artifact that would make it wrong to load is refused in
        `model_store._check_compatible`: the wrong environment, the wrong embedding model, the
        wrong width. Neither setting checked here is in that list and neither must be, because
        a mismatch does not make the model wrong. It makes the gate's numbers describe a set
        size nobody serves. `train_router.sql` measures coverage at the top `top_n`
        predictions and coverage_mass at `keep_probability_mass`, and the recall and saving in
        `gate_metrics` are measured there. Serve a different size and the artifact is still
        fine, the promise attached to it is not.

        Both are checked because both decide the size, and without menus the mass rule is
        usually the one that binds: on the reference corpus the cap kept 11 tools and the mass
        rule stopped at 8.8 (docs/training.md). An operator who moves either has moved the
        served set away from the measured one.

        An absent setting means an artifact published before that number was written into the
        metadata, and silence is the only honest answer there.

        The last line is the sharper one. With menus, `pruner.decide` takes a menu whole and
        reads neither setting, so an operator who lowers one expecting shorter declarations
        changes nothing at all. That is worth saying at the moment they have clearly tried.
        """
        if agent_name in self._skew_reported:
            return
        self._skew_reported.add(agent_name)

        skewed = False
        published_n = meta.gate_metrics.get("top_n")
        if published_n is not None and int(published_n) != self._cfg.max_tools_kept:
            skewed = True
            logger.warning(
                "smart_router: agent %s is serving max_tools_kept=%s against an artifact gated "
                "at top_n=%s. The model still predicts correctly; its published recall and "
                "saving were measured at a different set size, so the numbers in gate_metrics "
                "no longer describe what this process does. Match RouterConfig.max_tools_kept "
                "to top_n in sql/train_router.sql, or retrain at the size you want to serve "
                "(docs/training.md).",
                agent_name,
                self._cfg.max_tools_kept,
                int(published_n),
            )

        published_mass = meta.gate_metrics.get("keep_probability_mass")
        if published_mass is not None and not math.isclose(
            published_mass, self._cfg.keep_probability_mass, abs_tol=1e-9
        ):
            skewed = True
            logger.warning(
                "smart_router: agent %s is serving keep_probability_mass=%s against an artifact "
                "whose coverage_mass was measured at %s. Without menus this setting, not "
                "max_tools_kept, is usually the one that decides how many tools ship, so the "
                "served coverage in gate_metrics describes a different cut than this process "
                "makes. Match RouterConfig.keep_probability_mass to keep_probability_mass in "
                "sql/train_router.sql, or retrain at the mass you want to serve "
                "(docs/training.md#criterion-1-when-there-are-no-menus).",
                agent_name,
                self._cfg.keep_probability_mass,
                published_mass,
            )

        if skewed and meta.menus:
            logger.warning(
                "smart_router: and this artifact ships %s menus, of %s tools, which are taken "
                "whole. Neither max_tools_kept nor keep_probability_mass is read on that path, "
                "so changing them has no effect on what agent %s declares.",
                len(meta.menus),
                ", ".join(str(size) for size in sorted({len(menu) for menu in meta.menus})),
                agent_name,
            )

    def _scan_agents(self, agent: Any) -> None:  # noqa: ANN401 - ADK BaseAgent, duck-typed
        if agent is None or self._seen(agent):
            return
        self._mark_seen(agent)
        for node in agent_tree(agent):
            self._mark_seen(node)
            if not _gates_tools(node) or node.name in self._gated:
                continue
            self._gated.add(node.name)
            logger.warning(
                "smart_router: agent %s declares a before_tool_callback, so the tool result "
                "cache is disabled for it; a cache hit would skip that callback entirely "
                "(docs/threat-model.md#the-tool-result-cache)",
                node.name,
            )
