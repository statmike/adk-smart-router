"""When a tool call may be answered from memory, and when a result may be kept.

`tool_cache.py` owns the key, the clock and the bytes. This module owns the decision to use
them, which is a different job with a different failure mode: getting the key wrong corrupts
data, getting the eligibility wrong leaks it or skips a side effect.

Five rules gate a serve, and all five have to pass:

1. The router is on, and the tool was declared cacheable with a usable TTL.
2. The agent does not gate its own tools. A plugin-level hit short-circuits before the agent's
   own `before_tool_callback` chain runs, and that chain is where authorization usually lives
   (DESIGN.md 8.3).
3. The session service is not `InMemorySessionService`. AgentTool mints a fresh session per
   call over one of those, so session-scoped entries could never hit and would only accrete
   dead keys (DESIGN.md 7.6).
4. The caller has a non-empty `user_id`. Identity is inside the hash, so no identity means no
   cache, never a shared scope.
5. This call did not draw the refresh probe.

Storing has one extra rule that serving does not: a tool that wrote to `tool_context.actions`
is blocklisted on the spot. A declaration that a tool is cacheable is a human assertion, and
this is the mechanical verification of it (DESIGN.md 8.2).
"""

from __future__ import annotations

import copy
import logging
import random
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from google.adk.sessions import InMemorySessionService

from adk_smart_router.config import CACHE_HIT_KEY, Mode, RouterConfig
from adk_smart_router.provenance import CacheServeRecord, emit
from adk_smart_router.startup_checks import StartupChecks
from adk_smart_router.tool_cache import (
    ToolResultCache,
    actions_are_default,
    cache_key,
    user_scope,
)

if TYPE_CHECKING:
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

# Where `serve` leaves the note that `store` reads: an attribute on the ToolContext, which ADK
# builds fresh for each function call and passes to both callbacks (`functions.py`, Steps 1 and
# 4). Session state cannot carry this. `State.__setitem__` writes through to
# `invocation_context.session.state`, which is one dict shared by every tool call in the turn,
# and ADK runs a turn's calls concurrently under `asyncio.gather`. A turn that hit the cache on
# one tool and called a writer on another would have the writer read the reader's note and skip
# its purge, which is read-your-own-writes failing silently (DESIGN.md 8.4).
#
# Scoping it to the call object also means the router now writes no session state at all, so
# there is nothing for a co-registered logging plugin to see in `actions.state_delta`.
HIT_MARKER_ATTR = "_smart_router_served_from_cache"

# The second thing `serve` leaves for `store`, on the same object and for the same reason: the
# principal's purge count as it stood before the tool ran. Concurrency again, in the other
# direction. A reader that misses the cache and a writer that purges it can be in flight
# together, and if the reader's result lands after the purge it stores an answer computed
# before the write. Comparing the count closes that window (DESIGN.md 8.4).
PURGE_EPOCH_ATTR = "_smart_router_purge_epoch"


def tool_version(tool: BaseTool) -> str:
    """Whatever distinguishes two builds of the same tool name.

    `BaseTool` has no version field, so the implementing class is the best proxy available. The
    cache lives in one process, so a redeploy invalidates every entry anyway.
    """
    declared = getattr(tool, "version", None)
    return str(declared) if declared else type(tool).__name__


class CacheHooks:
    """The serve and store halves of the tool result cache, as one object.

    Both halves compute the cache key the same way, from the same six fields. They have to: a
    read site and a write site that disagree by one field produce a cache that stores forever
    and hits never, silently. That is why `_key_for` exists and why neither half builds a key
    of its own.
    """

    def __init__(
        self,
        cfg: RouterConfig,
        startup: StartupChecks,
        *,
        cache: ToolResultCache | None = None,
        clock: Callable[[], float] = time.monotonic,
        sampler: Callable[[], float] = random.random,
    ) -> None:
        self._cfg = cfg
        self._startup = startup
        self._clock = clock
        self._sampler = sampler
        self.cache = cache if cache is not None else ToolResultCache(cfg, clock=clock)

    def set_mode(self, mode: Mode) -> None:
        """Follow the plugin's kill switch. `RouterConfig` is frozen, so this rebinds it.

        The plugin owns the config and this object holds its own reference to it, so the two
        would drift the moment one of them was rebound alone. The plugin calls this; nothing
        else should.
        """
        self._cfg = self._cfg.with_mode(mode)

    # --- Serve ----------------------------------------------------------------

    def serve(
        self,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        *,
        decision_id: str,
    ) -> dict[str, Any] | None:
        """A memoized result for this call, or None to let the tool run for real."""
        # Stamped before any gate, because it dates the moment the tool call begins and the
        # gates below decide something else. A call that is not served here may still store.
        setattr(tool_context, PURGE_EPOCH_ATTR, self.cache.purge_epoch(self._scope(tool_context)))
        if not self._allowed(tool, tool_context):
            return None
        key = self._key_for(tool, tool_args, tool_context)
        if key is None:
            return None
        entry = self.cache.get(key)
        if entry is None:
            return None

        age_s = max(0.0, self._clock() - entry.stored_at)
        probe = self._sampler() < self._cfg.cache_probe_rate
        emit(
            tool_context,
            CacheServeRecord(
                decision_id=decision_id,
                agent_name=tool_context.agent_name,
                session_id=tool_context.session.id,
                tool_name=tool.name,
                entry_age_s=age_s,
                ttl_used=entry.ttl_used,
                ttl_source=entry.ttl_source,
                probe=probe,
            ),
        )
        if probe:
            # A refresh probe runs the tool for real. It restores unbiased inter-call timing for
            # the TTL estimator and is the only detector for a tool whose answer moved without
            # anyone noticing (DESIGN.md 8.6).
            return None

        # A bare flag, never the key: the key is a hash over the tool arguments, and arguments
        # are the one thing provenance deliberately never records, because a hash over a small
        # argument space is reversible. Nothing reads this value; only its presence matters.
        setattr(tool_context, HIT_MARKER_ATTR, True)
        # The envelope goes last so it cannot be overwritten by a payload key of the same name.
        # `get` already returned a copy. Copying again keeps that a property of this call site
        # rather than a promise the cache has to keep forever (DESIGN.md 8.5).
        return {
            **copy.deepcopy(dict(entry.payload)),
            CACHE_HIT_KEY: {"cache_hit": True, "age_ms": round(age_s * 1000.0)},
        }

    # --- Store ----------------------------------------------------------------

    def store(
        self,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> None:
        """Keep this result if the tool is eligible, and invalidate if it was a writer."""
        if self._take_hit_marker(tool_context):
            return  # a read must never extend an entry's life (DESIGN.md 8.5)
        if not self._cfg.cache_enabled:
            return

        scope = self._scope(tool_context)
        # The allowlist is tested before the actions check on purpose. A side-effecting tool
        # that nobody declared cacheable must still trigger the purge below, so it cannot be
        # short-circuited by an eligibility test first.
        if tool.name not in self._cfg.cacheable_tools:
            # Read-your-own-writes. A TTL bounds staleness against the world and does nothing
            # against the agent's own writes, which are the fastest invalidator of all
            # (DESIGN.md 8.4).
            self.cache.purge_user(scope)
            return

        if not actions_are_default(tool_context.actions):
            logger.warning(
                "smart_router: %s writes to tool_context.actions; refusing to cache it",
                tool.name,
            )
            self.cache.block(tool.name)  # drops what it already stored, permanently
            self.cache.purge_user(scope)  # it wrote somewhere: treat it as a writer, not a read
            return

        ttl = self.cache.ttl_for(tool.name)
        if ttl is None or not self._allowed(tool, tool_context):
            return
        if not isinstance(result, Mapping):
            return  # only a dict result can be replayed as a tool response
        if CACHE_HIT_KEY in result:
            # The reserved key is how the warehouse tells a cache hit from a real tool call, and
            # every training and TTL query filters on it. A tool that returns it, most likely by
            # echoing JSON from a third-party MCP server, could either exclude its own real
            # executions from the corpus or smuggle a forged label into it. Refuse the payload.
            logger.warning(
                "smart_router: %s returned the reserved key %r; refusing to cache it",
                tool.name,
                CACHE_HIT_KEY,
            )
            return

        key = self._key_for(tool, tool_args, tool_context)
        if key is None:
            return
        if getattr(tool_context, PURGE_EPOCH_ATTR, 0) != self.cache.purge_epoch(scope):
            # Something wrote for this principal while the tool was running, so this result may
            # already be the old answer. Storing it would reinstate what the purge just dropped.
            return
        self.cache.put(
            key,
            result,
            user_scope=scope,
            tool_name=tool.name,
            ttl_s=ttl,
            ttl_source="declared",
        )

    # --- Shared gates ----------------------------------------------------------

    def _scope(self, tool_context: ToolContext) -> str:
        """The principal whose cache this call reads and whose cache a writer would drop."""
        return user_scope(tool_context.session.app_name, tool_context.user_id)

    def _allowed(self, tool: BaseTool, tool_context: ToolContext) -> bool:
        """The eligibility rules that serve and store both apply."""
        if self._cfg.mode == "off" or not self._cfg.cache_enabled:
            return False
        if self._startup.gates_own_tools(tool_context.agent_name):
            return False
        if self.cache.ttl_for(tool.name) is None:
            return False
        service = getattr(tool_context.get_invocation_context(), "session_service", None)
        return not isinstance(service, InMemorySessionService)

    def _key_for(
        self, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> str | None:
        """The one place a cache key is built. None means "this call cannot be keyed".

        `cache_key` refuses an empty `user_id` by raising, which is the right behaviour for a
        function whose whole job is identity scoping. It is caught here rather than left to the
        plugin's fail-open wrapper, because an anonymous caller should mean "no caching", not a
        stack trace on every tool call.
        """
        try:
            return cache_key(
                tool_context.session.app_name,
                tool_context.user_id,
                tool_context.agent_name,
                tool.name,
                tool_version(tool),
                tool_args,
            )
        except ValueError:
            return None

    def _take_hit_marker(self, tool_context: ToolContext) -> bool:
        """Read and clear the "this call was served from cache" marker.

        Clearing is belt and braces: ADK builds one ToolContext per call and hands it to both
        callbacks once, so the marker could simply be left behind. Taking it means a context
        that is ever reused, by a retry or by a future ADK, cannot report a second hit.
        """
        if not getattr(tool_context, HIT_MARKER_ATTR, False):
            return False
        setattr(tool_context, HIT_MARKER_ATTR, False)
        return True
