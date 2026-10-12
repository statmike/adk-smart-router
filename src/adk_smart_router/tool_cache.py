"""Identity-scoped memoization of tool results. The security surface of the router.

The one rule that must never be relaxed: **`user_id` is inside the sha256, always**.
Most real tools are principal-scoped with arguments that never mention the principal
(`get_balance()`, `list_my_orders(status='open')`), so a key without the caller's identity
serves A's data to B. Identity is in the hash, never a filter applied afterwards
(DESIGN.md 8.1).

Three more leakage or corruption channels are closed here, and each one is load-bearing:

* Deep copy on write **and** on read. ADK hands the returned dict by reference to every
  downstream `after_tool_callback`; one consumer doing `result['n'] += 1` corrupts the stored
  entry, and the corruption compounds on every subsequent hit (DESIGN.md 8.5).
* `time.monotonic()`, never wall clock. An NTP step backward makes entries immortal, a step
  forward mass-expires the cache into a thundering herd.
* `absolute_deadline` is anchored to the first write and never moves, so an entry that is
  re-written on every read cannot become immortal.

Eligibility (which tools may be cached at all) and invalidation are the plugin's job; this
module only owns the key, the clock, and the bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from adk_smart_router.config import RouterConfig

# Fields are length-prefixed before hashing, so no field value can shift a boundary and forge
# another principal's key. The separator is cosmetic; the length prefix is the guarantee.
_FIELD_SEP = "\x1f"

# An entry may be legitimately re-written by a refresh probe, so the ceiling is a multiple of
# the TTL rather than the TTL itself. It is anchored to the first write and never moves.
_ABSOLUTE_TTL_FACTOR = 2.0

# How many principals' purge counts to remember. An int per scope is small, but the number of
# scopes a long-lived process sees is not bounded, so the oldest are forgotten. The window that
# matters is one tool call, and forgetting a scope takes this many other principals writing.
_MAX_TRACKED_SCOPES = 4096

# Every side-channel field on ADK 2.8.0's EventActions. A tool's contract is its return value
# PLUS everything it wrote here, and a cache hit reproduces the first and none of the second
# (DESIGN.md 8.2). This list is only the fallback for objects that cannot be dumped; the real
# check diffs against a pristine EventActions so a field added by a future ADK is caught
# without anyone remembering to edit this tuple.
_ACTION_FIELDS = (
    "agent_state",
    "artifact_delta",
    "compaction",
    "end_of_agent",
    "escalate",
    "render_ui_widgets",
    "requested_auth_configs",
    "requested_tool_confirmations",
    "rewind_before_invocation_id",
    "route",
    "set_model_response",
    "skip_summarization",
    "state_delta",
    "transfer_to_agent",
)


# --- Keying -------------------------------------------------------------------


def canonical_json(args: Mapping[str, Any]) -> str:
    """Serialize tool arguments deterministically.

    Sorted keys, no whitespace, floats via `repr`, explicit `null` for a present-but-None
    value, missing keys absent. Two dicts that differ only in insertion order serialize
    identically. Total by construction: an unserializable argument falls back to its `repr`
    rather than raising, because raising here would take down the tool call.
    """
    return _canon(args)


def _canon(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return "{" + ",".join(f"{_canon(str(k))}:{_canon(v)}" for k, v in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canon(item) for item in value) + "]"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(repr(value), ensure_ascii=False)


def cache_key(
    app_name: str,
    user_id: str,
    agent_name: str,
    tool_name: str,
    tool_version: str,
    args: Mapping[str, Any],
) -> str:
    """Hash the full identity of a tool call. `user_id` is unconditional.

    There is no opt-out and no GLOBAL scope in v1. Removing `user_id` from this tuple turns
    every ambient-identity tool into a cross-user data leak that is silent, intermittent, and
    unreproducible from logs (DESIGN.md 8.1).
    """
    if not user_id:
        # Fail closed. An empty principal would collapse every caller into one shared scope,
        # which is exactly the leak this key exists to prevent. The plugin wraps its callbacks,
        # so this degrades to "no caching", not to a failed run.
        # NOTE(contract): the interface does not specify this raise. Silently returning a key
        # here would be the single worst failure mode in the package, so it fails loudly.
        raise ValueError("cache_key requires a non-empty user_id; there is no anonymous scope")
    fields = (app_name, user_id, agent_name, tool_name, tool_version, canonical_json(args))
    joined = _FIELD_SEP.join(f"{len(field)}:{field}" for field in fields)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def user_scope(app_name: str, user_id: str) -> str:
    """The unit of `purge_user`: one principal in one app.

    Length-prefixed for the same reason `cache_key` is. A plain `f"{app}|{user}"` lets
    ("acme|eu", "bob") and ("acme", "eu|bob") produce one string, and while that cannot serve
    one principal's data to another (identity is in the key, not the scope), it does let one
    tenant's writer drop another tenant's entries.
    """
    return _FIELD_SEP.join(f"{len(field)}:{field}" for field in (app_name, user_id))


def actions_are_default(actions: Any) -> bool:  # noqa: ANN401 - ADK's EventActions, duck-typed
    """True only if the tool wrote nothing to `tool_context.actions`.

    Anything written there is part of the tool's contract and cannot be replayed from a cache
    entry: a saved artifact is not saved, a transfer does not transfer, an `escalate=True`
    safety path silently stops escalating, a rendered widget never renders (DESIGN.md 8.2).

    The check is a diff against the model's own defaults rather than a list of field names, so
    a field that a future ADK adds is caught the day it ships. A named list would fail open on
    it, which is the wrong direction for the one mechanical check standing between a human
    declaring a tool cacheable and that tool having side effects.

    Note that a pristine `EventActions` already dumps `{"state_delta": {}}`, because that field
    has a default factory. Emptiness is what matters here, not whether the field is present.
    """
    if actions is None:
        return True
    dump = getattr(actions, "model_dump", None)
    if callable(dump):
        try:
            written = dump(exclude_defaults=True, exclude_none=True)
        except Exception:  # noqa: BLE001 - a stub that only looks like a pydantic model
            written = None
        if isinstance(written, Mapping):
            return not any(bool(value) for value in written.values())
    return not any(getattr(actions, name, None) for name in _ACTION_FIELDS)


# --- Storage ------------------------------------------------------------------


@dataclass(frozen=True)
class CacheEntry:
    """One memoized tool result. All times are `time.monotonic()` seconds."""

    payload: Mapping[str, Any]
    user_scope: str  # what `user_scope()` returns, and the unit of `purge_user`
    tool_name: str
    stored_at: float
    expires_at: float  # jittered at write time
    absolute_deadline: float  # set on first write, never extended by a read or a re-write
    ttl_used: int
    # Always "declared" today: a tool is cacheable only by appearing in `cacheable_tools`,
    # and that mapping carries a TTL for every entry, so there is no other way for one to
    # arrive. The field is here because DESIGN.md 11.4 needs the warehouse to be able to
    # tell a hand-declared TTL from a measured one, and measured TTLs are blocked on real
    # traffic rather than on code (DESIGN.md R13). Emitting the constant now is what makes
    # that distinction free to start making later.
    ttl_source: str


def _default_jitter() -> float:
    """+/-10% spread on expiry so a cohort of entries does not expire in lockstep."""
    return random.uniform(0.9, 1.1)


def _copy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(payload))


class ToolResultCache:
    """Bounded, TTL'd, LRU, per-process store of tool results.

    Not thread-safe by design: one plugin instance serves all concurrent users, and every
    mutation here is a single dict operation under the GIL. Correctness never depends on two
    operations being atomic with respect to each other.
    """

    def __init__(
        self,
        cfg: RouterConfig,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = _default_jitter,
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        self._jitter = jitter
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._blocked: set[str] = set()
        self._epochs: OrderedDict[str, int] = OrderedDict()
        self._stats: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "purges": 0,
            "expired": 0,
            "refused_size": 0,
        }

    # --- Eligibility ----------------------------------------------------------

    def ttl_for(self, tool_name: str) -> int | None:
        """The TTL to use for this tool, or None if it must not be cached.

        None when the cache is off, when the tool was never declared cacheable, when it has
        been blocklisted for using `tool_context.actions`, or when the declared TTL is below
        `cache_min_ttl_s` (below the floor, caching is not worth the staleness). A declared TTL
        above `cache_max_ttl_s` is clamped rather than refused (DESIGN.md 8.6).

        NOTE(contract): `cache_enabled=False` and a declared TTL of 0 are also "not cacheable".
        The interface lists neither; both are refusals the caller should not have to remember.
        """
        if not self._cfg.cache_enabled or tool_name in self._blocked:
            return None
        declared = self._cfg.cacheable_tools.get(tool_name)
        # A hard 0 is the poller rule from DESIGN.md 8.6: permanent, no override.
        if declared is None or declared <= 0 or declared < self._cfg.cache_min_ttl_s:
            return None
        capped: int = min(declared, self._cfg.cache_max_ttl_s)
        return capped

    def block(self, tool_name: str) -> None:
        """Permanently refuse this tool for the life of the process, and drop what it stored.

        Called when a declared-cacheable tool is caught writing to `tool_context.actions`. Its
        existing entries go too: their siblings had side effects that a hit cannot reproduce.
        """
        self._blocked.add(tool_name)
        doomed = [key for key, entry in self._entries.items() if entry.tool_name == tool_name]
        for key in doomed:
            del self._entries[key]
        self._stats["purges"] += len(doomed)

    def is_blocked(self, tool_name: str) -> bool:
        return tool_name in self._blocked

    # --- Read / write ---------------------------------------------------------

    def get(self, key: str, now: float | None = None) -> CacheEntry | None:
        """Return a deep copy of a live entry, or None. Never extends the entry's life."""
        moment = self._clock() if now is None else now
        entry = self._entries.get(key)
        if entry is None:
            self._stats["misses"] += 1
            return None
        if moment >= entry.expires_at or moment >= entry.absolute_deadline:
            del self._entries[key]
            self._stats["expired"] += 1
            self._stats["misses"] += 1
            return None
        if entry.tool_name in self._blocked:
            del self._entries[key]
            self._stats["misses"] += 1
            return None
        self._entries.move_to_end(key)  # LRU recency only; expiry is untouched by a read.
        self._stats["hits"] += 1
        return replace(entry, payload=_copy_payload(entry.payload))

    def put(
        self,
        key: str,
        payload: Mapping[str, Any],
        *,
        user_scope: str,
        tool_name: str,
        ttl_s: int,
        ttl_source: str,
    ) -> bool:
        """Store a deep copy of `payload`. False means nothing was stored.

        The caller keeps its own reference to `payload` and downstream plugins may mutate it,
        so the copy is taken here rather than trusted from the caller.

        NOTE(contract): besides the documented oversize refusal, this also returns False when
        the cache is disabled, when `ttl_s <= 0`, and when the tool is blocklisted.
        """
        if not self._cfg.cache_enabled or ttl_s <= 0 or tool_name in self._blocked:
            return False
        if len(canonical_json(payload).encode("utf-8")) > self._cfg.cache_max_payload_bytes:
            self._stats["refused_size"] += 1
            return False

        now = self._clock()
        ttl_used = max(1, min(int(ttl_s), self._cfg.cache_max_ttl_s))
        previous = self._entries.get(key)
        deadline = (
            previous.absolute_deadline
            if previous is not None
            else now + ttl_used * _ABSOLUTE_TTL_FACTOR
        )
        self._entries[key] = CacheEntry(
            payload=_copy_payload(payload),
            user_scope=user_scope,
            tool_name=tool_name,
            stored_at=now,
            expires_at=now + ttl_used * self._jitter(),
            absolute_deadline=deadline,
            ttl_used=ttl_used,
            ttl_source=ttl_source,
        )
        self._entries.move_to_end(key)
        self._stats["stores"] += 1
        while len(self._entries) > self._cfg.cache_max_entries:
            self._entries.popitem(last=False)
            self._stats["evictions"] += 1
        return True

    def purge_user(self, scope: str) -> int:
        """Drop every entry for one principal. Returns how many went.

        The read-your-own-writes invalidator: any tool that is not on the cacheable allowlist
        is assumed to have side effects, and its user's whole cache is dropped after it runs
        (DESIGN.md 8.4).

        The epoch moves whether or not anything was dropped, because what it dates is the
        write, not the entries. A tool call already in flight has nothing in the cache yet and
        is exactly the case `purge_epoch` exists for.
        """
        doomed = [key for key, entry in self._entries.items() if entry.user_scope == scope]
        for key in doomed:
            del self._entries[key]
        self._stats["purges"] += len(doomed)
        self._epochs[scope] = self._epochs.get(scope, 0) + 1
        self._epochs.move_to_end(scope)
        while len(self._epochs) > _MAX_TRACKED_SCOPES:
            self._epochs.popitem(last=False)
        return len(doomed)

    def purge_epoch(self, scope: str) -> int:
        """How many times this principal's cache has been dropped. Zero if never, or forgotten.

        Read before a tool runs and compared after it returns, so that a result computed before
        a concurrent writer's purge is not stored after it (DESIGN.md 8.4). Only equality is
        ever asked, so an epoch evicted by `_MAX_TRACKED_SCOPES` reads back as 0, stops matching
        the stamp a caller is holding, and refuses that store. Forgetting fails closed.
        """
        return self._epochs.get(scope, 0)

    @property
    def stats(self) -> Mapping[str, int]:
        # NOTE(contract): "blocked" is a gauge (how many tools are blocklisted right now), not
        # a counter. "purges" counts entries dropped, by purge_user or by block().
        return {**self._stats, "blocked": len(self._blocked)}
