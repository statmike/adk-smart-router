"""Tests for tool_cache: the key, the copies, and the clock.

Every test here is named after the incident it prevents, not after the function it calls.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from adk_smart_router.config import RouterConfig
from adk_smart_router.tool_cache import (
    CacheEntry,
    ToolResultCache,
    actions_are_default,
    cache_key,
    canonical_json,
)


class FakeClock:
    """A monotonic clock the test drives by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeActions:
    """Stands in for ADK's EventActions. Only the side-channel fields matter."""

    def __init__(self, **fields: Any) -> None:
        defaults: dict[str, Any] = {
            "state_delta": {},
            "artifact_delta": {},
            "transfer_to_agent": None,
            "escalate": None,
            "skip_summarization": None,
            "requested_auth_configs": {},
            "requested_tool_confirmations": {},
        }
        for name, value in {**defaults, **fields}.items():
            setattr(self, name, value)


def make_cache(clock: FakeClock | None = None, **overrides: Any) -> ToolResultCache:
    fields: dict[str, Any] = {
        "cacheable_tools": {"get_weather": 300, "cheap_tool": 10, "long_tool": 99_999},
        "cache_max_entries": 3,
        "cache_max_payload_bytes": 256,
    }
    fields.update(overrides)
    return ToolResultCache(RouterConfig(**fields), clock=clock or FakeClock(), jitter=lambda: 1.0)


def key_for(user_id: str = "alice", args: dict[str, Any] | None = None, **over: str) -> str:
    fields: dict[str, Any] = {
        "app_name": "app",
        "user_id": user_id,
        "agent_name": "agent",
        "tool_name": "get_weather",
        "tool_version": "v1",
        "args": {} if args is None else args,
    }
    fields.update(over)
    return cache_key(**fields)


# --- The key ------------------------------------------------------------------


def test_user_id_in_key_prevents_cross_user_serve():
    """Two users, identical everything else, must never share a cache slot."""
    assert key_for("alice", {"city": "SF"}) != key_for("bob", {"city": "SF"})


def test_zero_arg_tool_does_not_collide_across_users():
    """The ambient-identity shape: get_balance() carries no identity in its args at all."""
    assert key_for("alice", {}) != key_for("bob", {})


def test_key_is_stable_across_dict_insertion_order():
    first = key_for(args={"a": 1, "b": {"x": 1, "y": 2}})
    second = key_for(args={"b": {"y": 2, "x": 1}, "a": 1})
    assert first == second


def test_key_separates_every_field():
    base = key_for()
    assert base != key_for(app_name="other")
    assert base != key_for(agent_name="other")
    assert base != key_for(tool_name="other")
    assert base != key_for(tool_version="v2")
    assert base != key_for(args={"city": "SF"})


def test_field_boundaries_cannot_be_forged_by_a_crafted_user_id():
    """Length prefixes stop a user_id containing a separator from posing as another scope."""
    shifted = cache_key("app", "a\x1fb", "c", "t", "v", {})
    honest = cache_key("app", "a", "b\x1fc", "t", "v", {})
    assert shifted != honest


def test_empty_user_id_is_refused():
    with pytest.raises(ValueError, match="user_id"):
        cache_key("app", "", "agent", "tool", "v1", {})


def test_canonical_json_distinguishes_null_from_missing():
    assert canonical_json({"a": None}) == '{"a":null}'
    assert canonical_json({}) == "{}"
    assert canonical_json({"a": None}) != canonical_json({})


def test_canonical_json_shapes():
    assert canonical_json({"b": 1, "a": True}) == '{"a":true,"b":1}'
    assert canonical_json({"f": 0.1}) == '{"f":0.1}'
    assert canonical_json({"xs": [1, "two", None]}) == '{"xs":[1,"two",null]}'


def test_canonical_json_is_total_on_unserializable_values():
    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    assert canonical_json({"w": Weird()}) == '{"w":"<weird>"}'


# --- Actions verification -----------------------------------------------------


def test_actions_are_default_on_a_clean_tool():
    assert actions_are_default(FakeActions()) is True
    assert actions_are_default(None) is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_delta", {"k": 1}),
        ("artifact_delta", {"policy.pdf": 1}),
        ("transfer_to_agent", "triage"),
        ("escalate", True),
        ("skip_summarization", True),
        ("requested_auth_configs", {"a": 1}),
        ("requested_tool_confirmations", {"c": 1}),
    ],
)
def test_any_side_channel_write_makes_a_tool_uncacheable(field: str, value: Any):
    assert actions_are_default(FakeActions(**{field: value})) is False


# --- Copies -------------------------------------------------------------------


def store(cache: ToolResultCache, key: str, payload: dict[str, Any], **over: Any) -> bool:
    kwargs: dict[str, Any] = {
        "user_scope": "app|alice",
        "tool_name": "get_weather",
        "ttl_s": 300,
        "ttl_source": "declared",
    }
    kwargs.update(over)
    return cache.put(key, payload, **kwargs)


def test_mutating_a_served_payload_does_not_corrupt_the_entry():
    """A downstream after_tool_callback doing result['n'] += 1 must not compound on every hit."""
    cache = make_cache()
    store(cache, "k", {"n": 1, "nested": {"xs": [1, 2]}})

    first = cache.get("k")
    assert first is not None
    served = cast("dict[str, Any]", first.payload)  # ADK hands this exact object downstream
    served["n"] += 1
    served["nested"]["xs"].append(3)

    second = cache.get("k")
    assert second is not None
    assert second.payload == {"n": 1, "nested": {"xs": [1, 2]}}


def test_mutating_the_dict_passed_to_put_does_not_corrupt_the_entry():
    cache = make_cache()
    payload: dict[str, Any] = {"n": 1, "nested": {"xs": [1]}}
    store(cache, "k", payload)
    payload["n"] = 99
    payload["nested"]["xs"].append(2)

    entry = cache.get("k")
    assert entry is not None
    assert entry.payload == {"n": 1, "nested": {"xs": [1]}}


# --- The clock ----------------------------------------------------------------


def test_entry_expires_and_is_evicted():
    clock = FakeClock()
    cache = make_cache(clock)
    store(cache, "k", {"a": 1}, ttl_s=100)

    clock.advance(99)
    assert cache.get("k") is not None
    clock.advance(2)
    assert cache.get("k") is None
    assert cache.stats["expired"] == 1
    assert cache.get("k") is None  # gone, not merely expired
    assert cache.stats["expired"] == 1


def test_reads_never_extend_expiry():
    clock = FakeClock()
    cache = make_cache(clock)
    store(cache, "k", {"a": 1}, ttl_s=100)
    for _ in range(9):
        clock.advance(10)
        assert cache.get("k") is not None
    clock.advance(11)
    assert cache.get("k") is None


def test_absolute_deadline_is_not_extended_by_rewrites():
    """A re-write on every read must not make a polled entry immortal (DESIGN.md 8.5)."""
    clock = FakeClock()
    cache = make_cache(clock)
    store(cache, "k", {"a": 1}, ttl_s=100)
    first = cache.get("k")
    assert first is not None
    deadline = first.absolute_deadline
    assert deadline > first.expires_at

    # The bug the deadline exists to bound: after_tool_callback re-writing the entry on a hit.
    while clock.now < deadline:
        clock.advance(50)
        store(cache, "k", {"a": 1}, ttl_s=100)
        again = cache.get("k")
        if again is None:
            break
        assert again.absolute_deadline == deadline

    assert cache.get("k") is None  # the entry died on schedule despite being re-written


def test_jitter_is_applied_at_write_time():
    clock = FakeClock()
    cfg = RouterConfig(cacheable_tools={})
    cache = ToolResultCache(cfg, clock=clock, jitter=lambda: 0.9)
    store(cache, "k", {"a": 1}, ttl_s=100)
    clock.advance(89)
    assert cache.get("k") is not None
    clock.advance(2)
    assert cache.get("k") is None


# --- Bounds and scopes --------------------------------------------------------


def test_oversized_payload_is_refused():
    cache = make_cache()
    assert store(cache, "big", {"blob": "x" * 1000}) is False
    assert cache.get("big") is None
    assert cache.stats["refused_size"] == 1
    assert cache.stats["stores"] == 0


def test_lru_evicts_the_least_recently_used_entry():
    cache = make_cache()  # cache_max_entries=3
    for name in ("a", "b", "c"):
        store(cache, name, {"v": name})
    assert cache.get("a") is not None  # refresh recency of "a"
    store(cache, "d", {"v": "d"})

    assert cache.get("b") is None  # "b" was the least recently used
    assert cache.get("a") is not None
    assert cache.get("c") is not None
    assert cache.get("d") is not None
    assert cache.stats["evictions"] == 1


def test_purge_user_only_purges_that_user():
    cache = make_cache()
    store(cache, "alice1", {"v": 1}, user_scope="app|alice")
    store(cache, "alice2", {"v": 2}, user_scope="app|alice")
    store(cache, "bob1", {"v": 3}, user_scope="app|bob")

    assert cache.purge_user("app|alice") == 2
    assert cache.get("alice1") is None
    assert cache.get("alice2") is None
    assert cache.get("bob1") is not None
    assert cache.stats["purges"] == 2


def test_purge_user_of_an_unknown_scope_is_a_noop():
    cache = make_cache()
    store(cache, "k", {"v": 1})
    assert cache.purge_user("app|nobody") == 0
    assert cache.get("k") is not None


# --- TTL policy ---------------------------------------------------------------


def test_undeclared_tool_is_not_cacheable():
    assert make_cache().ttl_for("create_ticket") is None


def test_blocklisted_tool_is_not_cacheable():
    cache = make_cache()
    assert cache.ttl_for("get_weather") == 300
    cache.block("get_weather")
    assert cache.is_blocked("get_weather") is True
    assert cache.ttl_for("get_weather") is None
    assert cache.stats["blocked"] == 1


def test_blocking_a_tool_drops_the_entries_it_already_stored():
    """Its siblings wrote artifacts or transfers that a hit cannot reproduce."""
    cache = make_cache()
    store(cache, "k", {"summary": "..."})
    cache.block("get_weather")
    assert cache.get("k") is None
    assert store(cache, "k", {"summary": "..."}) is False


def test_declared_ttl_below_the_floor_is_not_cacheable():
    cache = make_cache()  # cheap_tool declares 10s, cache_min_ttl_s is 30
    assert cache.ttl_for("cheap_tool") is None


def test_declared_ttl_above_the_ceiling_is_clamped():
    cache = make_cache()
    assert cache.ttl_for("long_tool") == RouterConfig().cache_max_ttl_s


def test_zero_ttl_is_permanently_uncacheable():
    """The poller rule: a status tool gets hard ttl=0, no override (DESIGN.md 8.6)."""
    cache = make_cache(cacheable_tools={"get_job_status": 0})
    assert cache.ttl_for("get_job_status") is None
    assert store(cache, "k", {"state": "RUNNING"}, tool_name="get_job_status", ttl_s=0) is False


def test_cache_disabled_refuses_everything():
    cache = make_cache(cache_enabled=False)
    assert cache.ttl_for("get_weather") is None
    assert store(cache, "k", {"a": 1}) is False
    assert cache.get("k") is None


# --- Stats --------------------------------------------------------------------


def test_stats_track_hits_and_misses():
    cache = make_cache()
    assert cache.get("missing") is None
    store(cache, "k", {"a": 1})
    assert cache.get("k") is not None
    assert cache.stats == {
        "hits": 1,
        "misses": 1,
        "stores": 1,
        "evictions": 0,
        "purges": 0,
        "expired": 0,
        "refused_size": 0,
        "blocked": 0,
    }


def test_entry_carries_its_ttl_provenance():
    clock = FakeClock()
    cache = make_cache(clock)
    store(cache, "k", {"a": 1}, ttl_s=300, ttl_source="declared")
    entry = cache.get("k")
    assert isinstance(entry, CacheEntry)
    assert entry.ttl_used == 300
    assert entry.ttl_source == "declared"
    assert entry.tool_name == "get_weather"
    assert entry.user_scope == "app|alice"
    assert entry.stored_at == clock.now
