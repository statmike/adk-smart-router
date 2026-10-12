"""Tests for the artifact re-poll: a process picks up a new model without a redeploy.

Until this existed, a loaded router was kept for the life of the process. Publishing a better
model put it in Cloud Storage and left it there, and every serving replica went on using the
one it started with until somebody restarted it. That is the gap DESIGN.md 11.1 named and 21.4
closes, and it is the difference between a plugin that improves an agent and a plugin that
improved an agent once.

Two properties carry the whole design and most of this file is about them.

**Nothing waits.** The turn that notices the interval has elapsed is served by the router
already loaded. The check is a clock comparison; the work is a task nobody awaits. A request
must never block on Cloud Storage for a version it is not going to use.

**Every failure keeps serving.** A bucket that cannot be listed, a build that fails, a refusal
on the compatibility contract: all of them leave the loaded router exactly where it was. There
is no state in the refresh path whose failure mode is passthrough, which is what makes the
whole mechanism safe to run unattended (§21.8).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from test_plugin import (
    TOOLS,
    FakeClock,
    FakeContext,
    declared_names,
    make_model,
    make_plugin,
    make_request,
)

OLD = "20260830T000000Z"
NEW = "20260906T000000Z"
INTERVAL = 900.0


class VersionedStore:
    """A store whose published version can move under a running process."""

    def __init__(self, version: str = OLD) -> None:
        self.version = version
        self.list_calls = 0
        self.load_calls = 0
        self.list_raises = False
        self.list_returns_none = False
        self.load_returns_none = False

    def latest_version(self, agent_name: str) -> str | None:
        self.list_calls += 1
        if self.list_raises:
            raise RuntimeError("bucket unreachable")
        if self.list_returns_none:
            return None
        return self.version

    def load(self, agent_name: str) -> Any:
        self.load_calls += 1
        if self.load_returns_none:
            return None
        return make_model(version=self.version)


def build(
    monkeypatch: pytest.MonkeyPatch, store: VersionedStore, **overrides: Any
) -> tuple[Any, FakeClock]:
    fields: dict[str, Any] = {"refresh_interval_s": INTERVAL}
    fields.update(overrides)
    clock = FakeClock()
    router = make_plugin(monkeypatch, clock=clock, **fields)
    router._store = store  # noqa: SLF001
    router._routers.clear()  # noqa: SLF001
    router._checked_at.clear()  # noqa: SLF001
    return router, clock


async def turn(router: Any, session: str = "s1") -> list[str]:
    ctx, req = FakeContext(session_id=session), make_request()
    await router.before_model_callback(callback_context=ctx, llm_request=req)
    return declared_names(req)


async def settle(router: Any) -> None:
    """Await whatever the last turn scheduled.

    Production never does this, which is the point of the mechanism. A test has to, because
    the assertion is about the state the task leaves behind rather than about the turn.
    """
    while router._refreshing:  # noqa: SLF001
        await asyncio.gather(*list(router._refreshing), return_exceptions=True)


def loaded(router: Any, agent: str = "root") -> str | None:
    return router.loaded_versions.get(agent)


# --- The interval -----------------------------------------------------------------


async def test_a_freshly_loaded_router_does_not_poll_on_the_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load is itself a read of the bucket, so the first poll is one interval later."""
    store = VersionedStore()
    router, _ = build(monkeypatch, store)

    for i in range(20):
        await turn(router, session=f"s{i}")
    await settle(router)

    assert store.list_calls == 0
    assert store.load_calls == 1


async def test_nothing_is_polled_before_the_interval_is_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)

    clock.advance(INTERVAL - 1)
    await turn(router, session="s2")
    await settle(router)

    assert store.list_calls == 0


async def test_the_poll_stops_at_the_list_when_nothing_has_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The common case, and the one the interval is priced against: one list, no download."""
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)

    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)

    assert store.list_calls == 1
    assert store.load_calls == 1, "downloaded an artifact it already had"


async def test_one_poll_per_interval_however_many_turns_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_checked_at` is stamped by the turn that schedules, not by the task that runs."""
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)

    clock.advance(INTERVAL + 1)
    for i in range(50):
        await turn(router, session=f"s{i}")
    await settle(router)

    assert store.list_calls == 1


# --- The swap ---------------------------------------------------------------------


async def test_a_new_version_is_picked_up_without_a_redeploy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of §21.4, in one test."""
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)
    assert loaded(router) == OLD

    store.version = NEW  # somebody published
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)

    assert loaded(router) == NEW
    assert router.model_swaps == 1


async def test_a_withdrawn_generation_rolls_the_process_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The version moving *down* is the rollback in §21.5, and it uses the same path.

    `_build_if_newer` compares for equality rather than for order precisely so that this
    works. A check for "strictly newer" would make the automatic rollback impossible and the
    bug would only show up the first time a bad generation had to be withdrawn.
    """
    store = VersionedStore(version=NEW)
    router, clock = build(monkeypatch, store)
    await turn(router)
    assert loaded(router) == NEW

    store.version = OLD  # NEW got a withdrawn marker, so OLD is current again
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)

    assert loaded(router) == OLD
    assert router.model_swaps == 1


async def test_the_turn_that_schedules_a_refresh_is_served_by_the_old_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing waits. The swap lands after the turn that noticed it was due."""
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)

    store.version = NEW
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    assert loaded(router) == OLD, "the turn blocked on the refresh"

    await settle(router)
    assert loaded(router) == NEW


async def test_a_swap_does_not_move_a_session_that_already_decided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session stability survives the swap, which is what keeps §18.22 from re-entering.

    A declared tool list that changed mid-session would be a new prefix at full price. The
    decision is cached per session, so a later turn of an old session reuses it whatever the
    process has loaded now.
    """
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    first = await turn(router, session="long")

    store.version = NEW
    clock.advance(INTERVAL + 1)
    await turn(router, session="other")
    await settle(router)
    assert loaded(router) == NEW

    assert await turn(router, session="long") == first


# --- Every failure keeps serving --------------------------------------------------


async def test_a_bucket_that_cannot_be_listed_leaves_the_router_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)

    store.list_raises = True
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)

    assert loaded(router) == OLD
    assert router.model_swaps == 0
    assert len(await turn(router, session="s3")) < len(TOOLS), "stopped routing after a poll"


async def test_a_bucket_with_no_complete_version_leaves_the_router_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`latest_version` answering None is "cannot tell", not "withdraw everything".

    The real store returns None both when the listing failed and when nothing is published,
    and a process must not throw away a working model over either.
    """
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)

    store.list_returns_none = True
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)

    assert loaded(router) == OLD


async def test_a_new_version_that_will_not_build_leaves_the_old_one_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A half-written artifact, a bad digest, or a mismatched embedding model.

    All three land here as `load` returning None, and none of them may cost the process the
    model it already has.
    """
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)

    store.version = NEW
    store.load_returns_none = True
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)

    assert loaded(router) == OLD
    assert router.model_swaps == 0


async def test_a_failed_refresh_is_not_negative_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_failed_at` is for a cold process with no model, and a refusal here is neither.

    Nothing is passing through while a refresh fails, so there is no damage to bound and no
    reason to stop looking. The next interval tries again.
    """
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)

    store.list_raises = True
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)
    assert store.list_calls == 1

    store.list_raises = False
    store.version = NEW
    clock.advance(INTERVAL + 1)
    await turn(router, session="s3")
    await settle(router)

    assert store.list_calls == 2
    assert loaded(router) == NEW


# --- When it is switched off ------------------------------------------------------


async def test_a_zero_interval_disables_the_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    store = VersionedStore()
    router, clock = build(monkeypatch, store, refresh_interval_s=0.0)
    await turn(router)

    store.version = NEW
    clock.advance(100_000)
    await turn(router, session="s2")
    await settle(router)

    assert store.list_calls == 0
    assert loaded(router) == OLD


async def test_a_pinned_version_disables_the_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pin means versions move on deploys, so there is nothing to poll for."""
    store = VersionedStore()
    router, clock = build(monkeypatch, store, pin_version=OLD)
    await turn(router)

    store.version = NEW
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)

    assert store.list_calls == 0
    assert loaded(router) == OLD


# --- The ordering that matters ----------------------------------------------------


async def test_an_aged_out_model_still_polls_for_a_fresher_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case the whole feature exists for, and the easiest one to break.

    A model past `max_model_age_days` makes `_resolve` return None and the agent pass
    through. If the poll sat behind that gate, the process would stop looking at exactly the
    moment it stopped routing, and would never recover without a redeploy. So the refresh is
    scheduled before the age check, not after it.
    """
    store = VersionedStore()
    router, clock = build(monkeypatch, store, max_model_age_days=0)
    assert await turn(router) == TOOLS, "an aged-out model should pass through"

    store.version = NEW
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)

    assert store.list_calls == 1
    assert loaded(router) == NEW


async def test_a_swap_lets_a_stale_warning_fire_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_stale_warned` is once per agent, and a new artifact is a new thing to warn about."""
    store = VersionedStore()
    router, clock = build(monkeypatch, store, max_model_age_days=0)
    await turn(router)
    assert "root" in router._stale_warned  # noqa: SLF001

    store.version = NEW
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)

    assert "root" not in router._stale_warned  # noqa: SLF001


async def test_a_swap_lets_a_demoted_agent_prune_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A republish is usually the repair for the demotion, not a continuation of it.

    The cache watch demotes an agent whose pruned calls bill more than its unpruned ones,
    and everything it measured belongs to the artifact that was serving at the time. Without
    this the operator's fix lands, the new menus are never tried, and the agent sits in
    passthrough until the process restarts. `test_a_new_artifact_is_one_more_chance_and_not_a
    _fresh_budget_of_them` in `test_cache_watch.py` is the other half: the swap does not
    hand the agent an unlimited supply of retries.
    """
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)
    router._watch._demote("root", "later")  # noqa: SLF001 - 50 real samples would say no more
    assert router._watch.suppressed("root") is True  # noqa: SLF001

    store.version = NEW
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    await settle(router)

    assert loaded(router) == NEW
    assert router._watch.suppressed("root") is False  # noqa: SLF001


async def test_the_refresh_task_is_held_until_it_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task with no strong reference can be collected mid-flight, and then never swaps."""
    store = VersionedStore()
    router, clock = build(monkeypatch, store)
    await turn(router)

    store.version = NEW
    clock.advance(INTERVAL + 1)
    await turn(router, session="s2")
    assert len(router._refreshing) == 1  # noqa: SLF001

    await settle(router)
    assert router._refreshing == set()  # noqa: SLF001
