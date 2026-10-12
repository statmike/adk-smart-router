"""Tests for the deadline on the artifact load and the expiry on a failed one.

`_resolve` used to be one-shot in three ways at once: it waited on GCS with no bound, it
never retried, and it cached a `None` for the life of the process. Each is defensible alone.
Together they meant the first few seconds of a process decided whether it ever routed, and
the only signal was one WARNING line. A bad minute in GCS, a slow cold start behind a
throttled network, a booster published half-written: any of them and the process was a
passthrough forever, looking installed and healthy (DESIGN.md 19.73).

Two mechanisms replace the three gaps. A deadline, so a stuck load cannot hold the first turn
open. An expiry on the negative entry, which is the retry: the next turn after the window
tries again, and it pays nothing on a turn that was going to pass through anyway.

What is deliberately unchanged is the positive side. A loaded router is kept for the life of
the process, so this is still not the artifact re-poll DESIGN 11.1 wants.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from test_plugin import (
    FakeClock,
    FakeContext,
    declared_names,
    make_model,
    make_plugin,
    make_request,
)

TIMEOUT = 30.0
WINDOW = 300.0


class CountingStore:
    """A store whose answer can change between attempts, which is the whole point."""

    def __init__(self, answers: list[Any]) -> None:
        self._answers = answers
        self.calls = 0

    def load(self, agent_name: str) -> Any:
        self.calls += 1
        # The last answer repeats, so a test only has to write the turns it cares about.
        index = min(self.calls - 1, len(self._answers) - 1)
        return self._answers[index]


class HangingStore:
    """Blocks a worker thread past the deadline, the way a stuck socket would."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls = 0

    def load(self, agent_name: str) -> Any:
        self.calls += 1
        # A real `time.sleep` in a real worker thread, so `wait_for` is racing something.
        import time

        time.sleep(self.seconds)
        return make_model()


def build(monkeypatch: pytest.MonkeyPatch, store: Any, **overrides: Any) -> Any:
    fields: dict[str, Any] = {"model_load_timeout_s": TIMEOUT, "model_retry_after_s": WINDOW}
    fields.update(overrides)
    clock = FakeClock()
    router = make_plugin(monkeypatch, clock=clock, **fields)
    router._store = store  # noqa: SLF001
    router._routers.clear()  # noqa: SLF001
    return router, clock


async def turn(router: Any, session: str = "s1") -> list[str]:
    """One request. Returns the tool names actually declared to the model.

    The session id matters more here than anywhere else in the suite. A session that was
    in flight during the outage has already sent a full tool list, so the prefix lock holds
    it in passthrough no matter what the retry finds. Recovery is visible on the next new
    session and only there, which is the honest shape of it (DESIGN.md 7.2, 19.73).
    """
    ctx, req = FakeContext(session_id=session), make_request()
    await router.before_model_callback(callback_context=ctx, llm_request=req)
    return declared_names(req)


# --- The window, in both directions -----------------------------------------------


async def test_a_failed_load_is_not_retried_on_the_very_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative entry still does its original job, which is not to hammer GCS."""
    store = CountingStore([None])
    router, _ = build(monkeypatch, store)

    for _ in range(20):
        await turn(router)
    assert store.calls == 1


async def test_a_failed_load_is_tried_again_once_the_window_is_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CountingStore([None])
    router, clock = build(monkeypatch, store)

    await turn(router)
    assert store.calls == 1

    clock.advance(WINDOW - 1)
    await turn(router)
    assert store.calls == 1, "retried before the window was up"

    clock.advance(2)
    await turn(router)
    assert store.calls == 2


async def test_a_transient_failure_heals_and_the_process_routes_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure this whole item exists for: the bucket was unreachable for a minute.

    Before the expiry, this process passed through until it was redeployed.
    """
    store = CountingStore([None, make_model()])
    router, clock = build(monkeypatch, store)

    assert await turn(router, "during") != ["tool_00"], "routed with no model"

    clock.advance(WINDOW + 1)
    assert await turn(router, "after") == ["tool_00"], "never recovered from a transient failure"


async def test_a_recovered_agent_is_not_looked_up_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once it loads it stays loaded. The expiry is on failure only, not a re-poll."""
    store = CountingStore([None, make_model()])
    router, clock = build(monkeypatch, store)

    await turn(router)
    clock.advance(WINDOW + 1)
    await turn(router)
    assert store.calls == 2

    for _ in range(5):
        clock.advance(WINDOW + 1)
        await turn(router)
    assert store.calls == 2, "a loaded router was re-fetched, which is a re-poll"


async def test_the_first_load_of_a_healthy_agent_happens_once_and_stays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = CountingStore([make_model()])
    router, clock = build(monkeypatch, store)

    for n in range(5):
        clock.advance(WINDOW * 10)
        assert await turn(router, f"s{n}") == ["tool_00"]
    assert store.calls == 1


# --- The deadline -----------------------------------------------------------------


async def test_a_load_that_overruns_the_deadline_passes_the_turn_through(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The turn must not wait on a stuck socket. It passes through and says why."""
    store = HangingStore(seconds=0.20)
    router, _ = build(monkeypatch, store, model_load_timeout_s=0.02)

    with caplog.at_level(logging.WARNING):
        names = await turn(router)

    assert names != ["tool_00"], "routed on an artifact that never arrived"
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "took longer than" in text
    assert "retrying in" in text  # the operator's next question, answered in the same line


async def test_a_timed_out_load_is_a_failure_with_an_expiry_not_a_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow first attempt is the most likely transient failure of all, so it must heal."""
    store = HangingStore(seconds=0.05)
    router, clock = build(monkeypatch, store, model_load_timeout_s=0.01)

    assert await turn(router, "during") != ["tool_00"]

    store.seconds = 0.0  # whatever was slow is no longer slow
    clock.advance(WINDOW + 1)
    assert await turn(router, "after") == ["tool_00"]


async def test_the_deadline_does_not_fire_on_a_load_that_is_merely_slow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2.2 s reference load against a 30 s bound. The bound is for stuck, not slow."""
    store = HangingStore(seconds=0.05)
    router, _ = build(monkeypatch, store, model_load_timeout_s=5.0)

    assert await turn(router) == ["tool_00"]


# --- Both at once -----------------------------------------------------------------


async def test_a_burst_at_the_window_boundary_still_loads_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-flight has to cover the retry too, or a busy process retries eight times.

    The store blocks in a worker thread, so the eight callers really do overlap rather than
    running one after another inside one event-loop step.
    """
    import threading

    class SlowFailingStore:
        def __init__(self) -> None:
            self.calls = 0
            self.answer: Any = None
            self._lock = threading.Lock()

        def load(self, agent_name: str) -> Any:
            with self._lock:
                self.calls += 1
            threading.Event().wait(0.05)
            return self.answer

    store = SlowFailingStore()
    router, clock = build(monkeypatch, store)

    await turn(router)
    assert store.calls == 1

    store.answer = make_model()
    clock.advance(WINDOW + 1)
    await asyncio.gather(*(turn(router) for _ in range(8)))
    assert store.calls == 2, f"{store.calls} loads where the retry should have been shared"


async def test_a_second_agent_is_not_held_back_by_the_first_ones_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window is per agent. One missing artifact must not mute a healthy sibling."""
    store = CountingStore([None])
    router, _ = build(monkeypatch, store)

    await router._resolve("support")  # noqa: SLF001
    assert store.calls == 1
    await router._resolve("billing")  # noqa: SLF001
    assert store.calls == 2, "the second agent reused the first agent's failure"


async def test_a_session_that_lived_through_the_outage_stays_in_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery is for new sessions. The one that was mid-conversation keeps its prefix.

    It already sent a full tool list, so pruning it now would not be a smaller prompt, it
    would be a different prefix at full price. This is the prefix lock doing its job, and it
    is the reason the retry window is not the same thing as instant recovery.
    """
    store = CountingStore([None, make_model()])
    router, clock = build(monkeypatch, store)

    assert await turn(router, "old") != ["tool_00"]
    clock.advance(WINDOW + 1)

    assert await turn(router, "old") != ["tool_00"], "re-prefixed a session mid-conversation"
    assert await turn(router, "new") == ["tool_00"]
