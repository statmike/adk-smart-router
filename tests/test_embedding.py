"""Tests for embedding.py. Fully offline: the client is always a fake object."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from adk_smart_router import embedding
from adk_smart_router.config import RouterConfig
from adk_smart_router.embedding import Embedder, l2_normalize, truncate_and_normalize

Handler = Callable[[str], Awaitable[Any]]


# --- Fakes --------------------------------------------------------------------


@dataclass
class FakeEmbedding:
    values: list[float] | None


@dataclass
class FakeResponse:
    embeddings: list[FakeEmbedding] | None


@dataclass
class FakeModels:
    handler: Handler
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def embed_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return await self.handler(contents)


@dataclass
class FakeAio:
    models: FakeModels


class FakeClient:
    def __init__(self, handler: Handler) -> None:
        self.aio = FakeAio(models=FakeModels(handler=handler))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.aio.models.calls


def returns(values: list[float]) -> Handler:
    async def handler(_: str) -> Any:
        return FakeResponse(embeddings=[FakeEmbedding(values=list(values))])

    return handler


def returns_response(response: Any) -> Handler:
    async def handler(_: str) -> Any:
        return response

    return handler


def raises(exc: Exception) -> Handler:
    async def handler(_: str) -> Any:
        raise exc

    return handler


def hangs() -> Handler:
    async def handler(_: str) -> Any:
        await asyncio.sleep(10)
        raise AssertionError("unreachable")

    return handler


def cfg(**overrides: Any) -> RouterConfig:
    base: dict[str, Any] = {
        "embedding_dims": 4,
        "embed_timeout_ms": 500,
        "embed_cache_size": 8,
        "embed_breaker_threshold": 2,
        "embed_breaker_cooldown_s": 30.0,
        "bq_project": "proj",
        "vertex_location": "us-central1",
    }
    base.update(overrides)
    return RouterConfig(**base)


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


# --- Parity primitives --------------------------------------------------------


def test_l2_normalize_produces_unit_length():
    out = l2_normalize([3.0, 4.0])
    assert out == pytest.approx([0.6, 0.8])
    assert norm(out) == pytest.approx(1.0)


def test_l2_normalize_leaves_zero_vector_alone():
    assert l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_truncate_happens_before_normalize():
    # [3, 4, 12] truncated to 2 is [3, 4] -> [0.6, 0.8].
    # Normalizing first would give [3/13, 4/13] and a slice of norm 0.385.
    # This ordering is the parity contract (DESIGN.md 9.1).
    out = truncate_and_normalize([3.0, 4.0, 12.0], 2)
    assert out == pytest.approx([0.6, 0.8])
    assert norm(out) == pytest.approx(1.0)
    assert out != pytest.approx([3 / 13, 4 / 13])


def test_truncate_and_normalize_rejects_short_vector():
    with pytest.raises(ValueError, match="need at least 4"):
        truncate_and_normalize([1.0, 2.0], 4)


def test_truncate_and_normalize_rejects_non_positive_dims():
    with pytest.raises(ValueError, match="must be positive"):
        truncate_and_normalize([1.0, 2.0], 0)


# --- Happy path and Vertex pinning --------------------------------------------


async def test_embed_returns_unit_vector_and_pins_dimensionality():
    client = FakeClient(returns([1.0, 1.0, 1.0, 1.0, 99.0]))
    out = await Embedder(cfg(), client).embed("where is order 5512")

    assert out is not None
    assert len(out) == 4  # the 5th component is truncated away, then renormalized
    assert norm(out) == pytest.approx(1.0)
    call = client.calls[0]
    assert call["model"] == "gemini-embedding-2"
    assert call["contents"] == "where is order 5512"
    assert call["config"].output_dimensionality == 4
    assert call["config"].task_type is None


async def test_task_type_is_sent_only_when_configured():
    client = FakeClient(returns([1.0, 0.0, 0.0, 0.0]))
    await Embedder(cfg(task_type="RETRIEVAL_QUERY"), client).embed("hi there")
    assert client.calls[0]["config"].task_type == "RETRIEVAL_QUERY"


async def test_lazy_client_is_pinned_to_vertex(monkeypatch: pytest.MonkeyPatch):
    """The highest-risk line in the package: vertexai=True, ADC, no key file."""
    seen: dict[str, Any] = {}

    class FakeGenaiModule:
        @staticmethod
        def Client(**kwargs: Any) -> FakeClient:  # noqa: N802 - mirrors genai.Client
            seen.update(kwargs)
            return FakeClient(returns([1.0, 0.0, 0.0, 0.0]))

    monkeypatch.setattr(embedding, "genai", FakeGenaiModule)
    out = await Embedder(cfg()).embed("prompt")

    assert out == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert seen == {"vertexai": True, "project": "proj", "location": "us-central1"}
    assert "credentials" not in seen


# --- Failure paths: always None, never a raise --------------------------------


async def test_empty_text_returns_none_without_calling_the_client():
    client = FakeClient(returns([1.0, 0.0, 0.0, 0.0]))
    embedder = Embedder(cfg(), client)
    assert await embedder.embed("") is None
    assert await embedder.embed("   \n ") is None
    assert client.calls == []


async def test_timeout_returns_none_and_counts():
    embedder = Embedder(cfg(embed_timeout_ms=10), FakeClient(hangs()))
    assert await embedder.embed("slow") is None
    assert embedder.stats["timeouts"] == 1
    assert embedder.stats["failures"] == 0


async def test_client_exception_returns_none():
    embedder = Embedder(cfg(), FakeClient(raises(RuntimeError("429 quota exceeded"))))
    assert await embedder.embed("boom") is None
    assert embedder.stats["failures"] == 1


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(embeddings=None),
        FakeResponse(embeddings=[]),
        FakeResponse(embeddings=[FakeEmbedding(values=None)]),
        FakeResponse(embeddings=[FakeEmbedding(values=[])]),
        object(),
        None,
    ],
)
async def test_malformed_response_returns_none(response: Any):
    embedder = Embedder(cfg(), FakeClient(returns_response(response)))
    assert await embedder.embed("prompt") is None
    assert embedder.stats["failures"] == 1


async def test_wrong_dimensionality_returns_none():
    embedder = Embedder(cfg(embedding_dims=8), FakeClient(returns([1.0, 2.0, 3.0, 4.0])))
    assert await embedder.embed("prompt") is None
    assert embedder.stats["failures"] == 1


async def test_non_numeric_values_return_none():
    embedder = Embedder(cfg(), FakeClient(returns(["a", "b", "c", "d"])))  # type: ignore[list-item]
    assert await embedder.embed("prompt") is None
    assert embedder.stats["failures"] == 1


# --- LRU ----------------------------------------------------------------------


async def test_cache_hit_does_not_call_the_client_twice():
    client = FakeClient(returns([1.0, 1.0, 0.0, 0.0]))
    embedder = Embedder(cfg(), client)

    first = await embedder.embed("same prompt")
    second = await embedder.embed("same prompt")

    assert first == second
    assert len(client.calls) == 1
    assert embedder.stats == {
        "hits": 1,
        "misses": 1,
        "timeouts": 0,
        "failures": 0,
        "skipped": 0,
    }


async def test_cache_is_keyed_on_the_exact_text():
    client = FakeClient(returns([1.0, 0.0, 0.0, 0.0]))
    embedder = Embedder(cfg(), client)
    await embedder.embed("prompt")
    await embedder.embed("prompt ")
    assert len(client.calls) == 2


async def test_cache_evicts_least_recently_used():
    client = FakeClient(returns([1.0, 0.0, 0.0, 0.0]))
    embedder = Embedder(cfg(embed_cache_size=2), client)

    await embedder.embed("a")
    await embedder.embed("b")
    await embedder.embed("a")  # refreshes "a", so "b" is now the eviction victim
    await embedder.embed("c")
    assert len(client.calls) == 3

    await embedder.embed("a")
    assert len(client.calls) == 3  # still cached
    await embedder.embed("b")
    assert len(client.calls) == 4  # evicted, re-fetched


async def test_zero_sized_cache_disables_caching():
    client = FakeClient(returns([1.0, 0.0, 0.0, 0.0]))
    embedder = Embedder(cfg(embed_cache_size=0), client)
    await embedder.embed("a")
    await embedder.embed("a")
    assert len(client.calls) == 2


async def test_mutating_the_returned_vector_does_not_corrupt_the_cache():
    client = FakeClient(returns([1.0, 0.0, 0.0, 0.0]))
    embedder = Embedder(cfg(), client)

    first = await embedder.embed("prompt")
    assert first is not None
    first[0] = 999.0

    second = await embedder.embed("prompt")
    assert second == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert len(client.calls) == 1


# --- Circuit breaker ----------------------------------------------------------


async def test_breaker_opens_after_threshold_and_short_circuits():
    client = FakeClient(raises(RuntimeError("down")))
    embedder = Embedder(cfg(embed_breaker_threshold=2), client)

    assert await embedder.embed("one") is None
    assert embedder.breaker_open is False
    assert await embedder.embed("two") is None
    assert embedder.breaker_open is True

    assert await embedder.embed("three") is None
    assert len(client.calls) == 2  # the third call never reached the client


async def test_timeouts_also_open_the_breaker():
    embedder = Embedder(cfg(embed_timeout_ms=5, embed_breaker_threshold=2), FakeClient(hangs()))
    await embedder.embed("one")
    await embedder.embed("two")
    assert embedder.breaker_open is True
    assert embedder.stats["timeouts"] == 2


async def test_cached_vectors_are_still_served_while_the_breaker_is_open():
    state = {"fail": False}

    async def handler(_: str) -> Any:
        if state["fail"]:
            raise RuntimeError("down")
        return FakeResponse(embeddings=[FakeEmbedding(values=[1.0, 0.0, 0.0, 0.0])])

    embedder = Embedder(cfg(embed_breaker_threshold=1), FakeClient(handler))
    warm = await embedder.embed("warm")
    state["fail"] = True
    assert await embedder.embed("cold") is None
    assert embedder.breaker_open is True
    assert await embedder.embed("warm") == warm


async def test_breaker_recovers_after_cooldown_and_a_success_closes_it():
    state = {"fail": True}

    async def handler(_: str) -> Any:
        if state["fail"]:
            raise RuntimeError("down")
        return FakeResponse(embeddings=[FakeEmbedding(values=[0.0, 3.0, 4.0, 0.0])])

    client = FakeClient(handler)
    embedder = Embedder(cfg(embed_breaker_threshold=1, embed_breaker_cooldown_s=0.02), client)

    assert await embedder.embed("a") is None
    assert embedder.breaker_open is True
    assert await embedder.embed("b") is None
    assert len(client.calls) == 1  # held back while cooling down

    await asyncio.sleep(0.03)
    assert embedder.breaker_open is False

    state["fail"] = False
    out = await embedder.embed("c")
    assert out == pytest.approx([0.0, 0.6, 0.8, 0.0])
    assert embedder.breaker_open is False

    # A success reset the consecutive-failure count, so one new failure is not
    # enough to reopen a threshold-2 breaker.
    embedder = Embedder(cfg(embed_breaker_threshold=2, embed_breaker_cooldown_s=0.02), client)
    state["fail"] = True
    await embedder.embed("d")
    state["fail"] = False
    await embedder.embed("e")
    state["fail"] = True
    await embedder.embed("f")
    assert embedder.breaker_open is False


async def test_a_breaker_skip_is_counted_and_announced_rather_than_silently_dropped():
    """The losses that used to be invisible. A latched breaker moves no counter at all."""
    client = FakeClient(raises(RuntimeError("down")))
    embedder = Embedder(cfg(embed_breaker_threshold=1), client)

    await embedder.embed("one")  # the failure that latches it
    assert embedder.breaker_open is True
    assert await embedder.embed("two") is None
    assert await embedder.embed("three") is None

    assert len(client.calls) == 1
    assert embedder.stats["skipped"] == 2
    assert embedder.stats["failures"] == 1
    assert embedder.stats["misses"] == 3, "a skipped attempt is still an attempt"


async def test_a_breaker_skip_does_not_restart_the_cooldown():
    """Counting skips toward the threshold would keep re-opening it on calls it never let out."""
    embedder = Embedder(
        cfg(embed_breaker_threshold=1, embed_breaker_cooldown_s=0.05),
        FakeClient(raises(RuntimeError("down"))),
    )
    await embedder.embed("a")
    assert embedder.breaker_open is True

    await asyncio.sleep(0.03)
    await embedder.embed("b")  # skipped, halfway through the cooldown
    await asyncio.sleep(0.03)

    assert embedder.breaker_open is False
    assert embedder.stats["skipped"] == 1


async def test_only_one_call_is_let_through_when_the_cooldown_elapses():
    """Half-open means one trial, not the whole waiting room.

    Without the claim, "the cooldown has elapsed" is true for every session queued behind
    the breaker at the same instant, so a dead endpoint collects a full `embed_timeout_ms`
    from all of them before the first return re-latches it, once per cooldown, forever.
    """
    client = FakeClient(hangs())
    embedder = Embedder(
        cfg(embed_timeout_ms=40, embed_breaker_threshold=1, embed_breaker_cooldown_s=0.02),
        client,
    )

    assert await embedder.embed("the one that opens it") is None
    assert embedder.breaker_open is True
    await asyncio.sleep(0.03)

    results = await asyncio.gather(*(embedder.embed(f"q{i}") for i in range(10)))

    assert results == [None] * 10
    assert len(client.calls) == 2, "the call that opened the breaker, and the one trial"
    assert embedder.stats["skipped"] == 9
    assert embedder.breaker_open is True, "the trial failed, so it re-latched"


async def test_a_cancelled_trial_releases_the_half_open_slot():
    """A cancelled turn is not an embedding failure, and must not become a permanent one.

    `CancelledError` deliberately escapes this module, so the release has to be in a
    `finally`. A trial that is never released holds the slot for the life of the process,
    which would make one cancelled turn cost every session after it.
    """
    state = {"mode": "fail"}

    async def handler(_: str) -> Any:
        if state["mode"] == "fail":
            raise RuntimeError("down")
        if state["mode"] == "hang":
            await asyncio.sleep(3600)
        return FakeResponse(embeddings=[FakeEmbedding(values=[1.0, 0.0, 0.0, 0.0])])

    embedder = Embedder(
        cfg(embed_timeout_ms=5000, embed_breaker_threshold=1, embed_breaker_cooldown_s=0.02),
        FakeClient(handler),
    )
    assert await embedder.embed("the one that opens it") is None
    await asyncio.sleep(0.03)

    state["mode"] = "hang"
    trial = asyncio.create_task(embedder.embed("the trial"))
    await asyncio.sleep(0.01)  # long enough to claim the slot and block on the endpoint
    trial.cancel()
    with pytest.raises(asyncio.CancelledError):
        await trial

    state["mode"] = "ok"
    assert await embedder.embed("the next one") == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert embedder.breaker_open is False


async def test_failures_are_logged_once_per_reason_then_rolled_up(
    caplog: pytest.LogCaptureFixture,
):
    """One line per kind, then a periodic re-report. Neither silence nor a line per turn."""
    embedder = Embedder(
        cfg(embed_breaker_threshold=99, embed_failure_log_every=4),
        FakeClient(raises(RuntimeError("down"))),
    )
    with caplog.at_level("WARNING", logger="adk_smart_router.embedding"):
        for i in range(9):
            await embedder.embed(f"prompt {i}")

    assert embedder.stats["failures"] == 9
    # The first, then the 4th and the 8th.
    assert len(caplog.records) == 3
    assert "8 of 8 attempts lost: 0 timeouts, 8 errors, 0 breaker skips" in (
        caplog.records[-1].getMessage()
    )


async def test_the_rollup_line_carries_every_kind_of_loss(caplog: pytest.LogCaptureFixture):
    """The counts are the whole point: they say which remediation the incident needs."""
    embedder = Embedder(
        cfg(embed_timeout_ms=5, embed_breaker_threshold=99, embed_failure_log_every=2),
        FakeClient(hangs()),
    )
    with caplog.at_level("WARNING", logger="adk_smart_router.embedding"):
        await embedder.embed("one")
        await embedder.embed("two")

    assert "timed out after 5 ms" in caplog.records[0].getMessage()
    assert "2 of 2 attempts lost: 2 timeouts, 0 errors, 0 breaker skips" in (
        caplog.records[-1].getMessage()
    )


async def test_a_log_every_of_zero_does_not_divide_by_zero(caplog: pytest.LogCaptureFixture):
    embedder = Embedder(
        cfg(embed_failure_log_every=0, embed_breaker_threshold=99),
        FakeClient(raises(RuntimeError("down"))),
    )
    with caplog.at_level("WARNING", logger="adk_smart_router.embedding"):
        for i in range(3):
            await embedder.embed(f"prompt {i}")
    assert len(caplog.records) == 3  # clamped to 1: every loss reports


# --- Why a None is a None -------------------------------------------------------
#
# Timeout, call failure and open breaker all used to reach BigQuery as `decision_source:
# "error"`, and the remediation for each is different: ab-20260907 lost 47 sessions to Vertex
# quota latching the breaker, ab-20260911m8 lost 40 to plain timeouts (DESIGN.md 19.23).


async def test_a_successful_embed_reports_no_reason():
    embedder = Embedder(cfg(), FakeClient(returns([1.0, 0.0, 0.0, 0.0])))
    vector, why = await embedder.embed_with_reason("prompt")
    assert vector is not None
    assert why == embedding.OK

    cached, why = await embedder.embed_with_reason("prompt")
    assert cached == vector
    assert why == embedding.OK


async def test_a_timeout_a_call_failure_and_an_open_breaker_report_different_reasons():
    slow = Embedder(cfg(embed_timeout_ms=5, embed_breaker_threshold=99), FakeClient(hangs()))
    assert await slow.embed_with_reason("slow") == (None, embedding.TIMEOUT)

    broken = Embedder(cfg(embed_breaker_threshold=1), FakeClient(raises(RuntimeError("429"))))
    assert await broken.embed_with_reason("boom") == (None, embedding.CALL_ERROR)
    assert await broken.embed_with_reason("next") == (None, embedding.BREAKER_OPEN)

    assert len({embedding.TIMEOUT, embedding.CALL_ERROR, embedding.BREAKER_OPEN}) == 3


async def test_blank_text_reports_its_own_reason_and_is_not_an_endpoint_failure():
    embedder = Embedder(cfg(), FakeClient(returns([1.0, 0.0, 0.0, 0.0])))
    assert await embedder.embed_with_reason("  \n ") == (None, embedding.EMPTY)
    assert embedder.stats["failures"] == 0
    assert embedder.stats["misses"] == 0


async def test_cancellation_still_propagates():
    """A cancelled turn is not an embedding failure, so CancelledError escapes."""
    embedder = Embedder(cfg(), FakeClient(raises(asyncio.CancelledError())))  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        await embedder.embed("prompt")


async def test_stats_shape_and_isolation():
    embedder = Embedder(cfg(), FakeClient(returns([1.0, 0.0, 0.0, 0.0])))
    snapshot = embedder.stats
    assert set(snapshot) == {"hits", "misses", "timeouts", "failures", "skipped"}
    await embedder.embed("prompt")
    assert snapshot["misses"] == 0  # the snapshot is a copy, not a live view
    assert embedder.stats["misses"] == 1


# --- Warm-up ------------------------------------------------------------------


async def test_warmup_calls_the_endpoint_and_reports_success():
    client = FakeClient(returns([1.0, 0.0, 0.0, 0.0]))
    embedder = Embedder(cfg(), client)
    assert await embedder.warmup() is True
    assert len(client.calls) == 1


async def test_warmup_leaves_the_cache_and_the_counters_alone():
    """It is not a prompt. Nothing it does may change what a later embed returns."""
    client = FakeClient(returns([1.0, 0.0, 0.0, 0.0]))
    embedder = Embedder(cfg(), client)
    await embedder.warmup()
    assert embedder.stats == {
        "hits": 0,
        "misses": 0,
        "timeouts": 0,
        "failures": 0,
        "skipped": 0,
    }

    await embedder.embed("warmup")  # the warm-up text itself must still be a miss
    assert embedder.stats["hits"] == 0
    assert embedder.stats["misses"] == 1
    assert len(client.calls) == 2


async def test_a_failed_warmup_returns_false_without_touching_the_breaker():
    """A warm-up that pre-opened the breaker would turn a slow start into a dead one."""
    embedder = Embedder(cfg(embed_breaker_threshold=1), FakeClient(raises(RuntimeError("down"))))
    assert await embedder.warmup() is False
    assert embedder.breaker_open is False
    assert embedder.stats["failures"] == 0


async def test_warmup_has_its_own_timeout_not_the_hot_path_one():
    """5 ms would time out; the warm-up budget is what applies."""
    client = FakeClient(returns([1.0, 0.0, 0.0, 0.0]))
    embedder = Embedder(cfg(embed_timeout_ms=5, embed_warmup_timeout_ms=5000), client)
    assert await embedder.warmup() is True


async def test_warmup_times_out_on_its_own_budget():
    embedder = Embedder(cfg(embed_warmup_timeout_ms=10), FakeClient(hangs()))
    assert await embedder.warmup() is False


async def test_a_failed_warmup_logs_once_and_does_not_raise(caplog: pytest.LogCaptureFixture):
    embedder = Embedder(cfg(), FakeClient(raises(RuntimeError("down"))))
    with caplog.at_level("WARNING", logger="adk_smart_router.embedding"):
        assert await embedder.warmup() is False
    assert len(caplog.records) == 1
    assert "warmup failed" in caplog.records[0].getMessage()
