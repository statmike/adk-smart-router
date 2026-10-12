"""Turn a prompt into a vector. The only place in the package that does that.

The one rule that must never be broken here: **the client is pinned to Vertex
AI**. Training embeds with BigQuery `AI.EMBED`, which resolves to Vertex.
`google-genai` defaults to the Gemini Developer API. Same model name, same
dimensions, no error, subtly different vectors, and a model that passes the
offline gate and predicts garbage in production (DESIGN.md 9.1). `vertexai=True`
plus truncate-then-normalize in that order is the serving half of the parity
contract; the training half lives in `sql/train_router.sql`.

A remote call is therefore settled policy rather than the current
implementation. Running a small model in-process instead would remove the only
network hop on the hot path, and it was measured twice and declined both times:
recall passes, cost does not, and a second embedding model breaks the pairing
above until a retrain and a fresh parity check pay for it (DESIGN.md 19.78).

Nothing in this module raises to its caller. Timeout, open breaker, malformed
response, wrong dimensionality: all of them return None and let the turn pass
through untouched (DESIGN.md 11.2). They differ in what fixes them, though, so
each None is labelled — `embed_with_reason` returns the label, and the counters
in `stats` carry the rate.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

from google import genai
from google.genai import types

from adk_smart_router.config import RouterConfig

logger = logging.getLogger(__name__)


# --- Parity primitives --------------------------------------------------------


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Scale a vector to unit length. An all-zero vector is returned unchanged."""
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0.0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def truncate_and_normalize(vector: Sequence[float], dims: int) -> list[float]:
    """Truncate to `dims`, then L2-normalize. This order is the parity contract.

    Normalizing first and truncating after leaves a vector of norm < 1 whose
    direction also differs, which no test of the model itself can detect
    (DESIGN.md 9.1). Raises ValueError if the vector is shorter than `dims`.

    Both steps are currently no-ops, and they stay anyway. The training SQL asks
    `AI.EMBED` for `dims` directly rather than slicing a wider vector, and
    `AI.EMBED` returns unit-norm output, measured at 1.0 +/- 3e-7 across the
    corpus. So there is nothing to truncate and nothing to rescale, and live
    parity comes out at cosine 1.000000000. Both steps are load-bearing the
    moment either of those facts changes, which is exactly the kind of change
    that arrives without an error.
    """
    if dims <= 0:
        raise ValueError(f"dims must be positive, got {dims}")
    if len(vector) < dims:
        raise ValueError(f"embedding has {len(vector)} dims, need at least {dims}")
    return l2_normalize(vector[:dims])


# --- The embedder -------------------------------------------------------------

# Why a call returned None. Three of these reach BigQuery as a `decision_source`, and
# they are kept apart because their remediation is: `timeout` wants a larger budget or a
# faster endpoint, `error` wants the endpoint's own error read, `breaker` means the
# router already gave up on this process and the fix is upstream of both. Two live
# incidents recorded 47 and 40 lost sessions respectively, arrived in the warehouse
# looking identical, and needed opposite responses (DESIGN.md 19.23).
OK = ""
TIMEOUT = "timeout"
CALL_ERROR = "error"
BREAKER_OPEN = "breaker"
EMPTY = "empty"  # nothing to embed; the plugin filters these out before it ever calls


class Embedder:
    """Prompt to unit vector, with an LRU, a hard timeout and a circuit breaker."""

    # `client` is Any because google-genai's Client is only duck-typed here: tests
    # inject a fake, and the real one is built lazily inside `_get_client`.
    def __init__(self, cfg: RouterConfig, client: Any | None = None) -> None:  # noqa: ANN401
        self._cfg = cfg
        self._client = client
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        # True while the one call the cooldown lets out is still in flight. See `_admit`.
        self._trial_in_flight = False
        self._logged: set[str] = set()
        self._stats: dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "timeouts": 0,
            "failures": 0,
            # Calls the open breaker turned away. Counted because they were previously
            # invisible to both the log and the counters: a latched endpoint lost one
            # session per prompt while `timeouts` and `failures` stayed frozen.
            "skipped": 0,
        }

    async def embed(self, text: str) -> list[float] | None:
        """Embed `text`, or return None. Never raises.

        The short form, for callers that only want the vector. `embed_with_reason` is
        the same call and additionally says why a None is a None.
        """
        vector, _ = await self.embed_with_reason(text)
        return vector

    async def embed_with_reason(self, text: str) -> tuple[list[float] | None, str]:
        """Embed `text`, and say why if it did not. Never raises.

        The reason is returned rather than left on the instance for the caller to read
        back, because sessions run concurrently: the A/B ran four at a time, and a
        `last_failure` attribute would have attributed one session's timeout to another
        session's quota error. That is worse than no attribution at all.
        """
        if not text.strip():
            return None, EMPTY

        cached = self._cache_get(text)
        if cached is not None:
            self._stats["hits"] += 1
            return cached, OK
        self._stats["misses"] += 1

        if not self._admit():
            # Not a fresh failure, so `_consecutive_failures` is deliberately untouched:
            # counting skips toward the threshold would keep re-arming the breaker on
            # calls it never let out.
            self._stats["skipped"] += 1
            self._log_failure(BREAKER_OPEN, "embedding skipped, breaker open")
            return None, BREAKER_OPEN

        # A non-positive timeout would disable embedding outright; clamp instead.
        timeout_s = max(self._cfg.embed_timeout_ms, 1) / 1000.0
        try:
            vector = await asyncio.wait_for(self._fetch(text), timeout_s)
        except TimeoutError:
            self._stats["timeouts"] += 1
            self._record_failure(TIMEOUT, f"embedding timed out after {timeout_s * 1000:.0f} ms")
            return None, TIMEOUT
        except Exception as exc:
            # Broad on purpose: this module owes its caller a None, never a raise.
            # CancelledError is a BaseException and deliberately still escapes,
            # because a cancelled turn is not an embedding failure.
            self._stats["failures"] += 1
            self._record_failure(CALL_ERROR, f"embedding call failed: {exc!r}")
            return None, CALL_ERROR
        finally:
            # In `finally` and not on each path out, because CancelledError leaves through
            # here too. A trial that is never released latches the breaker for the life of
            # the process, which would make one cancelled turn cost every later one.
            self._trial_in_flight = False

        self._consecutive_failures = 0
        self._opened_at = None
        self._cache_put(text, vector)
        return vector, OK

    async def warmup(self) -> bool:
        """Pay the cold-start cost off the hot path. Call once, before serving traffic.

        The first call this process ever makes to Vertex costs about 1.9 seconds:
        ADC token fetch, TLS handshake, HTTP/2 setup. Steady state is p50 147 ms.
        So a cold process blows `embed_timeout_ms` on its first prompt no matter
        how generous that budget is, and because a routing decision is
        session-stable and one embedding failure pins the session to passthrough
        (DESIGN.md 11.2), that single timeout costs an entire session, not one
        turn.

        Measured, not assumed: the first routed session against the reference
        agent passed through for exactly this reason (DESIGN.md 18.10).

        Deliberately not on the hot path and deliberately not automatic. It uses
        its own generous timeout, it does not touch the LRU, and a failure here
        does not count toward the breaker: a warmup that could pre-open the
        breaker would turn a slow start into a cold start that never routes.
        Returns whether the endpoint answered.

        Left exactly as it is on purpose. Re-measured 2026-09-05 over 19 fresh
        processes: it costs 1.8-2.6 s, it succeeded every time, and it removed the
        1954 ms cold call every time. Warming four connections in parallel instead
        of one measured no better and cost more, so that idea is closed
        (DESIGN.md 19.23).
        """
        timeout_s = max(self._cfg.embed_warmup_timeout_ms, 1) / 1000.0
        try:
            await asyncio.wait_for(self._fetch("warmup"), timeout_s)
        except Exception as exc:  # noqa: BLE001 - a failed warmup is a log line, never a raise
            logger.warning("smart_router embedding: warmup failed (%r)", exc)
            return False
        return True

    # NOTE(contract): the frozen __init__ takes no clock, unlike ToolResultCache,
    # so breaker-recovery tests have to sleep in real time. Harmless at these
    # cooldowns; worth an injected clock if the timings ever grow.
    @property
    def breaker_open(self) -> bool:
        """True while the breaker is holding calls back.

        A pure read: asking does not claim the trial. `_admit` is what decides which
        single call gets through once the cooldown has elapsed.
        """
        if self._opened_at is None:
            return False
        # Monotonic, never the wall clock: an NTP step must not pin the breaker open.
        cooldown: float = self._cfg.embed_breaker_cooldown_s
        if time.monotonic() - self._opened_at < cooldown:
            return True
        # Cooled off, but half-open: calls are still held back while the trial is out.
        return self._trial_in_flight

    def _admit(self) -> bool:
        """Whether this call runs, claiming the half-open trial if it is the one taking it.

        The claim is what makes it *one* trial. Without it, "the cooldown has elapsed" is
        true for every session waiting behind the breaker at the same instant, so a dead
        endpoint collects a full `embed_timeout_ms` from all of them before the first
        return re-latches it, once per cooldown, forever. That is the herd a breaker exists
        to stop.

        No lock. The claim and the check are one synchronous stretch with no `await`
        between them, so on the event loop this runs on they cannot interleave.
        """
        if self.breaker_open:
            return False
        if self._opened_at is not None:
            self._trial_in_flight = True
        return True

    @property
    def stats(self) -> Mapping[str, int]:
        return dict(self._stats)

    # --- Internals ------------------------------------------------------------

    async def _fetch(self, text: str) -> list[float]:
        """Call Vertex and shape the response. This raises; `embed` swallows."""
        response = await self._get_client().aio.models.embed_content(
            model=self._cfg.embedding_model,
            contents=text,
            config=self._embed_config(),
        )
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise ValueError("embedding response carried no embeddings")
        values = getattr(embeddings[0], "values", None)
        if not values:
            raise ValueError("embedding response carried no values")
        return truncate_and_normalize(values, self._cfg.embedding_dims)

    def _get_client(self) -> Any:  # noqa: ANN401 - duck-typed client, see __init__
        if self._client is None:
            # vertexai=True is the parity contract, not a preference (DESIGN.md 9.1).
            # No `credentials` argument, so the client inherits ADC.
            self._client = genai.Client(
                vertexai=True,
                project=self._cfg.bq_project,
                location=self._cfg.vertex_location,
            )
        return self._client

    def _embed_config(self) -> types.EmbedContentConfig:
        # task_type is omitted entirely when unset. gemini-embedding-2 is a
        # multimodal endpoint and AI.EMBED sends no task type, so sending one
        # here would be a train/serve divergence (DESIGN.md 9.1).
        if self._cfg.task_type is None:
            return types.EmbedContentConfig(output_dimensionality=self._cfg.embedding_dims)
        return types.EmbedContentConfig(
            output_dimensionality=self._cfg.embedding_dims,
            task_type=self._cfg.task_type,
        )

    def _record_failure(self, reason: str, message: str) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._cfg.embed_breaker_threshold:
            self._opened_at = time.monotonic()
            # Same key as a breaker skip on purpose: the latch line and the skips it
            # causes are one incident, and the reader wants it announced once.
            self._log_failure(BREAKER_OPEN, f"embedding breaker open after {message}")
            return
        self._log_failure(reason, message)

    def _log_failure(self, reason: str, message: str) -> None:
        """Warn the first time each failure kind appears, then re-report on a rollup.

        A degraded endpoint fails on every turn, so a line per failure buries the
        incident under its own noise. Logging only the first line hides its size,
        which is how a run lost 17.9% of its sessions behind one suppressed warning
        (DESIGN.md 19.23). So: first of each kind, then every
        `embed_failure_log_every` losses, always carrying the running counters.

        Counted rather than timed because this class deliberately holds no clock
        (see the NOTE above `breaker_open`). `misses` is the denominator: it counts
        every attempt that was not an LRU hit, breaker skips included.
        """
        lost = self._stats["timeouts"] + self._stats["failures"] + self._stats["skipped"]
        every = max(self._cfg.embed_failure_log_every, 1)
        if reason in self._logged and lost % every != 0:
            return
        self._logged.add(reason)
        logger.warning(
            "smart_router embedding: %s (%d of %d attempts lost: "
            "%d timeouts, %d errors, %d breaker skips)",
            message,
            lost,
            self._stats["misses"],
            self._stats["timeouts"],
            self._stats["failures"],
            self._stats["skipped"],
        )

    def _cache_get(self, text: str) -> list[float] | None:
        vector = self._cache.get(text)
        if vector is None:
            return None
        self._cache.move_to_end(text)
        return list(vector)

    def _cache_put(self, text: str, vector: list[float]) -> None:
        if self._cfg.embed_cache_size <= 0:
            return
        # Store a copy: a caller mutating the returned list must not corrupt it.
        self._cache[text] = list(vector)
        self._cache.move_to_end(text)
        while len(self._cache) > self._cfg.embed_cache_size:
            self._cache.popitem(last=False)
