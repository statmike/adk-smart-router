"""Two replicas on one artifact, run long enough to overflow every bounded map (ROADMAP H6).

Everything the plugin remembers is an LRU with a fixed capacity, and every one of those bounds
is a trade: a fixed memory ceiling in exchange for forgetting something. The unit tests prove
each map evicts. What they do not prove is what a process looks like after thousands of turns
have pushed every map past its capacity several times over, or what happens when two processes
serve the same sessions, which is the deployment this is actually for.

So this file runs the churn. Four hundred sessions of three turns each, split across two plugin
instances, with a deterministic embedder and predictor so a message always lands on the same
menu and two replicas seeing the same message have to agree.

Two things come out of it, and the second is the reason the file exists.

The maps stay bounded, and the two replicas agree on the same first message. That is the part
that had to be true.

A session that outlives the LRU, or that moves between replicas, is decided again from a later
message, and the second decision can be a different menu. Nothing stops it: the prefix lock only
records sessions that sent a *full* tool list, so a pruned session has no lock to trip, and the
lock is per process anyway (D8). This is a cost rather than a bug, and DESIGN.md D8 prices it as
one recomputation. The tests below are named after it so it stays visible.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from test_model_store import FakeBlob, FakeBucket, metadata_dict
from test_plugin import TOOLS, FakeContext, declared_names, make_plugin, make_request, records

from adk_smart_router import plugin as plugin_mod
from adk_smart_router.config import RouterConfig
from adk_smart_router.model_store import ModelStore
from adk_smart_router.predictor import Prediction

CAPACITY = 32
SESSIONS = 400
TURNS = 3


# --- A deterministic router ----------------------------------------------------


class TextEmbedder:
    """A stand-in whose vector is a function of the text, so a message maps to one menu.

    The real embedder is deterministic in exactly this way, and it matters here: the claim
    under test is that two replicas given the same first message declare the same tools, and
    an embedder that returned a constant would make that true for the wrong reason.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float] | None:
        vector, _ = await self.embed_with_reason(text)
        return vector

    async def embed_with_reason(self, text: str) -> tuple[list[float] | None, str]:
        self.calls += 1
        return ([float(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))], "")

    async def warmup(self) -> bool:
        return True

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": 0, "misses": self.calls, "timeouts": 0, "failures": 0, "skipped": 0}


class KeyedPredictor:
    """Three tools per vector, at a mass that keeps all three and prunes the other nineteen."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, vector: list[float]) -> tuple[Prediction, ...]:
        self.calls += 1
        seed = int(vector[0])
        picks = [TOOLS[(seed + step * 7) % len(TOOLS)] for step in range(3)]
        return (Prediction(picks[0], 0.6), Prediction(picks[1], 0.25), Prediction(picks[2], 0.1))


def replica(monkeypatch: Any, **overrides: Any) -> Any:
    """One plugin instance, standing in for one serving process."""
    fields: dict[str, Any] = {"session_decision_cache_size": CAPACITY}
    fields.update(overrides)
    router = make_plugin(monkeypatch, embedder=TextEmbedder(), **fields)
    # After `make_plugin`, which patches in a predictor of its own. The plugin builds its
    # predictor lazily on the first turn, so this is the one that gets used.
    monkeypatch.setattr(plugin_mod, "Predictor", lambda _model: KeyedPredictor())
    return router


async def turn(router: Any, session: str, prompt: str) -> tuple[Any, Any]:
    ctx = FakeContext(session_id=session)
    request = make_request(prompt=prompt)
    await router.before_model_callback(callback_context=ctx, llm_request=request)
    return ctx, request


def source(ctx: Any) -> str | None:
    written = records(ctx)
    return written[0]["decision_source"] if written else None


def sizes(router: Any) -> dict[str, int]:
    """Every map in the plugin that grows with traffic."""
    return {
        "decisions": len(router._decisions),  # noqa: SLF001
        "sent": len(router._sent),  # noqa: SLF001
        "prefix locks": len(router._lock._locked),  # noqa: SLF001
        "session calls": len(router._calls._seen),  # noqa: SLF001
    }


# --- The churn ------------------------------------------------------------------


async def test_every_bounded_map_stays_bounded_across_a_long_run(monkeypatch) -> None:
    """1,200 turns over 400 sessions on two replicas, and nothing grows without a ceiling.

    The high-water mark is taken after every turn rather than at the end, because a map that
    overshoots and then evicts back down would pass an end-of-run check while still having
    held whatever it held at the peak.

    The prefix lock stays empty for the whole run, and that is not an oversight in the test.
    It records sessions that sent a full tool list, and every session here is pruned. It is
    also the reason the two tests below have nothing to catch them.
    """
    replicas = [replica(monkeypatch), replica(monkeypatch)]
    high = dict.fromkeys(sizes(replicas[0]), 0)

    for number in range(SESSIONS):
        router = replicas[number % 2]
        for index in range(TURNS):
            await turn(router, f"s-{number}", f"session {number} turn {index}")
            for name, size in sizes(router).items():
                high[name] = max(high[name], size)

    assert high == {
        "decisions": CAPACITY,
        "sent": CAPACITY,
        "prefix locks": 0,
        "session calls": CAPACITY,
    }
    for router in replicas:
        assert dict(router.failopen_counts) == {}, "a callback failed open during the run"


async def test_the_run_never_declares_a_tool_the_host_did_not(monkeypatch) -> None:
    """The subset property, asserted over churn rather than over a randomised walk.

    `tests/test_pruner.py` pins `kept ⊆ live` inside the pruner. This asserts it on what
    actually reaches the request, on a shorter run of the same traffic, and after eviction
    has started recycling the maps.
    """
    router = replica(monkeypatch, session_decision_cache_size=8)

    for number in range(60):
        for index in range(2):
            _, request = await turn(router, f"s-{number}", f"session {number} turn {index}")
            declared = declared_names(request)
            assert declared, "a session was left with no tools at all"
            assert set(declared) <= set(TOOLS), "a tool appeared that the host never declared"


async def test_a_session_inside_the_lru_decides_once_and_reuses_it(monkeypatch) -> None:
    """The control for the two tests below: while a session is remembered, it is not re-decided."""
    router = replica(monkeypatch)
    embedder = router._embedder  # noqa: SLF001

    _, first = await turn(router, "s-1", "where is my invoice")
    ctx, second = await turn(router, "s-1", "and what about the refund")

    assert embedder.calls == 1
    assert source(ctx) == "reused"
    assert declared_names(second) == declared_names(first)


# --- What the bound costs -------------------------------------------------------


async def test_a_session_that_outlives_the_lru_is_decided_again_from_a_later_message(
    monkeypatch,
) -> None:
    """The price of a bounded map, stated as a test rather than as a footnote.

    Once a session is evicted, its next turn looks exactly like a first turn: no remembered
    decision, and no prefix lock either, because a pruned send never locks. So the router
    decides again, from whatever the user happened to say this time, and lands on a different
    menu. That is a new declaration list mid-conversation, which is a broken prefix cache and
    a full-price turn (DESIGN.md 18.22).

    D8 accepts this and prices it as one recomputation. The point of writing it down is that
    the frequency is not a property of the router at all: it is the LRU capacity against the
    sessions in flight on one replica, which is an operator's sizing decision.
    """
    router = replica(monkeypatch, session_decision_cache_size=4)
    _, first = await turn(router, "s-long", "where is my invoice")

    for number in range(8):
        await turn(router, f"s-{number}", f"unrelated traffic {number}")

    ctx, later = await turn(router, "s-long", "and what about the refund")

    assert source(ctx) == "fresh", "an evicted session should look new, not reused and not late"
    assert declared_names(later) != declared_names(first), "the menu moved, which is the cost"
    assert sizes(router)["prefix locks"] == 0, "nothing locked, so nothing could have caught it"


async def test_a_session_that_moves_replicas_is_decided_again_by_the_one_it_lands_on(
    monkeypatch,
) -> None:
    """The same cost, arriving the other way: a load balancer rather than an eviction.

    Both maps are in-process on purpose (D8), so a replica that did not serve turn one knows
    nothing about it. Sticky sessions avoid this entirely, which is worth saying out loud
    because it is the only mitigation and it lives in the deployment rather than in the code.
    """
    first_replica = replica(monkeypatch)
    second_replica = replica(monkeypatch)

    _, first = await turn(first_replica, "s-1", "where is my invoice")
    ctx, second = await turn(second_replica, "s-1", "and what about the refund")

    assert source(ctx) == "fresh"
    assert declared_names(second) != declared_names(first)


async def test_a_session_held_out_on_one_replica_can_be_pruned_by_another(monkeypatch) -> None:
    """The sharpest version: the guard that exists for this misses it, because it is local.

    A session that sent a full tool list has locked its prefix, and `_TOO_LATE` refuses to
    prune it however good the later prediction looks. The lock lives in the process that sent
    that turn. Land the next turn on a replica that did not, and the prune happens.

    The router is not wrong here so much as blind: it is the first turn as far as that process
    can see. It is in this file because the fix is sticky sessions, and an operator who reads
    only the holdout documentation would not know to ask for them.
    """
    holding_out = replica(monkeypatch, holdout_fraction=1.0)
    routing = replica(monkeypatch, holdout_fraction=0.0)

    held_ctx, held = await turn(holding_out, "s-1", "where is my invoice")
    ctx, later = await turn(routing, "s-1", "and what about the refund")

    assert source(held_ctx) == "holdout"
    assert sorted(declared_names(held)) == sorted(TOOLS)
    assert source(ctx) == "fresh"
    assert len(declared_names(later)) < len(TOOLS), "the second replica pruned a locked session"


async def test_two_replicas_given_the_same_first_message_declare_the_same_tools(
    monkeypatch,
) -> None:
    """The property that makes a fleet coherent, and the reason menus break ties on membership.

    Two servers handing one session two different declaration lists is the exact cost menus
    exist to remove (DESIGN.md, "Three properties make this safe to run in a replicated
    deployment"). Same message, same vector, same prediction, same order.
    """
    first_replica = replica(monkeypatch)
    second_replica = replica(monkeypatch)

    for number in range(20):
        prompt = f"a message that could arrive at either replica {number}"
        _, here = await turn(first_replica, f"s-{number}", prompt)
        _, there = await turn(second_replica, f"s-{number}", prompt)
        assert declared_names(here) == declared_names(there)


# --- One artifact, many loaders --------------------------------------------------

BIG = b"x" * 200_000
BIG_DIGEST = hashlib.sha256(BIG).hexdigest()
VERSION = "20260101T000000Z"


class ChunkedBlob(FakeBlob):
    """Writes the booster in small pieces, so threads sharing one tmp name really interleave.

    `FakeBlob` writes in one call, which the OS is free to make look atomic. The contract note
    in `model_store._ensure_local` is about what happens when it is not, and a test of that
    contract has to actually produce the interleaving.
    """

    def download_to_filename(self, path: str) -> None:
        data = self._bytes()  # noqa: SLF001
        with open(path, "wb") as handle:
            for start in range(0, len(data), 4096):
                handle.write(data[start : start + 4096])
                time.sleep(0)  # hand the GIL to another loader mid-write


class ChunkedBucket(FakeBucket):
    def blob(self, name: str) -> ChunkedBlob:
        return ChunkedBlob(name, self)

    def get_blob(self, name: str) -> ChunkedBlob | None:
        return ChunkedBlob(name, self) if name in self.blobs else None


class ChunkedClient:
    def __init__(self, bucket: ChunkedBucket) -> None:
        self._bucket = bucket

    def bucket(self, name: str) -> ChunkedBucket:
        return self._bucket


def test_eight_loaders_sharing_one_cache_dir_all_get_the_booster(tmp_path) -> None:
    """Two processes on one artifact, which on one machine is one `local_cache_dir`.

    This is the test that found something. Loaders used to share a fixed tmp filename, which
    fails safe, so every unit test of it passed: a loader whose file was written over by
    another gets a digest mismatch and returns None, and None means passthrough. Run eight of
    them at once and seven return None, which stops being a graceful degradation once you
    notice that `plugin._resolve` loads once per process and remembers the answer. Seven
    workers on that host would never route again.

    The tmp name is now unique per attempt, so all eight get the artifact, `os.replace` still
    publishes it, and no `.tmp` survives the run.
    """
    prefix = f"routers/planner/{VERSION}"
    blobs = {
        f"{prefix}/metadata.json": _metadata_bytes(),
        f"{prefix}/model.ubj": BIG,
    }
    bucket = ChunkedBucket(blobs)
    cache = tmp_path / "cache"
    stores = [
        ModelStore(
            RouterConfig(gcs_bucket="artifacts", local_cache_dir=str(cache)),
            storage_client=ChunkedClient(bucket),
        )
        for _ in range(8)
    ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = list(pool.map(lambda store: store.load("planner"), stores))

    assert [model for model in loaded if model is None] == [], "a loader lost the race"
    for model in loaded:
        assert hashlib.sha256(Path(model.booster_path).read_bytes()).hexdigest() == BIG_DIGEST

    version_dir = cache / "planner" / VERSION
    assert [path.name for path in version_dir.iterdir()] == ["model.ubj"], "a .tmp was left behind"

    # And the storm leaves the cache in a state a later process can use.
    recovered = stores[0].load("planner")
    assert recovered is not None
    assert Path(recovered.booster_path).read_bytes() == BIG


def _metadata_bytes() -> bytes:
    import json

    return json.dumps(
        metadata_dict(version=VERSION, sha256=BIG_DIGEST, size_bytes=len(BIG))
    ).encode()
