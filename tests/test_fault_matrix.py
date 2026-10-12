"""Every way the outside world can fail, and the one thing that must happen anyway (ROADMAP H2).

The router depends on three things it does not control: a GCS bucket holding an artifact, a
Vertex endpoint turning a prompt into a vector, and whatever the operator published into that
bucket. Each of those has its own unit tests, and each of them proves the same small thing:
the function returns None instead of raising. None of them prove the turn survives.

That is what this file is for. Every row below breaks one dependency, in one specific way, at
the real boundary — a real `ModelStore` with a broken bucket underneath it, a real `Embedder`
with a broken endpoint underneath it — and then runs a real ADK turn through a real `Runner`
against the reference agent's hundred tools. Three assertions per row, the same three every
time:

1. The user got an answer.
2. The model was shown every tool, so nothing was pruned on a decision that could not be made.
3. No callback fell into `fail_open`.

The third is the one that makes this more than a smoke test. `fail_open` is the last-resort
catch that stops any plugin exception aborting an ADK run, and every callback carries it. It
should never fire on a fault this file injects, because each of those is a fault the code was
written to handle by name. A row that passes the first two assertions and trips the third is a
row where the router got lucky rather than being right, and the difference is invisible to the
user until the day the decorator is not enough.

The control row at the top routes. Without it every other row would pass by doing nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import reference_agent.agent as reference
import xgboost
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.api_core import exceptions as gcs_errors
from google.genai import types
from reference_agent.tools import ALL_TOOLS, reset_fixtures
from test_end_to_end import ScriptedLlm, declared
from test_plugin import FakeClock

from adk_smart_router.config import RouterConfig
from adk_smart_router.embedding import Embedder
from adk_smart_router.model_store import ModelStore
from adk_smart_router.plugin import SmartRouterPlugin
from adk_smart_router.provenance import PROVENANCE_KEY
from adk_smart_router.tool_cache import ToolResultCache

APP = "support"
USER = "u-1"
AGENT = reference.AGENT_NAME
VERSION = "20260901T000000Z"

# Sixteen, not the shipped 768. Nothing here measures a vector, and a real booster over 768
# features costs seconds to fit for no gain. The artifact and the config agree on the number,
# which is what `_check_compatible` checks, and that is the only thing the width has to do
# with any row below.
DIMS = 16

TOOL_NAMES = tuple(fn.__name__ for fn in ALL_TOOLS)
CLASSES = (*TOOL_NAMES, "ABSTAIN")

PREFIX = f"routers/{AGENT}/{VERSION}"


# --- A real booster -----------------------------------------------------------


def train_booster(path: Path) -> None:
    """A genuine XGBoost multi-class booster, small enough to fit in a fixture.

    The control row needs the whole path to work, including `xgboost.Booster(model_file=...)`,
    and a file of plausible bytes will not do: loading one is the failure the corrupt-booster
    row injects on purpose. So this trains a real one, with the positional `f0..fN` feature
    naming the serving path relies on (DESIGN.md D5).
    """
    rows = len(CLASSES) * 2
    rng = np.random.default_rng(0)
    features = rng.normal(size=(rows, DIMS)).astype(np.float32)
    labels = np.array([n % len(CLASSES) for n in range(rows)])
    matrix = xgboost.DMatrix(features, label=labels, feature_names=[f"f{i}" for i in range(DIMS)])
    booster = xgboost.train(
        {"objective": "multi:softprob", "num_class": len(CLASSES), "max_depth": 2, "nthread": 1},
        matrix,
        num_boost_round=2,
    )
    booster.save_model(str(path))


@pytest.fixture(scope="module")
def booster_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("booster") / "model.ubj"
    train_booster(path)
    return path.read_bytes()


# --- A bucket that can be broken ----------------------------------------------

# An effect is an exception to raise or a number of seconds to sleep. Keyed by a fragment of
# the operation, which reads as `list:<prefix>` or `read:<blob name>`, so a row can break the
# listing, the metadata or the booster download independently of the other two.
Effect = Exception | float


class Blob:
    def __init__(self, name: str, bucket: Bucket) -> None:
        self._name = name
        self._bucket = bucket

    def download_as_bytes(self) -> bytes:
        self._bucket.interrupt(f"read:{self._name}")
        return self._bucket.blobs[self._name]

    def download_to_filename(self, path: str) -> None:
        self._bucket.interrupt(f"read:{self._name}")
        # A second key, so a row can break the transfer of the bytes without also breaking
        # the request that reads the object's size.
        self._bucket.interrupt(f"bytes:{self._name}")
        Path(path).write_bytes(self._bucket.blobs[self._name])

    @property
    def size(self) -> int:
        return len(self._bucket.blobs[self._name])


class Bucket:
    def __init__(self, blobs: dict[str, bytes], breaks: dict[str, Effect]) -> None:
        self.blobs = blobs
        self.breaks = breaks

    def interrupt(self, key: str) -> None:
        for fragment, effect in self.breaks.items():
            if fragment not in key:
                continue
            if isinstance(effect, Exception):
                raise effect
            time.sleep(effect)

    def list_blobs(self, prefix: str = "") -> list[Any]:
        self.interrupt(f"list:{prefix}")
        return [SimpleNamespace(name=n) for n in sorted(self.blobs) if n.startswith(prefix)]

    def blob(self, name: str) -> Blob:
        return Blob(name, self)

    def get_blob(self, name: str) -> Blob | None:
        self.interrupt(f"read:{name}")
        return Blob(name, self) if name in self.blobs else None


class StorageClient:
    def __init__(self, bucket: Bucket, breaks: dict[str, Effect]) -> None:
        self._bucket = bucket
        self._breaks = breaks

    def bucket(self, name: str) -> Bucket:
        effect = self._breaks.get("bucket")
        if isinstance(effect, Exception):
            raise effect
        return self._bucket


def metadata(booster: bytes, **overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "version": VERSION,
        "sha256": hashlib.sha256(booster).hexdigest(),
        "size_bytes": len(booster),
        "trained_at": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        "rows": 5000,
        "classes": list(CLASSES),
        "backend": "vertex",
        "environment": "prod",
        "embedding_model": "gemini-embedding-2",
        "embedding_dims": DIMS,
        "gate_metrics": {"coverage": 0.99},
    }
    raw.update(overrides)
    return raw


# --- An endpoint that can be broken -------------------------------------------


class Endpoint:
    def __init__(self, effect: Any) -> None:
        self.effect = effect
        self.calls = 0

    async def embed_content(self, **kwargs: Any) -> Any:  # noqa: ANN401
        self.calls += 1
        if isinstance(self.effect, int | float):
            await asyncio.sleep(self.effect)
        if isinstance(self.effect, Exception):
            raise self.effect
        return self.effect


class GenaiClient:
    def __init__(self, effect: Any) -> None:  # noqa: ANN401
        self.endpoint = Endpoint(effect)
        self.aio = SimpleNamespace(models=self.endpoint)


def good_vector() -> Any:  # noqa: ANN401
    return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * DIMS)])


# --- The agent under test -----------------------------------------------------


def build(
    tmp_path: Path,
    booster: bytes,
    *,
    meta: dict[str, Any] | None = None,
    blobs: dict[str, bytes] | None = None,
    breaks: dict[str, Effect] | None = None,
    embed: Any = None,  # noqa: ANN401
    **cfg_overrides: Any,
) -> Any:  # noqa: ANN401
    """The reference agent with a real store and a real embedder over breakable backends.

    Nothing about the plugin is faked. `ModelStore` is the shipped one and does its own
    digest check; `Embedder` is the shipped one and runs its own timeout and breaker. What is
    replaced is the GCS client and the genai client, which is the line the faults are injected
    on and the only line that touches a network in production.
    """
    reset_fixtures()
    fields: dict[str, Any] = {
        "mode": "active",
        "route_fraction": 1.0,
        "holdout_fraction": 0.0,
        "embedding_dims": DIMS,
        "gcs_bucket": "artifacts",
        "local_cache_dir": str(tmp_path / "cache"),
        # The reference agent's declarations are real and large, so a prune saves real tokens
        # and the decline rule has something to weigh. A miss is priced at zero so the rule
        # always says yes: what a menu is worth is measured elsewhere, and a row here that
        # declined on economics would look exactly like a row that declined on a fault.
        "miss_cost_tokens": 0,
    }
    fields.update(cfg_overrides)
    cfg = RouterConfig(**fields)
    contents = {
        f"{PREFIX}/metadata.json": json.dumps(
            meta if meta is not None else metadata(booster)
        ).encode(),
        f"{PREFIX}/model.ubj": booster,
    }
    if blobs is not None:
        contents = blobs
    bucket = Bucket(contents, breaks or {})
    client = GenaiClient(embed if embed is not None else good_vector())

    router = SmartRouterPlugin(
        cfg,
        model_store=ModelStore(cfg, storage_client=StorageClient(bucket, breaks or {})),
        embedder=Embedder(cfg, client=client),
        cache=ToolResultCache(cfg, clock=FakeClock(), jitter=lambda: 1.0),
        clock=FakeClock(),
        sampler=lambda: 1.0,
    )
    llm = ScriptedLlm(model="gemini-2.5-flash")
    llm.script = ["Your order is on its way."]
    llm.seen = []
    agent = LlmAgent(
        name=AGENT, model=llm, instruction=reference.INSTRUCTION, tools=list(ALL_TOOLS)
    )
    return router, llm, client, App(name=APP, root_agent=agent, plugins=[router])


async def run_one_turn(app: App, session_id: str = "s-1") -> list[Any]:
    service = InMemorySessionService()
    await service.create_session(app_name=APP, user_id=USER, session_id=session_id)
    runner = Runner(app=app, session_service=service)
    return [
        event
        async for event in runner.run_async(
            user_id=USER,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text="where is my order")]),
        )
    ]


def answered(events: list[Any]) -> bool:
    """Whether the user got words back. The only thing a fault must never take away."""
    return any(
        part.text
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.text
    )


def sources(events: list[Any]) -> list[str]:
    """Every `decision_source` the turn wrote, which is how a fault reaches the warehouse."""
    found = []
    for event in events:
        for record in (event.custom_metadata or {}).get(PROVENANCE_KEY, []):
            found.append(record.get("decision_source"))
    return found


def assert_served_unrouted(router: SmartRouterPlugin, llm: ScriptedLlm, events: list[Any]) -> None:
    """The assertion every row in the matrix shares."""
    assert answered(events), "the fault cost the user their answer"
    assert declared(llm.seen[0]) == sorted(TOOL_NAMES), "tools went missing on a broken path"
    assert dict(router.failopen_counts) == {}, (
        "a callback fell into fail_open, so this fault is survived by accident "
        f"rather than handled: {dict(router.failopen_counts)}"
    )


# --- The control --------------------------------------------------------------


async def test_the_control_row_actually_routes(tmp_path: Path, booster_bytes: bytes) -> None:
    """Nothing is broken, so the router downloads, verifies, embeds, predicts and prunes.

    Every other test in this file asserts that the full tool list went out. That is the
    correct outcome under a fault and the trivial outcome under a bug, and only this row can
    tell them apart. It is also the row that proves the fakes are wired to the real code: a
    real `ModelStore` verifies a real sha256 over real booster bytes, and a real
    `xgboost.Booster` scores a real vector.
    """
    router, llm, client, app = build(tmp_path, booster_bytes)

    events = await run_one_turn(app)

    assert answered(events)
    assert client.endpoint.calls == 1, "the embedder was never called"
    assert len(declared(llm.seen[0])) < len(TOOL_NAMES), "nothing was pruned"
    assert sources(events) == ["fresh"]
    assert dict(router.failopen_counts) == {}


# --- GCS: the bucket is unreachable, slow, or says no -------------------------

BUCKET_FAULTS = {
    "bucket unreachable": {"bucket": gcs_errors.ServiceUnavailable("backend unavailable")},
    "listing forbidden": {"list:": gcs_errors.Forbidden("caller lacks storage.objects.list")},
    "listing times out": {"list:": TimeoutError("deadline exceeded")},
    "metadata forbidden": {"metadata.json": gcs_errors.Forbidden("no read on metadata.json")},
    "metadata download fails": {"metadata.json": ConnectionResetError("connection reset")},
    "booster forbidden": {"model.ubj": gcs_errors.Forbidden("no read on model.ubj")},
    "booster download fails": {"model.ubj": ConnectionResetError("connection reset")},
    "transfer drops mid-download": {"bytes:": ConnectionResetError("connection reset")},
    "quota exhausted": {"list:": gcs_errors.TooManyRequests("rate limit exceeded")},
    "server error": {"list:": gcs_errors.InternalServerError("internal")},
}


@pytest.mark.parametrize("name", sorted(BUCKET_FAULTS))
async def test_a_broken_bucket_serves_the_full_tool_list(
    tmp_path: Path, booster_bytes: bytes, name: str
) -> None:
    """Ten ways GCS can fail, and the same outcome from all ten.

    Three of them break the listing, which is discovery; the rest break one of the reads.
    They are kept apart because they fail at different points of `_load` and an operator
    reading the log has to be able to tell which permission is missing. The last one breaks
    only the transfer of the booster's bytes, after the request that read its size has already
    succeeded, so the download is reached rather than refused ahead of it.
    """
    router, llm, _, app = build(tmp_path, booster_bytes, breaks=BUCKET_FAULTS[name])

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)


async def test_a_slow_bucket_inside_the_deadline_is_waited_for(
    tmp_path: Path, booster_bytes: bytes
) -> None:
    """This used to be the one cell that only passed eventually, with no limit on eventually.

    `_resolve` loads the artifact inside `asyncio.to_thread`, so a slow bucket never blocks the
    event loop and other sessions keep moving. It does block the session that triggered the
    load, and until §19.73 the only bound on that was the storage library's own retry policy,
    which is minutes. Half a second here, well inside the deadline: the turn waits, and a slow
    bucket is still a good artifact.
    """
    router, llm, _, app = build(
        tmp_path, booster_bytes, breaks={"list:": 0.5}, model_load_timeout_s=30.0
    )

    started = time.monotonic()
    events = await run_one_turn(app)
    waited = time.monotonic() - started

    assert answered(events)
    assert waited >= 0.5, "the load did not wait for the bucket, so this proved nothing"
    assert len(declared(llm.seen[0])) < len(TOOL_NAMES), "a slow bucket is still a good artifact"


async def test_a_bucket_slower_than_the_deadline_is_abandoned_rather_than_waited_out(
    tmp_path: Path, booster_bytes: bytes
) -> None:
    """The other half of the same cell, and the reason the qualifier is gone.

    The turn passes through instead of holding the user's request open on a socket that may
    never answer. Nothing is lost that was not already lost: a turn waiting on an artifact is
    a turn with no prediction either way (DESIGN.md 19.73).
    """
    router, llm, _, app = build(
        tmp_path, booster_bytes, breaks={"list:": 0.5}, model_load_timeout_s=0.05
    )

    started = time.monotonic()
    events = await run_one_turn(app)
    waited = time.monotonic() - started

    assert answered(events)
    assert waited < 0.5, "the deadline did not fire, so the turn waited the bucket out"
    assert len(declared(llm.seen[0])) == len(TOOL_NAMES), "routed on an artifact that never came"


# --- The artifact itself is wrong ---------------------------------------------


async def test_a_truncated_booster_fails_the_digest(tmp_path: Path, booster_bytes: bytes) -> None:
    """A partial download that the transport reported as a success.

    The metadata is honest and the file is short, which is the shape a truncated transfer
    takes. Nothing upstream notices; the sha256 does, and this is the check the whole
    supply-chain argument in `model_store.py` rests on.
    """
    blobs = {
        f"{PREFIX}/metadata.json": json.dumps(metadata(booster_bytes)).encode(),
        f"{PREFIX}/model.ubj": booster_bytes[: len(booster_bytes) // 2],
    }
    router, llm, _, app = build(tmp_path, booster_bytes, blobs=blobs)

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)
    assert not list(Path(tmp_path / "cache").rglob("*.ubj")), "a rejected booster was kept"


async def test_a_booster_xgboost_cannot_read(tmp_path: Path, booster_bytes: bytes) -> None:
    """Bytes that pass every check `model_store` makes and then fail to load.

    The digest matches because the metadata was published over these bytes. This is the one
    corruption the supply chain cannot catch, and it lands on `Predictor.__init__`, which is
    documented as raising rather than degrading (DESIGN.md 11.2). `_build` catches it and
    negative-caches, because a booster that will not load will not load next turn either.
    """
    rubbish = b"not a booster, but a self-consistent one" * 64
    blobs = {
        f"{PREFIX}/metadata.json": json.dumps(metadata(rubbish)).encode(),
        f"{PREFIX}/model.ubj": rubbish,
    }
    router, llm, _, app = build(tmp_path, booster_bytes, blobs=blobs)

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)


METADATA_FAULTS = {
    "not JSON at all": b"<html>403 Forbidden</html>",
    "JSON but not an object": b"[]",
    "empty file": b"",
}


@pytest.mark.parametrize("name", sorted(METADATA_FAULTS))
async def test_unreadable_metadata_serves_the_full_tool_list(
    tmp_path: Path, booster_bytes: bytes, name: str
) -> None:
    """An HTML error page served with a 200 is the realistic one of these three."""
    blobs = {
        f"{PREFIX}/metadata.json": METADATA_FAULTS[name],
        f"{PREFIX}/model.ubj": booster_bytes,
    }
    router, llm, _, app = build(tmp_path, booster_bytes, blobs=blobs)

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)


FIELD_FAULTS = {
    "no digest": {"sha256": None},
    "digest is not hex": {"sha256": "z" * 64},
    "no classes": {"classes": []},
    "a class is not a string": {"classes": ["get_invoice", 7]},
    "negative row count": {"rows": -1},
    "gate metrics are not an object": {"gate_metrics": "0.99"},
    "a menu of the wrong shape": {"menus": [["get_invoice"], "get_price"]},
    "more menus than the cap": {"menus": [[f"t{n}"] for n in range(64)]},
}


@pytest.mark.parametrize("name", sorted(FIELD_FAULTS))
async def test_malformed_metadata_fields_serve_the_full_tool_list(
    tmp_path: Path, booster_bytes: bytes, name: str
) -> None:
    """Eight ways `metadata.json` can be wrong, all of them refusing the artifact.

    These are the fields the serving path reads without checking again: `classes` becomes the
    label vector, `menus` becomes the tool declarations sent to the model. A permissive parse
    here is the difference between an artifact that will not load and an artifact that routes
    on half of itself.
    """
    router, llm, _, app = build(
        tmp_path, booster_bytes, meta=metadata(booster_bytes, **FIELD_FAULTS[name])
    )

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)


MISMATCH_FAULTS = {
    "staging artifact in a prod process": {"environment": "staging"},
    "a different embedding model": {"embedding_model": "text-embedding-004"},
    "a different embedding width": {"embedding_dims": DIMS * 2},
    "not trained on Vertex": {"backend": "gemini"},
    "over the size cap": {"size_bytes": 10**9},
}


@pytest.mark.parametrize("name", sorted(MISMATCH_FAULTS))
async def test_an_artifact_that_is_not_ours_serves_the_full_tool_list(
    tmp_path: Path, booster_bytes: bytes, name: str
) -> None:
    """A well-formed artifact that belongs to a different process than this one.

    The first is the dangerous one. A path collision between environments means staging's
    booster decides production's tool menus, and nothing about the file is corrupt, so no
    digest and no parser would ever object. `_check_compatible` is the only thing that does.
    """
    router, llm, _, app = build(
        tmp_path, booster_bytes, meta=metadata(booster_bytes, **MISMATCH_FAULTS[name])
    )

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)


async def test_an_artifact_past_its_age_cap_serves_the_full_tool_list(
    tmp_path: Path, booster_bytes: bytes
) -> None:
    """Staleness is checked per use rather than per load, so a long-lived process demotes
    itself the day its artifact ages out rather than at the next redeploy."""
    router, llm, _, app = build(
        tmp_path, booster_bytes, meta=metadata(booster_bytes, trained_at="2020-01-01T00:00:00Z")
    )

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)


NOTHING_PUBLISHED = {
    "an empty bucket": {},
    "metadata without a booster": {"m": b"{}"},
    "a booster without metadata": {f"{PREFIX}/model.ubj": b"x"},
    "another agent's artifact": {"routers/planner/v1/model.ubj": b"x"},
}


@pytest.mark.parametrize("name", sorted(NOTHING_PUBLISHED))
async def test_no_complete_version_serves_the_full_tool_list(
    tmp_path: Path, booster_bytes: bytes, name: str
) -> None:
    """A version directory counts only when both files are there, so a half-written publish
    is invisible rather than half-loaded (D11). An agent with nothing published is the
    permanent steady state of every agent before its first retrain."""
    blobs = dict(NOTHING_PUBLISHED[name])
    if "m" in blobs:
        blobs = {f"{PREFIX}/metadata.json": blobs.pop("m")}
    router, llm, _, app = build(tmp_path, booster_bytes, blobs=blobs)

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)


async def test_a_pinned_version_that_does_not_exist(tmp_path: Path, booster_bytes: bytes) -> None:
    """`SMART_ROUTER_PIN_VERSION` skips discovery, so a typo in it is a fault of its own and
    not one the listing can catch."""
    router, llm, _, app = build(tmp_path, booster_bytes, pin_version="20250101T000000Z")

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)


async def test_an_unconfigured_bucket_says_so_rather_than_guessing(
    tmp_path: Path, booster_bytes: bytes
) -> None:
    """The one fault that is a deployment mistake rather than an outage. Guessing a bucket
    name would be worse than not routing, so it refuses and names the variable."""
    router, llm, _, app = build(tmp_path, booster_bytes, gcs_bucket="")

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)


# --- The embedding endpoint ---------------------------------------------------

EMBED_FAULTS: dict[str, tuple[Any, str]] = {
    "times out": (2.0, "timeout"),
    "rate limited": (gcs_errors.TooManyRequests("429 quota exceeded"), "error"),
    "server error": (gcs_errors.InternalServerError("500"), "error"),
    "permission denied": (gcs_errors.Forbidden("caller cannot use aiplatform"), "error"),
    "connection dropped": (ConnectionResetError("connection reset"), "error"),
    "empty response": (SimpleNamespace(embeddings=[]), "error"),
    "no values in the response": (
        SimpleNamespace(embeddings=[SimpleNamespace(values=[])]),
        "error",
    ),
    "a vector of the wrong width": (
        SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * (DIMS - 1))]),
        "error",
    ),
    "values that are not numbers": (
        SimpleNamespace(embeddings=[SimpleNamespace(values=["nope"] * DIMS)]),
        "error",
    ),
}


@pytest.mark.parametrize("name", sorted(EMBED_FAULTS))
async def test_a_broken_embedder_serves_the_full_tool_list(
    tmp_path: Path, booster_bytes: bytes, name: str
) -> None:
    """Nine ways the one remote call on the critical path can fail.

    The artifact is good in every one of these, so the router has a booster and a live tool
    list and everything it needs except the vector. That is the interesting shape: the session
    is pinned to passthrough on its first turn and stays there, rather than routing on a
    vector that is not a vector.

    Each row also asserts the reason reaches provenance, because the warehouse is where an
    operator finds out. A timeout wants a bigger budget or a faster endpoint and an error
    wants the endpoint's own message read, and two live incidents that arrived looking
    identical needed opposite responses (DESIGN.md 19.23).
    """
    effect, expected = EMBED_FAULTS[name]
    router, llm, _, app = build(tmp_path, booster_bytes, embed=effect)

    events = await run_one_turn(app)

    assert_served_unrouted(router, llm, events)
    assert sources(events) == [expected]


async def test_a_latched_breaker_stops_calling_and_says_so(
    tmp_path: Path, booster_bytes: bytes
) -> None:
    """The endpoint is down, and after `embed_breaker_threshold` failures the router stops
    asking. Every session after that is a passthrough that cost nothing, which is the whole
    point of the breaker: a dead endpoint should not add its own latency to every turn.

    The `breaker` source is what tells an operator the router gave up on this process,
    as against the endpoint failing one call at a time. The fix is upstream of both.

    The call that latches the breaker still reports `error`, and that is right rather than
    off by one: it reached the endpoint and the endpoint failed it. Only the calls the router
    then refuses to make are `breaker`, so the two counts in the warehouse add up to what
    actually happened on the wire.
    """
    router, llm, client, app = build(
        tmp_path,
        booster_bytes,
        embed=gcs_errors.InternalServerError("500"),
        embed_breaker_threshold=2,
    )

    seen = []
    for n in range(4):
        events = await run_one_turn(app, session_id=f"s-{n}")
        seen += sources(events)

    assert seen == ["error", "error", "breaker", "breaker"]
    assert client.endpoint.calls == 2, "the breaker let calls through after latching"
    assert declared(llm.seen[-1]) == sorted(TOOL_NAMES)
    assert dict(router.failopen_counts) == {}


async def test_the_first_turn_of_a_cold_process_is_the_one_that_times_out(
    tmp_path: Path, booster_bytes: bytes
) -> None:
    """Why `warmup` exists, as a fault rather than as a note.

    The first call a process makes to Vertex pays an ADC token fetch, a TLS handshake and
    HTTP/2 setup, measured at about 1.9 seconds against a 600 ms budget (DESIGN.md 18.10).
    One embedding failure pins a session to passthrough for good, so that single cold call
    costs a whole session rather than a turn. Here the first call is slow and the rest are
    fast, and the first session is the one that loses.
    """
    router, llm, client, app = build(tmp_path, booster_bytes, embed=good_vector())
    slow = client.endpoint.embed_content

    async def cold_start(**kwargs: Any) -> Any:  # noqa: ANN401
        if client.endpoint.calls == 0:
            client.endpoint.calls += 1
            await asyncio.sleep(2.0)
        return await slow(**kwargs)

    client.endpoint.embed_content = cold_start  # type: ignore[method-assign]

    first = await run_one_turn(app, session_id="cold")
    second = await run_one_turn(app, session_id="warm")

    assert sources(first) == ["timeout"]
    assert declared(llm.seen[0]) == sorted(TOOL_NAMES)
    assert sources(second) == ["fresh"]
    assert len(declared(llm.seen[-1])) < len(TOOL_NAMES)
    assert dict(router.failopen_counts) == {}


# --- The warehouse ------------------------------------------------------------


def test_the_serving_path_never_imports_bigquery() -> None:
    """The empty column in the matrix, asserted rather than argued.

    "BigQuery down" is a cell in ROADMAP H2 and it has no test, because there is nothing to
    break: the plugin writes provenance onto an event and BQAA logs it. The router never
    opens a BigQuery client, so a warehouse outage costs telemetry and nothing else. The one
    module in this package that does query BigQuery is `openings`, which runs offline in
    `scripts/`, behind a lazy import.

    Asserted on the import graph because a future edit that adds a client would be a real
    change of the failure model and would otherwise be invisible until the day BigQuery
    is down.

    In a subprocess, because the honest version of this test is "import the plugin into a
    process that has imported nothing else", and the alternative is deleting the package out
    of `sys.modules` and importing it again. That leaves a second copy of every module behind
    and any test that later monkeypatches a module global patches the wrong one. Cheaper to
    pay for a Python start than to leave that trap in a file about faults.
    """
    probe = (
        "import sys; import adk_smart_router.plugin; "
        "print([m for m in ('google.cloud.bigquery', 'google.cloud.storage') if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]", f"the serving path imports {result.stdout.strip()}"


async def test_a_provenance_writer_that_raises_is_the_one_thing_fail_open_is_for(
    tmp_path: Path, booster_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other row in this file asserts `fail_open` did not fire. This one asserts it does.

    The decorator is the last resort, and a last resort nobody has ever seen fire is a
    hypothesis. So: break something the code has no named handling for, in the one callback
    where a raise would otherwise abort the whole ADK run, and check the turn still lands.
    The counter is how an operator finds out, since by construction there is nothing else to
    tell them.
    """
    from adk_smart_router import plugin as plugin_mod

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("the warehouse hook exploded")

    router, _, _, app = build(tmp_path, booster_bytes)
    monkeypatch.setattr(plugin_mod, "carry_onto", explode)

    events = await run_one_turn(app)

    assert answered(events), "an exploding telemetry hook cost the user their answer"
    assert dict(router.failopen_counts) == {"on_event_callback": 1}
