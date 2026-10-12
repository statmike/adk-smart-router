"""Supply-chain tests for model_store: what it refuses matters more than what it loads."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from adk_smart_router.config import RouterConfig
from adk_smart_router.model_store import ModelStore, RouterModel, parse_metadata

# --- Fake GCS -----------------------------------------------------------------


class FakeBlob:
    def __init__(self, name: str, store: FakeBucket) -> None:
        self._name = name
        self._store = store

    @property
    def size(self) -> int | None:
        """What GCS reports, which is the length of the object and not a claim about it."""
        blob = self._store.blobs.get(self._name)
        return None if blob is None else len(blob)

    def _bytes(self) -> bytes:
        if self._name not in self._store.blobs:
            raise FileNotFoundError(self._name)
        self._store.downloads.append(self._name)
        return self._store.blobs[self._name]

    def download_as_bytes(self) -> bytes:
        return self._bytes()

    def download_to_filename(self, path: str) -> None:
        Path(path).write_bytes(self._bytes())


class FakeBucket:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = blobs
        self.downloads: list[str] = []
        self.listings: list[str] = []

    def list_blobs(self, prefix: str = "") -> list[Any]:
        self.listings.append(prefix)
        return [SimpleNamespace(name=n) for n in sorted(self.blobs) if n.startswith(prefix)]

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(name, self)

    def get_blob(self, name: str) -> FakeBlob | None:
        """Missing means None, the way the real client behaves, not a raise."""
        return FakeBlob(name, self) if name in self.blobs else None


class FakeStorageClient:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.the_bucket = FakeBucket(blobs)

    def bucket(self, name: str) -> FakeBucket:
        return self.the_bucket


# --- Fixtures -----------------------------------------------------------------

BOOSTER = b"pretend this is an xgboost booster"
DIGEST = hashlib.sha256(BOOSTER).hexdigest()
V1 = "20260101T000000Z"
V2 = "20260201T000000Z"


def metadata_dict(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "version": V1,
        "sha256": DIGEST,
        "size_bytes": len(BOOSTER),
        "trained_at": "2026-01-01T00:00:00Z",
        "rows": 5000,
        "classes": ["search", "book_flight", "ABSTAIN"],
        "backend": "vertex",
        "environment": "prod",
        "embedding_model": "gemini-embedding-2",
        "embedding_dims": 768,
        "gate_metrics": {"coverage": 0.99, "baseline_delta": 0.12},
    }
    raw.update(overrides)
    return raw


def version_blobs(
    version: str, *, meta: dict[str, Any] | None = None, booster: bytes | None = BOOSTER
) -> dict[str, bytes]:
    prefix = f"routers/planner/{version}"
    blobs: dict[str, bytes] = {}
    if meta is not None:
        blobs[f"{prefix}/metadata.json"] = json.dumps(meta).encode()
    if booster is not None:
        blobs[f"{prefix}/model.ubj"] = booster
    return blobs


def make_store(tmp_path: Path, blobs: dict[str, bytes], **cfg_overrides: Any) -> ModelStore:
    cfg = RouterConfig(
        gcs_bucket="artifacts",
        local_cache_dir=str(tmp_path / "cache"),
        **cfg_overrides,
    )
    return ModelStore(cfg, storage_client=FakeStorageClient(blobs))


# --- Happy path ---------------------------------------------------------------


def test_load_verifies_and_caches_the_booster(tmp_path):
    store = make_store(tmp_path, version_blobs(V1, meta=metadata_dict()))

    model = store.load("planner")

    assert isinstance(model, RouterModel)
    assert model.agent_name == "planner"
    assert model.classes == ("search", "book_flight", "ABSTAIN")
    assert Path(model.booster_path).read_bytes() == BOOSTER
    assert model.booster_path.endswith(f"planner/{V1}/model.ubj")
    assert not Path(model.booster_path + ".tmp").exists()
    assert model.age_days > 0


def test_highest_complete_version_wins_and_incomplete_is_skipped(tmp_path):
    blobs = version_blobs(V1, meta=metadata_dict())
    # V2 is mid-publish: metadata written, booster not yet. Completeness, not ordering (D11).
    blobs.update(version_blobs(V2, meta=metadata_dict(version=V2), booster=None))
    store = make_store(tmp_path, blobs)

    model = store.load("planner")

    assert model is not None
    assert f"/{V1}/" in model.booster_path


def test_pin_version_bypasses_discovery(tmp_path):
    blobs = version_blobs(V1, meta=metadata_dict())
    blobs.update(version_blobs(V2, meta=metadata_dict(version=V2)))
    store = make_store(tmp_path, blobs, pin_version=V1)
    client: FakeStorageClient = store._client  # noqa: SLF001 - inspecting the injected fake

    model = store.load("planner")

    assert model is not None
    assert f"/{V1}/" in model.booster_path
    assert client.the_bucket.listings == []


def test_second_load_reuses_the_verified_local_file(tmp_path):
    store = make_store(tmp_path, version_blobs(V1, meta=metadata_dict()))
    client: FakeStorageClient = store._client  # noqa: SLF001

    first = store.load("planner")
    second = store.load("planner")

    assert first is not None and second is not None
    assert first.booster_path == second.booster_path
    assert client.the_bucket.downloads.count(f"routers/planner/{V1}/model.ubj") == 1


# --- The local cache is a cache, so something has to evict it -------------------


def age(directory: Path, *, hours: float) -> None:
    """Backdate a cached version, so the sweep reads it as superseded and not as in flight."""
    when = time.time() - hours * 3600
    for path in (directory / "model.ubj", directory):
        if path.exists():
            os.utime(path, (when, when))


def test_a_superseded_version_directory_is_collected(tmp_path):
    """`local_cache_dir` is memory on Cloud Run, and nothing else ever deletes a version."""
    blobs = version_blobs(V1, meta=metadata_dict())
    store = make_store(tmp_path, blobs)
    assert store.load("planner") is not None
    old = tmp_path / "cache" / "planner" / V1
    age(old, hours=2)

    blobs.update(version_blobs(V2, meta=metadata_dict(version=V2)))
    assert store.load("planner") is not None

    assert not old.exists()
    assert (tmp_path / "cache" / "planner" / V2 / "model.ubj").read_bytes() == BOOSTER


def test_a_version_installed_moments_ago_survives_a_newer_one(tmp_path):
    """The grace period. Another process can be opening that file right now."""
    blobs = version_blobs(V1, meta=metadata_dict())
    store = make_store(tmp_path, blobs)
    assert store.load("planner") is not None

    blobs.update(version_blobs(V2, meta=metadata_dict(version=V2)))
    assert store.load("planner") is not None

    assert (tmp_path / "cache" / "planner" / V1 / "model.ubj").exists()


def test_the_version_being_loaded_is_never_the_one_swept(tmp_path):
    """Old is not the same as unwanted: a pinned deployment sits on one version for months."""
    store = make_store(tmp_path, version_blobs(V1, meta=metadata_dict()))
    assert store.load("planner") is not None
    directory = tmp_path / "cache" / "planner" / V1
    age(directory, hours=99)

    model = store.load("planner")

    assert model is not None
    assert Path(model.booster_path).read_bytes() == BOOSTER


def test_an_interrupted_download_is_collected_too(tmp_path):
    """A directory with no artifact still holds the bytes of one: a `.tmp` a SIGKILL left."""
    blobs = version_blobs(V2, meta=metadata_dict(version=V2))
    store = make_store(tmp_path, blobs)
    stump = tmp_path / "cache" / "planner" / V1
    stump.mkdir(parents=True)
    (stump / "model.ubj.31337.1.tmp").write_bytes(b"half a booster")
    age(stump, hours=2)

    assert store.load("planner") is not None

    assert not stump.exists()


# --- Refusals -----------------------------------------------------------------


def test_digest_mismatch_refuses_and_leaves_no_file(tmp_path, caplog):
    blobs = version_blobs(V1, meta=metadata_dict(), booster=b"tampered payload")
    blobs[f"routers/planner/{V1}/metadata.json"] = json.dumps(
        metadata_dict(size_bytes=len(b"tampered payload"))
    ).encode()
    store = make_store(tmp_path, blobs)

    with caplog.at_level("WARNING"):
        assert store.load("planner") is None

    assert "sha256 mismatch" in caplog.text
    assert list((tmp_path / "cache" / "planner" / V1).iterdir()) == []


def test_oversized_artifact_refuses_before_downloading(tmp_path):
    store = make_store(
        tmp_path,
        version_blobs(V1, meta=metadata_dict(size_bytes=60_000_000)),
        max_artifact_bytes=50_000_000,
    )
    client: FakeStorageClient = store._client  # noqa: SLF001

    assert store.load("planner") is None
    assert f"routers/planner/{V1}/model.ubj" not in client.the_bucket.downloads


def test_an_object_larger_than_it_claims_refuses_before_downloading(tmp_path, caplog):
    """The cap has to be applied to the size GCS reports, not the size the artifact claims.

    Anyone who can write the bucket writes both files, so declaring a small `size_bytes`
    beside a huge booster would walk straight past `_check_compatible`. The digest would
    catch it, but only after `download_to_filename` had written the whole thing into
    `local_cache_dir`, which on Cloud Run is memory. So the object's own size is read first.
    """
    booster = b"x" * 4096
    blobs = version_blobs(V1, meta=metadata_dict(size_bytes=10), booster=booster)
    store = make_store(tmp_path, blobs, max_artifact_bytes=1024)
    client: FakeStorageClient = store._client  # noqa: SLF001

    with caplog.at_level("WARNING"):
        assert store.load("planner") is None

    assert "over the 1024 byte cap" in caplog.text
    assert f"routers/planner/{V1}/model.ubj" not in client.the_bucket.downloads


def test_an_object_that_disagrees_with_its_declared_size_refuses(tmp_path, caplog):
    """Under the cap, but not the file the metadata describes. That is a mispublish."""
    booster = b"x" * 4096
    blobs = version_blobs(V1, meta=metadata_dict(size_bytes=4095), booster=booster)
    store = make_store(tmp_path, blobs)
    client: FakeStorageClient = store._client  # noqa: SLF001

    with caplog.at_level("WARNING"):
        assert store.load("planner") is None

    assert "metadata declares 4095" in caplog.text
    assert f"routers/planner/{V1}/model.ubj" not in client.the_bucket.downloads


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"environment": "staging"}, "environment"),
        ({"embedding_model": "gemini-embedding-001"}, "embedding model"),
        ({"embedding_dims": 1536}, "embedding dims"),
        ({"backend": "onnx"}, "backend"),
    ],
)
def test_incompatible_artifacts_are_refused_before_download(tmp_path, caplog, overrides, reason):
    store = make_store(tmp_path, version_blobs(V1, meta=metadata_dict(**overrides)))
    client: FakeStorageClient = store._client  # noqa: SLF001

    with caplog.at_level("WARNING"):
        assert store.load("planner") is None

    assert reason in caplog.text
    assert f"routers/planner/{V1}/model.ubj" not in client.the_bucket.downloads


def test_no_complete_version_returns_none(tmp_path):
    store = make_store(tmp_path, version_blobs(V1, meta=None))
    assert store.load("planner") is None


def test_missing_agent_prefix_returns_none(tmp_path):
    store = make_store(tmp_path, version_blobs(V1, meta=metadata_dict()))
    assert store.load("researcher") is None


def test_pinned_version_that_does_not_exist_returns_none(tmp_path):
    store = make_store(tmp_path, version_blobs(V1, meta=metadata_dict()), pin_version=V2)
    assert store.load("planner") is None


def test_path_traversal_in_pin_version_is_refused(tmp_path):
    store = make_store(tmp_path, version_blobs(V1, meta=metadata_dict()), pin_version="../../etc")
    assert store.load("planner") is None


def test_corrupt_metadata_json_returns_none(tmp_path):
    blobs = version_blobs(V1, meta=metadata_dict())
    blobs[f"routers/planner/{V1}/metadata.json"] = b"{not json"
    store = make_store(tmp_path, blobs)
    assert store.load("planner") is None


# --- Has anything ever been published? ----------------------------------------
#
# `load` answers None to an empty prefix and to a broken artifact alike, which is right at
# serve time and wrong for a retrain cycle: the first one is a new agent and the second is a
# problem. These five tests are the ones that keep the two apart.


def test_an_agent_with_nothing_published_has_no_versions(tmp_path):
    store = make_store(tmp_path, version_blobs(V1, meta=metadata_dict()))
    assert store.published_versions("researcher") == ()


def test_a_trained_but_unpublished_agent_has_published_nothing(tmp_path):
    """The case this exists for, and the one that is easy to get wrong. Training exports its
    booster into the version directory publishing later fills in, so a brand-new agent that
    has just trained already has a directory. It has not published."""
    store = make_store(
        tmp_path,
        {
            f"routers/planner/{V1}/model.bst": BOOSTER,
            f"routers/planner/{V1}/main.py": b"",
            f"routers/planner/{V1}/assets/model_metadata.json": b"{}",
        },
    )

    assert store.load("planner") is None
    assert store.published_versions("planner") == ()


def test_a_version_read_back_wrong_is_published_all_the_same(tmp_path):
    """The other half. Both files are there, so something was shipped, and a digest that
    does not match is a problem rather than a fresh agent."""
    blobs = version_blobs(V1, meta=metadata_dict(sha256="0" * 64))
    store = make_store(tmp_path, blobs)

    assert store.load("planner") is None
    assert store.published_versions("planner") == (V1,)


def test_every_published_version_is_listed_oldest_first(tmp_path):
    blobs = version_blobs(V2, meta=metadata_dict(version=V2))
    blobs.update(version_blobs(V1, meta=metadata_dict()))
    store = make_store(tmp_path, blobs)

    assert store.published_versions("planner") == (V1, V2)


def test_an_unreadable_bucket_is_not_an_empty_one(tmp_path):
    """`load` swallows this because passthrough is the safe answer at serve time. Here it
    would read as "new agent, publish on the defaults", which is a different claim."""
    store = ModelStore(RouterConfig(gcs_bucket="", local_cache_dir=str(tmp_path)))

    with pytest.raises(Exception, match="gcs_bucket is not configured"):
        store.published_versions("planner")


# --- parse_metadata -----------------------------------------------------------


def test_parse_metadata_round_trips():
    meta = parse_metadata(metadata_dict())
    assert meta.classes == ("search", "book_flight", "ABSTAIN")
    assert meta.gate_metrics == {"coverage": 0.99, "baseline_delta": 0.12}
    assert meta.sha256 == DIGEST


def test_parse_metadata_accepts_uppercase_digest():
    assert parse_metadata(metadata_dict(sha256=DIGEST.upper())).sha256 == DIGEST


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": ""},
        {"sha256": "deadbeef"},
        {"size_bytes": -1},
        {"size_bytes": "1024"},
        {"rows": True},
        {"classes": []},
        {"classes": ["ok", 7]},
        {"embedding_dims": None},
        {"gate_metrics": [0.99]},
        {"gate_metrics": {"coverage": "high"}},
        {"menus": "search,book_flight"},
        {"menus": ["search"]},
        {"menus": [[]]},
        {"menus": [["search", 7]]},
        {"menus": [["search", ""]]},
        {"menus": [["tool"] * 65]},
        {"menus": [["tool"]] * 33},
    ],
)
def test_parse_metadata_rejects_malformed_fields(overrides):
    with pytest.raises(ValueError):
        parse_metadata(metadata_dict(**overrides))


def test_parse_metadata_rejects_a_missing_field():
    raw = metadata_dict()
    del raw["trained_at"]
    with pytest.raises(ValueError, match="trained_at"):
        parse_metadata(raw)


# --- Menus --------------------------------------------------------------------
#
# `menus` is the one optional field, because every artifact published before menus existed
# is still valid and still routes. Everything else about it is as strict as `classes`: the
# names go straight into the tool declarations sent to the model.


def test_an_artifact_without_menus_is_still_a_good_artifact():
    raw = metadata_dict()
    assert "menus" not in raw
    assert parse_metadata(raw).menus == ()


def test_menus_are_parsed_into_frozensets_in_published_order():
    meta = parse_metadata(metadata_dict(menus=[["search", "book_flight"], ["search"]]))
    assert meta.menus == (frozenset({"search", "book_flight"}), frozenset({"search"}))


def test_a_menu_published_twice_is_kept_once():
    """Harmless rather than hostile: two identical menus describe one prefix."""
    meta = parse_metadata(metadata_dict(menus=[["a", "b"], ["b", "a"]]))
    assert meta.menus == (frozenset({"a", "b"}),)


def test_the_model_hands_its_menus_to_the_caller():
    model = RouterModel(
        agent_name="planner",
        metadata=parse_metadata(metadata_dict(menus=[["search"]])),
        booster_path="/dev/null",
    )
    assert model.menus == (frozenset({"search"}),)


# --- RouterModel properties ---------------------------------------------------


def test_age_days_reads_unparseable_timestamps_as_infinitely_old():
    model = RouterModel(
        agent_name="planner",
        metadata=parse_metadata(metadata_dict(trained_at="last tuesday")),
        booster_path="/dev/null",
    )
    assert model.age_days == float("inf")


# --- the drift baseline rides along, and is optional --------------------------


def test_an_artifact_published_before_drift_existed_has_no_baseline():
    """The common first run. Absent is a state to print, not a field to default to zero."""
    assert parse_metadata(metadata_dict()).drift_baseline is None


def test_a_baseline_is_read_back_as_the_signals_that_wrote_it():
    baseline = {"unseen_share": 0.261, "declared": 100, "unclassed": 26, "confidence_p50": 0.49}
    got = parse_metadata(metadata_dict(drift_baseline=baseline)).drift_baseline
    assert got is not None
    assert got.unseen_share == 0.261 and got.confidence_p50 == 0.49


def test_a_hand_edited_baseline_does_not_stop_the_artifact_loading():
    """Serving never reads this field, so it must not be able to refuse a good booster."""
    meta = parse_metadata(metadata_dict(drift_baseline="nonsense"))
    assert meta.drift_baseline is None
    assert meta.version == metadata_dict()["version"]
