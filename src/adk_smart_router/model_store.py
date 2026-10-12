"""Fetch a router's XGBoost booster from GCS and prove it is the artifact we asked for.

This is a supply-chain boundary, not a download helper: whatever this module returns gets loaded
into every serving process. The order of the checks is the whole point. Size is checked before any
bytes are pulled and against the size GCS reports rather than the one the metadata claims, the
sha256 digest before the file is handed to xgboost, and `environment` before
anything is trusted at all, because a cross-environment path collision means staging serves
production (DESIGN.md 11.6).

There is no `latest.json` pointer. A version directory counts only once both `metadata.json` and
`model.ubj` are present, so atomicity comes from completeness, not from write ordering (D11). The
same rule runs backwards: writing a `withdrawn.json` beside them makes the directory stop counting,
which is how a generation is taken out of service without deleting anything (DESIGN.md 21.5).

`model.ubj`, not the `model.bst` that `EXPORT MODEL` writes. That file is XGBoost's pre-1.0 binary
format, which XGBoost 2.1 removed the ability to read; `scripts/publish_router.py` converts it and
publishes the result (DESIGN.md 18.9). A raw export directory is therefore not loadable here, which
is the correct outcome: it has not been through the label-order reconciliation either.

This is the only module allowed to import `google.cloud.storage`, and it imports it lazily so that
importing the package does not require the GCS library.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import re
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from adk_smart_router.config import RouterConfig
from adk_smart_router.drift import Signals

logger = logging.getLogger(__name__)

# The slice of google-cloud-storage we touch is tiny: client.bucket(name), bucket.list_blobs(),
# bucket.blob(name).download_*(). Aliasing it keeps the GCS types out of this module's signatures
# so the tests can inject an in-memory fake, and so nothing here imports GCS at import time.
GcsClient = Any
GcsBucket = Any

_METADATA_NAME = "metadata.json"
_BOOSTER_NAME = "model.ubj"

# The withdrawal marker. A version directory holding one is not a published version, so `load`
# falls back to the one before it and every serving process follows on its next re-poll
# (DESIGN.md 21.5). Public because `scripts/watch_generations.py` writes it and there must be
# exactly one spelling of the file that can take a generation out of service.
WITHDRAWN_NAME = "withdrawn.json"

# Version and agent names become local filesystem paths, so a listing (or a pinned env var) must
# never be able to walk out of `local_cache_dir`.
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")

# Bounds on the published menu list. Menus exist to keep the number of distinct tool
# declaration lists small, so an artifact naming hundreds of them has either been
# mispublished or tampered with, and either way it would undo the thing it is for.
_MAX_MENUS = 32
_MAX_MENU_SIZE = 64

# Distinguishes two downloads inside one process the way the pid distinguishes two processes.
# `itertools.count` is atomic under the GIL, which is all this needs.
_ATTEMPTS = itertools.count()


# --- Data ---------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactMetadata:
    """The `metadata.json` published alongside a booster. Every field is verified."""

    version: str
    sha256: str
    size_bytes: int
    trained_at: str  # ISO-8601 UTC
    rows: int
    classes: tuple[str, ...]
    backend: str
    environment: str
    embedding_model: str
    embedding_dims: int
    gate_metrics: Mapping[str, float]
    # Optional, and empty on every artifact published before menus existed. An empty
    # tuple tells the pruner to build a bespoke set per session, which is what it always
    # did (`pruner.decide`).
    menus: tuple[frozenset[str], ...] = ()
    # Optional, and None on every artifact published before drift was measured. What this
    # model's fit to its tool surface looked like on the day it shipped, so a later check
    # compares it against itself rather than against zero (DESIGN.md 20.4). Serving never
    # reads it; it is here because metadata.json has one parser in this project and a
    # second reader of the same file is how two things start disagreeing about it.
    drift_baseline: Signals | None = None


@dataclass(frozen=True)
class RouterModel:
    """A verified booster on local disk, plus the metadata that vouched for it."""

    agent_name: str
    metadata: ArtifactMetadata
    booster_path: str

    @property
    def classes(self) -> tuple[str, ...]:
        return self.metadata.classes

    @property
    def menus(self) -> tuple[frozenset[str], ...]:
        return self.metadata.menus

    @property
    def age_days(self) -> float:
        """Days since training. Reported here, enforced by the plugin (`max_model_age_days`).

        An unparseable timestamp reads as infinitely old, so a malformed artifact demotes itself
        rather than passing a freshness check by accident.
        """
        try:
            trained = datetime.fromisoformat(self.metadata.trained_at)
        except ValueError:
            return math.inf
        if trained.tzinfo is None:
            trained = trained.replace(tzinfo=UTC)
        return (datetime.now(UTC) - trained).total_seconds() / 86400.0


class _Refused(Exception):
    """A deliberate refusal. The message is the one line that reaches the WARNING log."""


# --- Metadata parsing ---------------------------------------------------------


def parse_metadata(raw: Mapping[str, Any]) -> ArtifactMetadata:
    """Validate a decoded `metadata.json`. Raises ValueError naming the offending field."""
    return ArtifactMetadata(
        version=_as_text(raw, "version"),
        sha256=_as_digest(raw, "sha256"),
        size_bytes=_as_count(raw, "size_bytes"),
        trained_at=_as_text(raw, "trained_at"),
        rows=_as_count(raw, "rows"),
        classes=_as_classes(raw, "classes"),
        backend=_as_text(raw, "backend"),
        environment=_as_text(raw, "environment"),
        embedding_model=_as_text(raw, "embedding_model"),
        embedding_dims=_as_count(raw, "embedding_dims"),
        gate_metrics=_as_metrics(raw, "gate_metrics"),
        menus=_as_menus(raw, "menus"),
        # Forgiving on purpose: an artifact published before this existed has no baseline,
        # and a hand-edited one is a state `smart-router status` still has to print.
        drift_baseline=Signals.from_dict(raw.get("drift_baseline")),
    )


def _as_text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"metadata field {key!r} must be a non-empty string, got {value!r}")
    return value


def _as_count(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    # bool is an int subclass and is never a plausible count here.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"metadata field {key!r} must be a non-negative integer, got {value!r}")
    return value


def _as_digest(raw: Mapping[str, Any], key: str) -> str:
    value = _as_text(raw, key).lower()
    if not _SHA256_HEX.fullmatch(value):
        raise ValueError(f"metadata field {key!r} must be 64 hex characters, got {value!r}")
    return value


def _as_classes(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"metadata field {key!r} must be a non-empty list, got {value!r}")
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"metadata field {key!r} contains a non-string label: {item!r}")
    return tuple(value)


def _as_menus(raw: Mapping[str, Any], key: str) -> tuple[frozenset[str], ...]:
    """Validate the optional menu list. Absent means "no menus", anything else must be right.

    As strict as `_as_classes`, and for the same reason: these names go straight into the
    tool declarations sent to the model. A missing key is fine, a malformed one refuses
    the artifact rather than routing on half of it.

    Duplicate menus are dropped rather than rejected. Two identical menus are harmless to
    serving, they simply describe the same prefix twice.
    """
    value = raw.get(key)
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError(f"metadata field {key!r} must be a list, got {value!r}")
    if len(value) > _MAX_MENUS:
        raise ValueError(
            f"metadata field {key!r} has {len(value)} menus, over the {_MAX_MENUS} cap"
        )

    menus: list[frozenset[str]] = []
    for menu in value:
        if not isinstance(menu, list | tuple) or not menu:
            raise ValueError(f"metadata field {key!r} contains a non-list menu: {menu!r}")
        if len(menu) > _MAX_MENU_SIZE:
            raise ValueError(f"metadata field {key!r} has a menu of {len(menu)} tools")
        for tool in menu:
            if not isinstance(tool, str) or not tool:
                raise ValueError(f"metadata field {key!r} contains a non-string tool: {tool!r}")
        frozen = frozenset(menu)
        if frozen not in menus:
            menus.append(frozen)
    return tuple(menus)


def _as_metrics(raw: Mapping[str, Any], key: str) -> Mapping[str, float]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"metadata field {key!r} must be an object, got {value!r}")
    metrics: dict[str, float] = {}
    for name, number in value.items():
        if not isinstance(name, str) or isinstance(number, bool):
            raise ValueError(f"metadata field {key!r} has a malformed entry: {name!r}={number!r}")
        if not isinstance(number, int | float):
            raise ValueError(f"metadata field {key!r} entry {name!r} must be a number")
        metrics[name] = float(number)
    return metrics


# --- Store --------------------------------------------------------------------


class ModelStore:
    """Discovers, verifies and locally caches one booster per agent."""

    def __init__(self, cfg: RouterConfig, storage_client: GcsClient | None = None) -> None:
        self._cfg = cfg
        self._client = storage_client  # injected for tests; otherwise built lazily under ADC

    def load(self, agent_name: str) -> RouterModel | None:
        """Return a verified model, or None. Never raises: a bad artifact means passthrough."""
        try:
            return self._load(agent_name)
        except Exception as exc:  # any failure is the same failure: no model for this agent
            logger.warning("smart_router: no model for agent %s: %s", agent_name, exc)
            return None

    def latest_version(self, agent_name: str) -> str | None:
        """The version `load` would choose, without downloading anything. None if unavailable.

        This is the re-poll's whole hot path (DESIGN.md 21.4). It is one `list_blobs` call and
        a `max`, and on the common case where nothing has changed it is also the only call the
        refresh makes. Downloading, verifying and building a `Predictor` happen afterwards and
        only when the answer differs from what is loaded.

        Never raises, for the same reason `load` does not: a poll that cannot reach the bucket
        must leave the process serving what it already has, not take a turn down with it. A
        pinned version short-circuits, because a pin means versions move on deploys and there
        is nothing to poll for.
        """
        if self._cfg.pin_version:
            return self._cfg.pin_version
        try:
            return self._latest_complete_version(self._bucket(), agent_name)
        except Exception as exc:
            logger.warning("smart_router: cannot list versions for agent %s: %s", agent_name, exc)
            return None

    def published_versions(self, agent_name: str) -> tuple[str, ...]:
        """Every version this agent has published, oldest first.

        Published, not trained. Training exports a booster into the same version directory
        publishing later writes the artifact into, so a directory exists from the moment a
        model is fit and its presence says nothing about whether anything was shipped. The
        two files this router actually reads are what say so.

        At serve time the only question is whether there is a model to route with, so `load`
        answers None to "nothing was ever published here" and to "what was published cannot
        be read" alike. Both mean passthrough and the difference does not matter.

        It matters to the tools that run a retrain cycle. A new agent has published nothing
        by definition, and its first cycle has to get past the tuner to reach the publisher,
        so treating that as a failure makes the first cycle impossible. A broken artifact is
        a real problem and has to stay one. Unlike `load`, this raises when the bucket cannot
        be read, because an unreadable bucket is not an empty one.
        """
        return tuple(sorted(self._complete_versions(self._bucket(), agent_name)))

    def _load(self, agent_name: str) -> RouterModel:
        _check_segment("agent name", agent_name)
        bucket = self._bucket()

        version = self._cfg.pin_version or self._latest_complete_version(bucket, agent_name)
        _check_segment("version", version)

        prefix = f"routers/{agent_name}/{version}"
        meta = _read_metadata(bucket, f"{prefix}/{_METADATA_NAME}")
        # NOTE(contract): meta.version is not cross-checked against the directory name. A mismatch
        # is a publishing bug the digest cannot catch, but the contract does not list it as a
        # refusal condition, so it is only logged here.
        if meta.version != version:
            logger.warning("smart_router: %s metadata declares version %s", prefix, meta.version)
        self._check_compatible(meta)

        path = self._ensure_local(bucket, agent_name, version, meta)
        return RouterModel(agent_name=agent_name, metadata=meta, booster_path=path)

    def _bucket(self) -> GcsBucket:
        # `gcs_bucket` has no default, because guessing a bucket name is worse than
        # not routing. Caught here so an unset SMART_ROUTER_GCS_BUCKET says so,
        # rather than surfacing as an IndexError from inside the GCS library.
        if not self._cfg.gcs_bucket:
            raise _Refused("gcs_bucket is not configured; set SMART_ROUTER_GCS_BUCKET")
        if self._client is None:
            # Lazy: the package imports without the GCS library. The ignore is because
            # google-cloud-bigquery ships `py.typed` and google-cloud-storage does not, so once
            # anything in this package types against `google.cloud`, mypy can see the namespace
            # and reports the untyped half of it missing. It is here, not there.
            from google.cloud import storage  # type: ignore[attr-defined]

            self._client = storage.Client()  # no arguments, so it inherits ADC
        return self._client.bucket(self._cfg.gcs_bucket)

    def withdrawn_versions(self, agent_name: str) -> tuple[str, ...]:
        """Versions taken out of service by a withdrawal marker, oldest first.

        `published_versions` cannot answer this, because withdrawing is precisely the act of
        making a version stop counting as published. The watch needs both lists: one to know
        what it may withdraw next, the other to know what it already has.
        """
        return tuple(sorted(self._listing(self._bucket(), agent_name)[1]))

    def _complete_versions(self, bucket: GcsBucket, agent_name: str) -> list[str]:
        """Version directories holding both files a published artifact is made of, less any
        that have been withdrawn.

        Directories holding only what training exported are not published versions, and the
        `assets/` subdirectory BQML writes beside them is not a version at all, which is why
        only two-part paths are counted.

        Withdrawal is subtraction from this one list, which is what makes it reach everything at
        once. `load`, `latest_version` and `published_versions` all read it, so one marker takes
        a generation out of serving, out of the re-poll's answer and out of what a retrain cycle
        considers shipped, with no fourth place to keep in step (DESIGN.md 21.5). It is also
        why withdrawing is reversible: deleting the marker puts the version back.
        """
        complete, withdrawn = self._listing(bucket, agent_name)
        return sorted(complete - withdrawn)

    def _listing(self, bucket: GcsBucket, agent_name: str) -> tuple[set[str], set[str]]:
        """One `list_blobs`, read two ways: what is complete, and what is withdrawn."""
        _check_segment("agent name", agent_name)
        prefix = f"routers/{agent_name}/"
        seen: dict[str, set[str]] = {}
        for blob in bucket.list_blobs(prefix=prefix):
            parts = blob.name[len(prefix) :].split("/")
            if len(parts) != 2 or not _SAFE_SEGMENT.fullmatch(parts[0]):
                continue
            seen.setdefault(parts[0], set()).add(parts[1])
        complete = {v for v, files in seen.items() if {_METADATA_NAME, _BOOSTER_NAME} <= files}
        withdrawn = {v for v, files in seen.items() if WITHDRAWN_NAME in files}
        return complete, withdrawn

    def _latest_complete_version(self, bucket: GcsBucket, agent_name: str) -> str:
        """Highest version directory holding both files. Version names sort chronologically."""
        complete = self._complete_versions(bucket, agent_name)
        if not complete:
            raise _Refused(f"no complete version directory under routers/{agent_name}/")
        return max(complete)

    def _check_compatible(self, meta: ArtifactMetadata) -> None:
        cfg = self._cfg
        if meta.environment != cfg.environment:
            raise _Refused(
                f"artifact environment {meta.environment!r} != configured {cfg.environment!r}"
            )
        if meta.embedding_model != cfg.embedding_model:
            raise _Refused(
                f"artifact embedding model {meta.embedding_model!r} "
                f"!= configured {cfg.embedding_model!r}"
            )
        if meta.embedding_dims != cfg.embedding_dims:
            raise _Refused(
                f"artifact embedding dims {meta.embedding_dims} != configured {cfg.embedding_dims}"
            )
        if meta.backend != "vertex":
            raise _Refused(f"artifact backend {meta.backend!r} is not 'vertex'")
        # Last check before the bytes move: refuse an oversized artifact without downloading it.
        if meta.size_bytes > cfg.max_artifact_bytes:
            raise _Refused(
                f"artifact is {meta.size_bytes} bytes, over the {cfg.max_artifact_bytes} byte cap"
            )

    def _ensure_local(
        self, bucket: GcsBucket, agent_name: str, version: str, meta: ArtifactMetadata
    ) -> str:
        directory = os.path.join(self._cfg.local_cache_dir, agent_name, version)
        final = os.path.join(directory, _BOOSTER_NAME)
        if os.path.exists(final) and _sha256_file(final) == meta.sha256:
            _sweep_superseded(directory, _sweep_grace_s(self._cfg))
            return final

        blob = _sized_blob(
            bucket, f"routers/{agent_name}/{version}/{_BOOSTER_NAME}", meta, self._cfg
        )

        os.makedirs(directory, exist_ok=True)
        # One tmp name per attempt, so loaders sharing `local_cache_dir` cannot write over each
        # other. They used to share a fixed name, which fails safe but fails often: eight
        # loaders starting together left seven of them on a mixed file, and `plugin._resolve`
        # loads once per process and remembers a failure, so those seven never routed again
        # (DESIGN.md 19.71). `os.replace` is still what publishes the file, so the final name
        # only ever holds a digest-verified artifact and a late writer replaces identical bytes.
        tmp = f"{final}.{os.getpid()}.{next(_ATTEMPTS)}.tmp"
        try:
            blob.download_to_filename(tmp)
            digest = _sha256_file(tmp)
            if digest != meta.sha256:
                raise _Refused(f"sha256 mismatch: expected {meta.sha256}, downloaded {digest}")
            os.replace(tmp, final)
        except BaseException:
            _discard(tmp)  # a unique name is a leak if nobody cleans it up
            raise
        _sweep_superseded(directory, _sweep_grace_s(self._cfg))
        return final


# --- Helpers ------------------------------------------------------------------


def _sized_blob(
    bucket: GcsBucket, path: str, meta: ArtifactMetadata, cfg: RouterConfig
) -> GcsBucket:
    """Fetch the booster's object metadata, and refuse it before a single byte moves.

    `_check_compatible` already caps `size_bytes`, but that number is a claim made by the
    same artifact the cap exists to defend against, and the digest that would catch a lie
    is computed after the whole file is on local disk. On Cloud Run `local_cache_dir` is
    memory, so "download it and then reject it" is how a mispublished 50 GB object turns
    into an OOM kill instead of a warning line. The size checked here is the one GCS
    reports, and it has to match the declared one as well as clear the cap.

    `get_blob` is one extra request, on the cold path only: a warm process returns above
    on the digest of the file it already has.
    """
    blob = bucket.get_blob(path)
    if blob is None:
        raise _Refused(f"{path} is missing")
    size = blob.size
    if not isinstance(size, int):
        raise _Refused(f"{path} reports no size")
    if size > cfg.max_artifact_bytes:
        raise _Refused(f"{path} is {size} bytes, over the {cfg.max_artifact_bytes} byte cap")
    if size != meta.size_bytes:
        raise _Refused(f"{path} is {size} bytes, metadata declares {meta.size_bytes}")
    return blob


def _read_metadata(bucket: GcsBucket, path: str) -> ArtifactMetadata:
    raw = json.loads(bucket.blob(path).download_as_bytes())
    if not isinstance(raw, dict):
        raise _Refused(f"{path} is not a JSON object")
    return parse_metadata(raw)


def _check_segment(label: str, value: str) -> None:
    if not _SAFE_SEGMENT.fullmatch(value):
        raise _Refused(f"unsafe {label}: {value!r}")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sweep_grace_s(cfg: RouterConfig) -> float:
    """How long a superseded version directory is left alone before it is collected.

    Four re-poll intervals or an hour, whichever is longer. The number that matters is not
    how stale the directory is but how long every process on this machine has had to finish
    converging on the version that replaced it, and the re-poll interval is the clock that
    convergence runs on.
    """
    return max(cfg.refresh_interval_s * 4, 3600.0)


def _sweep_superseded(kept: str, grace_s: float) -> None:
    """Delete this agent's other local version directories. Best effort, never raises.

    Nothing else ever removes one, and `local_cache_dir` defaults to `/tmp`, which on Cloud
    Run is memory rather than disk. A replica that lives for weeks downloads every version
    published in that time and keeps all of them. Each is capped by `max_artifact_bytes`, so
    the total is bounded per version and unbounded over the life of the process, and the way
    it ends is an OOM kill rather than a log line.

    `grace_s` is what keeps this from racing the process beside it. Another loader can be
    between "the digest matches" and "open the file" on a directory this one is about to
    remove, and what it gets for losing that race is a refused load and a passthrough window
    (DESIGN.md 19.71). Only directories whose artifact was installed longer ago than the
    grace period are touched, so the version everyone is currently converging on is never
    the one swept, and the cost of being wrong anyway is a re-download rather than a wrong
    answer: this is a cache, and the bucket is still the truth.
    """
    parent = os.path.dirname(kept)
    cutoff = time.time() - grace_s
    try:
        names = os.listdir(parent)
    except OSError:
        return  # nothing local yet, or a directory this process may not read
    for name in names:
        path = os.path.join(parent, name)
        if path == kept or not _SAFE_SEGMENT.fullmatch(name) or not os.path.isdir(path):
            continue
        # The booster's mtime is when the version was installed. A directory without one is
        # a download that was interrupted, possibly by a SIGKILL that also left its `.tmp`
        # behind, so it is aged by the directory instead: that mtime moves whenever a file
        # inside is created or removed, which is exactly the last time anyone worked on it.
        booster = os.path.join(path, _BOOSTER_NAME)
        try:
            if os.path.getmtime(booster if os.path.exists(booster) else path) > cutoff:
                continue
        except OSError:
            continue  # it went away underneath us, which is the outcome we wanted anyway
        shutil.rmtree(path, ignore_errors=True)


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass  # the download never got as far as creating it, which is not worth a line
    except OSError:
        logger.warning("smart_router: could not remove %s", path)
