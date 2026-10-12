"""Per-decision provenance records: which model shaped this turn.

With rolling deploys, different pods serve different model versions at the same
moment, so "what was active then" is not recoverable after the fact. Every
routing decision and every cache serve therefore stamps its own provenance onto
the turn (DESIGN.md 11.3).

Recording a turn takes two hops. `emit()` appends to the invocation's metadata
bag, and `carry_onto()` moves what has accumulated there onto the next event
*that BQAA will turn into a row*. Both are needed: ADK never copies the
invocation bag onto an event, and BQAA reads `event.custom_metadata`, so records
that stop at the first hop reach the warehouse never.

Four rules, all easy to get wrong:

1. `CacheServeRecord` never carries the args hash or any argument value.
   Arguments contain PII, and a hash of low-cardinality args is reversible.
2. `emit()` is total. It runs inside an ADK callback, and ADK re-raises any
   plugin exception as a RuntimeError that kills the run, so a provenance
   failure must never propagate. `carry_onto()` is total for the same reason.
3. `carry_onto()` moves rather than copies. Records go onto exactly one event,
   and the bag is left empty for the next one.
4. The event it moves onto has to be one BQAA logs, which is not the same as one
   ADK persists. ADK yields a function-call event, a function-response event and
   a final response event for a single tool-calling turn, and persists all three;
   BQAA reads `custom_metadata` off the last of them only. Draining onto the
   first therefore deleted the record. It cost 903 of 2,793 decisions in run
   ab-v2 — every session whose first turn called a tool lost its `fresh`
   record and every cache serve ever emitted (DESIGN.md 19.23).
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The key under which records accumulate in `Context.custom_metadata`. Add it to
# BQAA's `custom_metadata_allowlist` to land these rows in the warehouse.
PROVENANCE_KEY = "smart_router"

# One id per process, minted at import and never changed. It is the stratum the fleet-wide
# cache comparison groups by. Replicas hold different mixes of routed and holdout sessions,
# and pooling a cost comparison across groups whose mixes differ is how DESIGN.md 18.22's
# Simpson's paradox comes back wearing a new grouping variable, so the comparison is made
# within a replica and combined afterwards (DESIGN.md 19.75).
#
# Random, rather than a hostname or a pid. A pid is reused across containers and a hostname
# is the container id on Cloud Run, so neither is a stable identity for a process, and
# neither raises the question of whether it is infrastructure detail or something an
# operator would rather not have sitting in a warehouse. Forty-eight bits is far more than
# enough to keep a fleet's replicas apart.
REPLICA_ID = uuid.uuid4().hex[:12]

# Every value `decision_source` can take, in one place because it is copied into a dozen
# others: the CASE in `sql/dashboard.sql`, the censoring filter in `sql/train_router.sql` and
# `scripts/score_sets.py`, and the arm labels in eight reporting scripts. Those copies are
# deliberate rather than lazy. A report that scored on a different definition than the
# operator reads would be answering a different question quietly, so each one states its own
# definition. `tests/test_decision_sources.py` is what stops them drifting apart.
DECISION_SOURCES: frozenset[str] = frozenset(
    {
        "fresh",
        "reused",
        "holdout",
        "disabled",
        "cheap",
        "lowconf",
        "floor",
        "watch",
        "late",
        "noop",
        "small",
        "timeout",
        "error",
        "breaker",
    }
)

# The two that mean the router acted: a menu was chosen and it was applied. Every other
# source is a decision not to prune, so a session carrying one sent its full tool list and is
# unrouted traffic. This subset is what the training corpus censors on (DESIGN.md 11.4) and
# what every report means by "routed".
ROUTED_SOURCES: frozenset[str] = frozenset({"fresh", "reused"})


@dataclass(frozen=True)
class RoutingRecord:
    """One routing decision: what the model saw and what the router did about it."""

    decision_id: str
    agent_name: str
    session_id: str
    mode: str
    model_version: str | None
    model_digest: str | None
    top_label: str | None
    confidence: float | None
    kept_tools: tuple[str, ...]
    # One of `DECISION_SOURCES` above, and what each of them means. "cheap" is the decline
    # rule: what this menu would save is worth less than the chance of missing a tool it dropped
    # (DESIGN.md 19.5), and it is the reason most declined sessions decline. "floor" and
    # "watch" are the two cache guards and are recorded apart: one means the prompt is too
    # small to prune safely, the other means this agent has been switched off for the rest of
    # the process (DESIGN.md 18.21). "late" is neither: the session already sent a full tool
    # list, so the moment to choose one has passed (DESIGN.md 18.22). "timeout", "error" and
    # "breaker" are the three ways the embedding can fail, split apart for the same reason the
    # cache guards are: they were one value, and the run that lost 47 sessions to quota and the
    # run that lost 40 to latency were indistinguishable in the warehouse (DESIGN.md 19.23).
    #
    # Two values were retired in v1, "grown" and "budget", when the pruner's growth path was
    # deleted as structurally unreachable. Neither was ever written: zero occurrences in the
    # 13,474 records the warehouse holds. Nothing reading this column has to accept them, and
    # nothing re-adding them from an old schema would find rows to match.
    decision_source: str
    # True whenever the declared tool list moved mid-session: a change of kept set, and also
    # a change of whether the kept set was applied at all. The second half was missing, and
    # it is the one that fired in production (DESIGN.md 18.19).
    cache_break: bool
    # Which process decided this session. Defaulted rather than passed, because it is a fact
    # about the process and not about the decision, and a caller that could set it could set
    # it wrong. Not on `CacheServeRecord`: nothing asks a per-replica question of cache
    # serves, and that record's field list is a boundary rather than a convenience.
    replica_id: str = REPLICA_ID

    def as_dict(self) -> dict[str, Any]:
        # kept_tools becomes a list because this dict is JSON-serialized on its
        # way to BigQuery, and tuples are not JSON.
        return {
            "decision_id": self.decision_id,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "mode": self.mode,
            "model_version": self.model_version,
            "model_digest": self.model_digest,
            "top_label": self.top_label,
            "confidence": self.confidence,
            "kept_tools": list(self.kept_tools),
            "decision_source": self.decision_source,
            "cache_break": self.cache_break,
            "replica_id": self.replica_id,
        }


@dataclass(frozen=True)
class CacheServeRecord:
    """One tool result served from cache instead of executed."""

    decision_id: str
    agent_name: str
    session_id: str
    tool_name: str
    entry_age_s: float
    ttl_used: int
    ttl_source: str
    probe: bool

    def as_dict(self) -> dict[str, Any]:
        # Deliberately field-by-field rather than dataclasses.asdict(): this is
        # the PII boundary, so adding a field to the dataclass must not silently
        # add it to the emitted record. Never add args or an args hash here.
        return {
            "decision_id": self.decision_id,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "entry_age_s": self.entry_age_s,
            "ttl_used": self.ttl_used,
            "ttl_source": self.ttl_source,
            "probe": self.probe,
        }


# NOTE(contract): the two as_dict() shapes are emitted into one list with no
# discriminator field, so a consumer must tell them apart by the presence of
# "tool_name". A "kind" key would be cheaper to parse, but DESIGN.md 11.3 fixes
# these field sets and dashboards are built on them, so it is not added here.


def new_decision_id() -> str:
    """A fresh id linking a routing decision to the cache serves under it."""
    return uuid.uuid4().hex


def emit(ctx: Any, record: RoutingRecord | CacheServeRecord) -> None:  # noqa: ANN401
    """Record provenance on the turn, falling back to a structured log. Never raises.

    `ctx` is `Any` on purpose: the plugin passes an ADK `Context`, but nothing
    here needs the type, and importing ADK for an annotation would make
    provenance depend on the framework.
    """
    payload = record.as_dict()
    if _append_to_metadata(ctx, payload):
        return
    with contextlib.suppress(Exception):
        logger.info("smart_router provenance", extra={PROVENANCE_KEY: payload})


def logged_by_analytics(event: Any) -> bool:  # noqa: ANN401
    """Whether BQAA will read `custom_metadata` off this event. Total: never raises.

    Mirrors the guard BQAA applies before it writes an AGENT_RESPONSE row: not
    partial, no function calls, no function responses, and a final response. A
    NODE_ERROR row is written from its own source event, so an event carrying an
    error code counts too.

    Duck-typed on purpose, one `getattr` per attribute, so provenance keeps no ADK
    import and no ADK version pin. An event shape this rule does not recognise
    answers True, which is the behaviour that shipped before the rule existed: the
    failure mode of holding a record for an event that never comes is worse than
    the failure mode of stamping one on an event nobody reads.
    """
    try:
        if getattr(event, "partial", None) is True:
            return False
        for probe in ("get_function_calls", "get_function_responses"):
            reader = getattr(event, probe, None)
            if callable(reader) and reader():
                return False
        if getattr(event, "error_code", None):
            return True
        is_final = getattr(event, "is_final_response", None)
        return bool(is_final()) if callable(is_final) else True
    except Exception:
        logger.debug("smart_router: could not classify the event; assuming it is logged")
        return True


def carry_onto(ctx: Any, event: Any) -> int:  # noqa: ANN401
    """Move the invocation's accumulated records onto `event`. Returns how many. Never raises.

    ADK keeps `custom_metadata` on the invocation and on each event, and copies
    nothing between them. BQAA reads the event's, so this is the hop that gets a
    routing decision into `agent_events`. The plugin calls it from
    `on_event_callback`, which ADK runs for every event it yields.

    Every event is offered and most are declined. An event BQAA will not log is
    not a hop, it is a delete: the bag is drained onto an object nothing reads.
    So a declined event leaves the bag untouched and the records ride the next
    event that qualifies, which for a tool-calling turn is the final response.
    """
    if not logged_by_analytics(event):
        return 0
    bag = _bag(ctx)
    if bag is None:
        return 0
    try:
        pending = bag.get(PROVENANCE_KEY)
        if not isinstance(pending, list) or not pending:
            return 0
        carried = list(pending)
        existing = getattr(event, "custom_metadata", None) or {}
        prior = existing.get(PROVENANCE_KEY)
        event.custom_metadata = {
            **existing,
            PROVENANCE_KEY: ([*prior, *carried] if isinstance(prior, list) else carried),
        }
    except Exception:
        logger.debug("smart_router: could not stamp provenance onto the event", exc_info=True)
        return 0
    pending.clear()
    return len(carried)


def _bag(ctx: object) -> MutableMapping[str, Any] | None:
    """The invocation's mutable metadata dict, whichever context object is holding it.

    `Context` (the callback and tool context) exposes it as a property.
    `InvocationContext`, which is what `on_event_callback` is handed, has no
    public accessor, so this reaches for the same private attribute ADK's own
    `mcp_toolset` reaches for. `ReadonlyContext` returns a read-only view and is
    correctly rejected here.
    """
    try:
        metadata = getattr(ctx, "custom_metadata", None)
        if isinstance(metadata, MutableMapping):
            return metadata
        private = getattr(ctx, "_custom_metadata", None)
    except Exception:
        return None  # a property that raises is a logging problem, not a serving problem
    return private if isinstance(private, MutableMapping) else None


def _append_to_metadata(ctx: object, payload: dict[str, Any]) -> bool:
    """Append to `ctx.custom_metadata[PROVENANCE_KEY]`. False if that is not possible."""
    metadata = _bag(ctx)
    if metadata is None:
        return False
    try:
        bucket = metadata.get(PROVENANCE_KEY)
        if not isinstance(bucket, list):
            bucket = []
            metadata[PROVENANCE_KEY] = bucket
        bucket.append(payload)
    except Exception:
        # A hostile or exotic context (mutating mapping that rejects writes) is a
        # logging problem, not a serving problem.
        return False
    return True
