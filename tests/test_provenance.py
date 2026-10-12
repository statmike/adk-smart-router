"""Tests for provenance records. No network, no GCP, no ADK objects: plain fakes."""

from __future__ import annotations

import json
import logging
from types import MappingProxyType
from typing import Any

from adk_smart_router.provenance import (
    PROVENANCE_KEY,
    REPLICA_ID,
    CacheServeRecord,
    RoutingRecord,
    carry_onto,
    emit,
    logged_by_analytics,
    new_decision_id,
)


class FakeContext:
    """Stands in for ADK's Context, which exposes a real mutable custom_metadata dict."""

    def __init__(self, custom_metadata: Any = None) -> None:
        self.custom_metadata = {} if custom_metadata is None else custom_metadata


class FakeInvocationContext:
    """Stands in for ADK's InvocationContext, which keeps the same dict behind a private name.

    `on_event_callback` is handed one of these rather than a `Context`, and it has no public
    accessor for the metadata bag, so `carry_onto` has to find it either way.
    """

    def __init__(self, custom_metadata: Any = None) -> None:
        self._custom_metadata = {} if custom_metadata is None else custom_metadata

    def custom_metadata_bag(self) -> list[Any]:
        return self._custom_metadata.get(PROVENANCE_KEY, [])


class FakeEvent:
    """Stands in for ADK's Event, whose custom_metadata is None until something sets it.

    Deliberately does not implement `is_final_response`, `partial` or the function-part
    readers: an event shape `logged_by_analytics` does not recognise is accepted, which is
    what keeps every test written before that rule existed meaningful.
    """

    def __init__(self, custom_metadata: Any = None) -> None:
        self.custom_metadata = custom_metadata


class FakeAdkEvent:
    """The three shapes ADK yields for one tool-calling turn, as BQAA sees them.

    Only the last of the three becomes a row, so only the last one is a place a provenance
    record can survive (see `logged_by_analytics`).
    """

    def __init__(
        self,
        *,
        final: bool = True,
        partial: bool | None = None,
        function_calls: bool = False,
        function_responses: bool = False,
        error_code: str | None = None,
    ) -> None:
        self.custom_metadata: Any = None
        self.partial = partial
        self.error_code = error_code
        self._final = final
        self._calls = function_calls
        self._responses = function_responses

    def is_final_response(self) -> bool:
        return self._final

    def get_function_calls(self) -> list[str]:
        return ["get_order_status"] if self._calls else []

    def get_function_responses(self) -> list[str]:
        return ["get_order_status"] if self._responses else []


class ExplodingContext:
    """A context whose custom_metadata access raises, as a misconfigured proxy might."""

    @property
    def custom_metadata(self) -> dict[str, Any]:
        raise RuntimeError("no metadata here")


class RefusingMapping(dict[str, Any]):
    """A MutableMapping that refuses writes, like a frozen or quota-limited store."""

    def __setitem__(self, key: str, value: Any) -> None:
        raise RuntimeError("read only")


def a_routing_record(**overrides: Any) -> RoutingRecord:
    fields: dict[str, Any] = {
        "decision_id": "d1",
        "agent_name": "support",
        "session_id": "s1",
        "mode": "active",
        "model_version": "20260101T000000Z",
        "model_digest": "abc123",
        "top_label": "search_docs",
        "confidence": 0.91,
        "kept_tools": ("search_docs", "transfer_to_agent"),
        "decision_source": "fresh",
        "cache_break": False,
    }
    fields.update(overrides)
    return RoutingRecord(**fields)


def a_cache_serve_record(**overrides: Any) -> CacheServeRecord:
    fields: dict[str, Any] = {
        "decision_id": "d1",
        "agent_name": "support",
        "session_id": "s1",
        "tool_name": "lookup_order",
        "entry_age_s": 12.5,
        "ttl_used": 60,
        "ttl_source": "declared",
        "probe": False,
    }
    fields.update(overrides)
    return CacheServeRecord(**fields)


# --- emit reaches custom_metadata ---------------------------------------------


def test_record_reaches_custom_metadata():
    ctx = FakeContext()
    emit(ctx, a_routing_record())

    assert ctx.custom_metadata[PROVENANCE_KEY] == [a_routing_record().as_dict()]


def test_records_accumulate_in_order():
    ctx = FakeContext()
    emit(ctx, a_routing_record(decision_id="first"))
    emit(ctx, a_cache_serve_record(decision_id="second"))

    bucket = ctx.custom_metadata[PROVENANCE_KEY]
    assert [entry["decision_id"] for entry in bucket] == ["first", "second"]


def test_existing_unrelated_metadata_is_preserved():
    ctx = FakeContext({"other": "keep me"})
    emit(ctx, a_routing_record())

    assert ctx.custom_metadata["other"] == "keep me"
    assert len(ctx.custom_metadata[PROVENANCE_KEY]) == 1


def test_non_list_value_under_the_key_is_replaced_not_appended_to():
    ctx = FakeContext({PROVENANCE_KEY: "someone else's string"})
    emit(ctx, a_routing_record())

    assert ctx.custom_metadata[PROVENANCE_KEY] == [a_routing_record().as_dict()]


# --- emit is total ------------------------------------------------------------


def test_context_without_custom_metadata_falls_back_to_logging(caplog):
    caplog.set_level(logging.INFO, logger="adk_smart_router.provenance")
    emit(object(), a_routing_record())

    assert len(caplog.records) == 1
    assert getattr(caplog.records[0], PROVENANCE_KEY)["decision_id"] == "d1"


def test_read_only_mapping_falls_back_to_logging(caplog):
    caplog.set_level(logging.INFO, logger="adk_smart_router.provenance")
    ctx = FakeContext(MappingProxyType({}))
    emit(ctx, a_routing_record())

    assert len(caplog.records) == 1


def test_mapping_that_refuses_writes_falls_back_to_logging(caplog):
    caplog.set_level(logging.INFO, logger="adk_smart_router.provenance")
    ctx = FakeContext(RefusingMapping())
    emit(ctx, a_routing_record())

    assert len(caplog.records) == 1


def test_raising_context_does_not_raise(caplog):
    caplog.set_level(logging.INFO, logger="adk_smart_router.provenance")
    emit(ExplodingContext(), a_cache_serve_record())

    assert len(caplog.records) == 1


def test_none_context_does_not_raise():
    emit(None, a_routing_record())


def test_no_double_recording_when_metadata_write_succeeds(caplog):
    caplog.set_level(logging.INFO, logger="adk_smart_router.provenance")
    ctx = FakeContext()
    emit(ctx, a_routing_record())

    assert caplog.records == []
    assert len(ctx.custom_metadata[PROVENANCE_KEY]) == 1


# --- carry_onto: the hop that actually reaches BigQuery -------------------------
#
# ADK keeps one custom_metadata bag on the invocation and another on each event, and copies
# nothing between them. BQAA reads the event's. Records that stop on the invocation are records
# that never leave the process, so this hop is not a convenience.


def test_records_move_from_the_invocation_onto_the_event():
    ctx = FakeInvocationContext()
    emit(ctx, a_routing_record())
    event = FakeEvent()

    assert carry_onto(ctx, event) == 1
    assert event.custom_metadata[PROVENANCE_KEY] == [a_routing_record().as_dict()]


def test_carrying_empties_the_bag_so_the_next_event_gets_nothing():
    ctx = FakeInvocationContext()
    emit(ctx, a_routing_record())

    first, second = FakeEvent(), FakeEvent()
    assert carry_onto(ctx, first) == 1
    assert carry_onto(ctx, second) == 0
    assert second.custom_metadata is None
    assert ctx.custom_metadata_bag() == []


def test_a_context_object_works_as_well_as_an_invocation_context():
    ctx = FakeContext()
    emit(ctx, a_cache_serve_record())
    event = FakeEvent()

    assert carry_onto(ctx, event) == 1
    assert event.custom_metadata[PROVENANCE_KEY][0]["tool_name"] == "lookup_order"


def test_metadata_already_on_the_event_is_preserved():
    ctx = FakeInvocationContext()
    emit(ctx, a_routing_record())
    event = FakeEvent({"a2a:task_id": "t1", PROVENANCE_KEY: [{"decision_id": "earlier"}]})

    assert carry_onto(ctx, event) == 1
    assert event.custom_metadata["a2a:task_id"] == "t1"
    assert [r["decision_id"] for r in event.custom_metadata[PROVENANCE_KEY]] == ["earlier", "d1"]


def test_an_empty_bag_leaves_the_event_untouched():
    event = FakeEvent()

    assert carry_onto(FakeInvocationContext(), event) == 0
    assert event.custom_metadata is None


def test_carry_onto_is_total():
    """It runs inside a callback, so a failure here must never abort the run."""
    assert carry_onto(None, FakeEvent()) == 0
    assert carry_onto(object(), FakeEvent()) == 0
    assert carry_onto(ExplodingContext(), FakeEvent()) == 0
    assert carry_onto(FakeContext(MappingProxyType({})), FakeEvent()) == 0

    populated = FakeInvocationContext()
    emit(populated, a_routing_record())
    assert carry_onto(populated, object()) == 0
    assert populated.custom_metadata_bag(), "a failed carry must not lose the records"


# --- carrying only onto events BQAA logs ----------------------------------------
#
# ADK yields three events for one tool-calling turn and persists all three, but BQAA writes a
# row for the last one only. Draining the bag onto the first two deleted the records: 903 of
# 2,793 decisions in run ab-v2, including every `fresh` record of every session whose first
# turn called a tool, and every cache serve ever emitted (DESIGN.md 19.23).


def test_a_function_call_event_is_declined_and_the_bag_is_left_full():
    ctx = FakeInvocationContext()
    emit(ctx, a_routing_record())
    call_event = FakeAdkEvent(final=False, function_calls=True)

    assert carry_onto(ctx, call_event) == 0
    assert call_event.custom_metadata is None
    assert len(ctx.custom_metadata_bag()) == 1, "the record must wait, not be dropped"


def test_a_tool_round_trip_delivers_both_records_to_the_final_event_in_order():
    """The whole point: turn one emits `fresh`, calls a tool, then emits `reused`.

    Three events go past — function call, function response, final response — and only the
    third is a row in `agent_events`. Both records have to be on that third one, in the order
    they happened, or the session's first surviving record is `reused` and the decision looks
    like it was made by a turn that never happened.
    """
    ctx = FakeInvocationContext()
    emit(ctx, a_routing_record(decision_source="fresh"))
    call_event = FakeAdkEvent(final=False, function_calls=True)
    response_event = FakeAdkEvent(final=False, function_responses=True)
    final_event = FakeAdkEvent()

    assert carry_onto(ctx, call_event) == 0
    emit(ctx, a_cache_serve_record())
    assert carry_onto(ctx, response_event) == 0
    emit(ctx, a_routing_record(decision_source="reused"))

    assert carry_onto(ctx, final_event) == 3
    carried = final_event.custom_metadata[PROVENANCE_KEY]
    assert [r.get("decision_source") or r["tool_name"] for r in carried] == [
        "fresh",
        "lookup_order",
        "reused",
    ]
    assert ctx.custom_metadata_bag() == []
    assert call_event.custom_metadata is None
    assert response_event.custom_metadata is None


def test_a_partial_streaming_event_is_declined():
    """A streamed turn fires this callback per chunk. Only the assembled event is a row."""
    ctx = FakeInvocationContext()
    emit(ctx, a_routing_record())

    assert carry_onto(ctx, FakeAdkEvent(partial=True)) == 0
    assert carry_onto(ctx, FakeAdkEvent(partial=False)) == 1


def test_a_non_final_event_with_no_function_parts_is_declined():
    ctx = FakeInvocationContext()
    emit(ctx, a_routing_record())

    assert carry_onto(ctx, FakeAdkEvent(final=False)) == 0
    assert len(ctx.custom_metadata_bag()) == 1


def test_an_error_event_is_accepted_even_though_it_is_not_a_final_response():
    """BQAA writes NODE_ERROR from its own source event, and a failed run still decided."""
    ctx = FakeInvocationContext()
    emit(ctx, a_routing_record())
    event = FakeAdkEvent(final=False, error_code="MALFORMED_FUNCTION_CALL")

    assert carry_onto(ctx, event) == 1
    assert len(event.custom_metadata[PROVENANCE_KEY]) == 1


def test_an_unrecognised_event_shape_is_accepted():
    """Holding a record for an event that never comes loses more than stamping a spare one."""
    assert logged_by_analytics(FakeEvent()) is True
    assert logged_by_analytics(object()) is True
    assert logged_by_analytics(None) is True


def test_logged_by_analytics_is_total_against_a_hostile_event():
    class Hostile:
        @property
        def partial(self) -> bool:
            raise RuntimeError("no")

    assert logged_by_analytics(Hostile()) is True


# --- the PII boundary ---------------------------------------------------------


def test_cache_serve_record_carries_no_args_or_args_hash():
    payload = a_cache_serve_record().as_dict()

    assert set(payload) == {
        "decision_id",
        "agent_name",
        "session_id",
        "tool_name",
        "entry_age_s",
        "ttl_used",
        "ttl_source",
        "probe",
    }
    assert not [k for k in payload if "arg" in k or "hash" in k or "key" in k]


def test_cache_serve_record_has_no_args_field_to_leak():
    # If someone adds args to the dataclass, as_dict() must still not emit them.
    assert not hasattr(a_cache_serve_record(), "args")


# --- record shapes ------------------------------------------------------------


def test_routing_record_kept_tools_is_json_serializable():
    payload = a_routing_record().as_dict()

    assert payload["kept_tools"] == ["search_docs", "transfer_to_agent"]
    assert json.loads(json.dumps(payload)) == payload


def test_optional_routing_fields_survive_as_none():
    payload = a_routing_record(
        model_version=None, model_digest=None, top_label=None, confidence=None
    ).as_dict()

    assert payload["model_version"] is None
    assert payload["confidence"] is None
    assert payload["decision_source"] == "fresh"


def test_every_routing_record_carries_the_process_that_made_it():
    """The stratum the fleet-wide cache comparison groups by. Without it, one replica's mix
    of routed and holdout sessions is indistinguishable from another's, and a pooled cost
    comparison across them can say pruning is free while every replica is losing money
    (DESIGN.md 19.75)."""
    payload = a_routing_record().as_dict()

    assert payload["replica_id"] == REPLICA_ID
    assert len(REPLICA_ID) == 12


def test_the_replica_id_is_defaulted_rather_than_asked_for():
    """It is a fact about the process, not about the decision. A caller that had to pass it
    is a caller that could pass the wrong one."""
    assert a_routing_record().replica_id == REPLICA_ID


def test_a_cache_serve_record_carries_no_replica_id():
    """Nothing asks a per-replica question of cache serves, and that record's field list is
    the PII boundary rather than a convenience."""
    assert "replica_id" not in a_cache_serve_record().as_dict()


def test_records_are_frozen():
    record = a_routing_record()
    try:
        record.decision_id = "mutated"  # type: ignore[misc]
    except Exception as exc:
        assert isinstance(exc, AttributeError)
    else:  # pragma: no cover - would mean the dataclass lost frozen=True
        raise AssertionError("RoutingRecord must be frozen")


def test_new_decision_id_is_unique_and_stringy():
    ids = {new_decision_id() for _ in range(1000)}

    assert len(ids) == 1000
    assert all(isinstance(i, str) and i.isalnum() for i in ids)
