"""The session-service tier from DESIGN.md 13: run the state boundary against both real services.

The router keeps its session decisions in process and never in ADK session state. DESIGN.md D8
gives two reasons, and this file holds the first one to the fire: a `frozenset` written to state
comes back a `frozenset` from `InMemorySessionService` and a `list` from `DatabaseSessionService`.
A suite that only ever ran on InMemory would pass on a router that stored its decision in state,
and then fail in production against Postgres, where `kept` is suddenly a list and every `in`
check still works, every set operation does not.

So these tests are parameterized over both services, and both services are real. A fake would
reproduce whichever behaviour we assumed, which is the assumption under test.

`sqlalchemy` and `aiosqlite` are dev dependencies for exactly this file. The database is a
temporary sqlite file, so the suite stays offline and needs no credentials.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.sessions.base_session_service import BaseSessionService

APP = "support"
USER = "u-1"
SESSION = "s-1"


@pytest.fixture(params=["in_memory", "database"])
async def service(request: pytest.FixtureRequest) -> AsyncIterator[BaseSessionService]:
    """One real session service per parameter. The database one is a throwaway sqlite file."""
    if request.param == "in_memory":
        yield InMemorySessionService()
        return
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "sessions.db"
        yield DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db}")


async def write_state(service: BaseSessionService, delta: dict[str, object]) -> dict[str, object]:
    """Commit a state delta through a real event and read the session back."""
    session = await service.create_session(app_name=APP, user_id=USER, session_id=SESSION)
    await service.append_event(
        session,
        Event(author="agent", invocation_id="inv-1", actions=EventActions(state_delta=delta)),
    )
    reloaded = await service.get_session(app_name=APP, user_id=USER, session_id=SESSION)
    assert reloaded is not None
    return dict(reloaded.state)


# --- The bug class D8 avoids ----------------------------------------------------


async def test_a_frozenset_does_not_survive_both_services_as_a_frozenset() -> None:
    """The reason the session decision cache is in process, demonstrated rather than asserted.

    This test is the evidence for the note in `plugin._remember`. If ADK ever normalizes the two
    services to one behaviour this test fails, and that is the right time to revisit D8 rather
    than to discover it from a production incident.
    """
    kept = frozenset({"tool_00", "tool_01"})

    in_memory = await write_state(InMemorySessionService(), {"kept": kept})
    assert in_memory["kept"] == kept
    assert isinstance(in_memory["kept"], frozenset)

    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "sessions.db"
        database = await write_state(
            DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{db}"), {"kept": kept}
        )
    assert isinstance(database["kept"], list)
    assert sorted(database["kept"]) == sorted(kept)


async def test_the_router_writes_no_session_state_of_its_own(
    service: BaseSessionService,
) -> None:
    """The decision is not in state, so nothing the router owns can be reshaped by a service.

    The router writes no session state at all, which is a stronger claim than it needs for D8
    and is kept because it is free. `test_a_cache_hit_writes_no_session_state` in
    `test_cache_safety.py` holds the serving path to it against a real callback round trip.
    """
    state = await write_state(service, {"app:unrelated": 1})
    assert not [key for key in state if key.startswith("smart_router")]
    assert not [key for key in state if "kept" in key or "decision" in key]


# --- The other half of D8 --------------------------------------------------------


async def test_a_second_process_does_not_inherit_a_decision_through_the_database() -> None:
    """The decision is worthless to a replica that did not make it, so it must not travel.

    Two `DatabaseSessionService` instances over one file are the closest offline stand-in for two
    replicas behind a load balancer. Nothing router-owned crosses between them, which is what
    lets a redeploy change the model without a half-migrated session serving a mix of both.
    """
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "sessions.db"
        url = f"sqlite+aiosqlite:///{db}"

        first = DatabaseSessionService(db_url=url)
        session = await first.create_session(app_name=APP, user_id=USER, session_id=SESSION)
        await first.append_event(
            session,
            Event(
                author="agent",
                invocation_id="inv-1",
                actions=EventActions(state_delta={"app:turns": 1}),
            ),
        )

        second = DatabaseSessionService(db_url=url)
        seen = await second.get_session(app_name=APP, user_id=USER, session_id=SESSION)

    assert seen is not None
    assert seen.state["app:turns"] == 1, "the two services are not sharing a database"
    assert not [key for key in seen.state if key.startswith(("smart_router", "temp:"))]
