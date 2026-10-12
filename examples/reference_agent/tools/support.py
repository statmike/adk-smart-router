"""Support tools: FAQ, ticketing, human escalation.

`escalate_to_human` sets `tool_context.actions.escalate`. Its contract is therefore its
return value *plus* that action, which is exactly what a cache hit would drop: the model
would be told a human is coming and no human would be paged. DESIGN.md 8.2 requires the
plugin to blocklist it automatically at write time, and this tool is what proves it.

`get_faq` is the other extreme: near-static text, safe to cache for hours.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from . import _data

_PRIORITIES: tuple[str, ...] = ("low", "normal", "high", "urgent")


def get_faq(topic: str) -> dict[str, Any]:
    """Look up the help-centre answer for a topic.

    Args:
        topic: One of "returns", "shipping", "payments", "account" or "warranty".

    Returns the published answer text. This content changes rarely.
    """
    key = topic.strip().lower()
    answer = _data.FAQS.get(key)
    if answer is None:
        return {
            "topic": key,
            "known_topics": sorted(_data.FAQS),
            "error": f"no FAQ entry for {key!r}",
        }
    return {"topic": key, "answer": answer}


def create_ticket(
    subject: str, body: str, priority: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Open a support ticket for something that needs follow-up by email.

    Args:
        subject: One-line summary of the problem.
        body: The details, including any order id.
        priority: "low", "normal", "high" or "urgent".

    Returns the ticket id and the response time the customer should expect. Use this
    for asynchronous follow-up, not for a customer who is asking for a person now.
    """
    level = priority.strip().lower()
    if level not in _PRIORITIES:
        level = "normal"

    user_id = _data.current_user_id(tool_context)
    ticket_id = f"TCK-{_data.stable_int(f'{user_id}|{subject}', 10000, 99999)}"
    ticket = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "subject": subject.strip(),
        "body": body.strip(),
        "priority": level,
        "status": "OPEN",
        "opened_on": str(_data.TODAY),
    }
    _data.TICKETS.append(ticket)
    response_hours = {"low": 72, "normal": 24, "high": 8, "urgent": 2}[level]
    return {
        "ticket_id": ticket_id,
        "created": True,
        "priority": level,
        "response_within_hours": response_hours,
    }


def escalate_to_human(reason: str, tool_context: ToolContext) -> dict[str, Any]:
    """Hand the conversation to a human agent right now.

    Args:
        reason: Why a person is needed, in one sentence.

    Use this when the customer asks for a person, is distressed, or the request is
    outside what these tools can do. Returns the queue and the wait estimate.
    """
    # The side channel that makes this tool uncacheable (DESIGN.md 8.2). A replayed
    # result would say a human is coming while nothing was escalated.
    tool_context.actions.escalate = True

    user_id = _data.current_user_id(tool_context)
    handoff_id = f"ESC-{_data.stable_int(f'{user_id}|{reason}', 10000, 99999)}"
    _data.TICKETS.append(
        {
            "ticket_id": handoff_id,
            "user_id": user_id,
            "subject": "escalated to a human agent",
            "body": reason.strip(),
            "priority": "urgent",
            "status": "ESCALATED",
            "opened_on": str(_data.TODAY),
        }
    )
    return {
        "handoff_id": handoff_id,
        "escalated": True,
        "queue": "tier-2 support",
        "estimated_wait_minutes": _data.stable_int(f"wait:{handoff_id}", 2, 12),
    }
