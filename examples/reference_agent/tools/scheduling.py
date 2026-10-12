"""Scheduling tools: callbacks, delivery windows and the slots behind both.

One of the twelve domains added in DESIGN.md 19.15. Slots are generated from the frozen
TODAY and a stable hash, so the same day always offers the same times and a regenerated
corpus matches. Only `_data` is imported, never another domain.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from google.adk.tools import ToolContext

from . import _data

SLOT_TIMES: tuple[str, ...] = ("09:00", "11:00", "13:00", "15:00", "17:00")
CALLBACK_TOPICS: tuple[str, ...] = (
    "order_problem",
    "return_help",
    "billing_question",
    "technical_support",
    "complaint",
)


def _seed_callbacks() -> dict[str, dict[str, Any]]:
    return {}


def _seed_delivery_changes() -> dict[str, dict[str, Any]]:
    return {}


CALLBACKS: dict[str, dict[str, Any]] = _seed_callbacks()
DELIVERY_WINDOWS: dict[str, dict[str, Any]] = _seed_delivery_changes()


def _reset() -> None:
    CALLBACKS.clear()
    DELIVERY_WINDOWS.clear()


_data.register_reset(_reset)


def _next_business_days(count: int) -> list[date]:
    """The next `count` weekdays after the frozen TODAY."""
    days: list[date] = []
    cursor = _data.TODAY
    while len(days) < count:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            days.append(cursor)
    return days


def _slots_for(day: date, taken: set[str]) -> list[dict[str, Any]]:
    """The bookable slots on one day, with a stable subset already full."""
    slots = []
    for time in SLOT_TIMES:
        key = f"{day.isoformat()} {time}"
        full = key in taken or _data.stable_int(key, 0, 4) == 0
        slots.append({"date": day.isoformat(), "time": time, "available": not full})
    return slots


def list_available_slots(days_ahead: int) -> dict[str, Any]:
    """List the appointment and callback slots open over the next few working days.

    Read this before offering a customer a time, so you offer one that exists. Slots
    run at 09:00, 11:00, 13:00, 15:00 and 17:00 on weekdays only.

    Args:
        days_ahead: How many working days to look ahead. Values above 10 are treated as
            10, since the diary is not open further out than that.

    Returns:
        A list of days, each with its date, the day of the week, and its five slots
        marked available or full. Also returns the total number of open slots and the
        earliest one, which is the answer to the question most customers are actually
        asking. Weekends are left out entirely rather than shown as closed.
    """
    horizon = max(1, min(10, days_ahead))
    booked = {f"{record['date']} {record['time']}" for record in CALLBACKS.values()}
    days = []
    for day in _next_business_days(horizon):
        slots = _slots_for(day, booked)
        days.append(
            {
                "date": day.isoformat(),
                "weekday": day.strftime("%A"),
                "open_count": sum(1 for slot in slots if slot["available"]),
                "slots": slots,
            }
        )
    open_slots = [slot for day in days for slot in day["slots"] if slot["available"]]
    return {
        "days_ahead": horizon,
        "open_slot_count": len(open_slots),
        "earliest": open_slots[0] if open_slots else None,
        "days": days,
    }


def book_callback(
    slot_date: str, slot_time: str, topic: str, phone: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Book a time for someone to ring the signed-in customer back.

    Offer this when a problem cannot be finished in the current conversation, rather
    than leaving the customer to ring again. Check `list_available_slots` first.

    Args:
        slot_date: The day, as an ISO date, for example "2026-09-07".
        slot_time: The time, in 24-hour form. One of "09:00", "11:00", "13:00", "15:00"
            or "17:00".
        topic: What the call is about. One of "order_problem", "return_help",
            "billing_question", "technical_support" or "complaint". Decides which team
            takes the call, so getting it right saves a transfer.
        phone: The number to ring, in the customer's own format.

    Returns:
        A `booked` flag, the callback reference, the confirmed date and time, the team
        that will call, and the window they will call within. Returns `booked: false`
        with a reason when the slot is already taken, when the time is not one of the
        five, when the date is in the past or a weekend, or when the topic is unknown.
    """
    key = topic.strip().lower()
    if key not in CALLBACK_TOPICS:
        return {"booked": False, "valid_topics": list(CALLBACK_TOPICS), "error": "unknown topic"}
    if slot_time not in SLOT_TIMES:
        return {"booked": False, "valid_times": list(SLOT_TIMES), "error": "unknown slot time"}
    try:
        day = date.fromisoformat(slot_date.strip())
    except ValueError:
        return {"booked": False, "detail": f"{slot_date!r} is not an ISO date like 2026-09-07"}
    if day <= _data.TODAY:
        return {"booked": False, "detail": "callbacks have to be booked at least a day ahead"}
    if day.weekday() >= 5:
        return {"booked": False, "detail": "callbacks run on weekdays only"}

    slot_key = f"{day.isoformat()} {slot_time}"
    taken = {f"{record['date']} {record['time']}" for record in CALLBACKS.values()}
    if slot_key in taken or _data.stable_int(slot_key, 0, 4) == 0:
        return {"booked": False, "date": day.isoformat(), "time": slot_time, "detail": "slot full"}

    reference = f"CB-{_data.stable_int(slot_key + key, 10000, 99999)}"
    CALLBACKS[reference] = {
        "reference": reference,
        "user_id": _data.current_user_id(tool_context),
        "date": day.isoformat(),
        "time": slot_time,
        "topic": key,
        "phone": phone.strip(),
        "cancelled": False,
    }
    return {
        "reference": reference,
        "booked": True,
        "date": day.isoformat(),
        "time": slot_time,
        "topic": key,
        "team": key.replace("_", " "),
        "calling_within_minutes": 30,
    }


def get_callback(reference: str) -> dict[str, Any]:
    """Look up a booked callback.

    Args:
        reference: The callback reference, for example "CB-40318". Issued by
            `book_callback` and starting with "CB-".

    Returns:
        The date, the time, the topic, the number to be rung with the middle digits
        masked, the team taking the call, whether it has been cancelled, and how many
        days away it is. Unlike the status tools elsewhere, this does not advance
        anything: reading a booking twice returns the same booking.
    """
    record = CALLBACKS.get(reference)
    if record is None:
        return _data.not_found("callback", reference)

    phone = record["phone"]
    masked = f"{phone[:3]}***{phone[-2:]}" if len(phone) > 5 else "on file"
    return {
        "reference": reference,
        "date": record["date"],
        "time": record["time"],
        "topic": record["topic"],
        "team": record["topic"].replace("_", " "),
        "phone": masked,
        "cancelled": record["cancelled"],
        "days_away": (date.fromisoformat(record["date"]) - _data.TODAY).days,
    }


def cancel_callback(reference: str, reason: str) -> dict[str, Any]:
    """Release a booked callback slot.

    Args:
        reference: The callback to cancel, for example "CB-40318".
        reason: Why it is no longer needed, in the customer's own words. Recorded so
            repeat cancellations on one topic can be looked at.

    Returns:
        A `cancelled` flag, the slot that has been released, and confirmation that the
        customer will not be rung. Returns `cancelled: false` when it was already
        cancelled, so a customer is not told twice that something has been undone.
    """
    record = CALLBACKS.get(reference)
    if record is None:
        return _data.not_found("callback", reference)
    if record["cancelled"]:
        return {
            "reference": reference,
            "cancelled": False,
            "detail": "this callback was already cancelled",
        }

    record["cancelled"] = True
    record["cancel_reason"] = reason.strip()
    return {
        "reference": reference,
        "cancelled": True,
        "slot_released": f"{record['date']} {record['time']}",
        "will_be_called": False,
    }


def reschedule_delivery(order_id: str, new_date: str, window: str) -> dict[str, Any]:
    """Move a delivery to a day and time window that suits the customer.

    Use this when a customer will not be in. It is cheaper than a failed delivery and a
    redelivery, so offer it before anyone suggests leaving a parcel with a neighbour.

    Args:
        order_id: The order to move, for example "A1001".
        new_date: The requested day as an ISO date, for example "2026-09-08". Must be a
            weekday and at least one day ahead.
        window: The time window. One of "morning" for 08:00 to 12:00, "afternoon" for
            12:00 to 17:00, or "evening" for 17:00 to 20:00.

    Returns:
        A `rescheduled` flag, the new date and window, the carrier, and whether a
        charge applies, which it does for a second change on the same order. Returns
        `rescheduled: false` when the order has already been delivered, since there is
        nothing left to move, and when the day requested is a weekend or in the past.
    """
    order = _data.ORDERS.get(order_id)
    if order is None:
        return _data.not_found("order", order_id)

    windows = {"morning": "08:00-12:00", "afternoon": "12:00-17:00", "evening": "17:00-20:00"}
    which = window.strip().lower()
    if which not in windows:
        return {"rescheduled": False, "valid_windows": sorted(windows), "error": "unknown window"}
    if _data.order_status(order) == "DELIVERED":
        return {"order_id": order_id, "rescheduled": False, "detail": "this order was delivered"}
    try:
        day = date.fromisoformat(new_date.strip())
    except ValueError:
        return {"rescheduled": False, "detail": f"{new_date!r} is not an ISO date like 2026-09-08"}
    if day <= _data.TODAY or day.weekday() >= 5:
        return {
            "order_id": order_id,
            "rescheduled": False,
            "detail": "pick a weekday at least one day ahead",
        }

    previous = DELIVERY_WINDOWS.get(order_id)
    changes = (previous["changes"] if previous else 0) + 1
    DELIVERY_WINDOWS[order_id] = {
        "order_id": order_id,
        "date": day.isoformat(),
        "window": which,
        "changes": changes,
    }
    shipment = _data.shipment_for_order(order_id)
    return {
        "order_id": order_id,
        "rescheduled": True,
        "date": day.isoformat(),
        "window": windows[which],
        "carrier": shipment["carrier"] if shipment else "UPS",
        "change_number": changes,
        "fee": 0.0 if changes == 1 else 5.00,
        "currency": _data.CURRENCY,
    }
