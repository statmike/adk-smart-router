"""Warranty and repair tools: cover, claims, service centres and appointments.

One of the twelve domains added in DESIGN.md 19.15. Warranty records are keyed by the
order and SKU they cover, so a claim can never point at something the customer did not
buy. Reaches into `_data` for orders and products only, and imports no other domain.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from . import _data

# How long each product category is covered for, in months.
COVER_MONTHS: dict[str, int] = {
    "electronics": 24,
    "home": 12,
    "apparel": 6,
    "accessories": 6,
}

REPAIR_STAGES: tuple[str, ...] = (
    "RECEIVED",
    "DIAGNOSING",
    "AWAITING_PARTS",
    "REPAIRING",
    "SHIPPED_BACK",
)


def _seed_service_centres() -> dict[str, dict[str, Any]]:
    return {
        "SC-DETROIT": {
            "centre_id": "SC-DETROIT",
            "name": "Detroit Service Centre",
            "city": "Detroit",
            "state": "MI",
            "postcode": "48226",
            "handles": ["electronics", "home"],
            "walk_ins": True,
        },
        "SC-CHICAGO": {
            "centre_id": "SC-CHICAGO",
            "name": "Chicago Service Centre",
            "city": "Chicago",
            "state": "IL",
            "postcode": "60601",
            "handles": ["electronics"],
            "walk_ins": False,
        },
        "SC-COLUMBUS": {
            "centre_id": "SC-COLUMBUS",
            "name": "Columbus Repair Depot",
            "city": "Columbus",
            "state": "OH",
            "postcode": "43215",
            "handles": ["electronics", "home", "accessories"],
            "walk_ins": True,
        },
    }


def _seed_registrations() -> dict[str, dict[str, Any]]:
    """One product already registered, so the "already covered" path has a fixture."""
    return {
        "A1001:SKU-HDP-04": {
            "order_id": "A1001",
            "sku": "SKU-HDP-04",
            "registered_on": "2026-08-25",
            "serial_number": "HDP-04-7731840",
        }
    }


SERVICE_CENTRES: dict[str, dict[str, Any]] = _seed_service_centres()
REGISTRATIONS: dict[str, dict[str, Any]] = _seed_registrations()
REPAIRS: dict[str, dict[str, Any]] = {}
APPOINTMENTS: dict[str, dict[str, Any]] = {}


def _reset() -> None:
    SERVICE_CENTRES.clear()
    SERVICE_CENTRES.update(_seed_service_centres())
    REGISTRATIONS.clear()
    REGISTRATIONS.update(_seed_registrations())
    REPAIRS.clear()
    APPOINTMENTS.clear()


_data.register_reset(_reset)


def _line_for(order_id: str, sku: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The order and the line on it, or (order, None) when the SKU is not on the order."""
    order = _data.ORDERS.get(order_id)
    if order is None:
        return None, None
    return order, next((item for item in order["items"] if item["sku"] == sku), None)


def get_warranty_status(order_id: str, sku: str) -> dict[str, Any]:
    """Check whether a purchased item is still under warranty.

    Use this first whenever the customer says something has broken, stopped working or
    developed a fault, before offering a repair or a replacement. Cover length depends
    on the product category, not on the price paid.

    Args:
        order_id: The order the item was bought on, for example "A1001".
        sku: The product code of the item, for example "SKU-HDP-04".

    Returns:
        Whether the item is covered, the cover length in months for its category, the
        purchase date, the expiry date, how many days of cover remain, and whether the
        item has been registered for extended cover. A negative days-remaining value
        means cover has lapsed and the answer is a paid repair quote rather than a
        claim. Returns an error field when the SKU is not on that order.
    """
    order, line = _line_for(order_id, sku)
    if order is None:
        return _data.not_found("order", order_id)
    if line is None:
        return {"order_id": order_id, "sku": sku, "error": f"{sku} is not on order {order_id}"}

    category = _data.PRODUCTS.get(sku, {}).get("category", "accessories")
    months = COVER_MONTHS.get(category, 6)
    purchased = date.fromisoformat(order["placed_on"])
    expires = purchased + timedelta(days=months * 30)
    remaining = (expires - _data.TODAY).days
    return {
        "order_id": order_id,
        "sku": sku,
        "category": category,
        "covered": remaining >= 0,
        "cover_months": months,
        "purchased_on": order["placed_on"],
        "expires_on": str(expires),
        "days_remaining": remaining,
        "registered": f"{order_id}:{sku}" in REGISTRATIONS,
    }


def register_warranty(order_id: str, sku: str, serial_number: str) -> dict[str, Any]:
    """Register a purchased item for its extended warranty.

    Registration is optional but adds three months of cover, so offer it whenever a
    customer asks about protecting a recent purchase.

    Args:
        order_id: The order the item was bought on, for example "A1002".
        sku: The product code being registered, for example "SKU-LMP-03".
        serial_number: The serial number printed on the item or its box. Pass an empty
            string if the customer cannot find one, and a placeholder is issued.

    Returns:
        A `registered` flag, the serial number recorded, the new expiry date and the
        extra months added. Returns `registered: false` when the item is already
        registered, along with the date it was registered on.
    """
    order, line = _line_for(order_id, sku)
    if order is None:
        return _data.not_found("order", order_id)
    if line is None:
        return {"order_id": order_id, "sku": sku, "error": f"{sku} is not on order {order_id}"}

    key = f"{order_id}:{sku}"
    existing = REGISTRATIONS.get(key)
    if existing is not None:
        return {
            "order_id": order_id,
            "sku": sku,
            "registered": False,
            "detail": f"already registered on {existing['registered_on']}",
            "serial_number": existing["serial_number"],
        }

    serial = serial_number.strip() or f"{sku}-{_data.stable_int(key, 1000000, 9999999)}"
    REGISTRATIONS[key] = {
        "order_id": order_id,
        "sku": sku,
        "registered_on": str(_data.TODAY),
        "serial_number": serial,
    }
    status = get_warranty_status(order_id, sku)
    return {
        "order_id": order_id,
        "sku": sku,
        "registered": True,
        "serial_number": serial,
        "extra_months": 3,
        "expires_on": status["expires_on"],
    }


def start_repair_request(order_id: str, sku: str, fault_description: str) -> dict[str, Any]:
    """Open a repair for a faulty item the customer already owns.

    Check `get_warranty_status` first. A covered item is repaired free; an item out of
    cover gets a quote before any work starts, and the customer can decline it.

    Args:
        order_id: The order the faulty item was bought on, for example "A1001".
        sku: The product code of the faulty item, for example "SKU-HDP-04".
        fault_description: What is wrong, in the customer's own words. Passed to the
            technician, so keep the customer's detail rather than summarising it away.

    Returns:
        The new repair id, whether the work is covered by warranty, the quoted cost if
        it is not, the service centre the item is going to, and the postage label
        status. Returns `started: false` when a repair is already open for that item,
        with the existing repair id so it can be looked up instead.
    """
    order, line = _line_for(order_id, sku)
    if order is None:
        return _data.not_found("order", order_id)
    if line is None:
        return {"order_id": order_id, "sku": sku, "error": f"{sku} is not on order {order_id}"}

    key = f"{order_id}:{sku}"
    open_repair = next(
        (r for r in REPAIRS.values() if r["key"] == key and r["stage"] != "SHIPPED_BACK"), None
    )
    if open_repair is not None:
        return {
            "order_id": order_id,
            "sku": sku,
            "started": False,
            "repair_id": open_repair["repair_id"],
            "detail": "a repair is already open for this item",
        }

    status = get_warranty_status(order_id, sku)
    covered = bool(status["covered"])
    repair_id = f"RPR-{_data.stable_int(key + fault_description, 10000, 99999)}"
    centre = "SC-COLUMBUS" if status["category"] != "electronics" else "SC-DETROIT"
    REPAIRS[repair_id] = {
        "repair_id": repair_id,
        "key": key,
        "order_id": order_id,
        "sku": sku,
        "fault": fault_description,
        "stage": "RECEIVED",
        "covered": covered,
        "quote": 0.0 if covered else round(line["unit_price"] * 0.35, 2),
        "centre_id": centre,
        "opened_on": str(_data.TODAY),
    }
    return {
        "repair_id": repair_id,
        "started": True,
        "covered": covered,
        "quote": REPAIRS[repair_id]["quote"],
        "currency": _data.CURRENCY,
        "centre_id": centre,
        "postage_label": "emailed",
        "next_step": "print the label and post the item within 14 days",
    }


def get_repair_status(repair_id: str) -> dict[str, Any]:
    """Check where a repair has got to.

    Args:
        repair_id: The repair identifier, for example "RPR-40318". These start with
            "RPR-" and are issued by `start_repair_request`.

    Returns:
        The current stage, which runs RECEIVED, DIAGNOSING, AWAITING_PARTS, REPAIRING
        then SHIPPED_BACK; the item being repaired; whether it is covered; the quote if
        it is not; the service centre holding it; and the estimated return date. Each
        call advances the repair one stage, mirroring the polling behaviour of
        `get_order_status`.
    """
    record = REPAIRS.get(repair_id)
    if record is None:
        return _data.not_found("repair", repair_id)

    index = REPAIR_STAGES.index(record["stage"])
    if index < len(REPAIR_STAGES) - 1:
        record["stage"] = REPAIR_STAGES[index + 1]
    return {
        "repair_id": repair_id,
        "order_id": record["order_id"],
        "sku": record["sku"],
        "stage": record["stage"],
        "covered": record["covered"],
        "quote": record["quote"],
        "currency": _data.CURRENCY,
        "centre_id": record["centre_id"],
        "estimated_return_on": _data.business_days_from_today(
            len(REPAIR_STAGES) - REPAIR_STAGES.index(record["stage"])
        ),
    }


def list_service_centres(category: str, state: str) -> dict[str, Any]:
    """Find service centres that can handle a category of product.

    Use this when a customer would rather bring an item in than post it.

    Args:
        category: The product category needing service. One of "electronics", "home",
            "apparel" or "accessories".
        state: A two-letter US state code to narrow the list, for example "MI". Pass an
            empty string to list every centre that handles the category.

    Returns:
        A count and a list of centres, each with its id, name, city, state, postcode,
        the categories it handles and whether it takes walk-ins. Centres that do not
        take walk-ins need an appointment booked with `book_repair_appointment`.
        Returns an empty list when nothing matches, which is not an error.
    """
    wanted_category = category.strip().lower()
    wanted_state = state.strip().upper()
    matches = [
        dict(centre)
        for centre in SERVICE_CENTRES.values()
        if wanted_category in centre["handles"]
        and (not wanted_state or centre["state"] == wanted_state)
    ]
    return {
        "count": len(matches),
        "category": wanted_category,
        "state": wanted_state or "any",
        "centres": matches,
    }


def book_repair_appointment(repair_id: str, centre_id: str, preferred_date: str) -> dict[str, Any]:
    """Book a slot to bring a repair in to a service centre.

    Args:
        repair_id: The open repair the appointment is for, for example "RPR-40318".
        centre_id: The service centre to visit, for example "SC-DETROIT". Get valid
            ids from `list_service_centres`.
        preferred_date: The ISO date the customer would like, for example
            "2026-09-15". If the centre is full that day the next working day is
            offered instead.

    Returns:
        A `booked` flag, the appointment reference, the confirmed date and a time
        window, the centre's address, and whether the requested date was moved.
        Returns `booked: false` when the centre does not handle that item's category,
        naming the centres that do.
    """
    record = REPAIRS.get(repair_id)
    if record is None:
        return _data.not_found("repair", repair_id)
    centre = SERVICE_CENTRES.get(centre_id)
    if centre is None:
        return _data.not_found("service centre", centre_id)

    category = _data.PRODUCTS.get(record["sku"], {}).get("category", "accessories")
    if category not in centre["handles"]:
        able = [c["centre_id"] for c in SERVICE_CENTRES.values() if category in c["handles"]]
        return {
            "repair_id": repair_id,
            "booked": False,
            "detail": f"{centre['name']} does not handle {category}",
            "centres_that_do": able,
        }

    slot = _data.stable_int(repair_id + centre_id, 0, 3)
    confirmed = preferred_date.strip() or _data.business_days_from_today(3)
    moved = slot == 3
    if moved:
        confirmed = _data.business_days_from_today(4)
    reference = f"APT-{_data.stable_int(repair_id + confirmed, 1000, 9999)}"
    APPOINTMENTS[reference] = {
        "reference": reference,
        "repair_id": repair_id,
        "centre_id": centre_id,
        "date": confirmed,
    }
    return {
        "repair_id": repair_id,
        "booked": True,
        "reference": reference,
        "centre_id": centre_id,
        "date": confirmed,
        "time_window": ["09:00-11:00", "11:00-13:00", "13:00-15:00", "15:00-17:00"][slot],
        "address": f"{centre['city']}, {centre['state']} {centre['postcode']}",
        "date_moved": moved,
    }


def get_warranty_terms(category: str) -> dict[str, Any]:
    """Read the warranty terms for a product category.

    This is near-static reference text. Use it for questions about what the warranty
    covers in general, rather than about one customer's item.

    Args:
        category: The product category. One of "electronics", "home", "apparel" or
            "accessories".

    Returns:
        The cover length in months, what is included, what is excluded, whether
        registration extends the cover, and how a claim is made. Returns the list of
        known categories alongside an error field when the category is not one of
        them.
    """
    key = category.strip().lower()
    months = COVER_MONTHS.get(key)
    if months is None:
        return {
            "category": key,
            "known_categories": sorted(COVER_MONTHS),
            "error": f"no warranty terms for category {key!r}",
        }
    return {
        "category": key,
        "cover_months": months,
        "includes": "manufacturing faults, component failure under normal use",
        "excludes": "accidental damage, water damage, wear to finishes, consumable parts",
        "registration_bonus_months": 3,
        "claim_route": "start_repair_request, then post the item with the emailed label",
    }
