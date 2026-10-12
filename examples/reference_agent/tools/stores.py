"""Physical store tools: find one, check it, hold stock, collect, cancel.

One of the twelve domains added in DESIGN.md 19.15. Stores own their own inventory
table keyed by store and SKU, so in-store stock can honestly disagree with the online
stock in `_data.PRODUCTS`. That disagreement is the point: it is what a customer rings
up about.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from . import _data

PICKUP_STAGES: tuple[str, ...] = ("RESERVED", "PICKING", "READY", "COLLECTED")


def _seed_stores() -> dict[str, dict[str, Any]]:
    return {
        "ST-A2": {
            "store_id": "ST-A2",
            "name": "Ann Arbor Main Street",
            "city": "Ann Arbor",
            "state": "MI",
            "postcode": "48104",
            "opens": "09:00",
            "closes": "20:00",
            "sunday_hours": "11:00-17:00",
            "pickup": True,
        },
        "ST-DT": {
            "store_id": "ST-DT",
            "name": "Detroit Riverfront",
            "city": "Detroit",
            "state": "MI",
            "postcode": "48226",
            "opens": "10:00",
            "closes": "21:00",
            "sunday_hours": "closed",
            "pickup": True,
        },
        "ST-CH": {
            "store_id": "ST-CH",
            "name": "Chicago Loop",
            "city": "Chicago",
            "state": "IL",
            "postcode": "60601",
            "opens": "08:00",
            "closes": "22:00",
            "sunday_hours": "10:00-18:00",
            "pickup": False,
        },
    }


def _seed_inventory() -> dict[str, int]:
    """In-store stock per store and SKU, deliberately unlike the online figures."""
    return {
        f"{store}:{sku}": _data.stable_int(store + sku, 0, 9)
        for store in ("ST-A2", "ST-DT", "ST-CH")
        for sku in _data.PRODUCTS
    }


STORES: dict[str, dict[str, Any]] = _seed_stores()
STORE_STOCK: dict[str, int] = _seed_inventory()
RESERVATIONS: dict[str, dict[str, Any]] = {}


def _reset() -> None:
    STORES.clear()
    STORES.update(_seed_stores())
    STORE_STOCK.clear()
    STORE_STOCK.update(_seed_inventory())
    RESERVATIONS.clear()


_data.register_reset(_reset)


def find_stores_nearby(postcode: str, radius_miles: int) -> dict[str, Any]:
    """Find stores near a postcode.

    Use this when a customer would rather collect than wait for delivery, or asks
    whether there is a shop near them.

    Args:
        postcode: The customer's postcode, for example "48104". US five-digit codes.
        radius_miles: How far the customer is willing to travel, in miles. Values
            above 100 are treated as 100.

    Returns:
        A count and a list of stores ordered by distance. Each carries its store id,
        name, city, state, postcode, distance in miles and whether it offers collection.
        Distance is derived from the postcodes rather than measured, so treat it as
        approximate and say so if the customer is deciding on a few miles. Returns an
        empty list when nothing is in range.
    """
    code = "".join(ch for ch in postcode if ch.isdigit())
    if len(code) != 5:
        return {"error": f"{postcode!r} is not a five-digit postcode"}

    radius = max(1, min(100, radius_miles))
    found = []
    for store in STORES.values():
        distance = abs(int(store["postcode"]) - int(code)) % 97
        if distance <= radius:
            found.append(
                {
                    "store_id": store["store_id"],
                    "name": store["name"],
                    "city": store["city"],
                    "state": store["state"],
                    "postcode": store["postcode"],
                    "distance_miles": distance,
                    "pickup": store["pickup"],
                }
            )
    found.sort(key=lambda entry: entry["distance_miles"])
    return {"postcode": code, "radius_miles": radius, "count": len(found), "stores": found}


def get_store_details(store_id: str) -> dict[str, Any]:
    """Get everything known about one store.

    Args:
        store_id: The store identifier, for example "ST-A2". Get valid ids from
            `find_stores_nearby`.

    Returns:
        The store's name, full address, weekday opening and closing times, Sunday
        hours, whether it offers collection, and the services it runs. Use
        `get_store_hours` instead when the customer only wants to know if it is open.
    """
    store = STORES.get(store_id)
    if store is None:
        return _data.not_found("store", store_id)
    return {
        **store,
        "address": f"{store['name']}, {store['city']}, {store['state']} {store['postcode']}",
        "services": ["collection", "returns", "gift wrapping"] if store["pickup"] else ["returns"],
    }


def get_store_hours(store_id: str, day: str) -> dict[str, Any]:
    """Check when a store is open on a given day.

    Args:
        store_id: The store to check, for example "ST-DT".
        day: The day of the week in lower case, for example "sunday". Pass an empty
            string for the standard weekday hours.

    Returns:
        The opening and closing times for that day, whether the store is open at all,
        and a note when the day's hours differ from the weekday pattern. Sunday is the
        day that usually differs, and one store is closed on Sundays entirely.
    """
    store = STORES.get(store_id)
    if store is None:
        return _data.not_found("store", store_id)

    which = day.strip().lower()
    if which == "sunday":
        hours = store["sunday_hours"]
        if hours == "closed":
            return {"store_id": store_id, "day": which, "open": False, "hours": "closed"}
        opens, _, closes = hours.partition("-")
        return {
            "store_id": store_id,
            "day": which,
            "open": True,
            "opens": opens,
            "closes": closes,
            "note": "Sunday hours are shorter than the rest of the week",
        }
    return {
        "store_id": store_id,
        "day": which or "weekday",
        "open": True,
        "opens": store["opens"],
        "closes": store["closes"],
    }


def check_in_store_availability(store_id: str, sku: str) -> dict[str, Any]:
    """Check whether a store has a product on the shelf right now.

    In-store stock is counted separately from online stock, so an item that is sold
    out online can still be in a shop and the other way round. Always check here
    before promising a customer they can collect something today.

    Args:
        store_id: The store to check, for example "ST-A2".
        sku: The product code, for example "SKU-MUG-02".

    Returns:
        The units on the shelf, whether the item can be reserved for collection, the
        online stock for comparison, and the product's name and price. When the store
        has none, the nearest stores that do are listed so the customer has somewhere
        to go. Returns an error field when the store or the product is unknown.
    """
    store = STORES.get(store_id)
    if store is None:
        return _data.not_found("store", store_id)
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)

    units = STORE_STOCK.get(f"{store_id}:{sku}", 0)
    elsewhere = [
        other for other in STORES if other != store_id and STORE_STOCK.get(f"{other}:{sku}", 0) > 0
    ]
    return {
        "store_id": store_id,
        "sku": sku,
        "name": product["name"],
        "unit_price": product["unit_price"],
        "currency": _data.CURRENCY,
        "units_in_store": units,
        "reservable": units > 0 and store["pickup"],
        "online_stock": product["stock"],
        "also_in_stock_at": elsewhere,
    }


def reserve_for_pickup(
    store_id: str, sku: str, quantity: int, tool_context: ToolContext
) -> dict[str, Any]:
    """Hold stock at a store for the signed-in customer to collect.

    A reservation holds the item for 48 hours and is not a purchase, so the customer
    pays in store. Check `check_in_store_availability` first.

    Args:
        store_id: The store to hold the item at, for example "ST-A2".
        sku: The product to hold, for example "SKU-CBL-06".
        quantity: How many units to hold. Must be at least 1 and no more than the
            store has on the shelf.

    Returns:
        A `reserved` flag, the reservation id, the collection deadline, what the
        customer will pay in store, and where in the store to collect. Returns
        `reserved: false` with a reason when the store does not offer collection, when
        it has fewer units than asked for, or when the quantity is not positive.
    """
    store = STORES.get(store_id)
    if store is None:
        return _data.not_found("store", store_id)
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)
    if not store["pickup"]:
        return {"store_id": store_id, "reserved": False, "detail": "this store has no collection"}
    if quantity < 1:
        return {"store_id": store_id, "reserved": False, "detail": "quantity must be at least 1"}

    key = f"{store_id}:{sku}"
    units = STORE_STOCK.get(key, 0)
    if quantity > units:
        return {
            "store_id": store_id,
            "sku": sku,
            "reserved": False,
            "units_in_store": units,
            "detail": f"the store only has {units} on the shelf",
        }

    STORE_STOCK[key] = units - quantity
    reservation_id = f"PU-{_data.stable_int(key + str(quantity), 10000, 99999)}"
    RESERVATIONS[reservation_id] = {
        "reservation_id": reservation_id,
        "user_id": _data.current_user_id(tool_context),
        "store_id": store_id,
        "sku": sku,
        "quantity": quantity,
        "stage": "RESERVED",
        "reserved_on": str(_data.TODAY),
    }
    return {
        "reservation_id": reservation_id,
        "reserved": True,
        "store_id": store_id,
        "quantity": quantity,
        "pay_in_store": round(product["unit_price"] * quantity, 2),
        "currency": _data.CURRENCY,
        "collect_by": _data.business_days_from_today(2),
        "collect_from": "the collection desk near the entrance",
    }


def get_pickup_status(reservation_id: str) -> dict[str, Any]:
    """Check whether a collection order is ready yet.

    Args:
        reservation_id: The reservation identifier, for example "PU-40318". Issued by
            `reserve_for_pickup` and starting with "PU-".

    Returns:
        The current stage, which runs RESERVED, PICKING, READY then COLLECTED; the
        store; the item and quantity; and the deadline for collection. Each call
        advances the reservation one stage, so a customer chasing it will see it move
        towards READY.
    """
    record = RESERVATIONS.get(reservation_id)
    if record is None:
        return _data.not_found("pickup reservation", reservation_id)

    index = PICKUP_STAGES.index(record["stage"])
    if index < len(PICKUP_STAGES) - 1:
        record["stage"] = PICKUP_STAGES[index + 1]
    return {
        "reservation_id": reservation_id,
        "store_id": record["store_id"],
        "sku": record["sku"],
        "quantity": record["quantity"],
        "stage": record["stage"],
        "ready": record["stage"] in {"READY", "COLLECTED"},
        "collect_by": _data.business_days_from_today(2),
    }


def cancel_pickup_reservation(reservation_id: str, reason: str) -> dict[str, Any]:
    """Release a held item back to the shop floor.

    Args:
        reservation_id: The reservation to cancel, for example "PU-40318".
        reason: Why the customer is not collecting, in their own words. Recorded so
            stores can see whether holds are being wasted.

    Returns:
        A `cancelled` flag, the units released back to the shelf and the store's new
        shelf count. Returns `cancelled: false` when the item has already been
        collected, since there is nothing left to release.
    """
    record = RESERVATIONS.get(reservation_id)
    if record is None:
        return _data.not_found("pickup reservation", reservation_id)
    if record["stage"] == "COLLECTED":
        return {
            "reservation_id": reservation_id,
            "cancelled": False,
            "detail": "this reservation was already collected",
        }

    key = f"{record['store_id']}:{record['sku']}"
    STORE_STOCK[key] = STORE_STOCK.get(key, 0) + record["quantity"]
    record["stage"] = "COLLECTED"
    record["cancel_reason"] = reason
    return {
        "reservation_id": reservation_id,
        "cancelled": True,
        "units_released": record["quantity"],
        "units_in_store": STORE_STOCK[key],
    }
