"""The single in-memory fixture behind all 25 stub tools (DESIGN.md 3.2).

One module owns the data so orders, shipments, returns, invoices and accounts stay
mutually consistent: a return always points at a real order, a tracking number always
belongs to the order whose status it mirrors.

Everything here is deterministic and offline. No network, no database, no clock, no
`random`. Where a tool needs a plausible-looking number it comes from `stable_int`,
which is a hash of the arguments, so the same call always yields the same answer and
the generated corpus can be regenerated for free.

The two mutable things are intentional:

* write tools mutate these dicts, which is what DESIGN.md 8.4 (read-your-own-writes)
  needs in order to be demonstrated end to end;
* `get_order_status` advances an order one stage per call, which is the poller trap
  from DESIGN.md 8.6.

Call `reset_fixtures()` between tests to get back to the seeded state.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from google.adk.tools import ToolContext

# A frozen "today". Real clocks would make the corpus irreproducible.
TODAY: date = date(2026, 8, 31)

# Identity is read from session state, never from an argument, for the ambient-identity
# tools in DESIGN.md 8.1. The default only applies when a runner forgot to seed state.
USER_ID_STATE_KEY = "user_id"
DEFAULT_USER_ID = "u_1001"

# `get_order_status` walks an order one step down this list per call.
ORDER_STATUS_FLOW: tuple[str, ...] = (
    "PLACED",
    "PACKED",
    "IN_TRANSIT",
    "OUT_FOR_DELIVERY",
    "DELIVERED",
)

# Shipment status is derived from the order, never advanced on its own, so the two
# views of a parcel can never disagree.
SHIPMENT_STATUS: dict[str, str] = {
    "PLACED": "LABEL_CREATED",
    "PACKED": "LABEL_CREATED",
    "IN_TRANSIT": "IN_TRANSIT",
    "OUT_FOR_DELIVERY": "OUT_FOR_DELIVERY",
    "DELIVERED": "DELIVERED",
}

CURRENCY = "USD"


# --- Seeds --------------------------------------------------------------------------
# Each seed function returns fresh objects so `reset_fixtures()` cannot alias mutated
# state back into the "pristine" copy.


def _seed_accounts() -> dict[str, dict[str, Any]]:
    return {
        "u_1001": {
            "user_id": "u_1001",
            "name": "Ada Ellis",
            "email": "ada@example.com",
            "tier": "gold",
            "member_since": "2021-03-14",
            "loyalty_points": 1240,
            "store_credit": 500.00,
            "addresses": [
                {
                    "label": "home",
                    "line1": "18 Prospect Row",
                    "city": "Ann Arbor",
                    "region": "MI",
                    "postal_code": "48104",
                    "country": "US",
                    "default": True,
                },
                {
                    "label": "work",
                    "line1": "900 Victors Way, Suite 200",
                    "city": "Ann Arbor",
                    "region": "MI",
                    "postal_code": "48108",
                    "country": "US",
                    "default": False,
                },
            ],
            "payment_methods": [
                {
                    "method_id": "pm_visa_4242",
                    "brand": "visa",
                    "last4": "4242",
                    "expires": "2029-04",
                    "default": True,
                    "expired": False,
                },
                {
                    "method_id": "pm_amex_0005",
                    "brand": "amex",
                    "last4": "0005",
                    "expires": "2024-11",
                    "default": False,
                    "expired": True,
                },
            ],
        },
        "u_1002": {
            "user_id": "u_1002",
            "name": "Bo Nguyen",
            "email": "bo@example.com",
            "tier": "standard",
            "member_since": "2025-11-02",
            "loyalty_points": 85,
            "store_credit": 40.00,
            "addresses": [
                {
                    "label": "home",
                    "line1": "4471 Cedar Bend",
                    "city": "Austin",
                    "region": "TX",
                    "postal_code": "78745",
                    "country": "US",
                    "default": True,
                }
            ],
            "payment_methods": [
                {
                    "method_id": "pm_mc_7781",
                    "brand": "mastercard",
                    "last4": "7781",
                    "expires": "2028-01",
                    "default": True,
                    "expired": False,
                }
            ],
        },
    }


def _seed_products() -> dict[str, dict[str, Any]]:
    return {
        "SKU-TEE-01": {
            "sku": "SKU-TEE-01",
            "name": "Cotton crew tee",
            "category": "apparel",
            "unit_price": 24.00,
            "stock": 42,
            "keywords": ["tee", "shirt", "cotton", "apparel"],
        },
        "SKU-MUG-02": {
            "sku": "SKU-MUG-02",
            "name": "Ceramic mug, 12oz",
            "category": "home",
            "unit_price": 12.50,
            "stock": 0,
            "keywords": ["mug", "cup", "ceramic", "kitchen"],
        },
        "SKU-LMP-03": {
            "sku": "SKU-LMP-03",
            "name": "Desk lamp, warm LED",
            "category": "home",
            "unit_price": 59.00,
            "stock": 7,
            "keywords": ["lamp", "light", "desk", "led"],
        },
        "SKU-HDP-04": {
            "sku": "SKU-HDP-04",
            "name": "Over-ear headphones",
            "category": "electronics",
            "unit_price": 149.00,
            "stock": 3,
            "keywords": ["headphones", "audio", "electronics", "bluetooth"],
        },
        "SKU-BAG-05": {
            "sku": "SKU-BAG-05",
            "name": "Canvas tote bag",
            "category": "accessories",
            "unit_price": 32.00,
            "stock": 18,
            "keywords": ["bag", "tote", "canvas"],
        },
        "SKU-CBL-06": {
            "sku": "SKU-CBL-06",
            "name": "USB-C cable, 2m",
            "category": "electronics",
            "unit_price": 9.99,
            "stock": 250,
            "keywords": ["cable", "usb", "charger", "electronics"],
        },
    }


def _seed_orders() -> dict[str, dict[str, Any]]:
    return {
        "A1001": {
            "order_id": "A1001",
            "user_id": "u_1001",
            "placed_on": "2026-08-24",
            "status_index": 2,  # IN_TRANSIT
            "items": [{"sku": "SKU-HDP-04", "quantity": 1, "unit_price": 149.00}],
            "total": 149.00,
            "shipping_address": "18 Prospect Row, Ann Arbor, MI 48104, US",
            "shipping_speed": "standard",
            "cancelled": False,
        },
        "A1002": {
            "order_id": "A1002",
            "user_id": "u_1001",
            "placed_on": "2026-08-11",
            "status_index": 4,  # DELIVERED
            "items": [
                {"sku": "SKU-TEE-01", "quantity": 2, "unit_price": 24.00},
                {"sku": "SKU-MUG-02", "quantity": 1, "unit_price": 12.50},
            ],
            "total": 60.50,
            "shipping_address": "18 Prospect Row, Ann Arbor, MI 48104, US",
            "shipping_speed": "standard",
            "cancelled": False,
        },
        "A1003": {
            "order_id": "A1003",
            "user_id": "u_1001",
            "placed_on": "2026-08-30",
            "status_index": 0,  # PLACED
            "items": [{"sku": "SKU-LMP-03", "quantity": 1, "unit_price": 59.00}],
            "total": 59.00,
            "shipping_address": "900 Victors Way, Suite 200, Ann Arbor, MI 48108, US",
            "shipping_speed": "express",
            "cancelled": False,
        },
        "B2001": {
            "order_id": "B2001",
            "user_id": "u_1002",
            "placed_on": "2026-08-28",
            "status_index": 1,  # PACKED
            "items": [{"sku": "SKU-BAG-05", "quantity": 1, "unit_price": 32.00}],
            "total": 32.00,
            "shipping_address": "4471 Cedar Bend, Austin, TX 78745, US",
            "shipping_speed": "standard",
            "cancelled": False,
        },
        "B2002": {
            "order_id": "B2002",
            "user_id": "u_1002",
            "placed_on": "2026-07-19",
            "status_index": 4,  # DELIVERED
            "items": [{"sku": "SKU-CBL-06", "quantity": 3, "unit_price": 9.99}],
            "total": 29.97,
            "shipping_address": "4471 Cedar Bend, Austin, TX 78745, US",
            "shipping_speed": "standard",
            "cancelled": False,
        },
    }


def _seed_shipments() -> dict[str, dict[str, Any]]:
    return {
        "1Z-A1001": {
            "tracking_number": "1Z-A1001",
            "order_id": "A1001",
            "carrier": "UPS",
            "shipped_on": "2026-08-26",
            "eta": "2026-09-01",
            "last_scan": "Louisville, KY",
        },
        "1Z-A1002": {
            "tracking_number": "1Z-A1002",
            "order_id": "A1002",
            "carrier": "UPS",
            "shipped_on": "2026-08-12",
            "eta": "2026-08-15",
            "last_scan": "Ann Arbor, MI",
        },
        "1Z-A1003": {
            "tracking_number": "1Z-A1003",
            "order_id": "A1003",
            "carrier": "FedEx",
            "shipped_on": "",
            "eta": "2026-09-02",
            "last_scan": "",
        },
        "1Z-B2001": {
            "tracking_number": "1Z-B2001",
            "order_id": "B2001",
            "carrier": "USPS",
            "shipped_on": "",
            "eta": "2026-09-03",
            "last_scan": "",
        },
        "1Z-B2002": {
            "tracking_number": "1Z-B2002",
            "order_id": "B2002",
            "carrier": "USPS",
            "shipped_on": "2026-07-20",
            "eta": "2026-07-24",
            "last_scan": "Austin, TX",
        },
    }


def _seed_returns() -> dict[str, dict[str, Any]]:
    return {
        "RMA-A1002": {
            "return_id": "RMA-A1002",
            "order_id": "A1002",
            "user_id": "u_1001",
            "sku": "SKU-MUG-02",
            "reason": "arrived chipped",
            "status": "REFUND_ISSUED",
            "opened_on": "2026-08-16",
            "refund_amount": 12.50,
        }
    }


def _seed_invoices() -> dict[str, dict[str, Any]]:
    return {
        order_id: {
            "invoice_id": f"INV-{order_id}",
            "order_id": order_id,
            "user_id": order["user_id"],
            "issued_on": order["placed_on"],
            "subtotal": order["total"],
            "tax": round(order["total"] * 0.06, 2),
            "total": round(order["total"] * 1.06, 2),
            "currency": CURRENCY,
        }
        for order_id, order in _seed_orders().items()
    }


def _seed_faqs() -> dict[str, str]:
    return {
        "returns": (
            "Most items can be returned within 30 days of delivery in original condition. "
            "Start a return from your order and we email a prepaid label."
        ),
        "shipping": (
            "Standard shipping is 3-5 business days and free over $50. Express is 1-2 "
            "business days. We ship to the US and Canada."
        ),
        "payments": (
            "We accept Visa, Mastercard, Amex and store credit. Cards are charged when the "
            "order ships, not when it is placed."
        ),
        "account": (
            "You can change your email, addresses and payment methods at any time. Loyalty "
            "points are earned at 1 point per dollar and never expire."
        ),
        "warranty": (
            "Electronics carry a 1 year manufacturer warranty. Apparel and home goods are "
            "covered against defects for 90 days."
        ),
    }


def _seed_return_policies() -> dict[str, dict[str, Any]]:
    return {
        "general": {
            "window_days": 30,
            "restocking_fee_pct": 0,
            "condition": "unused, original packaging",
            "refund_method": "original payment method",
        },
        "apparel": {
            "window_days": 30,
            "restocking_fee_pct": 0,
            "condition": "unworn, tags attached",
            "refund_method": "original payment method",
        },
        "electronics": {
            "window_days": 14,
            "restocking_fee_pct": 10,
            "condition": "all accessories included",
            "refund_method": "original payment method",
        },
        "final_sale": {
            "window_days": 0,
            "restocking_fee_pct": 0,
            "condition": "not returnable",
            "refund_method": "none",
        },
    }


# --- Live fixture state -------------------------------------------------------------

ACCOUNTS: dict[str, dict[str, Any]] = _seed_accounts()
PRODUCTS: dict[str, dict[str, Any]] = _seed_products()
ORDERS: dict[str, dict[str, Any]] = _seed_orders()
SHIPMENTS: dict[str, dict[str, Any]] = _seed_shipments()
RETURNS: dict[str, dict[str, Any]] = _seed_returns()
INVOICES: dict[str, dict[str, Any]] = _seed_invoices()
FAQS: dict[str, str] = _seed_faqs()
RETURN_POLICIES: dict[str, dict[str, Any]] = _seed_return_policies()
TICKETS: list[dict[str, Any]] = []

# Each domain module added in DESIGN.md 19.15 owns its own fixture and registers a
# function that reseeds it. The dependency runs one way: domain modules import this
# one, this one never imports them, and `reset_fixtures` still resets everything.
_DOMAIN_RESETS: list[Callable[[], None]] = []


def register_reset(reset: Callable[[], None]) -> None:
    """Add a domain module's reseed function to `reset_fixtures`. Called at import."""
    _DOMAIN_RESETS.append(reset)


def reset_fixtures() -> None:
    """Restore every fixture to its seeded state. For tests and corpus reruns."""
    for reset in _DOMAIN_RESETS:
        reset()
    for live, seed in (
        (ACCOUNTS, _seed_accounts()),
        (PRODUCTS, _seed_products()),
        (ORDERS, _seed_orders()),
        (SHIPMENTS, _seed_shipments()),
        (RETURNS, _seed_returns()),
        (INVOICES, _seed_invoices()),
        (FAQS, _seed_faqs()),
        (RETURN_POLICIES, _seed_return_policies()),
    ):
        live.clear()
        live.update(seed)  # type: ignore[arg-type]
    TICKETS.clear()


# --- Helpers ------------------------------------------------------------------------


def stable_int(seed: str, low: int, high: int) -> int:
    """A repeatable pseudo-random int in [low, high] derived only from `seed`.

    Used instead of `random` so a tool's "plausible" numbers depend on its arguments
    and nothing else.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return low + int.from_bytes(digest[:8], "big") % (high - low + 1)


def current_user_id(tool_context: ToolContext) -> str:
    """Read the caller's identity from session state (DESIGN.md 8.1, ambient identity)."""
    return str(tool_context.state.get(USER_ID_STATE_KEY) or DEFAULT_USER_ID)


def account_for(tool_context: ToolContext) -> dict[str, Any]:
    """The account record of the current user, created empty if state names an unknown user."""
    user_id = current_user_id(tool_context)
    return ACCOUNTS.setdefault(
        user_id,
        {
            "user_id": user_id,
            "name": "Unknown",
            "email": "",
            "tier": "standard",
            "member_since": str(TODAY),
            "loyalty_points": 0,
            "store_credit": 0.00,
            "addresses": [],
            "payment_methods": [],
        },
    )


def order_status(order: dict[str, Any]) -> str:
    """The order's current stage. Does not advance it."""
    if order.get("cancelled"):
        return "CANCELLED"
    return ORDER_STATUS_FLOW[order["status_index"]]


def shipment_for_order(order_id: str) -> dict[str, Any] | None:
    """The shipment record attached to an order, if one exists."""
    return next((s for s in SHIPMENTS.values() if s["order_id"] == order_id), None)


def not_found(kind: str, identifier: str) -> dict[str, Any]:
    """The one shape every tool uses to report a bad identifier."""
    return {"error": f"no {kind} found with id {identifier!r}"}


def business_days_from_today(days: int) -> str:
    """An ISO date `days` business days after the frozen TODAY."""
    result = TODAY
    remaining = days
    while remaining > 0:
        result += timedelta(days=1)
        if result.weekday() < 5:
            remaining -= 1
    return result.isoformat()
