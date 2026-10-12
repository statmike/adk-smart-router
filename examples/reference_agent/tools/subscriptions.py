"""Recurring-order tools: see them, change the schedule, stop them, bill them.

One of the twelve domains added in DESIGN.md 19.15. Subscriptions own their own fixture
and reach into `_data` only for accounts and products, by identifier. Nothing in this
module imports another domain module.

The declarations here are deliberately thicker than the original 25 tools': named enum
values, worked examples and a full return contract. That is what a tool in a real
inventory costs, and the declaration token count is the whole point of the experiment.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from google.adk.tools import ToolContext

from . import _data

# How often a subscription can be delivered, and how many days that is.
FREQUENCY_DAYS: dict[str, int] = {
    "weekly": 7,
    "biweekly": 14,
    "monthly": 30,
    "quarterly": 91,
}

STATUSES: tuple[str, ...] = ("active", "paused", "cancelled")


def _seed_subscriptions() -> dict[str, dict[str, Any]]:
    """Four subscriptions across two customers, covering every status."""
    rows = [
        ("SUB-4101", "u_1001", "SKU-CBL-06", "monthly", "active", 3, 9.99, 10),
        ("SUB-4102", "u_1001", "SKU-MUG-02", "quarterly", "paused", 1, 12.50, 0),
        ("SUB-4103", "u_1002", "SKU-TEE-01", "biweekly", "active", 2, 24.00, 15),
        ("SUB-4104", "u_1002", "SKU-BAG-05", "monthly", "cancelled", 1, 32.00, 0),
    ]
    return {
        sub_id: {
            "subscription_id": sub_id,
            "user_id": user_id,
            "sku": sku,
            "frequency": frequency,
            "status": status,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_percent": discount,
            "started_on": str(_data.TODAY - timedelta(days=180)),
            "next_charge_on": _data.business_days_from_today(
                FREQUENCY_DAYS[frequency] if status == "active" else 0
            ),
            "resume_on": None,
            "deliveries_made": _data.stable_int(sub_id, 3, 18),
        }
        for sub_id, user_id, sku, frequency, status, quantity, unit_price, discount in rows
    }


SUBSCRIPTIONS: dict[str, dict[str, Any]] = _seed_subscriptions()


def _reset() -> None:
    SUBSCRIPTIONS.clear()
    SUBSCRIPTIONS.update(_seed_subscriptions())


_data.register_reset(_reset)


def _charge(record: dict[str, Any]) -> float:
    """What one delivery of this subscription costs after its discount."""
    gross = record["unit_price"] * record["quantity"]
    return round(gross * (1 - record["discount_percent"] / 100), 2)


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    """The fields every subscription tool returns, so the views cannot disagree."""
    return {
        "subscription_id": record["subscription_id"],
        "sku": record["sku"],
        "frequency": record["frequency"],
        "status": record["status"],
        "quantity": record["quantity"],
        "charge_per_delivery": _charge(record),
        "currency": _data.CURRENCY,
        "next_charge_on": record["next_charge_on"],
    }


def list_subscriptions(status_filter: str, tool_context: ToolContext) -> dict[str, Any]:
    """List the signed-in customer's recurring orders.

    Use this when the customer talks about a repeat delivery, a standing order or a
    plan they pay for on a schedule, and has not said which one. The customer is
    whoever is signed in, so there is no argument for identity.

    Args:
        status_filter: Which subscriptions to include. One of "active" for the ones
            still billing, "paused" for ones temporarily stopped, "cancelled" for ones
            already ended, or "all" for every subscription on the account.

    Returns:
        A count and a list. Each entry carries the subscription id, the product code,
        the delivery frequency, the status, the quantity per delivery, the charge per
        delivery in the account currency, and the date of the next charge. A cancelled
        subscription reports its last charge date instead. Returns an empty list rather
        than an error when the customer has no subscriptions matching the filter.
    """
    wanted = status_filter.strip().lower() or "all"
    if wanted not in {*STATUSES, "all"}:
        allowed = sorted({*STATUSES, "all"})
        return {"error": f"status_filter must be one of {allowed}, not {wanted!r}"}

    user_id = _data.current_user_id(tool_context)
    matches = [
        _summary(record)
        for record in SUBSCRIPTIONS.values()
        if record["user_id"] == user_id and (wanted == "all" or record["status"] == wanted)
    ]
    return {"count": len(matches), "status_filter": wanted, "subscriptions": matches}


def get_subscription(subscription_id: str) -> dict[str, Any]:
    """Get everything known about one recurring order.

    Use this once the customer has named a specific subscription, or after
    `list_subscriptions` has narrowed it to one.

    Args:
        subscription_id: The subscription identifier, for example "SUB-4101". These
            always start with "SUB-" followed by four digits.

    Returns:
        The subscription's product code and product name, delivery frequency, status,
        quantity, unit price, any standing discount percentage, the charge per
        delivery, the date it started, how many deliveries have been made so far, and
        the next charge date. A paused subscription also carries the date it is due to
        resume, which may be null if the customer paused it indefinitely. Returns an
        error field when the identifier is not one of this customer's subscriptions.
    """
    record = SUBSCRIPTIONS.get(subscription_id)
    if record is None:
        return _data.not_found("subscription", subscription_id)

    product = _data.PRODUCTS.get(record["sku"], {})
    return {
        **_summary(record),
        "product_name": product.get("name", record["sku"]),
        "unit_price": record["unit_price"],
        "discount_percent": record["discount_percent"],
        "started_on": record["started_on"],
        "deliveries_made": record["deliveries_made"],
        "resume_on": record["resume_on"],
    }


def pause_subscription(subscription_id: str, resume_on: str) -> dict[str, Any]:
    """Temporarily stop a recurring order without cancelling it.

    Prefer this over `cancel_subscription` whenever the customer says they are going
    away, want a break, or want to skip a delivery. Pausing keeps the standing
    discount; cancelling loses it.

    Args:
        subscription_id: The subscription to pause, for example "SUB-4101".
        resume_on: The ISO date the subscription should start billing again, for
            example "2026-11-01". Pass an empty string to pause indefinitely, which
            leaves the customer to resume it by hand.

    Returns:
        A `paused` flag, the subscription id, its new status, the resume date if one
        was given, and the charge that was skipped. Returns `paused: false` with a
        detail line when the subscription is already paused or has been cancelled,
        because neither can be paused again.
    """
    record = SUBSCRIPTIONS.get(subscription_id)
    if record is None:
        return _data.not_found("subscription", subscription_id)
    if record["status"] != "active":
        return {
            "subscription_id": subscription_id,
            "paused": False,
            "status": record["status"],
            "detail": f"only an active subscription can be paused, this one is {record['status']}",
        }

    record["status"] = "paused"
    record["resume_on"] = resume_on.strip() or None
    return {
        "subscription_id": subscription_id,
        "paused": True,
        "status": "paused",
        "resume_on": record["resume_on"],
        "skipped_charge": _charge(record),
        "currency": _data.CURRENCY,
    }


def resume_subscription(subscription_id: str) -> dict[str, Any]:
    """Start a paused recurring order billing again.

    Args:
        subscription_id: The paused subscription to restart, for example "SUB-4102".

    Returns:
        A `resumed` flag, the new status, the date of the next charge and what that
        charge will be. The next charge is one full delivery interval away, so
        resuming never bills the customer immediately. Returns `resumed: false` when
        the subscription is already active or was cancelled, since a cancelled
        subscription has to be created again rather than resumed.
    """
    record = SUBSCRIPTIONS.get(subscription_id)
    if record is None:
        return _data.not_found("subscription", subscription_id)
    if record["status"] != "paused":
        return {
            "subscription_id": subscription_id,
            "resumed": False,
            "status": record["status"],
            "detail": f"only a paused subscription can be resumed, this one is {record['status']}",
        }

    record["status"] = "active"
    record["resume_on"] = None
    record["next_charge_on"] = _data.business_days_from_today(FREQUENCY_DAYS[record["frequency"]])
    return {
        "subscription_id": subscription_id,
        "resumed": True,
        "status": "active",
        "next_charge_on": record["next_charge_on"],
        "next_charge": _charge(record),
        "currency": _data.CURRENCY,
    }


def change_subscription_frequency(subscription_id: str, frequency: str) -> dict[str, Any]:
    """Change how often a recurring order is delivered.

    Use this when the customer says they are receiving too much or too little, rather
    than cancelling and starting again, which would lose their standing discount.

    Args:
        subscription_id: The subscription to change, for example "SUB-4101".
        frequency: The new delivery interval. One of "weekly", "biweekly", "monthly"
            or "quarterly".

    Returns:
        A `changed` flag, the previous and new frequency, the recalculated next charge
        date, and the charge amount, which does not change because frequency affects
        timing rather than price. Returns an error field when the frequency is not one
        of the four supported values, and lists the ones that are.
    """
    record = SUBSCRIPTIONS.get(subscription_id)
    if record is None:
        return _data.not_found("subscription", subscription_id)

    wanted = frequency.strip().lower()
    if wanted not in FREQUENCY_DAYS:
        return {
            "subscription_id": subscription_id,
            "changed": False,
            "supported_frequencies": sorted(FREQUENCY_DAYS),
            "error": f"{frequency!r} is not a delivery frequency we offer",
        }

    previous = record["frequency"]
    record["frequency"] = wanted
    record["next_charge_on"] = _data.business_days_from_today(FREQUENCY_DAYS[wanted])
    return {
        "subscription_id": subscription_id,
        "changed": True,
        "previous_frequency": previous,
        "frequency": wanted,
        "next_charge_on": record["next_charge_on"],
        "charge_per_delivery": _charge(record),
        "currency": _data.CURRENCY,
    }


def cancel_subscription(subscription_id: str, reason: str) -> dict[str, Any]:
    """End a recurring order permanently.

    This is not reversible and it loses any standing discount, so offer
    `pause_subscription` first whenever the customer's reason sounds temporary.

    Args:
        subscription_id: The subscription to end, for example "SUB-4103".
        reason: Why the customer is cancelling, in their own words. Recorded against
            the account and used for retention reporting.

    Returns:
        A `cancelled` flag, the final status, the date of the last delivery that will
        still be sent, and whether any prepaid balance is being refunded. Returns
        `cancelled: false` when the subscription was already cancelled.
    """
    record = SUBSCRIPTIONS.get(subscription_id)
    if record is None:
        return _data.not_found("subscription", subscription_id)
    if record["status"] == "cancelled":
        return {
            "subscription_id": subscription_id,
            "cancelled": False,
            "status": "cancelled",
            "detail": "this subscription was already cancelled",
        }

    prepaid = record["status"] == "paused"
    record["status"] = "cancelled"
    record["resume_on"] = None
    record["cancel_reason"] = reason
    return {
        "subscription_id": subscription_id,
        "cancelled": True,
        "status": "cancelled",
        "final_delivery_on": record["next_charge_on"],
        "refund_issued": prepaid,
        "refund_amount": _charge(record) if prepaid else 0.0,
        "currency": _data.CURRENCY,
    }


def get_subscription_invoice(subscription_id: str, billing_period: str) -> dict[str, Any]:
    """Get the billing detail for one period of a recurring order.

    Use this when the customer questions an amount they were charged for a repeat
    delivery, rather than `get_invoice`, which covers one-off orders.

    Args:
        subscription_id: The subscription being queried, for example "SUB-4101".
        billing_period: The month to bill for, as "YYYY-MM", for example "2026-07".

    Returns:
        The gross amount, the standing discount applied, tax, the net charged, the
        payment method's last four digits and the charge date. Returns an error field
        when the period is not in "YYYY-MM" form or falls before the subscription
        started, since no charge exists to explain.
    """
    record = SUBSCRIPTIONS.get(subscription_id)
    if record is None:
        return _data.not_found("subscription", subscription_id)

    period = billing_period.strip()
    parts = period.split("-")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return {
            "subscription_id": subscription_id,
            "error": f"billing_period must look like '2026-07', not {billing_period!r}",
        }
    if period < record["started_on"][:7]:
        return {
            "subscription_id": subscription_id,
            "billing_period": period,
            "error": f"this subscription started in {record['started_on'][:7]}",
        }

    gross = round(record["unit_price"] * record["quantity"], 2)
    discount = round(gross - _charge(record), 2)
    tax = round((gross - discount) * 0.08, 2)
    return {
        "subscription_id": subscription_id,
        "billing_period": period,
        "gross_amount": gross,
        "discount_amount": discount,
        "tax_amount": tax,
        "net_charged": round(gross - discount + tax, 2),
        "currency": _data.CURRENCY,
        "card_last4": f"{_data.stable_int(subscription_id + period, 1000, 9999)}",
        "charged_on": f"{period}-05",
    }
