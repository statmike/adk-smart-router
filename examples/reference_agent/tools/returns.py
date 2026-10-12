"""Return tools: open a return, check one, read the policy.

`get_return_policy` is the long-TTL end of DESIGN.md 8.6: it is near-static reference
text keyed only by category, so caching it for hours is safe.
"""

from __future__ import annotations

from typing import Any

from . import _data


def start_return(order_id: str, sku: str, reason: str) -> dict[str, Any]:
    """Open a return for one item of a delivered order.

    Args:
        order_id: The delivered order the item came from.
        sku: The product code being returned, for example "SKU-MUG-02".
        reason: Why the customer is returning it, in their own words.

    Returns the new return id, the refund amount and the next step. Only delivered
    orders can be returned.
    """
    order = _data.ORDERS.get(order_id)
    if order is None:
        return _data.not_found("order", order_id)

    status = _data.order_status(order)
    if status != "DELIVERED":
        return {
            "order_id": order_id,
            "started": False,
            "status": status,
            "detail": "the order has not been delivered yet",
        }

    line = next((item for item in order["items"] if item["sku"] == sku), None)
    if line is None:
        return {
            "order_id": order_id,
            "started": False,
            "detail": f"{sku} is not on this order",
        }

    return_id = f"RMA-{order_id}-{sku}"
    refund_amount = round(line["unit_price"] * line["quantity"], 2)
    _data.RETURNS[return_id] = {
        "return_id": return_id,
        "order_id": order_id,
        "user_id": order["user_id"],
        "sku": sku,
        "reason": reason,
        "status": "LABEL_SENT",
        "opened_on": str(_data.TODAY),
        "refund_amount": refund_amount,
    }
    return {
        "return_id": return_id,
        "started": True,
        "status": "LABEL_SENT",
        "refund_amount": refund_amount,
        "currency": _data.CURRENCY,
        "next_step": "print the emailed label and drop the parcel at any carrier point",
    }


def get_return_status(return_id: str) -> dict[str, Any]:
    """Check where a return has got to.

    Args:
        return_id: The return identifier, for example "RMA-A1002".

    Returns the return's status, the item, the refund amount and when it was opened.
    """
    record = _data.RETURNS.get(return_id)
    if record is None:
        return _data.not_found("return", return_id)
    return {
        "return_id": return_id,
        "order_id": record["order_id"],
        "sku": record["sku"],
        "status": record["status"],
        "opened_on": record["opened_on"],
        "refund_amount": record["refund_amount"],
        "currency": _data.CURRENCY,
    }


def get_return_policy(category: str) -> dict[str, Any]:
    """Look up the return policy for a product category.

    Args:
        category: One of "general", "apparel", "electronics" or "final_sale".

    Returns the return window in days, any restocking fee, the required condition and
    how the refund is paid. This text changes rarely.
    """
    key = category.strip().lower() or "general"
    policy = _data.RETURN_POLICIES.get(key)
    if policy is None:
        return {
            "category": key,
            "known_categories": sorted(_data.RETURN_POLICIES),
            "error": f"no policy for category {key!r}",
        }
    return {"category": key, **policy}
