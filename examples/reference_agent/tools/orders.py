"""Order tools: status, listing, cancellation, address changes.

`get_order_status` is deliberately a poller: every call moves the order one stage
further down `ORDER_STATUS_FLOW`. That makes it the tool from DESIGN.md 8.6 for which
any nonzero cache TTL is wrong, and it makes the wrongness observable in a test.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from . import _data


def get_order_status(order_id: str) -> dict[str, Any]:
    """Get the current fulfilment status of one order.

    Args:
        order_id: The order identifier, for example "A1001".

    Returns the status (PLACED, PACKED, IN_TRANSIT, OUT_FOR_DELIVERY or DELIVERED),
    the order total and the destination address. The status changes as the order moves,
    so always re-read it rather than reusing an earlier answer.
    """
    order = _data.ORDERS.get(order_id)
    if order is None:
        return _data.not_found("order", order_id)

    # The poller trap (DESIGN.md 8.6): one stage per call, until delivered.
    if not order["cancelled"]:
        order["status_index"] = min(order["status_index"] + 1, len(_data.ORDER_STATUS_FLOW) - 1)

    shipment = _data.shipment_for_order(order_id)
    return {
        "order_id": order_id,
        "status": _data.order_status(order),
        "placed_on": order["placed_on"],
        "total": order["total"],
        "currency": _data.CURRENCY,
        "shipping_address": order["shipping_address"],
        "tracking_number": shipment["tracking_number"] if shipment else None,
    }


def list_orders(status: str, tool_context: ToolContext) -> dict[str, Any]:
    """List the signed-in customer's orders, newest first.

    Args:
        status: "open" for orders not yet delivered, "delivered", or "all".

    Returns one summary row per order: id, placed date, status and total.
    """
    user_id = _data.current_user_id(tool_context)
    wanted = status.strip().lower() or "all"

    rows = []
    for order in _data.ORDERS.values():
        if order["user_id"] != user_id:
            continue
        current = _data.order_status(order)
        if wanted == "open" and current in ("DELIVERED", "CANCELLED"):
            continue
        if wanted == "delivered" and current != "DELIVERED":
            continue
        rows.append(
            {
                "order_id": order["order_id"],
                "placed_on": order["placed_on"],
                "status": current,
                "total": order["total"],
                "currency": _data.CURRENCY,
            }
        )

    rows.sort(key=lambda row: row["placed_on"], reverse=True)
    return {"filter": wanted, "count": len(rows), "orders": rows}


def cancel_order(order_id: str, reason: str) -> dict[str, Any]:
    """Cancel an order that has not shipped yet.

    Args:
        order_id: The order to cancel, for example "A1003".
        reason: Short free-text reason from the customer.

    Cancellation only works while the order is PLACED or PACKED. Returns whether it was
    cancelled and, if not, why not.
    """
    order = _data.ORDERS.get(order_id)
    if order is None:
        return _data.not_found("order", order_id)

    current = _data.order_status(order)
    if current not in ("PLACED", "PACKED"):
        return {
            "order_id": order_id,
            "cancelled": False,
            "status": current,
            "detail": "already shipped; start a return once it arrives",
        }

    order["cancelled"] = True
    order["cancel_reason"] = reason
    return {
        "order_id": order_id,
        "cancelled": True,
        "status": "CANCELLED",
        "refund_to": "original payment method",
        "refund_amount": order["total"],
        "currency": _data.CURRENCY,
    }


def modify_shipping_address(order_id: str, address: str) -> dict[str, Any]:
    """Change the delivery address of an order that has not shipped yet.

    Args:
        order_id: The order to update.
        address: The full new address on one line, including postal code and country.

    Returns the stored address, or an explanation if the order is already in transit.
    """
    order = _data.ORDERS.get(order_id)
    if order is None:
        return _data.not_found("order", order_id)

    current = _data.order_status(order)
    if current not in ("PLACED", "PACKED"):
        return {
            "order_id": order_id,
            "updated": False,
            "status": current,
            "detail": "parcel already left the warehouse; ask the carrier to redirect",
        }

    order["shipping_address"] = address.strip()
    return {
        "order_id": order_id,
        "updated": True,
        "shipping_address": order["shipping_address"],
        "status": current,
    }
