"""Payment tools: cards on file, invoices, refunds.

`request_refund` reads the account's store credit and then writes it. Together with
`get_account` it is the money path of DESIGN.md 8.4: cache `get_account`, refund, ask
again, and a cache that is not purged on write answers with the pre-refund balance.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from . import _data


def get_payment_methods(include_expired: bool, tool_context: ToolContext) -> dict[str, Any]:
    """List the signed-in customer's saved payment methods.

    Args:
        include_expired: True to also return cards past their expiry date.

    Returns brand, last four digits, expiry and which one is the default. Full card
    numbers are never available.
    """
    account = _data.account_for(tool_context)
    methods = [
        method for method in account["payment_methods"] if include_expired or not method["expired"]
    ]
    return {"count": len(methods), "payment_methods": methods}


def add_payment_method(
    brand: str, last4: str, expires: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Save a new card on the signed-in customer's account.

    Args:
        brand: Card brand, for example "visa" or "mastercard".
        last4: The last four digits of the card number.
        expires: Expiry as "YYYY-MM".

    Returns the new method id. The first card saved becomes the default.
    """
    account = _data.account_for(tool_context)
    method_id = f"pm_{brand.strip().lower()}_{last4}"
    if any(m["method_id"] == method_id for m in account["payment_methods"]):
        return {"method_id": method_id, "added": False, "detail": "card already on file"}

    method = {
        "method_id": method_id,
        "brand": brand.strip().lower(),
        "last4": last4,
        "expires": expires,
        "default": not account["payment_methods"],
        "expired": False,
    }
    account["payment_methods"].append(method)
    return {"method_id": method_id, "added": True, "payment_method": method}


def get_invoice(order_id: str) -> dict[str, Any]:
    """Fetch the invoice for an order.

    Args:
        order_id: The order the invoice belongs to, for example "A1002".

    Returns the invoice id, issue date, subtotal, tax and total.
    """
    invoice = _data.INVOICES.get(order_id)
    if invoice is None:
        return _data.not_found("invoice for order", order_id)
    return dict(invoice)


def request_refund(
    order_id: str, amount: float, reason: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Refund part or all of an order to the customer's store credit.

    Args:
        order_id: The order being refunded.
        amount: Amount to refund in USD. Must not exceed the order total.
        reason: Why the refund is being issued.

    Returns the new store credit balance. This changes the balance, so any earlier
    account balance you were told is now out of date.
    """
    order = _data.ORDERS.get(order_id)
    if order is None:
        return _data.not_found("order", order_id)

    account = _data.account_for(tool_context)
    if order["user_id"] != account["user_id"]:
        return _data.not_found("order", order_id)

    if amount <= 0 or amount > order["total"]:
        return {
            "order_id": order_id,
            "refunded": False,
            "detail": f"amount must be between 0 and the order total of {order['total']}",
        }

    # Read-your-own-writes (DESIGN.md 8.4): read the balance, then move it.
    balance_before = account["store_credit"]
    account["store_credit"] = round(balance_before + amount, 2)
    order.setdefault("refunds", []).append({"amount": amount, "reason": reason})
    return {
        "order_id": order_id,
        "refunded": True,
        "amount": amount,
        "currency": _data.CURRENCY,
        "balance_before": balance_before,
        "store_credit": account["store_credit"],
    }
