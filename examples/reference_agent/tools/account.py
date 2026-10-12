"""Account tools, and the ambient-identity shape from DESIGN.md 8.1.

`get_account`, `get_loyalty_points` and `list_addresses` take no arguments at all. They
read the customer from `tool_context.state`, so the cache key's args hash carries no
identity whatsoever. If `user_id` ever drops out of the key, these three are the first
tools to serve one customer's data to another. The cross-user isolation test names them.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from . import _data


def get_account(tool_context: ToolContext) -> dict[str, Any]:
    """Get the signed-in customer's account profile.

    Takes no arguments: the customer is whoever is signed in.

    Returns name, email, membership tier, join date, loyalty points and store credit
    balance. The balance changes whenever a refund is issued.
    """
    account = _data.account_for(tool_context)
    return {
        "user_id": account["user_id"],
        "name": account["name"],
        "email": account["email"],
        "tier": account["tier"],
        "member_since": account["member_since"],
        "loyalty_points": account["loyalty_points"],
        "store_credit": account["store_credit"],
        "currency": _data.CURRENCY,
    }


def update_email(new_email: str, tool_context: ToolContext) -> dict[str, Any]:
    """Change the signed-in customer's email address.

    Args:
        new_email: The new address, for example "ada.ellis@example.com".

    Returns the old and new address. A confirmation link is sent to the new address.
    """
    address = new_email.strip()
    if "@" not in address or address.startswith("@") or address.endswith("@"):
        return {"updated": False, "error": f"{new_email!r} is not a valid email address"}

    account = _data.account_for(tool_context)
    previous = account["email"]
    account["email"] = address
    return {
        "updated": True,
        "previous_email": previous,
        "email": address,
        "detail": "confirmation link sent to the new address",
    }


def get_loyalty_points(tool_context: ToolContext) -> dict[str, Any]:
    """Get the signed-in customer's loyalty point balance.

    Takes no arguments: the customer is whoever is signed in.

    Returns the point balance, its cash value, the tier and the points still needed for
    the next tier.
    """
    account = _data.account_for(tool_context)
    points = account["loyalty_points"]
    next_tier_at = 2000 if account["tier"] != "platinum" else points
    return {
        "loyalty_points": points,
        "cash_value": round(points * 0.01, 2),
        "currency": _data.CURRENCY,
        "tier": account["tier"],
        "points_to_next_tier": max(0, next_tier_at - points),
    }


def list_addresses(tool_context: ToolContext) -> dict[str, Any]:
    """List the signed-in customer's saved delivery addresses.

    Takes no arguments: the customer is whoever is signed in.

    Returns each address with its label and whether it is the default.
    """
    account = _data.account_for(tool_context)
    addresses = [dict(address) for address in account["addresses"]]
    return {"count": len(addresses), "addresses": addresses}
