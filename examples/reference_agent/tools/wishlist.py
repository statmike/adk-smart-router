"""Wishlist and saved cart tools: keep things for later, and pick them up again.

One of the twelve domains added in DESIGN.md 19.15. Wishlists are per customer and read
identity from session state, so they sit alongside the ambient-identity tools of
DESIGN.md 8.1. Products are read from `_data` by SKU and no other domain is imported.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from . import _data


def _seed_wishlists() -> dict[str, list[dict[str, Any]]]:
    return {
        "u_1001": [
            {"sku": "SKU-LMP-03", "added_on": "2026-08-11", "note": "for the study"},
            {"sku": "SKU-HDP-04", "added_on": "2026-08-19", "note": ""},
        ],
        "u_1002": [{"sku": "SKU-BAG-05", "added_on": "2026-07-30", "note": "birthday idea"}],
    }


def _seed_saved_carts() -> dict[str, dict[str, Any]]:
    return {
        "CART-7781": {
            "cart_id": "CART-7781",
            "user_id": "u_1001",
            "saved_on": "2026-08-27",
            "items": [
                {"sku": "SKU-TEE-01", "quantity": 2},
                {"sku": "SKU-CBL-06", "quantity": 1},
            ],
        }
    }


WISHLISTS: dict[str, list[dict[str, Any]]] = _seed_wishlists()
SAVED_CARTS: dict[str, dict[str, Any]] = _seed_saved_carts()


def _reset() -> None:
    WISHLISTS.clear()
    WISHLISTS.update(_seed_wishlists())
    SAVED_CARTS.clear()
    SAVED_CARTS.update(_seed_saved_carts())


_data.register_reset(_reset)


def _decorate(entry: dict[str, Any]) -> dict[str, Any]:
    """A wishlist entry with the product's current name, price and stock attached."""
    product = _data.PRODUCTS.get(entry["sku"], {})
    return {
        **entry,
        "name": product.get("name", entry["sku"]),
        "unit_price": product.get("unit_price"),
        "in_stock": bool(product.get("stock", 0)),
    }


def get_wishlist(tool_context: ToolContext) -> dict[str, Any]:
    """List everything the signed-in customer has saved for later.

    Takes no arguments beyond identity: the customer is whoever is signed in. Use this
    when a customer refers to something they saved, starred or added to a list, and
    has not said which item.

    Returns:
        A count and a list. Each entry carries the product code, the current name and
        price, whether it is in stock right now, the date it was saved and any note
        the customer left. Prices are read live rather than stored, so an item saved
        before a price change reports the new price and the customer sees the truth.
    """
    user_id = _data.current_user_id(tool_context)
    entries = [_decorate(entry) for entry in WISHLISTS.get(user_id, [])]
    return {
        "count": len(entries),
        "currency": _data.CURRENCY,
        "in_stock_count": sum(1 for entry in entries if entry["in_stock"]),
        "items": entries,
    }


def add_to_wishlist(sku: str, note: str, tool_context: ToolContext) -> dict[str, Any]:
    """Save a product to the signed-in customer's wishlist.

    Offer this whenever a customer likes something but is not buying today, and
    especially when an item is out of stock, since a saved item can be watched.

    Args:
        sku: The product to save, for example "SKU-LMP-03".
        note: A short reminder in the customer's own words, for example "for the
            study". Pass an empty string when they have not said why.

    Returns:
        An `added` flag, the product name, the wishlist's new length and whether the
        item is currently in stock. Returns `added: false` when the item is already on
        the list, along with the date it was originally saved, so the customer is not
        told something changed when nothing did.
    """
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)

    user_id = _data.current_user_id(tool_context)
    entries = WISHLISTS.setdefault(user_id, [])
    existing = next((entry for entry in entries if entry["sku"] == sku), None)
    if existing is not None:
        return {
            "sku": sku,
            "added": False,
            "detail": f"already on the wishlist since {existing['added_on']}",
            "count": len(entries),
        }

    entries.append({"sku": sku, "added_on": str(_data.TODAY), "note": note.strip()})
    return {
        "sku": sku,
        "added": True,
        "name": product["name"],
        "count": len(entries),
        "in_stock": bool(product["stock"]),
    }


def remove_from_wishlist(sku: str, tool_context: ToolContext) -> dict[str, Any]:
    """Take a product off the signed-in customer's wishlist.

    Args:
        sku: The product to remove, for example "SKU-HDP-04".

    Returns:
        A `removed` flag, the product name and the wishlist's new length. Returns
        `removed: false` when the item was not on the list, which usually means the
        customer is thinking of a different product or a different account.
    """
    user_id = _data.current_user_id(tool_context)
    entries = WISHLISTS.get(user_id, [])
    remaining = [entry for entry in entries if entry["sku"] != sku]
    if len(remaining) == len(entries):
        return {"sku": sku, "removed": False, "detail": "that is not on the wishlist"}

    WISHLISTS[user_id] = remaining
    return {
        "sku": sku,
        "removed": True,
        "name": _data.PRODUCTS.get(sku, {}).get("name", sku),
        "count": len(remaining),
    }


def get_saved_carts(tool_context: ToolContext) -> dict[str, Any]:
    """List baskets the signed-in customer abandoned or saved to finish later.

    Takes no arguments beyond identity. Use this when a customer says they had a
    basket ready, or asks what happened to the things they were buying.

    Returns:
        A count and a list of carts. Each carries its cart id, the date it was saved,
        the items with their quantities and current prices, the cart total at today's
        prices, and whether every item is still in stock. A cart with an out-of-stock
        item can still be restored, but the missing item is flagged.
    """
    user_id = _data.current_user_id(tool_context)
    carts = []
    for cart in SAVED_CARTS.values():
        if cart["user_id"] != user_id:
            continue
        items = []
        total = 0.0
        for line in cart["items"]:
            product = _data.PRODUCTS.get(line["sku"], {})
            price = product.get("unit_price", 0.0)
            total += price * line["quantity"]
            items.append(
                {
                    "sku": line["sku"],
                    "name": product.get("name", line["sku"]),
                    "quantity": line["quantity"],
                    "unit_price": price,
                    "in_stock": bool(product.get("stock", 0)),
                }
            )
        carts.append(
            {
                "cart_id": cart["cart_id"],
                "saved_on": cart["saved_on"],
                "items": items,
                "total": round(total, 2),
                "all_in_stock": all(item["in_stock"] for item in items),
            }
        )
    return {"count": len(carts), "currency": _data.CURRENCY, "carts": carts}


def restore_saved_cart(cart_id: str, tool_context: ToolContext) -> dict[str, Any]:
    """Put a saved basket back so the customer can check out from it.

    Args:
        cart_id: The saved cart to restore, for example "CART-7781". Get valid ids
            from `get_saved_carts`.

    Returns:
        A `restored` flag, the items moved back, the basket total at today's prices,
        and any items that were dropped because they are now out of stock. Returns
        `restored: false` when the cart belongs to a different account, which is the
        one case that must never silently succeed.
    """
    cart = SAVED_CARTS.get(cart_id)
    if cart is None:
        return _data.not_found("saved cart", cart_id)

    user_id = _data.current_user_id(tool_context)
    if cart["user_id"] != user_id:
        return {"cart_id": cart_id, "restored": False, "detail": "that cart is on another account"}

    restored, dropped, total = [], [], 0.0
    for line in cart["items"]:
        product = _data.PRODUCTS.get(line["sku"], {})
        if not product.get("stock", 0):
            dropped.append(line["sku"])
            continue
        total += product["unit_price"] * line["quantity"]
        restored.append({"sku": line["sku"], "quantity": line["quantity"]})

    return {
        "cart_id": cart_id,
        "restored": True,
        "items": restored,
        "dropped_out_of_stock": dropped,
        "total": round(total, 2),
        "currency": _data.CURRENCY,
    }
