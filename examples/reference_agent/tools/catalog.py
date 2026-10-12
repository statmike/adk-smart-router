"""Catalog tools: search, product detail, stock, price.

`check_stock` and `get_price` are the interesting middle of DESIGN.md 8.6: they change
often enough that a long TTL is wrong and often enough repeated within a session that
a short one pays.
"""

from __future__ import annotations

from typing import Any

from . import _data

# Quantity break -> discount percentage. Applied at get_price.
_VOLUME_BREAKS: tuple[tuple[int, int], ...] = ((10, 15), (5, 10), (3, 5))


def search_products(query: str, max_results: int) -> dict[str, Any]:
    """Search the catalog by name or keyword.

    Args:
        query: What the customer is looking for, for example "headphones".
        max_results: Maximum number of products to return.

    Returns matching products with sku, name, category and price, in catalog order.
    """
    needle = query.strip().lower()
    matches = [
        {
            "sku": product["sku"],
            "name": product["name"],
            "category": product["category"],
            "unit_price": product["unit_price"],
            "currency": _data.CURRENCY,
        }
        for product in sorted(_data.PRODUCTS.values(), key=lambda p: p["sku"])
        if not needle
        or needle in product["name"].lower()
        or needle in product["category"]
        or any(needle in keyword for keyword in product["keywords"])
    ]
    limit = max(1, max_results)
    return {"query": query, "count": len(matches[:limit]), "products": matches[:limit]}


def get_product_details(sku: str) -> dict[str, Any]:
    """Get the full description of one product.

    Args:
        sku: The product code, for example "SKU-LMP-03".

    Returns name, category, price, keywords and whether it is currently in stock.
    """
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)
    return {
        "sku": sku,
        "name": product["name"],
        "category": product["category"],
        "unit_price": product["unit_price"],
        "currency": _data.CURRENCY,
        "keywords": list(product["keywords"]),
        "in_stock": product["stock"] > 0,
    }


def check_stock(sku: str, quantity: int) -> dict[str, Any]:
    """Check whether a product is available in the quantity the customer wants.

    Args:
        sku: The product code.
        quantity: How many units they want.

    Returns availability, units on hand and, when short, the restock date. Stock moves
    during the day, so do not rely on an answer from earlier in the conversation.
    """
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)

    on_hand = product["stock"]
    wanted = max(1, quantity)
    available = on_hand >= wanted
    result: dict[str, Any] = {
        "sku": sku,
        "requested": wanted,
        "units_on_hand": on_hand,
        "available": available,
    }
    if not available:
        # Restock lead time is derived from the sku, so the answer is repeatable.
        result["restock_date"] = _data.business_days_from_today(
            _data.stable_int(f"restock:{sku}", 3, 21)
        )
    return result


def get_price(sku: str, quantity: int) -> dict[str, Any]:
    """Price a quantity of a product, including any volume discount.

    Args:
        sku: The product code.
        quantity: How many units to price.

    Returns unit price, discount percentage and the line total.
    """
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)

    units = max(1, quantity)
    discount_pct = next((pct for threshold, pct in _VOLUME_BREAKS if units >= threshold), 0)
    unit_price = round(product["unit_price"] * (1 - discount_pct / 100), 2)
    return {
        "sku": sku,
        "quantity": units,
        "list_price": product["unit_price"],
        "discount_pct": discount_pct,
        "unit_price": unit_price,
        "line_total": round(unit_price * units, 2),
        "currency": _data.CURRENCY,
    }
