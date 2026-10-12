"""Promotion tools: codes, price matching, clearance and bundles.

One of the twelve domains added in DESIGN.md 19.15. Promotions read products and orders
from `_data` by identifier and own their own code table. Nothing here imports another
domain module.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from . import _data


def _seed_codes() -> dict[str, dict[str, Any]]:
    """Five codes covering percentage off, money off, free shipping, expired, used up."""
    return {
        "SAVE10": {
            "kind": "percent_off",
            "value": 10.0,
            "minimum_spend": 0.0,
            "categories": [],
            "expires_on": "2026-12-31",
            "uses_left": 500,
        },
        "TAKE15OFF": {
            "kind": "amount_off",
            "value": 15.0,
            "minimum_spend": 75.0,
            "categories": [],
            "expires_on": "2026-10-31",
            "uses_left": 120,
        },
        "FREESHIP": {
            "kind": "free_shipping",
            "value": 0.0,
            "minimum_spend": 25.0,
            "categories": [],
            "expires_on": "2026-11-30",
            "uses_left": 1000,
        },
        "AUDIO20": {
            "kind": "percent_off",
            "value": 20.0,
            "minimum_spend": 0.0,
            "categories": ["electronics"],
            "expires_on": "2026-09-30",
            "uses_left": 40,
        },
        "SPRING5": {
            "kind": "percent_off",
            "value": 5.0,
            "minimum_spend": 0.0,
            "categories": [],
            "expires_on": "2026-05-31",
            "uses_left": 0,
        },
    }


CODES: dict[str, dict[str, Any]] = _seed_codes()
PRICE_MATCH_CLAIMS: dict[str, dict[str, Any]] = {}


def _reset() -> None:
    CODES.clear()
    CODES.update(_seed_codes())
    PRICE_MATCH_CLAIMS.clear()


_data.register_reset(_reset)

# Retailers whose prices we will match, and how long after purchase a claim can be made.
MATCHABLE_RETAILERS: tuple[str, ...] = ("bigbox", "shopmart", "netcart", "valuehouse")
PRICE_MATCH_WINDOW_DAYS = 14


def _problem(code: str, record: dict[str, Any], order_total: float, category: str) -> str | None:
    """Why this code cannot be used here, or None when it can."""
    if record["expires_on"] < str(_data.TODAY):
        return f"{code} expired on {record['expires_on']}"
    if record["uses_left"] <= 0:
        return f"{code} has been fully redeemed"
    if order_total < record["minimum_spend"]:
        return f"{code} needs a basket of at least {record['minimum_spend']}"
    if record["categories"] and category not in record["categories"]:
        return f"{code} only applies to {', '.join(record['categories'])}"
    return None


def _discount(record: dict[str, Any], order_total: float) -> float:
    if record["kind"] == "percent_off":
        return round(order_total * record["value"] / 100, 2)
    if record["kind"] == "amount_off":
        return round(min(record["value"], order_total), 2)
    return 0.0


def list_active_promotions(category: str) -> dict[str, Any]:
    """List the promotions a customer could use right now.

    Use this when a customer asks whether there is a deal on, or whether they are
    missing a discount, rather than guessing a code.

    Args:
        category: Narrow the list to promotions that apply to one product category,
            for example "electronics". Pass an empty string for every live promotion,
            including the ones that apply to everything.

    Returns:
        A count and a list. Each promotion carries its code, what kind of discount it
        is ("percent_off", "amount_off" or "free_shipping"), the value, any minimum
        spend, the categories it is limited to, and its expiry date. Codes that have
        expired or been fully redeemed are left out, so anything returned here can
        actually be applied.
    """
    wanted = category.strip().lower()
    live = []
    for code, record in CODES.items():
        if record["expires_on"] < str(_data.TODAY) or record["uses_left"] <= 0:
            continue
        if wanted and record["categories"] and wanted not in record["categories"]:
            continue
        live.append({"code": code, **{k: v for k, v in record.items() if k != "uses_left"}})
    return {"count": len(live), "category": wanted or "any", "promotions": live}


def validate_promo_code(code: str, order_total: float, category: str) -> dict[str, Any]:
    """Check whether a promo code would work on a basket, without applying it.

    Use this when a customer reads out a code and wants to know if it is any good.
    It changes nothing, so it is safe to call before the customer has decided.

    Args:
        code: The code the customer has, for example "SAVE10". Case is ignored.
        order_total: The basket total before any discount, in the account currency.
        category: The main product category in the basket, for example "electronics".
            Pass an empty string when the basket spans categories.

    Returns:
        A `valid` flag, the discount the code would give, the total after it, and the
        code's terms. When the code cannot be used, `valid` is false and a `reason`
        explains which condition failed: expired, fully redeemed, below the minimum
        spend, or limited to a category the basket is not in.
    """
    key = code.strip().upper()
    record = CODES.get(key)
    if record is None:
        return {"code": key, "valid": False, "reason": f"{key} is not a code we issued"}

    reason = _problem(key, record, order_total, category.strip().lower())
    if reason is not None:
        return {"code": key, "valid": False, "reason": reason}

    discount = _discount(record, order_total)
    return {
        "code": key,
        "valid": True,
        "kind": record["kind"],
        "discount": discount,
        "free_shipping": record["kind"] == "free_shipping",
        "total_after": round(order_total - discount, 2),
        "currency": _data.CURRENCY,
        "expires_on": record["expires_on"],
    }


def apply_promo_code(order_id: str, code: str) -> dict[str, Any]:
    """Apply a promo code to an order that has not shipped yet.

    This spends one use of the code and reduces what the customer pays, so validate it
    first if there is any doubt. Only one code can be on an order at a time.

    Args:
        order_id: The order to discount, for example "A1002".
        code: The promo code to apply, for example "TAKE15OFF". Case is ignored.

    Returns:
        An `applied` flag, the discount, the order's new total and the refund due if
        the order has already been paid for. Returns `applied: false` with a reason
        when the code fails its conditions or the order has already shipped, since a
        shipped order is discounted by refund rather than by repricing.
    """
    order = _data.ORDERS.get(order_id)
    if order is None:
        return _data.not_found("order", order_id)

    status = _data.order_status(order)
    if status not in {"PLACED", "PACKED"}:
        return {
            "order_id": order_id,
            "applied": False,
            "status": status,
            "detail": "this order has already shipped, a discount now has to be a refund",
        }

    key = code.strip().upper()
    record = CODES.get(key)
    if record is None:
        return {"order_id": order_id, "applied": False, "reason": f"{key} is not a code we issued"}

    first_sku = order["items"][0]["sku"] if order["items"] else ""
    category = _data.PRODUCTS.get(first_sku, {}).get("category", "")
    reason = _problem(key, record, order["total"], category)
    if reason is not None:
        return {"order_id": order_id, "applied": False, "reason": reason}

    discount = _discount(record, order["total"])
    record["uses_left"] -= 1
    order["total"] = round(order["total"] - discount, 2)
    order["promo_code"] = key
    return {
        "order_id": order_id,
        "applied": True,
        "code": key,
        "discount": discount,
        "new_total": order["total"],
        "currency": _data.CURRENCY,
        "refund_due": discount if status == "PACKED" else 0.0,
    }


def request_price_match(
    order_id: str, sku: str, retailer: str, their_price: float
) -> dict[str, Any]:
    """Claim the difference when a customer finds an item cheaper elsewhere.

    Check `get_price_match_policy` if the customer questions the rules. Claims are
    reviewed rather than paid instantly, because the competitor's price has to be
    verified.

    Args:
        order_id: The order the item was bought on, for example "A1001".
        sku: The product code being matched, for example "SKU-HDP-04".
        retailer: The competitor's name in lower case, for example "bigbox". Only a
            fixed list of retailers is matchable.
        their_price: The competitor's price for the same item, in the account currency.

    Returns:
        A `claim_opened` flag, the claim id, the difference being claimed and how the
        refund would be paid. Returns `claim_opened: false` with a reason when the
        retailer is not on the matchable list, when their price is not lower, or when
        the purchase is outside the claim window.
    """
    order = _data.ORDERS.get(order_id)
    if order is None:
        return _data.not_found("order", order_id)
    line = next((item for item in order["items"] if item["sku"] == sku), None)
    if line is None:
        return {"order_id": order_id, "sku": sku, "error": f"{sku} is not on order {order_id}"}

    name = retailer.strip().lower()
    if name not in MATCHABLE_RETAILERS:
        return {
            "order_id": order_id,
            "claim_opened": False,
            "matchable_retailers": list(MATCHABLE_RETAILERS),
            "reason": f"we do not price match {retailer!r}",
        }

    age_days = (_data.TODAY - date.fromisoformat(order["placed_on"])).days
    if age_days > PRICE_MATCH_WINDOW_DAYS:
        return {
            "order_id": order_id,
            "claim_opened": False,
            "reason": f"the claim window is {PRICE_MATCH_WINDOW_DAYS} days and this order is "
            f"{age_days} days old",
        }

    difference = round(line["unit_price"] - their_price, 2)
    if difference <= 0:
        return {
            "order_id": order_id,
            "claim_opened": False,
            "our_price": line["unit_price"],
            "reason": "their price is not lower than ours",
        }

    claim_id = f"PM-{_data.stable_int(order_id + sku + name, 10000, 99999)}"
    PRICE_MATCH_CLAIMS[claim_id] = {
        "claim_id": claim_id,
        "order_id": order_id,
        "sku": sku,
        "retailer": name,
        "their_price": their_price,
        "difference": difference,
        "status": "UNDER_REVIEW",
        "opened_on": str(_data.TODAY),
    }
    return {
        "claim_id": claim_id,
        "claim_opened": True,
        "difference": difference,
        "currency": _data.CURRENCY,
        "status": "UNDER_REVIEW",
        "paid_as": "store credit within 3 working days of approval",
    }


def get_price_match_policy() -> dict[str, Any]:
    """Read the standing price match rules.

    Near-static reference text. Use it for general questions about whether we match
    prices, rather than for one customer's claim.

    Returns:
        The retailers we match, how many days after purchase a claim can be made, what
        is excluded, how an approved claim is paid, and whether price matching can be
        combined with a promo code. Changes rarely and applies to every customer.
    """
    return {
        "matchable_retailers": list(MATCHABLE_RETAILERS),
        "claim_window_days": PRICE_MATCH_WINDOW_DAYS,
        "excludes": "marketplace sellers, clearance stock, bundle prices, member-only pricing",
        "paid_as": "store credit",
        "combinable_with_promo_code": False,
        "identical_item_required": True,
    }


def get_clearance_items(category: str) -> dict[str, Any]:
    """List products currently on clearance.

    Clearance stock is final sale, so say so when offering it. Use this when a
    customer asks what is cheap, what is being discontinued, or what is in the sale.

    Args:
        category: Narrow to one product category, for example "home". Pass an empty
            string for everything on clearance.

    Returns:
        A count and a list of products with their code, name, category, normal price,
        clearance price, the percentage off and the stock left. Clearance items cannot
        be returned, which is stated on each entry so the answer carries its own
        warning.
    """
    wanted = category.strip().lower()
    items = []
    for sku, product in _data.PRODUCTS.items():
        if wanted and product["category"] != wanted:
            continue
        if _data.stable_int(sku, 0, 2) != 0:
            continue
        clearance = round(product["unit_price"] * 0.7, 2)
        items.append(
            {
                "sku": sku,
                "name": product["name"],
                "category": product["category"],
                "normal_price": product["unit_price"],
                "clearance_price": clearance,
                "percent_off": 30,
                "stock": product["stock"],
                "final_sale": True,
            }
        )
    return {"count": len(items), "category": wanted or "any", "items": items}


def get_bundle_offers(sku: str) -> dict[str, Any]:
    """Find bundles that include a given product.

    Use this when a customer is buying one thing and a cheaper pairing exists, or when
    they ask whether buying two of something is cheaper.

    Args:
        sku: The product the customer is looking at, for example "SKU-HDP-04".

    Returns:
        A count and a list of bundles. Each carries a bundle id, the products in it,
        the total if bought separately, the bundle price, the saving and whether the
        bundle can be combined with a promo code. Returns an empty list when nothing
        is bundled with that product, which is not an error.
    """
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)

    partners = [
        other
        for other_sku, other in _data.PRODUCTS.items()
        if other_sku != sku and other["category"] == product["category"]
    ]
    bundles = []
    for partner in partners[:2]:
        separate = round(product["unit_price"] + partner["unit_price"], 2)
        price = round(separate * 0.85, 2)
        bundles.append(
            {
                "bundle_id": f"BND-{sku[-2:]}{partner['sku'][-2:]}",
                "includes": [sku, partner["sku"]],
                "names": [product["name"], partner["name"]],
                "separate_total": separate,
                "bundle_price": price,
                "saving": round(separate - price, 2),
                "combinable_with_promo_code": False,
            }
        )
    return {"sku": sku, "count": len(bundles), "currency": _data.CURRENCY, "bundles": bundles}
