"""International tools: where we ship, what it costs, duty, customs and currency.

One of the twelve domains added in DESIGN.md 19.15. Rates and rules live in this module
so the domestic shipping tools in `shipping.py` keep working exactly as they did. Only
`_data` is imported, never another domain.
"""

from __future__ import annotations

from typing import Any

from . import _data

# Destination country, its duty-free threshold in USD, the duty rate above it, and how
# many business days delivery takes once the parcel clears customs.
DESTINATIONS: dict[str, dict[str, Any]] = {
    "CA": {"name": "Canada", "duty_free_under": 150.0, "duty_rate": 0.05, "transit_days": 6},
    "GB": {
        "name": "United Kingdom",
        "duty_free_under": 175.0,
        "duty_rate": 0.08,
        "transit_days": 8,
    },
    "DE": {"name": "Germany", "duty_free_under": 165.0, "duty_rate": 0.09, "transit_days": 8},
    "AU": {"name": "Australia", "duty_free_under": 1000.0, "duty_rate": 0.10, "transit_days": 11},
    "JP": {"name": "Japan", "duty_free_under": 110.0, "duty_rate": 0.07, "transit_days": 9},
}

# Indicative rates against USD. Frozen like everything else in the fixture, so a quote
# generated today can be regenerated next year and match.
EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0,
    "CAD": 1.37,
    "GBP": 0.79,
    "EUR": 0.92,
    "AUD": 1.52,
    "JPY": 151.0,
}

# Categories that cannot cross certain borders. The list is short on purpose: it exists
# so a customer can be told no with a reason rather than discovering it at the border.
RESTRICTIONS: dict[str, tuple[str, ...]] = {
    "JP": ("electronics",),
    "AU": ("home",),
}

CUSTOMS_STAGES: tuple[str, ...] = ("SUBMITTED", "IN_CUSTOMS", "CLEARED", "DELIVERED")


def _seed_declarations() -> dict[str, dict[str, Any]]:
    return {}


DECLARATIONS: dict[str, dict[str, Any]] = _seed_declarations()


def _reset() -> None:
    DECLARATIONS.clear()
    DECLARATIONS.update(_seed_declarations())


_data.register_reset(_reset)


def _country(code: str) -> tuple[str, dict[str, Any] | None]:
    """Normalise a country code and return its record, or None if we do not ship there."""
    key = code.strip().upper()
    return key, DESTINATIONS.get(key)


def list_shipping_destinations() -> dict[str, Any]:
    """List every country we ship to and what each one involves.

    Near-static reference. Use this when a customer asks whether we deliver to them, or
    is comparing where to send a gift.

    Returns:
        A count and a list of destinations. Each carries the two-letter country code,
        the country name, how long delivery takes in business days once the parcel
        clears customs, the value below which no duty is charged, the duty rate above
        that, and any product categories that cannot be sent there. Anything not on
        this list cannot be shipped at all.
    """
    countries = [
        {
            "country_code": code,
            "country": record["name"],
            "transit_days": record["transit_days"],
            "duty_free_under": record["duty_free_under"],
            "duty_rate_pct": round(record["duty_rate"] * 100, 1),
            "restricted_categories": list(RESTRICTIONS.get(code, ())),
        }
        for code, record in sorted(DESTINATIONS.items())
    ]
    return {"count": len(countries), "currency": _data.CURRENCY, "destinations": countries}


def quote_international_shipping(country_code: str, sku: str, quantity: int) -> dict[str, Any]:
    """Price a shipment abroad, including postage and estimated duty.

    Give a customer this before they order, not after. The duty line is what surprises
    people, so quote it even when it comes out at zero.

    Args:
        country_code: The two-letter destination code, for example "CA" or "GB". Get
            valid codes from `list_shipping_destinations`.
        sku: The product being sent, for example "SKU-HDP-04".
        quantity: How many units. Must be at least 1.

    Returns:
        The goods value, the shipping cost, the estimated duty, the estimated tax, the
        all-in total the customer pays, and the expected delivery date. Also returns
        whether the goods fall under the duty-free threshold, which is the number worth
        quoting when a customer is deciding how much to order. Returns an error field
        when the country is not served or the product cannot be sent there.
    """
    code, destination = _country(country_code)
    if destination is None:
        return {
            "error": f"we do not ship to {code!r}",
            "served": sorted(DESTINATIONS),
        }
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)
    if quantity < 1:
        return {"error": "quantity must be at least 1"}
    if product["category"] in RESTRICTIONS.get(code, ()):
        return {
            "country_code": code,
            "sku": sku,
            "shippable": False,
            "detail": f"{product['category']} cannot be sent to {destination['name']}",
        }

    goods = round(product["unit_price"] * quantity, 2)
    shipping = round(18.0 + 4.5 * quantity, 2)
    dutiable = goods > destination["duty_free_under"]
    duty = round(goods * destination["duty_rate"], 2) if dutiable else 0.0
    tax = round(goods * 0.06, 2)
    return {
        "country_code": code,
        "country": destination["name"],
        "sku": sku,
        "quantity": quantity,
        "shippable": True,
        "goods_value": goods,
        "shipping": shipping,
        "duty": duty,
        "tax": tax,
        "total": round(goods + shipping + duty + tax, 2),
        "currency": _data.CURRENCY,
        "under_duty_free_threshold": not dutiable,
        "duty_free_under": destination["duty_free_under"],
        "expected_delivery": _data.business_days_from_today(destination["transit_days"]),
    }


def estimate_customs_duty(
    country_code: str, declared_value: float, category: str
) -> dict[str, Any]:
    """Estimate the duty and tax on goods already being sent abroad.

    Use this when a customer has been billed at the border and wants to know whether
    the figure is right, or when they are sending something we did not sell them.

    Args:
        country_code: The two-letter destination code, for example "DE".
        declared_value: The value of the goods in USD, as declared on the paperwork.
        category: What kind of goods, for example "electronics", "apparel", "home" or
            "accessories". Affects nothing today but is recorded on the declaration.

    Returns:
        The duty-free threshold for that country, whether this shipment is over it, the
        duty, the handling fee the carrier adds, the estimated tax and the total the
        customer will be asked for on delivery. Figures are estimates: the border
        decides, and it is worth saying so.
    """
    code, destination = _country(country_code)
    if destination is None:
        return {"error": f"we do not ship to {code!r}", "served": sorted(DESTINATIONS)}
    if declared_value < 0:
        return {"error": "declared value cannot be negative"}

    dutiable = declared_value > destination["duty_free_under"]
    duty = round(declared_value * destination["duty_rate"], 2) if dutiable else 0.0
    handling = 12.0 if dutiable else 0.0
    tax = round(declared_value * 0.06, 2)
    return {
        "country_code": code,
        "country": destination["name"],
        "category": category.strip().lower(),
        "declared_value": round(declared_value, 2),
        "duty_free_under": destination["duty_free_under"],
        "over_threshold": dutiable,
        "duty": duty,
        "carrier_handling_fee": handling,
        "tax": tax,
        "payable_on_delivery": round(duty + handling + tax, 2),
        "currency": _data.CURRENCY,
        "estimate_only": True,
    }


def submit_customs_declaration(
    order_id: str, contents: str, declared_value: float
) -> dict[str, Any]:
    """File the customs paperwork for an order going abroad.

    A parcel cannot leave the country without this. File it as soon as the customer
    confirms the contents description, since a vague description is the usual cause of
    a parcel sitting in customs.

    Args:
        order_id: The order being sent, for example "A1001".
        contents: A plain description of what is in the parcel, for example "one pair
            of over-ear headphones". Must be specific: "gift" or "goods" is rejected.
        declared_value: The value of the contents in USD.

    Returns:
        A `filed` flag, the declaration reference, the stage, and the expected clearance
        date. Returns `filed: false` with a reason when the contents description is too
        vague, when the value is not positive, or when a declaration already exists for
        that order, in which case the existing reference is returned instead.
    """
    order = _data.ORDERS.get(order_id)
    if order is None:
        return _data.not_found("order", order_id)

    description = contents.strip()
    if len(description) < 8 or description.lower() in {"gift", "goods", "items", "presents"}:
        return {
            "order_id": order_id,
            "filed": False,
            "detail": "customs needs a specific description of the contents",
        }
    if declared_value <= 0:
        return {"order_id": order_id, "filed": False, "detail": "declared value must be positive"}

    existing = next((d for d in DECLARATIONS.values() if d["order_id"] == order_id), None)
    if existing is not None:
        return {
            "order_id": order_id,
            "filed": False,
            "detail": "a declaration is already on file for this order",
            "reference": existing["reference"],
            "stage": existing["stage"],
        }

    reference = f"CUS-{_data.stable_int(order_id + description, 10000, 99999)}"
    DECLARATIONS[reference] = {
        "reference": reference,
        "order_id": order_id,
        "contents": description,
        "declared_value": round(declared_value, 2),
        "stage": "SUBMITTED",
        "filed_on": str(_data.TODAY),
    }
    return {
        "reference": reference,
        "order_id": order_id,
        "filed": True,
        "stage": "SUBMITTED",
        "declared_value": round(declared_value, 2),
        "expected_clearance": _data.business_days_from_today(3),
    }


def get_customs_status(reference: str) -> dict[str, Any]:
    """Check where a parcel has got to in customs.

    Args:
        reference: The declaration reference, for example "CUS-40318". Issued by
            `submit_customs_declaration` and starting with "CUS-".

    Returns:
        The current stage, which runs SUBMITTED, IN_CUSTOMS, CLEARED then DELIVERED;
        the order it belongs to; the declared value; the date it was filed; and any
        charge outstanding at the border. Each call advances the declaration one stage,
        so a customer chasing a stuck parcel will see it move.
    """
    record = DECLARATIONS.get(reference)
    if record is None:
        return _data.not_found("customs declaration", reference)

    index = CUSTOMS_STAGES.index(record["stage"])
    if index < len(CUSTOMS_STAGES) - 1:
        record["stage"] = CUSTOMS_STAGES[index + 1]
    held = record["stage"] == "IN_CUSTOMS"
    return {
        "reference": reference,
        "order_id": record["order_id"],
        "stage": record["stage"],
        "declared_value": record["declared_value"],
        "filed_on": record["filed_on"],
        "held_for_payment": held,
        "outstanding_charge": round(record["declared_value"] * 0.07, 2) if held else 0.0,
        "currency": _data.CURRENCY,
    }


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
    """Convert a price between currencies at our indicative rate.

    Use this when a customer abroad asks what something will cost them in their own
    money. The rate is indicative, not the one their card issuer will use, and it is
    worth saying so when the amount is large.

    Args:
        amount: The amount to convert.
        from_currency: The three-letter code to convert from, for example "USD". One of
            USD, CAD, GBP, EUR, AUD or JPY.
        to_currency: The three-letter code to convert to, for example "GBP".

    Returns:
        The converted amount rounded to two decimal places, the rate used, both
        currency codes, and a note that the rate is indicative. Returns an error field
        listing the supported currencies when either code is not one of them.
    """
    source = from_currency.strip().upper()
    target = to_currency.strip().upper()
    if source not in EXCHANGE_RATES or target not in EXCHANGE_RATES:
        return {
            "error": "unsupported currency",
            "supported": sorted(EXCHANGE_RATES),
        }

    rate = EXCHANGE_RATES[target] / EXCHANGE_RATES[source]
    return {
        "amount": round(amount, 2),
        "from_currency": source,
        "to_currency": target,
        "rate": round(rate, 4),
        "converted": round(amount * rate, 2),
        "indicative_only": True,
    }
