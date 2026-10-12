"""Shipping tools: track a parcel, price the options, estimate a delivery date.

Shipment status is derived from the order it belongs to rather than stored twice, so
tracking and order status can never contradict each other. Unlike `get_order_status`,
tracking does not advance anything.
"""

from __future__ import annotations

from typing import Any

from . import _data

# Speed -> (business days, flat price, free-over threshold).
_SHIPPING_SPEEDS: dict[str, tuple[int, float, float]] = {
    "standard": (4, 5.95, 50.00),
    "express": (2, 14.95, 150.00),
    "overnight": (1, 29.95, 500.00),
}


def track_shipment(tracking_number: str) -> dict[str, Any]:
    """Track a parcel by its carrier tracking number.

    Args:
        tracking_number: The tracking number, for example "1Z-A1001".

    Returns the carrier, the shipment status, the last scan location and the estimated
    arrival date.
    """
    shipment = _data.SHIPMENTS.get(tracking_number)
    if shipment is None:
        return _data.not_found("shipment", tracking_number)

    order = _data.ORDERS[shipment["order_id"]]
    order_state = _data.order_status(order)
    return {
        "tracking_number": tracking_number,
        "order_id": shipment["order_id"],
        "carrier": shipment["carrier"],
        "status": _data.SHIPMENT_STATUS.get(order_state, order_state),
        "shipped_on": shipment["shipped_on"] or None,
        "last_scan": shipment["last_scan"] or None,
        "estimated_arrival": shipment["eta"],
    }


def get_shipping_options(destination_zip: str, weight_kg: float) -> dict[str, Any]:
    """List the shipping speeds available to a destination, with prices.

    Args:
        destination_zip: Destination postal code, for example "48104".
        weight_kg: Parcel weight in kilograms.

    Returns one row per speed: name, business days in transit, price, and the order
    value above which that speed is free.
    """
    surcharge = round(max(0.0, weight_kg - 2.0) * 1.50, 2)
    options = [
        {
            "speed": speed,
            "business_days": days,
            "price": round(price + surcharge, 2),
            "currency": _data.CURRENCY,
            "free_over": free_over,
        }
        for speed, (days, price, free_over) in _SHIPPING_SPEEDS.items()
    ]
    return {
        "destination_zip": destination_zip,
        "weight_kg": weight_kg,
        "weight_surcharge": surcharge,
        "options": options,
    }


def estimate_delivery(destination_zip: str, speed: str) -> dict[str, Any]:
    """Estimate when an order shipped today would arrive.

    Args:
        destination_zip: Destination postal code, for example "78745".
        speed: "standard", "express" or "overnight".

    Returns the estimated delivery date and the transit days it is based on.
    """
    key = speed.strip().lower() or "standard"
    if key not in _SHIPPING_SPEEDS:
        return {
            "speed": key,
            "known_speeds": sorted(_SHIPPING_SPEEDS),
            "error": f"unknown shipping speed {key!r}",
        }

    base_days = _SHIPPING_SPEEDS[key][0]
    # Distance stands in for a routing lookup: derived from the zip, so it is stable.
    zone_days = _data.stable_int(f"zone:{destination_zip}", 0, 2)
    total_days = base_days + zone_days
    return {
        "destination_zip": destination_zip,
        "speed": key,
        "business_days": total_days,
        "estimated_delivery_date": _data.business_days_from_today(total_days),
        "assumes": "shipped today",
    }
