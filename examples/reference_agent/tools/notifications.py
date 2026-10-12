"""Notification tools: what we send, where we send it, and how to make it stop.

One of the twelve domains added in DESIGN.md 19.15. Preferences are per customer and
read identity from session state, so this domain sits in the ambient-identity class of
DESIGN.md 8.1. It imports `_data` and no other domain module.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from google.adk.tools import ToolContext

from . import _data

CHANNELS: tuple[str, ...] = ("email", "sms", "push", "post")
TOPICS: tuple[str, ...] = (
    "order_updates",
    "delivery_alerts",
    "price_drops",
    "back_in_stock",
    "marketing",
    "security_alerts",
)
# Two topics are transactional. A customer cannot switch them off, and saying so plainly
# is better than accepting the change and quietly ignoring it.
ALWAYS_ON: frozenset[str] = frozenset({"order_updates", "security_alerts"})


def _seed_preferences() -> dict[str, dict[str, Any]]:
    return {
        "u_1001": {
            "email": True,
            "sms": True,
            "push": False,
            "post": False,
            "topics": {
                "order_updates": True,
                "delivery_alerts": True,
                "price_drops": True,
                "back_in_stock": False,
                "marketing": True,
                "security_alerts": True,
            },
            "quiet_hours": "22:00-07:00",
        },
        "u_1002": {
            "email": True,
            "sms": False,
            "push": True,
            "post": False,
            "topics": {
                "order_updates": True,
                "delivery_alerts": False,
                "price_drops": False,
                "back_in_stock": True,
                "marketing": False,
                "security_alerts": True,
            },
            "quiet_hours": "",
        },
    }


def _seed_history() -> list[dict[str, Any]]:
    """A short send history so a customer can be shown what they actually received."""
    return [
        {
            "user_id": "u_1001",
            "sent_on": "2026-08-26",
            "channel": "email",
            "topic": "order_updates",
            "subject": "Your order A1001 is on its way",
            "delivered": True,
        },
        {
            "user_id": "u_1001",
            "sent_on": "2026-08-24",
            "channel": "sms",
            "topic": "delivery_alerts",
            "subject": "Delivery scheduled for Monday",
            "delivered": True,
        },
        {
            "user_id": "u_1001",
            "sent_on": "2026-08-20",
            "channel": "email",
            "topic": "marketing",
            "subject": "New arrivals for autumn",
            "delivered": False,
        },
        {
            "user_id": "u_1002",
            "sent_on": "2026-08-28",
            "channel": "push",
            "topic": "order_updates",
            "subject": "Order B2001 packed",
            "delivered": True,
        },
    ]


PREFERENCES: dict[str, dict[str, Any]] = _seed_preferences()
HISTORY: list[dict[str, Any]] = _seed_history()
STOCK_ALERTS: dict[str, list[str]] = {}


def _reset() -> None:
    PREFERENCES.clear()
    PREFERENCES.update(_seed_preferences())
    HISTORY.clear()
    HISTORY.extend(_seed_history())
    STOCK_ALERTS.clear()


_data.register_reset(_reset)


def _preferences_for(user_id: str) -> dict[str, Any]:
    """This customer's preferences, defaulted to email-only if they have never set any."""
    return PREFERENCES.setdefault(
        user_id,
        {
            "email": True,
            "sms": False,
            "push": False,
            "post": False,
            "topics": {topic: topic in ALWAYS_ON for topic in TOPICS},
            "quiet_hours": "",
        },
    )


def get_notification_preferences(tool_context: ToolContext) -> dict[str, Any]:
    """Read what the signed-in customer has agreed to be contacted about.

    Takes no arguments: the customer is whoever is signed in. Read this before changing
    anything, so the customer can be told what is on today rather than guessing.

    Returns:
        Which channels are on out of email, sms, push and post; which topics are on out
        of order_updates, delivery_alerts, price_drops, back_in_stock, marketing and
        security_alerts; the customer's quiet hours if they have set any; and the list
        of topics that cannot be switched off because they are transactional.
    """
    user_id = _data.current_user_id(tool_context)
    prefs = _preferences_for(user_id)
    return {
        "channels": {channel: prefs[channel] for channel in CHANNELS},
        "topics": dict(prefs["topics"]),
        "quiet_hours": prefs["quiet_hours"] or "none set",
        "cannot_be_disabled": sorted(ALWAYS_ON),
    }


def update_notification_preferences(
    channel: str, topic: str, enabled: bool, tool_context: ToolContext
) -> dict[str, Any]:
    """Switch one channel or one topic on or off for the signed-in customer.

    Change one thing per call so the customer can be told exactly what happened. To
    switch a channel on or off pass the channel and an empty topic; to switch a topic
    on or off pass the topic and an empty channel.

    Args:
        channel: The channel to change. One of "email", "sms", "push" or "post". Pass
            an empty string when changing a topic instead.
        topic: The topic to change. One of "order_updates", "delivery_alerts",
            "price_drops", "back_in_stock", "marketing" or "security_alerts". Pass an
            empty string when changing a channel instead.
        enabled: True to switch it on, False to switch it off.

    Returns:
        An `updated` flag, what was changed, its value before and after, and the full
        preference set afterwards so nothing needs re-reading. Returns `updated: false`
        when the topic is transactional and cannot be switched off, when neither a
        channel nor a topic was named, or when the name is not one of the valid ones.
    """
    which_channel = channel.strip().lower()
    which_topic = topic.strip().lower()
    if not which_channel and not which_topic:
        return {"updated": False, "detail": "name either a channel or a topic to change"}

    user_id = _data.current_user_id(tool_context)
    prefs = _preferences_for(user_id)

    if which_topic:
        if which_topic not in TOPICS:
            return {"updated": False, "valid_topics": list(TOPICS), "error": "unknown topic"}
        if which_topic in ALWAYS_ON and not enabled:
            return {
                "updated": False,
                "topic": which_topic,
                "detail": "this topic is transactional and stays on while the account is open",
            }
        was = prefs["topics"][which_topic]
        prefs["topics"][which_topic] = enabled
        return {
            "updated": True,
            "topic": which_topic,
            "was": was,
            "now": enabled,
            "topics": dict(prefs["topics"]),
        }

    if which_channel not in CHANNELS:
        return {"updated": False, "valid_channels": list(CHANNELS), "error": "unknown channel"}
    was = prefs[which_channel]
    prefs[which_channel] = enabled
    return {
        "updated": True,
        "channel": which_channel,
        "was": was,
        "now": enabled,
        "channels": {name: prefs[name] for name in CHANNELS},
    }


def unsubscribe_all_marketing(tool_context: ToolContext) -> dict[str, Any]:
    """Switch off every optional message for the signed-in customer at once.

    Use this when a customer says they want it all to stop. Transactional messages
    about their own orders and account security keep going, and it is worth saying so
    rather than promising silence you cannot deliver.

    Returns:
        An `unsubscribed` flag, the list of topics that were switched off, the list
        that stays on because it is transactional, and how long the change takes to
        reach messages already queued. Returns `unsubscribed: false` when everything
        optional was already off, so the customer is not told something changed.
    """
    user_id = _data.current_user_id(tool_context)
    prefs = _preferences_for(user_id)
    turned_off = [
        topic for topic in TOPICS if topic not in ALWAYS_ON and prefs["topics"].get(topic)
    ]
    for topic in turned_off:
        prefs["topics"][topic] = False

    if not turned_off:
        return {
            "unsubscribed": False,
            "detail": "every optional message was already switched off",
            "still_on": sorted(ALWAYS_ON),
        }
    return {
        "unsubscribed": True,
        "switched_off": turned_off,
        "still_on": sorted(ALWAYS_ON),
        "takes_effect_within_hours": 24,
    }


def get_notification_history(days: int, tool_context: ToolContext) -> dict[str, Any]:
    """List what has actually been sent to the signed-in customer recently.

    Use this when a customer says they never got an email, or asks why they are getting
    so many. It shows what was sent and whether it was delivered, which usually settles
    the question before anything is changed.

    Args:
        days: How far back to look, in days. Values above 60 are treated as 60, since
            send history is not kept longer than that.

    Returns:
        A count and a list of messages, newest first, each with the date, the channel,
        the topic, the subject line and whether it reached the customer. Also returns
        how many failed to deliver, which is the number that explains a complaint about
        missing messages.
    """
    window = max(1, min(60, days))
    user_id = _data.current_user_id(tool_context)
    cutoff = str(_data.TODAY - timedelta(days=window))
    sent = [
        {k: v for k, v in entry.items() if k != "user_id"}
        for entry in HISTORY
        if entry["user_id"] == user_id and entry["sent_on"] >= cutoff
    ]
    sent.sort(key=lambda entry: entry["sent_on"], reverse=True)
    return {
        "days": window,
        "count": len(sent),
        "undelivered_count": sum(1 for entry in sent if not entry["delivered"]),
        "messages": sent,
    }


def set_stock_alert(sku: str, tool_context: ToolContext) -> dict[str, Any]:
    """Tell the signed-in customer when an out-of-stock product comes back.

    This is the right answer whenever a customer wants something that is sold out. It
    turns a dead end into a promise you can keep.

    Args:
        sku: The product to watch, for example "SKU-MUG-02".

    Returns:
        An `alert_set` flag, the product name, the channel the alert will use, and the
        expected restock date. Returns `alert_set: false` when the item is already in
        stock, since the customer can just buy it, and when an alert is already set on
        that product for this customer.
    """
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)

    user_id = _data.current_user_id(tool_context)
    if product["stock"] > 0:
        return {
            "sku": sku,
            "alert_set": False,
            "detail": f"{product['name']} is in stock now, with {product['stock']} available",
        }

    watching = STOCK_ALERTS.setdefault(user_id, [])
    if sku in watching:
        return {"sku": sku, "alert_set": False, "detail": "an alert is already set on this item"}

    watching.append(sku)
    prefs = _preferences_for(user_id)
    channel = "email" if prefs["email"] else next((c for c in CHANNELS if prefs[c]), "email")
    return {
        "sku": sku,
        "alert_set": True,
        "name": product["name"],
        "channel": channel,
        "expected_restock": _data.business_days_from_today(_data.stable_int(sku, 3, 21)),
    }


def list_stock_alerts(tool_context: ToolContext) -> dict[str, Any]:
    """List the out-of-stock products the signed-in customer is waiting on.

    Takes no arguments beyond identity. Use it when a customer asks what they are
    watching, or before setting an alert they may already have.

    Returns:
        A count and a list of alerts. Each carries the product code, its name, whether
        it is back in stock already, the current price and the expected restock date.
        An item that is back in stock is still listed, flagged as ready, because that
        is the one the customer wants to hear about.
    """
    user_id = _data.current_user_id(tool_context)
    alerts = []
    for sku in STOCK_ALERTS.get(user_id, []):
        product = _data.PRODUCTS.get(sku, {})
        alerts.append(
            {
                "sku": sku,
                "name": product.get("name", sku),
                "back_in_stock": bool(product.get("stock", 0)),
                "unit_price": product.get("unit_price"),
                "expected_restock": _data.business_days_from_today(_data.stable_int(sku, 3, 21)),
            }
        )
    return {
        "count": len(alerts),
        "currency": _data.CURRENCY,
        "ready_count": sum(1 for alert in alerts if alert["back_in_stock"]),
        "alerts": alerts,
    }
