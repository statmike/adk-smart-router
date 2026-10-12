"""Gift card and store credit tools: balances, redemption, transfers, terms.

One of the twelve domains added in DESIGN.md 19.15. Store credit lives on the account
record in `_data`, so a credit issued here shows up in `get_account` immediately, which
is the read-your-own-writes path of DESIGN.md 8.4 running through a second domain.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from . import _data


def _seed_cards() -> dict[str, dict[str, Any]]:
    """Four cards: one full, one part-spent, one empty, one expired."""
    return {
        "GC-4412-8890": {"balance": 50.00, "issued_on": "2026-06-01", "expires_on": "2027-06-01"},
        "GC-7731-2044": {"balance": 12.35, "issued_on": "2025-12-20", "expires_on": "2026-12-20"},
        "GC-9005-1177": {"balance": 0.00, "issued_on": "2026-01-15", "expires_on": "2027-01-15"},
        "GC-2200-6631": {"balance": 25.00, "issued_on": "2024-03-10", "expires_on": "2025-03-10"},
    }


CARDS: dict[str, dict[str, Any]] = _seed_cards()
CREDIT_LEDGER: list[dict[str, Any]] = []


def _reset() -> None:
    CARDS.clear()
    CARDS.update(_seed_cards())
    CREDIT_LEDGER.clear()


_data.register_reset(_reset)


def _expired(card: dict[str, Any]) -> bool:
    return card["expires_on"] < str(_data.TODAY)


def check_gift_card_balance(card_number: str) -> dict[str, Any]:
    """Look up what is left on a gift card.

    Use this before promising a customer they can pay with a card, and before
    `redeem_gift_card`, because an expired card still reports a balance it can no
    longer spend.

    Args:
        card_number: The full card number as printed, for example "GC-4412-8890".
            Four groups separated by hyphens. Spaces are ignored.

    Returns:
        The balance in the account currency, the issue and expiry dates, whether the
        card has expired, and whether it can currently be spent. An expired card
        reports `spendable: false` with its balance intact, because the balance can
        still be moved to store credit by an agent. Returns an error field when the
        number does not match a card we issued.
    """
    number = card_number.replace(" ", "").strip().upper()
    card = CARDS.get(number)
    if card is None:
        return _data.not_found("gift card", card_number)
    return {
        "card_number": number,
        "balance": card["balance"],
        "currency": _data.CURRENCY,
        "issued_on": card["issued_on"],
        "expires_on": card["expires_on"],
        "expired": _expired(card),
        "spendable": not _expired(card) and card["balance"] > 0,
    }


def redeem_gift_card(card_number: str, amount: float, tool_context: ToolContext) -> dict[str, Any]:
    """Move money off a gift card and onto the signed-in customer's store credit.

    Once redeemed the value sits on the account and can be spent on any order, so this
    is also the way to rescue a card the customer keeps losing.

    Args:
        card_number: The card to redeem, for example "GC-4412-8890".
        amount: How much to move, in the account currency. Pass 0 to move the whole
            remaining balance, which is what a customer usually means by "use it up".

    Returns:
        A `redeemed` flag, the amount moved, the card's remaining balance and the
        account's new store credit total. Returns `redeemed: false` with a reason when
        the card has expired, has nothing left, or when the amount asked for is more
        than the balance.
    """
    number = card_number.replace(" ", "").strip().upper()
    card = CARDS.get(number)
    if card is None:
        return _data.not_found("gift card", card_number)
    if _expired(card):
        return {"card_number": number, "redeemed": False, "detail": "this card expired"}
    if card["balance"] <= 0:
        return {"card_number": number, "redeemed": False, "detail": "this card is empty"}

    moving = round(card["balance"] if amount <= 0 else min(amount, card["balance"]), 2)
    if amount > card["balance"]:
        return {
            "card_number": number,
            "redeemed": False,
            "balance": card["balance"],
            "detail": f"the card only has {card['balance']} on it",
        }

    account = _data.account_for(tool_context)
    card["balance"] = round(card["balance"] - moving, 2)
    account["store_credit"] = round(account["store_credit"] + moving, 2)
    CREDIT_LEDGER.append(
        {
            "user_id": account["user_id"],
            "change": moving,
            "reason": f"gift card {number} redeemed",
            "on": str(_data.TODAY),
        }
    )
    return {
        "card_number": number,
        "redeemed": True,
        "amount": moving,
        "card_balance": card["balance"],
        "store_credit": account["store_credit"],
        "currency": _data.CURRENCY,
    }


def get_store_credit_balance(tool_context: ToolContext) -> dict[str, Any]:
    """Get the signed-in customer's store credit balance and how it got there.

    Takes no arguments beyond identity: the customer is whoever is signed in. This is
    one of the ambient-identity tools described in DESIGN.md 8.1.

    Returns:
        The balance in the account currency, the number of ledger entries behind it,
        and the three most recent movements with their amount, reason and date. A
        customer querying an unexpected balance usually wants the ledger, not the
        total, so both come back together.
    """
    account = _data.account_for(tool_context)
    mine = [row for row in CREDIT_LEDGER if row["user_id"] == account["user_id"]]
    return {
        "store_credit": account["store_credit"],
        "currency": _data.CURRENCY,
        "ledger_entries": len(mine),
        "recent": mine[-3:],
    }


def issue_store_credit(amount: float, reason: str, tool_context: ToolContext) -> dict[str, Any]:
    """Add store credit to the signed-in customer's account as a goodwill gesture.

    Use this for service failures a refund does not cover, such as a late delivery on
    an order that did eventually arrive. It is real money, so the reason is recorded
    against the account and is auditable.

    Args:
        amount: How much to add, in the account currency. Must be positive, and single
            gestures above 50 are declined and need a supervisor.
        reason: Why the credit is being given, in one sentence. Recorded on the ledger.

    Returns:
        An `issued` flag, the amount added, the new balance and the ledger entry that
        was written. Returns `issued: false` with a detail line when the amount is not
        positive or is above the goodwill limit, in which case escalate instead.
    """
    if amount <= 0:
        return {"issued": False, "detail": "amount must be positive"}
    if amount > 50:
        return {
            "issued": False,
            "limit": 50,
            "detail": "goodwill credit above 50 needs a supervisor, escalate instead",
        }

    account = _data.account_for(tool_context)
    account["store_credit"] = round(account["store_credit"] + amount, 2)
    entry = {
        "user_id": account["user_id"],
        "change": round(amount, 2),
        "reason": reason,
        "on": str(_data.TODAY),
    }
    CREDIT_LEDGER.append(entry)
    return {
        "issued": True,
        "amount": round(amount, 2),
        "store_credit": account["store_credit"],
        "currency": _data.CURRENCY,
        "entry": entry,
    }


def transfer_store_credit(
    recipient_email: str, amount: float, tool_context: ToolContext
) -> dict[str, Any]:
    """Send some of the signed-in customer's store credit to another customer.

    Customers use this to pass a refund on as a gift. The recipient has to be an
    existing account, and the transfer cannot be reversed once sent.

    Args:
        recipient_email: The email address on the receiving account, for example
            "sam.reyes@example.com".
        amount: How much to send, in the account currency. Must be positive and no
            more than the sender's balance.

    Returns:
        A `sent` flag, the amount, the sender's remaining balance and the recipient's
        masked email. Returns `sent: false` when the recipient has no account, when
        the amount is not positive, or when the sender cannot cover it.
    """
    sender = _data.account_for(tool_context)
    email = recipient_email.strip().lower()
    recipient = next((a for a in _data.ACCOUNTS.values() if a["email"].lower() == email), None)
    if recipient is None:
        return {"sent": False, "detail": f"no account found for {recipient_email!r}"}
    if recipient["user_id"] == sender["user_id"]:
        return {"sent": False, "detail": "that is the signed-in customer's own address"}
    if amount <= 0:
        return {"sent": False, "detail": "amount must be positive"}
    if amount > sender["store_credit"]:
        return {
            "sent": False,
            "store_credit": sender["store_credit"],
            "detail": "not enough store credit to cover that",
        }

    moving = round(amount, 2)
    sender["store_credit"] = round(sender["store_credit"] - moving, 2)
    recipient["store_credit"] = round(recipient["store_credit"] + moving, 2)
    local, _, domain = email.partition("@")
    return {
        "sent": True,
        "amount": moving,
        "store_credit": sender["store_credit"],
        "currency": _data.CURRENCY,
        "recipient": f"{local[:2]}***@{domain}",
    }


def get_gift_card_terms() -> dict[str, Any]:
    """Read the standing terms that apply to gift cards and store credit.

    Near-static reference text. Use it for general questions about how cards work
    rather than questions about one customer's card or balance.

    Returns:
        How long cards last, whether balances expire, whether cards can be refunded to
        a payment method, the transfer rules, and what happens to store credit when an
        account is closed. This text changes rarely and is the same for every customer.
    """
    return {
        "card_validity_months": 12,
        "store_credit_expires": False,
        "refundable_to_card": False,
        "transferable": True,
        "on_account_closure": "unspent store credit is paid out by cheque after 30 days",
        "combinable_with_promotions": True,
    }
