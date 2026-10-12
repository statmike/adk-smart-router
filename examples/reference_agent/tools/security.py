"""Account security tools: posture, locks, sign-in history, fraud cases, verification.

One of the twelve domains added in DESIGN.md 19.15. Every tool here reads identity from
session state rather than from an argument, which puts the whole domain in the
ambient-identity class of DESIGN.md 8.1. If `user_id` ever falls out of the cache key,
these leak first and they leak the worst, so the cross-user isolation test covers them.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from google.adk.tools import ToolContext

from . import _data

FRAUD_STAGES: tuple[str, ...] = ("OPEN", "INVESTIGATING", "RESOLVED")
FRAUD_KINDS: tuple[str, ...] = (
    "unrecognised_charge",
    "account_takeover",
    "phishing_email",
    "card_stolen",
    "delivery_theft",
)
VERIFICATION_METHODS: tuple[str, ...] = ("email", "sms", "id_document")


def _seed_logins() -> dict[str, list[dict[str, Any]]]:
    """Recent sign-ins, one of them from an unusual place for each seeded customer."""
    return {
        "u_1001": [
            {"on": "2026-08-30", "city": "Ann Arbor", "device": "Chrome on macOS", "usual": True},
            {"on": "2026-08-28", "city": "Ann Arbor", "device": "iOS app", "usual": True},
            {"on": "2026-08-21", "city": "Lisbon", "device": "Chrome on Windows", "usual": False},
        ],
        "u_1002": [
            {"on": "2026-08-29", "city": "Detroit", "device": "Android app", "usual": True},
            {"on": "2026-08-15", "city": "Detroit", "device": "Firefox on Linux", "usual": True},
        ],
    }


LOGINS: dict[str, list[dict[str, Any]]] = _seed_logins()
LOCKS: dict[str, dict[str, Any]] = {}
FRAUD_CASES: dict[str, dict[str, Any]] = {}
VERIFICATIONS: dict[str, dict[str, Any]] = {}


def _reset() -> None:
    LOGINS.clear()
    LOGINS.update(_seed_logins())
    LOCKS.clear()
    FRAUD_CASES.clear()
    VERIFICATIONS.clear()


_data.register_reset(_reset)


def get_account_security_status(tool_context: ToolContext) -> dict[str, Any]:
    """Get the security posture of the signed-in customer's account.

    Takes no arguments: the customer is whoever is signed in. Start here whenever a
    customer says they think something is wrong with their account, before locking
    anything or opening a case.

    Returns:
        Whether the account is locked and why, whether two-factor authentication is
        on, the date the password was last changed, how many sign-ins in the last
        thirty days came from an unusual place, whether any fraud case is open, and a
        plain-language risk level of "low", "medium" or "high". A high level means the
        account should be locked before anything else is discussed.
    """
    account = _data.account_for(tool_context)
    user_id = account["user_id"]
    unusual = sum(1 for entry in LOGINS.get(user_id, []) if not entry["usual"])
    open_cases = [
        c for c in FRAUD_CASES.values() if c["user_id"] == user_id and c["stage"] != "RESOLVED"
    ]
    lock = LOCKS.get(user_id)
    risk = "high" if open_cases else "medium" if unusual else "low"
    return {
        "locked": lock is not None,
        "lock_reason": lock["reason"] if lock else None,
        "two_factor_enabled": _data.stable_int(user_id, 0, 1) == 1,
        "password_changed_on": str(
            _data.TODAY - timedelta(days=_data.stable_int(user_id, 20, 400))
        ),
        "unusual_sign_ins_30d": unusual,
        "open_fraud_cases": len(open_cases),
        "risk_level": risk,
    }


def list_recent_logins(days: int, tool_context: ToolContext) -> dict[str, Any]:
    """List where and when the signed-in customer's account has been used.

    Use this when a customer suspects someone else has been in their account, so they
    can point at a session they do not recognise.

    Args:
        days: How far back to look, in days. Values above 90 are treated as 90, since
            sign-in history is not kept longer than that.

    Returns:
        A count and a list of sign-ins, newest first, each with the date, the city,
        the device and browser, and whether it matches the customer's usual pattern.
        Also returns how many were unusual, which is the number worth reading out
        rather than the whole list.
    """
    window = max(1, min(90, days))
    user_id = _data.current_user_id(tool_context)
    cutoff = str(_data.TODAY - timedelta(days=window))
    entries = [entry for entry in LOGINS.get(user_id, []) if entry["on"] >= cutoff]
    entries.sort(key=lambda entry: entry["on"], reverse=True)
    return {
        "days": window,
        "count": len(entries),
        "unusual_count": sum(1 for entry in entries if not entry["usual"]),
        "sign_ins": entries,
    }


def lock_account(reason: str, tool_context: ToolContext) -> dict[str, Any]:
    """Lock the signed-in customer's account against further sign-ins.

    Do this immediately when a customer reports their account has been taken over, or
    when they have lost a device that stays signed in. It is reversible and locking
    early costs nothing, so prefer it over waiting for proof.

    Args:
        reason: Why the account is being locked, in one sentence. Shown to the
            customer on their next sign-in attempt and recorded for the fraud team.

    Returns:
        A `locked` flag, when the lock took effect, what it stops, and how the
        customer gets back in. Returns `locked: false` when the account was already
        locked, along with the original reason and date, so nobody is told a fresh
        lock was applied when it was not.
    """
    account = _data.account_for(tool_context)
    user_id = account["user_id"]
    existing = LOCKS.get(user_id)
    if existing is not None:
        return {
            "locked": False,
            "detail": "this account was already locked",
            "lock_reason": existing["reason"],
            "locked_on": existing["on"],
        }

    LOCKS[user_id] = {"reason": reason.strip(), "on": str(_data.TODAY)}
    return {
        "locked": True,
        "locked_on": str(_data.TODAY),
        "stops": "new sign-ins, password resets, address changes and new payment methods",
        "does_not_stop": "orders already placed and in transit",
        "unlock_route": "identity verification, then unlock_account",
    }


def unlock_account(verification_reference: str, tool_context: ToolContext) -> dict[str, Any]:
    """Lift a lock on the signed-in customer's account.

    A lock can only be lifted once identity has been verified, so call
    `request_identity_verification` first and pass its reference here.

    Args:
        verification_reference: The reference from a completed identity check, for
            example "IDV-4471". Passing an empty string is refused rather than
            treated as a bypass.

    Returns:
        An `unlocked` flag, how long the account was locked for, and what the customer
        should do next. Returns `unlocked: false` when the account is not locked, when
        no reference was given, or when the reference does not belong to this account,
        which is the case that must never succeed.
    """
    account = _data.account_for(tool_context)
    user_id = account["user_id"]
    lock = LOCKS.get(user_id)
    if lock is None:
        return {"unlocked": False, "detail": "this account is not locked"}

    reference = verification_reference.strip().upper()
    if not reference:
        return {"unlocked": False, "detail": "identity has to be verified before unlocking"}
    record = VERIFICATIONS.get(reference)
    if record is None or record["user_id"] != user_id:
        return {
            "unlocked": False,
            "detail": f"{reference} is not a completed check on this account",
        }

    del LOCKS[user_id]
    return {
        "unlocked": True,
        "was_locked_on": lock["on"],
        "verification_reference": reference,
        "next_step": "change the password, then review recent sign-ins",
    }


def report_fraud(
    kind: str, detail: str, amount: float, tool_context: ToolContext
) -> dict[str, Any]:
    """Open a fraud case on the signed-in customer's account.

    Opening a case starts a formal investigation and freezes any disputed charge, so
    it is the right response to a customer who says money has gone missing.

    Args:
        kind: What kind of fraud is being reported. One of "unrecognised_charge",
            "account_takeover", "phishing_email", "card_stolen" or "delivery_theft".
        detail: What happened, in the customer's own words. Given to the investigator
            verbatim, so keep specifics like dates and amounts the customer mentions.
        amount: The disputed amount in the account currency. Pass 0 when no money is
            involved, which is usual for a phishing report.

    Returns:
        A `case_opened` flag, the case id, the stage, whether the disputed amount has
        been provisionally credited, and when the customer will hear back. Cases with
        an amount above 100 are escalated automatically. Returns an error field naming
        the valid kinds when the kind is not one of them.
    """
    key = kind.strip().lower()
    if key not in FRAUD_KINDS:
        return {
            "case_opened": False,
            "valid_kinds": list(FRAUD_KINDS),
            "error": f"{kind!r} is not a fraud kind",
        }

    account = _data.account_for(tool_context)
    user_id = account["user_id"]
    case_id = f"FRD-{_data.stable_int(user_id + key + detail, 10000, 99999)}"
    provisional = amount > 0 and amount <= 100
    FRAUD_CASES[case_id] = {
        "case_id": case_id,
        "user_id": user_id,
        "kind": key,
        "detail": detail.strip(),
        "amount": round(amount, 2),
        "stage": "OPEN",
        "opened_on": str(_data.TODAY),
    }
    if provisional:
        account["store_credit"] = round(account["store_credit"] + amount, 2)
    return {
        "case_id": case_id,
        "case_opened": True,
        "kind": key,
        "stage": "OPEN",
        "provisional_credit": round(amount, 2) if provisional else 0.0,
        "escalated": amount > 100,
        "response_by": _data.business_days_from_today(5),
    }


def get_fraud_case_status(case_id: str) -> dict[str, Any]:
    """Check where a fraud investigation has got to.

    Args:
        case_id: The case identifier, for example "FRD-40318". Issued by
            `report_fraud` and starting with "FRD-".

    Returns:
        The current stage, which runs OPEN, INVESTIGATING then RESOLVED; what was
        reported; the disputed amount; the date it was opened; and the outcome once
        resolved. Each call advances an unresolved case one stage, so a customer
        chasing it will see movement.
    """
    record = FRAUD_CASES.get(case_id)
    if record is None:
        return _data.not_found("fraud case", case_id)

    index = FRAUD_STAGES.index(record["stage"])
    if index < len(FRAUD_STAGES) - 1:
        record["stage"] = FRAUD_STAGES[index + 1]
    resolved = record["stage"] == "RESOLVED"
    return {
        "case_id": case_id,
        "kind": record["kind"],
        "stage": record["stage"],
        "amount": record["amount"],
        "currency": _data.CURRENCY,
        "opened_on": record["opened_on"],
        "outcome": "refunded in full" if resolved else None,
    }


def request_identity_verification(method: str, tool_context: ToolContext) -> dict[str, Any]:
    """Start an identity check on the signed-in customer.

    Needed before unlocking an account, changing an email on a locked account, or
    releasing a held refund. Pick the lightest method the situation allows.

    Args:
        method: How to verify. One of "email" for a link to the address on file, "sms"
            for a code to the phone on file, or "id_document" for a photo of
            identification, which is slowest and only needed after a takeover.

    Returns:
        A `started` flag, the verification reference to quote later, the method used,
        where the code or link was sent with the destination masked, and how long it
        stays valid. The reference is what `unlock_account` needs. Returns an error
        field naming the valid methods when the method is not one of them.
    """
    key = method.strip().lower()
    if key not in VERIFICATION_METHODS:
        return {
            "started": False,
            "valid_methods": list(VERIFICATION_METHODS),
            "error": f"{method!r} is not a verification method",
        }

    account = _data.account_for(tool_context)
    reference = f"IDV-{_data.stable_int(account['user_id'] + key, 1000, 9999)}"
    VERIFICATIONS[reference] = {
        "reference": reference,
        "user_id": account["user_id"],
        "method": key,
        "on": str(_data.TODAY),
    }
    local, _, domain = account["email"].partition("@")
    destination = f"{local[:2]}***@{domain}" if key == "email" else "the phone on file"
    return {
        "started": True,
        "reference": reference,
        "method": key,
        "sent_to": destination if key != "id_document" else "the secure upload link",
        "valid_minutes": 30 if key != "id_document" else 1440,
    }
