"""Business account tools: quotes, purchase orders, bulk pricing, terms, tax exemption.

One of the twelve domains added in DESIGN.md 19.15. Business accounts are keyed by their
own account id rather than by the signed-in customer, because the person on the phone is
often a buyer acting for a company. Only `_data` is imported, never another domain.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from . import _data

# Volume breaks. A quantity at or above the first number takes the discount beside it.
VOLUME_BREAKS: tuple[tuple[int, float], ...] = (
    (500, 0.20),
    (250, 0.15),
    (100, 0.10),
    (25, 0.05),
)

PAYMENT_TERMS: tuple[str, ...] = ("prepay", "net15", "net30", "net60")
PO_STAGES: tuple[str, ...] = ("RECEIVED", "APPROVED", "FULFILLING", "INVOICED")


def _seed_business_accounts() -> dict[str, dict[str, Any]]:
    return {
        "BIZ-4410": {
            "account_id": "BIZ-4410",
            "company": "Fenwick Design Studio",
            "buyer_email": "buying@fenwick.example",
            "terms": "net30",
            "credit_limit": 25000.00,
            "credit_used": 4180.00,
            "tax_exempt": True,
            "exemption_certificate": "MI-TX-88213",
            "contract_discount_pct": 8.0,
        },
        "BIZ-7702": {
            "account_id": "BIZ-7702",
            "company": "Northline Logistics",
            "buyer_email": "ap@northline.example",
            "terms": "net15",
            "credit_limit": 10000.00,
            "credit_used": 9650.00,
            "tax_exempt": False,
            "exemption_certificate": "",
            "contract_discount_pct": 0.0,
        },
    }


BUSINESS_ACCOUNTS: dict[str, dict[str, Any]] = _seed_business_accounts()
QUOTES: dict[str, dict[str, Any]] = {}
PURCHASE_ORDERS: dict[str, dict[str, Any]] = {}


def _reset() -> None:
    BUSINESS_ACCOUNTS.clear()
    BUSINESS_ACCOUNTS.update(_seed_business_accounts())
    QUOTES.clear()
    PURCHASE_ORDERS.clear()


_data.register_reset(_reset)


def _volume_discount(quantity: int) -> float:
    """The fraction off for this quantity, 0.0 below the first break."""
    for threshold, discount in VOLUME_BREAKS:
        if quantity >= threshold:
            return discount
    return 0.0


def get_business_account(account_id: str) -> dict[str, Any]:
    """Read a business account: who they are, what they owe and what they have agreed.

    Start here on any business call. Everything else in this domain depends on the
    terms and the credit headroom this returns.

    Args:
        account_id: The business account identifier, for example "BIZ-4410". These
            start with "BIZ-" and are quoted on every invoice.

    Returns:
        The company name, the buyer's email, the payment terms out of prepay, net15,
        net30 and net60, the credit limit, how much of it is used, how much is left,
        whether the account is tax exempt and its certificate number, and any contract
        discount. Credit remaining is the number that decides whether an order can go
        through on terms.
    """
    account = BUSINESS_ACCOUNTS.get(account_id)
    if account is None:
        return _data.not_found("business account", account_id)
    return {
        **account,
        "credit_remaining": round(account["credit_limit"] - account["credit_used"], 2),
        "currency": _data.CURRENCY,
    }


def get_bulk_pricing(sku: str, quantity: int) -> dict[str, Any]:
    """Price a product at volume, with the quantity breaks applied.

    Use this whenever a customer asks about buying more than a handful. Show them the
    next break as well as their own price: the usual outcome is that they buy up to it.

    Args:
        sku: The product being priced, for example "SKU-CBL-06".
        quantity: How many units they want. Breaks are at 25, 100, 250 and 500 units,
            worth 5, 10, 15 and 20 percent respectively.

    Returns:
        The list price, the discount applied, the unit price at that discount, the line
        total and the amount saved. Also returns the next quantity break, how many more
        units it takes to reach it and what it would be worth, so the customer can
        decide. Returns the list price with no discount below 25 units.
    """
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)
    if quantity < 1:
        return {"error": "quantity must be at least 1"}

    list_price = product["unit_price"]
    discount = _volume_discount(quantity)
    unit = round(list_price * (1 - discount), 2)
    higher = [break_at for break_at, _ in reversed(VOLUME_BREAKS) if break_at > quantity]
    next_break = higher[0] if higher else None
    next_discount = _volume_discount(next_break) if next_break else 0.0
    return {
        "sku": sku,
        "name": product["name"],
        "quantity": quantity,
        "list_unit_price": list_price,
        "discount_pct": round(discount * 100, 1),
        "unit_price": unit,
        "line_total": round(unit * quantity, 2),
        "saving": round((list_price - unit) * quantity, 2),
        "currency": _data.CURRENCY,
        "next_break_at": next_break,
        "units_to_next_break": next_break - quantity if next_break else None,
        "next_break_discount_pct": round(next_discount * 100, 1) if next_break else None,
    }


def request_quote(account_id: str, sku: str, quantity: int) -> dict[str, Any]:
    """Produce a formal, dated quote a business can take to their finance team.

    A quote holds the price for 30 days and is what a buyer needs before they can raise
    a purchase order. Volume and contract discounts both apply, and tax is left off for
    exempt accounts.

    Args:
        account_id: The business account the quote is for, for example "BIZ-4410".
        sku: The product being quoted, for example "SKU-HDP-04".
        quantity: How many units.

    Returns:
        The quote reference, the unit price after both discounts, the subtotal, the tax
        or zero when the account is exempt, the total, the expiry date and the payment
        terms. Quote the reference to `submit_purchase_order` to turn it into an order.
        Returns an error field when the account or the product is unknown.
    """
    account = BUSINESS_ACCOUNTS.get(account_id)
    if account is None:
        return _data.not_found("business account", account_id)
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)
    if quantity < 1:
        return {"error": "quantity must be at least 1"}

    volume = _volume_discount(quantity)
    contract = account["contract_discount_pct"] / 100
    unit = round(product["unit_price"] * (1 - volume) * (1 - contract), 2)
    subtotal = round(unit * quantity, 2)
    tax = 0.0 if account["tax_exempt"] else round(subtotal * 0.06, 2)
    reference = f"QTE-{_data.stable_int(account_id + sku + str(quantity), 10000, 99999)}"
    QUOTES[reference] = {
        "reference": reference,
        "account_id": account_id,
        "sku": sku,
        "quantity": quantity,
        "unit_price": unit,
        "subtotal": subtotal,
        "tax": tax,
        "total": round(subtotal + tax, 2),
        "issued_on": str(_data.TODAY),
        "expires_on": str(_data.TODAY + timedelta(days=30)),
    }
    return {
        **QUOTES[reference],
        "company": account["company"],
        "volume_discount_pct": round(volume * 100, 1),
        "contract_discount_pct": account["contract_discount_pct"],
        "tax_exempt": account["tax_exempt"],
        "payment_terms": account["terms"],
        "currency": _data.CURRENCY,
    }


def submit_purchase_order(account_id: str, quote_reference: str, po_number: str) -> dict[str, Any]:
    """Turn an accepted quote into an order against the account's credit.

    This is the point where money is committed, so it checks the quote is still valid
    and that the account has the headroom. If credit is short, say so with the numbers
    rather than refusing flatly.

    Args:
        account_id: The business account placing the order, for example "BIZ-4410".
        quote_reference: The quote being accepted, for example "QTE-40318".
        po_number: The buyer's own purchase order number, in their format. Printed on
            the invoice so their finance team can match it.

    Returns:
        An `accepted` flag, our order reference, the stage, the amount committed, the
        credit left afterwards, the payment due date and the expected despatch date.
        Returns `accepted: false` with the shortfall when the order exceeds available
        credit, when the quote has expired, or when the quote belongs to another
        account, which must never go through.
    """
    account = BUSINESS_ACCOUNTS.get(account_id)
    if account is None:
        return _data.not_found("business account", account_id)
    quote = QUOTES.get(quote_reference.strip().upper())
    if quote is None:
        return _data.not_found("quote", quote_reference)
    if quote["account_id"] != account_id:
        return {
            "accepted": False,
            "detail": "that quote was issued to a different account",
        }
    if quote["expires_on"] < str(_data.TODAY):
        return {
            "accepted": False,
            "detail": f"quote {quote['reference']} expired on {quote['expires_on']}",
        }

    remaining = account["credit_limit"] - account["credit_used"]
    if quote["total"] > remaining and account["terms"] != "prepay":
        return {
            "accepted": False,
            "detail": "this order is over the available credit",
            "order_total": quote["total"],
            "credit_remaining": round(remaining, 2),
            "shortfall": round(quote["total"] - remaining, 2),
        }

    account["credit_used"] = round(account["credit_used"] + quote["total"], 2)
    order_reference = f"PO-{_data.stable_int(account_id + po_number, 10000, 99999)}"
    net_days = {"prepay": 0, "net15": 15, "net30": 30, "net60": 60}[account["terms"]]
    PURCHASE_ORDERS[order_reference] = {
        "reference": order_reference,
        "account_id": account_id,
        "po_number": po_number.strip(),
        "quote_reference": quote["reference"],
        "total": quote["total"],
        "stage": "RECEIVED",
        "received_on": str(_data.TODAY),
    }
    return {
        "reference": order_reference,
        "accepted": True,
        "po_number": po_number.strip(),
        "stage": "RECEIVED",
        "committed": quote["total"],
        "credit_remaining": round(account["credit_limit"] - account["credit_used"], 2),
        "payment_due": str(_data.TODAY + timedelta(days=net_days)),
        "expected_despatch": _data.business_days_from_today(4),
        "currency": _data.CURRENCY,
    }


def get_purchase_order_status(reference: str) -> dict[str, Any]:
    """Check where a business order has got to.

    Args:
        reference: Our order reference, for example "PO-40318". Issued by
            `submit_purchase_order`. The buyer's own PO number will not work here, so
            look it up with `list_business_orders` if that is all they have.

    Returns:
        The current stage, which runs RECEIVED, APPROVED, FULFILLING then INVOICED; the
        buyer's PO number; the total; the date received; and the invoice number once it
        has been raised. Each call advances the order one stage, so a buyer chasing it
        will see progress.
    """
    record = PURCHASE_ORDERS.get(reference)
    if record is None:
        return _data.not_found("purchase order", reference)

    index = PO_STAGES.index(record["stage"])
    if index < len(PO_STAGES) - 1:
        record["stage"] = PO_STAGES[index + 1]
    invoiced = record["stage"] == "INVOICED"
    return {
        "reference": reference,
        "po_number": record["po_number"],
        "stage": record["stage"],
        "total": record["total"],
        "currency": _data.CURRENCY,
        "received_on": record["received_on"],
        "invoice_number": f"INV-{reference}" if invoiced else None,
    }


def list_business_orders(account_id: str) -> dict[str, Any]:
    """List every order placed on a business account.

    Use this when a buyer has their own PO number but not ours, or when finance is
    reconciling and wants the whole picture.

    Args:
        account_id: The business account to list, for example "BIZ-4410".

    Returns:
        A count, the total committed across all orders, and the orders themselves with
        our reference, the buyer's PO number, the stage, the total and the date. Also
        returns the account's credit remaining, since that is the question behind most
        of these calls. Returns an empty list when nothing has been ordered yet.
    """
    account = BUSINESS_ACCOUNTS.get(account_id)
    if account is None:
        return _data.not_found("business account", account_id)

    orders = [
        {
            "reference": record["reference"],
            "po_number": record["po_number"],
            "stage": record["stage"],
            "total": record["total"],
            "received_on": record["received_on"],
        }
        for record in PURCHASE_ORDERS.values()
        if record["account_id"] == account_id
    ]
    orders.sort(key=lambda entry: entry["received_on"], reverse=True)
    return {
        "account_id": account_id,
        "company": account["company"],
        "count": len(orders),
        "committed_total": round(sum(order["total"] for order in orders), 2),
        "credit_remaining": round(account["credit_limit"] - account["credit_used"], 2),
        "currency": _data.CURRENCY,
        "orders": orders,
    }


def update_tax_exemption(account_id: str, certificate_number: str) -> dict[str, Any]:
    """Record or clear a tax exemption certificate on a business account.

    Every quote raised after this reflects it, but nothing already invoiced changes, so
    a buyer asking for tax back on an old invoice needs a credit note instead.

    Args:
        account_id: The business account to update, for example "BIZ-7702".
        certificate_number: The exemption certificate, for example "MI-TX-88213". Pass
            an empty string to remove an exemption, which makes the account taxable
            again from the next quote.

    Returns:
        An `updated` flag, whether the account is now exempt, the certificate on file,
        and a note that existing invoices are unaffected. Returns `updated: false` when
        the certificate given is the one already on file, so nothing is logged as a
        change when nothing changed.
    """
    account = BUSINESS_ACCOUNTS.get(account_id)
    if account is None:
        return _data.not_found("business account", account_id)

    certificate = certificate_number.strip().upper()
    if certificate == account["exemption_certificate"]:
        return {
            "account_id": account_id,
            "updated": False,
            "detail": "that certificate is already on file",
            "tax_exempt": account["tax_exempt"],
        }

    account["exemption_certificate"] = certificate
    account["tax_exempt"] = bool(certificate)
    return {
        "account_id": account_id,
        "updated": True,
        "tax_exempt": account["tax_exempt"],
        "certificate": certificate or None,
        "applies_from": str(_data.TODAY),
        "existing_invoices_unchanged": True,
    }
