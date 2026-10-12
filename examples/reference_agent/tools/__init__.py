"""The reference agent's 100 stub tools (DESIGN.md 3.2 and 19.15), grouped by domain.

Plain functions. ADK wraps each one as a `FunctionTool`, so the type hints become the
schema and the docstring becomes the description the model reads.

All of them are deterministic and offline: no network, no database, no clock, no
`random`. The original seven domains share one fixture in `_data.py`, which is what keeps
orders, shipments, returns and accounts consistent with each other. Each of the twelve
domains added in DESIGN.md 19.15 owns its own fixture and registers it with `_data` so
`reset_fixtures()` still puts everything back at once.

Conventions:

* Reads return their fields directly; a bad identifier returns `{"error": ...}`.
* Writes return a boolean outcome flag (`created`, `updated`, `refunded`, ...) so the
  model can tell a refusal from a success.
* Tools that take no argument but `tool_context` read identity from session state. That
  is the ambient-identity shape of DESIGN.md 8.1 and it is deliberate. The whole
  `security` domain works this way, which is why it is the one the cross-user isolation
  test leans on.
* Docstrings in the twelve new domains are deliberately fuller than in the original
  seven. Declaration tokens are the variable the routing experiment turns on, and a
  realistic tool surface is written this way (DESIGN.md 19.15).

`ALL_TOOLS` is the list to hand to the root agent. `reset_fixtures()` puts the
in-memory data back to its seeded state, which tests and corpus reruns need because the
write tools really do mutate it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._data import reset_fixtures
from .account import get_account, get_loyalty_points, list_addresses, update_email
from .business import (
    get_bulk_pricing,
    get_business_account,
    get_purchase_order_status,
    list_business_orders,
    request_quote,
    submit_purchase_order,
    update_tax_exemption,
)
from .catalog import check_stock, get_price, get_product_details, search_products
from .giftcards import (
    check_gift_card_balance,
    get_gift_card_terms,
    get_store_credit_balance,
    issue_store_credit,
    redeem_gift_card,
    transfer_store_credit,
)
from .international import (
    convert_currency,
    estimate_customs_duty,
    get_customs_status,
    list_shipping_destinations,
    quote_international_shipping,
    submit_customs_declaration,
)
from .notifications import (
    get_notification_history,
    get_notification_preferences,
    list_stock_alerts,
    set_stock_alert,
    unsubscribe_all_marketing,
    update_notification_preferences,
)
from .orders import cancel_order, get_order_status, list_orders, modify_shipping_address
from .payments import add_payment_method, get_invoice, get_payment_methods, request_refund
from .promotions import (
    apply_promo_code,
    get_bundle_offers,
    get_clearance_items,
    get_price_match_policy,
    list_active_promotions,
    request_price_match,
    validate_promo_code,
)
from .returns import get_return_policy, get_return_status, start_return
from .reviews import (
    get_product_reviews,
    get_review_guidelines,
    get_review_status,
    report_review,
    submit_review,
)
from .scheduling import (
    book_callback,
    cancel_callback,
    get_callback,
    list_available_slots,
    reschedule_delivery,
)
from .security import (
    get_account_security_status,
    get_fraud_case_status,
    list_recent_logins,
    lock_account,
    report_fraud,
    request_identity_verification,
    unlock_account,
)
from .shipping import estimate_delivery, get_shipping_options, track_shipment
from .stores import (
    cancel_pickup_reservation,
    check_in_store_availability,
    find_stores_nearby,
    get_pickup_status,
    get_store_details,
    get_store_hours,
    reserve_for_pickup,
)
from .subscriptions import (
    cancel_subscription,
    change_subscription_frequency,
    get_subscription,
    get_subscription_invoice,
    list_subscriptions,
    pause_subscription,
    resume_subscription,
)
from .support import create_ticket, escalate_to_human, get_faq
from .warranty import (
    book_repair_appointment,
    get_repair_status,
    get_warranty_status,
    get_warranty_terms,
    list_service_centres,
    register_warranty,
    start_repair_request,
)
from .wishlist import (
    add_to_wishlist,
    get_saved_carts,
    get_wishlist,
    remove_from_wishlist,
    restore_saved_cart,
)

ALL_TOOLS: list[Callable[..., dict[str, Any]]] = [
    # Orders
    get_order_status,
    list_orders,
    cancel_order,
    modify_shipping_address,
    # Returns
    start_return,
    get_return_status,
    get_return_policy,
    # Shipping
    track_shipment,
    get_shipping_options,
    estimate_delivery,
    # Payments
    get_payment_methods,
    add_payment_method,
    get_invoice,
    request_refund,
    # Catalog
    search_products,
    get_product_details,
    check_stock,
    get_price,
    # Account
    get_account,
    update_email,
    get_loyalty_points,
    list_addresses,
    # Support
    get_faq,
    create_ticket,
    escalate_to_human,
    # Subscriptions
    list_subscriptions,
    get_subscription,
    pause_subscription,
    resume_subscription,
    change_subscription_frequency,
    cancel_subscription,
    get_subscription_invoice,
    # Warranty and repairs
    get_warranty_status,
    register_warranty,
    start_repair_request,
    get_repair_status,
    list_service_centres,
    book_repair_appointment,
    get_warranty_terms,
    # Gift cards and store credit
    check_gift_card_balance,
    redeem_gift_card,
    get_store_credit_balance,
    issue_store_credit,
    transfer_store_credit,
    get_gift_card_terms,
    # Promotions and price matching
    list_active_promotions,
    validate_promo_code,
    apply_promo_code,
    request_price_match,
    get_price_match_policy,
    get_clearance_items,
    get_bundle_offers,
    # Reviews
    get_product_reviews,
    submit_review,
    get_review_status,
    report_review,
    get_review_guidelines,
    # Wishlist and saved carts
    get_wishlist,
    add_to_wishlist,
    remove_from_wishlist,
    get_saved_carts,
    restore_saved_cart,
    # Physical stores
    find_stores_nearby,
    get_store_details,
    get_store_hours,
    check_in_store_availability,
    reserve_for_pickup,
    get_pickup_status,
    cancel_pickup_reservation,
    # Account security and fraud
    get_account_security_status,
    list_recent_logins,
    lock_account,
    unlock_account,
    report_fraud,
    get_fraud_case_status,
    request_identity_verification,
    # Notifications
    get_notification_preferences,
    update_notification_preferences,
    unsubscribe_all_marketing,
    get_notification_history,
    set_stock_alert,
    list_stock_alerts,
    # International
    list_shipping_destinations,
    quote_international_shipping,
    estimate_customs_duty,
    submit_customs_declaration,
    get_customs_status,
    convert_currency,
    # Business accounts
    get_business_account,
    get_bulk_pricing,
    request_quote,
    submit_purchase_order,
    get_purchase_order_status,
    list_business_orders,
    update_tax_exemption,
    # Scheduling
    list_available_slots,
    book_callback,
    get_callback,
    cancel_callback,
    reschedule_delivery,
]

__all__ = [
    "ALL_TOOLS",
    "add_payment_method",
    "add_to_wishlist",
    "apply_promo_code",
    "book_callback",
    "book_repair_appointment",
    "cancel_callback",
    "cancel_order",
    "cancel_pickup_reservation",
    "cancel_subscription",
    "change_subscription_frequency",
    "check_gift_card_balance",
    "check_in_store_availability",
    "check_stock",
    "convert_currency",
    "create_ticket",
    "escalate_to_human",
    "estimate_customs_duty",
    "estimate_delivery",
    "find_stores_nearby",
    "get_account",
    "get_account_security_status",
    "get_bulk_pricing",
    "get_bundle_offers",
    "get_business_account",
    "get_callback",
    "get_clearance_items",
    "get_customs_status",
    "get_faq",
    "get_fraud_case_status",
    "get_gift_card_terms",
    "get_invoice",
    "get_loyalty_points",
    "get_notification_history",
    "get_notification_preferences",
    "get_order_status",
    "get_payment_methods",
    "get_pickup_status",
    "get_price",
    "get_price_match_policy",
    "get_product_details",
    "get_product_reviews",
    "get_purchase_order_status",
    "get_repair_status",
    "get_return_policy",
    "get_return_status",
    "get_review_guidelines",
    "get_review_status",
    "get_saved_carts",
    "get_shipping_options",
    "get_store_credit_balance",
    "get_store_details",
    "get_store_hours",
    "get_subscription",
    "get_subscription_invoice",
    "get_warranty_status",
    "get_warranty_terms",
    "get_wishlist",
    "issue_store_credit",
    "list_active_promotions",
    "list_addresses",
    "list_available_slots",
    "list_business_orders",
    "list_orders",
    "list_recent_logins",
    "list_service_centres",
    "list_shipping_destinations",
    "list_stock_alerts",
    "list_subscriptions",
    "lock_account",
    "modify_shipping_address",
    "pause_subscription",
    "quote_international_shipping",
    "redeem_gift_card",
    "register_warranty",
    "remove_from_wishlist",
    "report_fraud",
    "report_review",
    "request_identity_verification",
    "request_price_match",
    "request_quote",
    "request_refund",
    "reschedule_delivery",
    "reserve_for_pickup",
    "reset_fixtures",
    "restore_saved_cart",
    "resume_subscription",
    "search_products",
    "set_stock_alert",
    "start_repair_request",
    "start_return",
    "submit_customs_declaration",
    "submit_purchase_order",
    "submit_review",
    "track_shipment",
    "transfer_store_credit",
    "unlock_account",
    "unsubscribe_all_marketing",
    "update_email",
    "update_notification_preferences",
    "update_tax_exemption",
    "validate_promo_code",
]
