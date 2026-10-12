"""Review tools: read reviews, write one, chase it, report one, read the rules.

One of the twelve domains added in DESIGN.md 19.15. Reviews are keyed by product and
author and check the author actually bought the item, which is the only cross-domain
read this module makes and it goes through `_data`, not through another domain module.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools import ToolContext

from . import _data

MODERATION_STAGES: tuple[str, ...] = ("SUBMITTED", "IN_MODERATION", "PUBLISHED")
REPORT_REASONS: tuple[str, ...] = ("spam", "offensive", "off_topic", "not_a_buyer", "spoiler")


def _seed_reviews() -> dict[str, dict[str, Any]]:
    """Three published reviews across two products, so averages are not degenerate."""
    return {
        "REV-1001": {
            "review_id": "REV-1001",
            "sku": "SKU-HDP-04",
            "user_id": "u_1002",
            "rating": 5,
            "title": "Better than the price suggests",
            "body": "Comfortable for long calls and the battery genuinely lasts a week.",
            "stage": "PUBLISHED",
            "written_on": "2026-07-14",
            "verified_purchase": True,
        },
        "REV-1002": {
            "review_id": "REV-1002",
            "sku": "SKU-HDP-04",
            "user_id": "u_1003",
            "rating": 3,
            "title": "Good sound, tight fit",
            "body": "Sound is great. The headband pinches if you wear glasses.",
            "stage": "PUBLISHED",
            "written_on": "2026-08-02",
            "verified_purchase": True,
        },
        "REV-1003": {
            "review_id": "REV-1003",
            "sku": "SKU-LMP-03",
            "user_id": "u_1001",
            "rating": 4,
            "title": "Warm light, small footprint",
            "body": "Exactly the colour temperature I wanted. The base is a little light.",
            "stage": "PUBLISHED",
            "written_on": "2026-06-28",
            "verified_purchase": True,
        },
    }


REVIEWS: dict[str, dict[str, Any]] = _seed_reviews()
REPORTS: list[dict[str, Any]] = []


def _reset() -> None:
    REVIEWS.clear()
    REVIEWS.update(_seed_reviews())
    REPORTS.clear()


_data.register_reset(_reset)


def _bought(user_id: str, sku: str) -> bool:
    """Whether this customer has an order containing this product."""
    return any(
        order["user_id"] == user_id and any(item["sku"] == sku for item in order["items"])
        for order in _data.ORDERS.values()
    )


def get_product_reviews(sku: str, minimum_rating: int) -> dict[str, Any]:
    """Read the published reviews for a product.

    Use this when a customer is deciding whether to buy something and asks what other
    people thought, or when they dispute a rating they have seen.

    Args:
        sku: The product code to read reviews for, for example "SKU-HDP-04".
        minimum_rating: Only return reviews at or above this many stars, from 1 to 5.
            Pass 1 for every review, which is the usual case.

    Returns:
        The product name, the number of published reviews, the average rating rounded
        to one decimal place, a breakdown of how many reviews gave each star rating,
        and the reviews themselves with their id, rating, title, body, date and
        whether the author bought the item. Reviews still in moderation are left out.
        Returns an empty list when the product has no reviews yet.
    """
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)

    floor = max(1, min(5, minimum_rating))
    published = [
        record
        for record in REVIEWS.values()
        if record["sku"] == sku and record["stage"] == "PUBLISHED"
    ]
    shown = [r for r in published if r["rating"] >= floor]
    ratings = [r["rating"] for r in published]
    return {
        "sku": sku,
        "product_name": product["name"],
        "review_count": len(published),
        "average_rating": round(sum(ratings) / len(ratings), 1) if ratings else None,
        "breakdown": {str(star): ratings.count(star) for star in range(1, 6)},
        "minimum_rating": floor,
        "reviews": [
            {k: v for k, v in record.items() if k not in {"user_id", "stage"}} for record in shown
        ],
    }


def submit_review(
    sku: str, rating: int, title: str, body: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Write a review of a product on behalf of the signed-in customer.

    Reviews go to moderation before they appear, so tell the customer it will not be
    visible straight away. A customer who has bought the item gets a verified badge.

    Args:
        sku: The product being reviewed, for example "SKU-LMP-03".
        rating: How many stars, from 1 to 5.
        title: A short headline for the review, in the customer's words.
        body: The review itself. At least 20 characters, or moderation rejects it as
            too thin to be useful.

    Returns:
        A `submitted` flag, the new review id, the moderation stage, whether the
        purchase was verified, and how long moderation usually takes. Returns
        `submitted: false` with a detail line when the rating is outside 1 to 5, when
        the body is too short, or when this customer has already reviewed this product.
    """
    product = _data.PRODUCTS.get(sku)
    if product is None:
        return _data.not_found("product", sku)
    if not 1 <= rating <= 5:
        return {"sku": sku, "submitted": False, "detail": "rating must be between 1 and 5"}
    if len(body.strip()) < 20:
        return {"sku": sku, "submitted": False, "detail": "the review body is too short to publish"}

    user_id = _data.current_user_id(tool_context)
    if any(r["sku"] == sku and r["user_id"] == user_id for r in REVIEWS.values()):
        return {
            "sku": sku,
            "submitted": False,
            "detail": "this customer has already reviewed this product",
        }

    review_id = f"REV-{_data.stable_int(user_id + sku, 2000, 9999)}"
    REVIEWS[review_id] = {
        "review_id": review_id,
        "sku": sku,
        "user_id": user_id,
        "rating": rating,
        "title": title.strip(),
        "body": body.strip(),
        "stage": "SUBMITTED",
        "written_on": str(_data.TODAY),
        "verified_purchase": _bought(user_id, sku),
    }
    return {
        "review_id": review_id,
        "submitted": True,
        "stage": "SUBMITTED",
        "verified_purchase": REVIEWS[review_id]["verified_purchase"],
        "moderation_days": 2,
    }


def get_review_status(review_id: str) -> dict[str, Any]:
    """Check whether a submitted review has been published yet.

    Args:
        review_id: The review identifier issued by `submit_review`, for example
            "REV-4820". These start with "REV-".

    Returns:
        The current stage, which runs SUBMITTED, IN_MODERATION then PUBLISHED; the
        product it is about; the rating; the date it was written; and whether it has
        been reported by anyone. Each call advances an unpublished review one stage,
        so a customer chasing it twice will see it move.
    """
    record = REVIEWS.get(review_id)
    if record is None:
        return _data.not_found("review", review_id)

    index = MODERATION_STAGES.index(record["stage"])
    if index < len(MODERATION_STAGES) - 1:
        record["stage"] = MODERATION_STAGES[index + 1]
    return {
        "review_id": review_id,
        "sku": record["sku"],
        "stage": record["stage"],
        "rating": record["rating"],
        "written_on": record["written_on"],
        "reported": any(r["review_id"] == review_id for r in REPORTS),
        "published": record["stage"] == "PUBLISHED",
    }


def report_review(review_id: str, reason: str, detail: str) -> dict[str, Any]:
    """Flag a published review for moderator attention.

    Use this when a customer says a review is abusive, fake, about the wrong product,
    or gives away something it should not.

    Args:
        review_id: The review being reported, for example "REV-1002".
        reason: Why it is being reported. One of "spam", "offensive", "off_topic",
            "not_a_buyer" or "spoiler".
        detail: Anything the customer wants the moderator to know, in their own words.
            Pass an empty string when they have nothing to add.

    Returns:
        A `reported` flag, a report reference, how many times this review has now been
        reported, and the review's current stage. A review reported three times is
        hidden automatically pending review, which is reflected in the stage. Returns
        an error field naming the valid reasons when the reason is not one of them.
    """
    record = REVIEWS.get(review_id)
    if record is None:
        return _data.not_found("review", review_id)

    key = reason.strip().lower()
    if key not in REPORT_REASONS:
        return {
            "review_id": review_id,
            "reported": False,
            "valid_reasons": list(REPORT_REASONS),
            "error": f"{reason!r} is not a report reason",
        }

    REPORTS.append({"review_id": review_id, "reason": key, "detail": detail.strip()})
    count = sum(1 for r in REPORTS if r["review_id"] == review_id)
    if count >= 3:
        record["stage"] = "IN_MODERATION"
    return {
        "review_id": review_id,
        "reported": True,
        "reference": f"RPT-{_data.stable_int(review_id + key, 1000, 9999)}",
        "report_count": count,
        "stage": record["stage"],
        "hidden_pending_review": count >= 3,
    }


def get_review_guidelines() -> dict[str, Any]:
    """Read the rules a review has to follow to be published.

    Near-static reference text. Use it when a customer asks why their review was
    rejected, or what they are allowed to say.

    Returns:
        The minimum body length, how long moderation takes, what gets a review
        rejected, whether a verified purchase is required, whether reviews can be
        edited after publication, and how a rejected review can be appealed. The same
        for every customer and every product.
    """
    return {
        "minimum_body_characters": 20,
        "moderation_days": 2,
        "rejected_for": "personal details, abusive language, competitor links, unrelated content",
        "verified_purchase_required": False,
        "editable_after_publication": False,
        "appeal_route": "create_ticket with the review id",
    }
