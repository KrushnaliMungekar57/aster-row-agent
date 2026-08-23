"""
Order lookup tool over data/orders.json.

The model never sees the full orders file. `OrderLookup.lookup()` is the
only path from the dataset to the model, and it:
  - normalizes harmless input differences (case, whitespace, punctuation)
  - returns only the customer-safe fields listed in
    data/orders-data-dictionary.md
  - never returns customer PII or the `internal` block
  - attaches a deterministic `guidance` string that encodes the
    status-precedence rules from the data dictionary (stale ETA on
    cancelled/returned orders, missing ETA on shipped orders, exception
    handling) so correctness does not depend solely on the model
    interpreting raw fields correctly.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ORDER_ID_RE = re.compile(r"^ORD-\d+$")

SAFE_FIELDS = [
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
]


def normalize_order_id(raw: str) -> str:
    """Uppercase, strip whitespace, and strip common surrounding punctuation.

    Does not attempt to guess a different order ID -- it only removes
    harmless formatting noise.
    """
    if raw is None:
        return ""
    cleaned = raw.strip().strip(".,;:!?'\"()[]").strip()
    cleaned = re.sub(r"\s+", "", cleaned)  # "ord 1007" -> "ord1007"? handled below
    return cleaned.upper()


def _loose_normalize(raw: str) -> str:
    """A second, looser normalization pass: collapse internal whitespace
    around the hyphen (e.g. 'ord 1007' or 'ord- 1007' -> 'ORD-1007')."""
    cleaned = raw.strip().upper()
    cleaned = re.sub(r"[^A-Z0-9-]", "", cleaned)
    cleaned = re.sub(r"^ORD0*", "ORD-", cleaned) if cleaned.startswith("ORD") and "-" not in cleaned else cleaned
    if not cleaned.startswith("ORD-") and re.match(r"^ORD\d+$", cleaned):
        cleaned = "ORD-" + cleaned[3:]
    return cleaned


@dataclass
class LookupResult:
    found: bool
    order_id_queried: str
    normalized_id: str
    data: Optional[dict] = None
    guidance: str = ""
    error: Optional[str] = None

    def to_tool_output(self) -> dict:
        """What actually gets sent back to the model. Internal-only and PII
        fields are structurally excluded -- they are never even loaded into
        this object."""
        out = {
            "found": self.found,
            "queried_as": self.order_id_queried,
            "normalized_order_id": self.normalized_id,
        }
        if self.found:
            out["order"] = self.data
            out["guidance"] = self.guidance
        else:
            out["error"] = self.error
        return out


class OrderLookup:
    def __init__(self, orders_path: Path):
        self.orders_path = Path(orders_path)
        raw = json.loads(self.orders_path.read_text(encoding="utf-8"))
        self.snapshot_at = raw.get("snapshot_at")
        self._orders_by_id = {o["order_id"]: o for o in raw.get("orders", [])}

    def _sanitize(self, order: dict) -> dict:
        return {k: order.get(k) for k in SAFE_FIELDS}

    def _guidance(self, order: dict) -> str:
        status = order.get("status")
        eta = order.get("estimated_delivery")
        notes = []

        if status == "cancelled":
            notes.append(
                "Order is CANCELLED. It will not be shipped or delivered. "
                "Any estimated_delivery or carrier/tracking values present are stale "
                "operational leftovers -- do not present them as an active delivery estimate."
            )
        elif status == "returned":
            notes.append(
                "Order is RETURNED. It has already been sent back and is not in transit. "
                "Do not present estimated_delivery as an active/upcoming date."
            )
        elif status == "exception":
            notes.append(
                "Order has a carrier EXCEPTION. This requires support review. "
                "Recommend a human handoff and do not invent a resolution or new delivery date."
            )
        elif status == "shipped" and not eta:
            notes.append(
                "Order has SHIPPED but no delivery estimate is available yet. "
                "State that it has shipped and that an estimate is not currently available. "
                "Do not calculate or guess a date."
            )
        elif status == "shipped" and eta:
            notes.append(
                "Order has SHIPPED. Explicitly state that it has shipped, and give the "
                "carrier and estimated_delivery date as provided."
            )
        elif status == "delayed":
            notes.append(
                "Order is DELAYED. Use customer_safe_message for the current explanation "
                "and estimated_delivery for the current date; do not reference the original date."
            )
        elif status in ("pending", "processing"):
            notes.append(
                f"Order is {status.upper()} and has not shipped yet. "
                "Do not state a carrier or tracking number (none exists yet)."
            )
        elif status == "delivered":
            notes.append("Order is DELIVERED. Use delivered_at as the delivery date, not estimated_delivery.")

        return " ".join(notes) if notes else "Use the status and dates as provided."

    def lookup(self, raw_order_id: str) -> LookupResult:
        if not raw_order_id or not raw_order_id.strip():
            return LookupResult(
                found=False,
                order_id_queried=raw_order_id or "",
                normalized_id="",
                error="No order ID was supplied.",
            )

        candidates = {normalize_order_id(raw_order_id), _loose_normalize(raw_order_id)}
        normalized = None
        for cand in candidates:
            if ORDER_ID_RE.match(cand) and cand in self._orders_by_id:
                normalized = cand
                break

        if normalized is None:
            # Still report a best-effort normalized form for transparency,
            # but do not guess a different, existing order ID.
            best_guess = _loose_normalize(raw_order_id)
            return LookupResult(
                found=False,
                order_id_queried=raw_order_id,
                normalized_id=best_guess,
                error=(
                    "No order was found for this ID. It may be mistyped, or it may not be "
                    "in this system. Ask the customer to double-check the order ID, or "
                    "recommend contacting support."
                ),
            )

        order = self._orders_by_id[normalized]
        return LookupResult(
            found=True,
            order_id_queried=raw_order_id,
            normalized_id=normalized,
            data=self._sanitize(order),
            guidance=self._guidance(order),
        )