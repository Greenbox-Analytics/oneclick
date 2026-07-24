"""Overage → Stripe pending InvoiceItems (spec §7).

Pending items fold into the next subscription invoice automatically (monthly
plans); the sweep creates standalone invoices for annual plans (Task 12).
Idempotency: one InvoiceItem per ledger row, recorded in metadata.invoice_item_id.
NEVER called on the request path — only from the sweep and invoice.created.
"""

import logging
import os

import stripe as stripe_errors

import subscriptions.stripe_client as stripe_client_module

logger = logging.getLogger(__name__)


def _overage_usd_per_credit() -> float:
    return float(os.getenv("CREDIT_OVERAGE_USD", "0.02"))


def bill_overage_row(supabase, user_id: str, ledger_row: dict, *, invoice_id: str | None = None) -> str | None:
    """Create a pending InvoiceItem for one overage_debit ledger row. Returns item id.

    invoice_id: pass the draft invoice's id from the invoice.created handler —
    pending items created AFTER a draft invoice exists are NOT pulled into it
    automatically, so stragglers must attach explicitly.
    """
    if ledger_row.get("metadata", {}).get("invoice_item_id"):
        return ledger_row["metadata"]["invoice_item_id"]  # already billed
    sub_res = supabase.table("subscriptions").select("stripe_customer_id").eq("user_id", user_id).execute()
    customer = sub_res.data[0].get("stripe_customer_id") if sub_res.data else None
    if not customer:
        logger.warning("bill_overage_row: no stripe customer for user %s", user_id)
        return None
    # Overage rows carry delta=0 (they don't drain the wallet — double-charge
    # fix); the billable amount rides metadata.credits_billed. -delta fallback
    # tolerates any legacy-shaped row.
    credits = (ledger_row.get("metadata") or {}).get("credits_billed") or -ledger_row["delta"]
    if credits <= 0:
        logger.warning("bill_overage_row: non-positive billable credits row=%s", ledger_row.get("id"))
        return None
    amount_cents = round(credits * _overage_usd_per_credit() * 100)
    stripe = stripe_client_module.get_stripe()
    kwargs = {
        "customer": customer,
        "amount": amount_cents,
        "currency": "usd",
        "description": f"Pay-per-use credits: {credits} × ${_overage_usd_per_credit():.2f} ({ledger_row.get('action') or 'usage'})",
        "metadata": {"ledger_row_id": ledger_row["id"], "user_id": user_id},
    }
    if invoice_id:
        kwargs["invoice"] = invoice_id
    # Stripe-side idempotency key: a concurrent sweep + invoice.created can
    # both pass the check-then-act above, but Stripe rejects reuse of the
    # key, so the race cannot double-charge.
    item = stripe.InvoiceItem.create(idempotency_key=f"overage:{ledger_row['id']}", **kwargs)
    new_meta = {**ledger_row.get("metadata", {}), "invoice_item_id": item.id}
    if invoice_id:
        # Attached directly to a real invoice → already consumed. Without this
        # stamp the annual sweep would fold it into a standalone invoice that
        # Stripe rejects as empty (invoice_no_customer_line_items) every day.
        new_meta["swept"] = True
    supabase.table("credit_ledger").update({"metadata": new_meta}).eq("id", ledger_row["id"]).execute()
    try:
        from analytics import capture as analytics_capture

        analytics_capture(
            user_id,
            "overage_charge_accrued",
            {"credits": credits, "usd": amount_cents / 100, "action": ledger_row.get("action")},
        )
    except Exception:
        pass  # analytics must never break billing
    return item.id


def bill_pending_overage(supabase, user_id: str, *, invoice_id: str | None = None) -> int:
    """Bill every unbilled overage_debit row for this user. Returns count billed."""
    wallet_res = (
        supabase.table("credit_wallets").select("id").eq("owner_type", "user").eq("owner_id", user_id).execute()
    )
    if not wallet_res.data:
        return 0
    rows = (
        supabase.table("credit_ledger")
        .select("*")
        .eq("wallet_id", wallet_res.data[0]["id"])
        .eq("kind", "overage_debit")
        # PostgREST JSON-path filter bounds the scan to unbilled rows; the
        # Python-side re-check in the loop below stays as belt-and-suspenders.
        .is_("metadata->invoice_item_id", "null")
        .execute()
    )
    billed = 0
    for row in rows.data or []:
        if not (row.get("metadata") or {}).get("invoice_item_id"):
            try:
                if bill_overage_row(supabase, user_id, row, invoice_id=invoice_id):
                    billed += 1
            except Exception:
                logger.exception("bill_overage_row failed row=%s", row.get("id"))
    return billed


def invoice_unswept_items(supabase, wallet_id: str, customer_id: str, *, idempotency_key: str | None = None) -> dict:
    """Collect this wallet's floating pending InvoiceItems onto ONE standalone
    auto-advancing invoice, then stamp the rows `swept`.

    Rows counted: overage_debit / storage_bill rows with a backfilled
    invoice_item_id (proves a real pending Stripe item exists) and no `swept`
    stamp. Returns {"invoiced": bool, "stamped": int}.

    If Stripe reports nothing to invoice (invoice_no_customer_line_items — the
    items were already consumed by another invoice, e.g. attached to a renewal
    by handle_invoice_created), rows are stamped swept anyway: the money IS
    billed, only our bookkeeping lagged. Without this the caller would retry a
    doomed empty Invoice.create forever.
    """
    rows_res = (
        supabase.table("credit_ledger")
        .select("id, kind, metadata")
        .eq("wallet_id", wallet_id)
        .in_("kind", ["overage_debit", "storage_bill"])
        .execute()
    )
    unswept = [
        r
        for r in (rows_res.data or [])
        if not (r.get("metadata") or {}).get("swept") and (r.get("metadata") or {}).get("invoice_item_id")
    ]
    if not unswept:
        return {"invoiced": False, "stamped": 0}

    stripe = stripe_client_module.get_stripe()
    kwargs = {"customer": customer_id, "auto_advance": True}
    if idempotency_key:
        kwargs["idempotency_key"] = idempotency_key
    invoiced = True
    try:
        stripe.Invoice.create(**kwargs)
    except stripe_errors.InvalidRequestError as exc:
        if getattr(exc, "code", None) != "invoice_no_customer_line_items":
            raise
        # Invariant: Stripe raises this code only when the customer has ZERO
        # floating items (we never pass explicit line items), so stamping the
        # whole unswept set below cannot orphan a still-billable item.
        invoiced = False  # items consumed elsewhere — stamp and stop retrying

    for r in unswept:
        supabase.table("credit_ledger").update({"metadata": {**(r.get("metadata") or {}), "swept": True}}).eq(
            "id", r["id"]
        ).execute()
    return {"invoiced": invoiced, "stamped": len(unswept)}
