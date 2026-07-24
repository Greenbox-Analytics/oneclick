"""FastAPI router for OneClick royalties read and payout endpoints.

Endpoints:
  GET  /payees?base=USD              → list[PayeeSummary]
  GET  /payees/{payee_id}?base=USD   → PayeeDetail
  GET  /periods?base=USD             → PeriodLedger
  POST /payouts                      → create_payouts
  POST /payouts/{payout_id}/pay      → mark_paid
  POST /payouts/{payout_id}/cancel   → cancel_payout
  GET  /payouts                      → list_payouts
  GET  /payouts/{payout_id}          → get_payout
  POST /payouts/{payout_id}/paypal/order    → create PayPal checkout order
  POST /payouts/{payout_id}/paypal/capture  → capture order, mark paid

All endpoints require authentication and are gated on Action.USE_ONECLICK (→ 402).
Payee ownership is enforced: a 404 is returned if the payee doesn't belong to the caller.
PermissionError → 403, cancel ValueError → 409.
"""

import re
import sys
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_current_user_id
from oneclick.royalties import emails, paypal_client, pdf, service
from oneclick.royalties.models import (
    CreatePayoutRequest,
    PatchPayeeRequest,
    SaveReceiptRequest,
    SplitPayeeRequest,
)
from subscriptions.enforcement import gated_feature
from subscriptions.models import Action

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


def _verify_owns_project(user_id: str, pid: str) -> bool:
    from main import verify_user_owns_project

    return verify_user_owns_project(user_id, pid)


@router.get("/payees")
def list_payees(
    base: str = "USD",
    user_id: str = Depends(get_current_user_id),
):
    """Return aggregated PayeeSummary list for all payees owned by the caller."""
    gated_feature(user_id, Action.USE_ONECLICK)
    return service.payee_summary(_get_supabase(), user_id, base)


@router.get("/payees/{payee_id}")
def get_payee(
    payee_id: str,
    base: str = "USD",
    user_id: str = Depends(get_current_user_id),
):
    """Return PayeeDetail for a single payee.  404 if payee not owned by caller."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        return service.payee_detail(_get_supabase(), user_id, payee_id, base)
    except PermissionError:
        raise HTTPException(status_code=404, detail="Payee not found")


@router.patch("/payees/{payee_id}")
def patch_payee(
    payee_id: str,
    body: PatchPayeeRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Patch mutable fields (payout_currency, registry_user_id, email) on a payee."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        return service.patch_payee(_get_supabase(), user_id, payee_id, body.model_dump(exclude_unset=True))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/payees/{payee_id}/split")
def split_payee(
    payee_id: str,
    body: SplitPayeeRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Reassign selected lines to a new payee.  409 if any line is in a paid bucket."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        return service.split_payee(_get_supabase(), user_id, payee_id, body.line_ids, body.new_display_name)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.delete("/projects/{project_id}/entries")
async def delete_project_entries(
    project_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Delete a project's royalty calculations (lines + statement_rows cascade via FK) and payout coverage.

    Returns 403 if the caller does not own the project.
    Note: royalty_payouts and invoices are intentionally NOT deleted; orphan_state is derived at read time.
    """
    gated_feature(user_id, Action.USE_ONECLICK)
    if not _verify_owns_project(user_id, project_id):
        raise HTTPException(status_code=403, detail="Access denied")
    return service.delete_project_royalty_entries(_get_supabase(), user_id, project_id)


@router.get("/contracts/{contract_id}/impact")
def contract_ledger_impact(
    contract_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Preview what deleting a contract would do to the ledger: how many lines
    assert it as a source, the backed amount per currency, and how many of the
    affected (payee, statement, project) buckets already have paid coverage.

    No separate ownership check needed: every query below is scoped to the
    caller's own user_id, so a foreign contract_id simply matches nothing.
    """
    gated_feature(user_id, Action.USE_ONECLICK)
    db = _get_supabase()
    lines = (
        db.table("royalty_lines")
        .select("*")
        .eq("user_id", user_id)
        .contains("source_contracts", [{"id": contract_id}])
        .execute()
    ).data or []
    backed: dict[str, float] = {}
    buckets = set()
    for line in lines:
        ccy = (line.get("statement_currency") or "USD").upper()
        backed[ccy] = backed.get(ccy, 0.0) + float(line.get("amount_owed") or 0)
        buckets.add((line.get("payee_id"), line.get("royalty_statement_id"), line.get("project_id")))
    paid_buckets = 0
    if buckets:
        # Scope the coverage load to this user's paid payouts — the service-role
        # client bypasses RLS; per-endpoint filtering is the only authz here.
        paid_ids = [p["id"] for p in service._load_payouts(db, user_id) if p.get("status") == "paid"]
        covered = set()
        if paid_ids:
            cov = (
                db.table("royalty_payout_coverage")
                .select("payee_id, royalty_statement_id, project_id")
                .in_("payout_id", paid_ids)
                .execute()
            ).data or []
            covered = {(c["payee_id"], c["royalty_statement_id"], c["project_id"]) for c in cov}
        paid_buckets = len(buckets & covered)
    return {"lines": len(lines), "backed": backed, "buckets_with_paid_coverage": paid_buckets}


@router.get("/periods")
def get_periods(
    base: str = "USD",
    user_id: str = Depends(get_current_user_id),
):
    """Return PeriodLedger (payee × statement-period matrix)."""
    gated_feature(user_id, Action.USE_ONECLICK)
    return service.periods_ledger(_get_supabase(), user_id, base)


# ---------------------------------------------------------------------------
# Payout endpoints
# ---------------------------------------------------------------------------


@router.post("/payouts")
def create_payouts(
    body: CreatePayoutRequest,
    force: bool = False,
    user_id: str = Depends(get_current_user_id),
):
    """Create draft payouts for one or more payees.  Skips payees with no owed amount.
    409 with {"stale_lines": [...]} if any owed line has no live source contract —
    retry with ?force=true to proceed anyway."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        return service.create_payouts(
            _get_supabase(),
            user_id,
            body.payee_ids,
            body.idempotency_key,
            body.note,
            force=force,
        )
    except service.StaleSourcesError as exc:
        raise HTTPException(status_code=409, detail={"stale_lines": exc.lines})
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/payouts/{payout_id}/pay")
def mark_paid(
    payout_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Mark a draft payout as paid."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        return service.mark_paid(_get_supabase(), user_id, payout_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


@router.post("/payouts/{payout_id}/cancel")
def cancel_payout(
    payout_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Cancel (delete) a draft payout.  Coverage cascades.  409 if not draft."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        service.cancel_payout(_get_supabase(), user_id, payout_id)
        return {"status": "canceled"}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/payouts/{payout_id}/revert")
def revert_payout(
    payout_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Revert a manually-completed payout back to draft (undo an accidental
    mark-paid).  409 if not paid or if it was completed through PayPal."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        return service.revert_payout_to_draft(_get_supabase(), user_id, payout_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


def _handle_paypal_endpoint(fn, user_id: str, payout_id: str, stage: str):
    """Run a PayPal service call with shared error mapping + failure analytics.

    PermissionError → 404 (match get_payout — don't reveal foreign payout ids),
    PayoutStateError → 409, ValueError → 400 (actionable UI copy),
    PayPalError → 502 with a generic message (real body is logged server-side).
    """
    try:
        return fn()
    except PermissionError:
        raise HTTPException(status_code=404, detail="Payout not found")
    except service.PayoutStateError as exc:
        analytics_capture(
            user_id, "paypal_payment_failed", {"payout_id": payout_id, "stage": stage, "reason": str(exc)}
        )
        raise HTTPException(status_code=409, detail=str(exc))
    except paypal_client.PayPalError as exc:
        analytics_capture(
            user_id,
            "paypal_payment_failed",
            {"payout_id": payout_id, "stage": stage, "reason": exc.issue or str(exc)},
        )
        raise HTTPException(status_code=502, detail="PayPal couldn't process this payment. Please try again.")
    except ValueError as exc:
        analytics_capture(
            user_id, "paypal_payment_failed", {"payout_id": payout_id, "stage": stage, "reason": str(exc)}
        )
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/payouts/{payout_id}/paypal/order")
def create_paypal_order(
    payout_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Create a PayPal checkout order for a draft payout. Returns {paypal_order_id}."""
    gated_feature(user_id, Action.USE_ONECLICK)
    result = _handle_paypal_endpoint(
        lambda: service.create_paypal_order_for_payout(_get_supabase(), user_id, payout_id),
        user_id,
        payout_id,
        stage="create_order",
    )
    analytics_capture(user_id, "paypal_payment_started", {"payout_id": payout_id})
    return result


def _send_receipt_email_background(payout: dict, payee_email: str, payee_name: str, payer_name: str | None):
    """Generate the receipt PDF and email it to the payee. Never raises."""
    try:
        receipt_bytes = pdf.generate_receipt_pdf(payout, payer_name=payer_name).read()
        emails.send_payout_receipt_email(
            recipient_email=payee_email,
            payee_name=payee_name,
            payer_name=payer_name or "A Msanii user",
            amount_str=pdf.fmt_money(payout.get("total_amount", 0), payout.get("pay_currency") or ""),
            paid_at=payout.get("paid_at"),
            paypal_capture_id=payout.get("paypal_capture_id"),
            receipt_pdf_bytes=receipt_bytes,
        )
    except Exception as exc:
        print(f"Warning: Failed to send payout receipt email: {exc}")


def _fetch_payer_name(db, user_id: str) -> str | None:
    try:
        res = db.table("profiles").select("full_name").eq("id", user_id).execute()
        rows = res.data or []
        return rows[0].get("full_name") if rows else None
    except Exception:
        return None


def _schedule_receipt_email(background_tasks: BackgroundTasks, db, user_id: str, payout: dict) -> None:
    """Queue the payee receipt email after a verified PayPal capture.

    Best-effort: any lookup failure just skips the email. An idempotent
    re-capture (double-click) may re-send the receipt — harmless.
    """
    try:
        payee_res = db.table("royalty_payees").select("email, display_name").eq("id", payout.get("payee_id")).execute()
        payee_rows = payee_res.data or []
        payee = payee_rows[0] if payee_rows else {}
        payee_email = (payee.get("email") or "").strip()
        if not payee_email:
            return
        background_tasks.add_task(
            _send_receipt_email_background,
            payout=payout,
            payee_email=payee_email,
            payee_name=payee.get("display_name") or "there",
            payer_name=_fetch_payer_name(db, user_id),
        )
    except Exception as exc:
        print(f"Warning: Failed to schedule payout receipt email: {exc}")


@router.post("/payouts/{payout_id}/paypal/capture")
def capture_paypal_order(
    payout_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    """Capture the payout's approved PayPal order and mark the payout paid."""
    gated_feature(user_id, Action.USE_ONECLICK)
    db = _get_supabase()
    payout = _handle_paypal_endpoint(
        lambda: service.capture_paypal_order_for_payout(db, user_id, payout_id),
        user_id,
        payout_id,
        stage="capture",
    )
    analytics_capture(
        user_id,
        "paypal_payment_completed",
        {
            "payout_id": payout_id,
            "currency": payout.get("pay_currency"),
            "amount": payout.get("total_amount"),
            "paypal_capture_id": payout.get("paypal_capture_id"),
        },
    )
    _schedule_receipt_email(background_tasks, db, user_id, payout)
    return payout


@router.get("/payouts")
def list_payouts(
    user_id: str = Depends(get_current_user_id),
):
    """Return all payouts for the caller with derived orphan_state."""
    gated_feature(user_id, Action.USE_ONECLICK)
    return service.list_payouts(_get_supabase(), user_id)


@router.get("/payouts/{payout_id}")
def get_payout(
    payout_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Return a single payout.  404 if not found or not the caller's."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        return service.get_payout(_get_supabase(), user_id, payout_id)
    except PermissionError:
        raise HTTPException(status_code=404, detail="Payout not found")


# ---------------------------------------------------------------------------
# Payout documents: analysis breakdown + payment receipt
# ---------------------------------------------------------------------------


def _safe_filename_part(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name or "payout")


def _payee_name_from_snapshot(payout: dict) -> str:
    snapshot = payout.get("breakdown_snapshot") or {}
    return (snapshot.get("payee") or {}).get("display_name") or "payee"


def _pdf_response(buffer, filename: str) -> StreamingResponse:
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/payouts/{payout_id}/breakdown.pdf")
def export_breakdown_pdf(
    payout_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Download the payout analysis breakdown as a PDF (any payout status)."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        payout = service.get_payout(_get_supabase(), user_id, payout_id)
    except PermissionError:
        raise HTTPException(status_code=404, detail="Payout not found")

    filename = f"Payout_Breakdown_{_safe_filename_part(_payee_name_from_snapshot(payout))}_{payout.get('fx_rate_date', '')}.pdf"
    return _pdf_response(pdf.generate_breakdown_pdf(payout), filename)


@router.get("/payouts/{payout_id}/receipt.pdf")
def export_receipt_pdf(
    payout_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Download a payment receipt PDF. Only available once the payout is paid."""
    gated_feature(user_id, Action.USE_ONECLICK)
    db = _get_supabase()
    try:
        payout = service.get_paid_payout(db, user_id, payout_id)
    except PermissionError:
        raise HTTPException(status_code=404, detail="Payout not found")
    except service.PayoutStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    paid_date = str(payout.get("paid_at") or "")[:10]
    filename = f"Receipt_{_safe_filename_part(_payee_name_from_snapshot(payout))}_{paid_date}.pdf"
    return _pdf_response(pdf.generate_receipt_pdf(payout, payer_name=_fetch_payer_name(db, user_id)), filename)


@router.post("/payouts/{payout_id}/receipt/save")
def save_receipt_to_project(
    payout_id: str,
    body: SaveReceiptRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Generate the payout receipt PDF and save it into a project's files."""
    gated_feature(user_id, Action.USE_ONECLICK)
    from main import verify_user_owns_artist

    db = _get_supabase()
    try:
        payout = service.get_paid_payout(db, user_id, payout_id)
    except PermissionError:
        raise HTTPException(status_code=404, detail="Payout not found")
    except service.PayoutStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if not verify_user_owns_artist(user_id, body.artist_id) or not _verify_owns_project(user_id, body.project_id):
        raise HTTPException(status_code=403, detail="Access denied")

    paid_date = str(payout.get("paid_at") or "")[:10]
    filename = f"Receipt_{_safe_filename_part(_payee_name_from_snapshot(payout))}_{paid_date}.pdf"
    file_bytes = pdf.generate_receipt_pdf(payout, payer_name=_fetch_payer_name(db, user_id)).read()

    try:
        storage_path = f"{body.artist_id}/{body.project_id}/receipts/{int(time.time())}_{filename}"
        db.storage.from_("project-files").upload(
            storage_path, file_bytes, file_options={"content-type": "application/pdf"}
        )
        file_url = db.storage.from_("project-files").get_public_url(storage_path)

        # folder_category must be one of the four the project Files tab
        # renders (contract / split_sheet / royalty_statement / other) —
        # anything else would make the saved receipt invisible.
        db_record = {
            "project_id": body.project_id,
            "folder_category": "other",
            "file_name": filename,
            "file_url": file_url,
            "file_path": storage_path,
            "file_size": len(file_bytes),
            "file_type": "application/pdf",
        }
        insert_res = db.table("project_files").insert(db_record).execute()
        rows = insert_res.data or []
        return rows[0] if rows else db_record
    except Exception as exc:
        print(f"Warning: Failed to save payout receipt to project: {exc}")
        raise HTTPException(status_code=500, detail="Failed to save the receipt to the project. Please try again.")
