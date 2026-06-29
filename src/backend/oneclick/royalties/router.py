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

All endpoints require authentication and are gated on Action.USE_ONECLICK (→ 402).
Payee ownership is enforced: a 404 is returned if the payee doesn't belong to the caller.
PermissionError → 403, cancel ValueError → 409.
"""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from oneclick.royalties import service
from oneclick.royalties.models import CreatePayoutRequest, PatchPayeeRequest, SplitPayeeRequest
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
    user_id: str = Depends(get_current_user_id),
):
    """Create draft payouts for one or more payees.  Skips payees with no owed amount."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        return service.create_payouts(
            _get_supabase(),
            user_id,
            body.payee_ids,
            body.idempotency_key,
            body.note,
        )
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
