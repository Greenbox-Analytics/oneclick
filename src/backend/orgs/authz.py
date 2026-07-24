"""Access-control helpers for organizations & seats (Licensing Phase B).

The backend uses the Supabase service-role client, which BYPASSES RLS — so
these per-request checks are the real authorization gate (RLS is
defense-in-depth for any direct frontend reads). Spec §4/§5.

Unlike teams/authz.py (which queries team_members directly), these call the
is_org_member/is_org_admin SQL helpers via `rpc` — the same SECURITY DEFINER
functions the RLS policies use (supabase/migrations/20260721000001_licensing_core.sql)
— so the backend's membership answer and RLS's membership answer can never
drift apart. Both SQL helpers only count ACTIVE rows (status='active'); a
suspended or removed seat confers no membership or admin rights.

Argument order is (db, user_id, org_id) to match teams/authz.py and the SQL
helpers is_org_member(p_user_id, p_org_id) / is_org_admin(p_user_id, p_org_id).
"""

from fastapi import HTTPException
from supabase import Client


def is_org_member(db: Client, user_id: str, org_id: str) -> bool:
    """True if user_id holds an ACTIVE seat (any role) in org_id."""
    res = db.rpc("is_org_member", {"p_user_id": user_id, "p_org_id": org_id}).execute()
    return bool(res.data)


def is_org_admin(db: Client, user_id: str, org_id: str) -> bool:
    """True if user_id is an ACTIVE admin of org_id."""
    res = db.rpc("is_org_admin", {"p_user_id": user_id, "p_org_id": org_id}).execute()
    return bool(res.data)


def require_member(db: Client, user_id: str, org_id: str) -> None:
    """Raise 404 if the caller doesn't hold an active seat in org_id.

    404 (not 403) so a random/foreign org_id can't be distinguished from one
    that simply doesn't exist — same no-existence-oracle stance as the
    billing-context resolution (spec §5) and teams' require_board_access.
    """
    if not is_org_member(db, user_id, org_id):
        raise HTTPException(status_code=404, detail="Organization not found")


def require_admin(db: Client, user_id: str, org_id: str) -> None:
    """Raise 403 if the caller isn't an active admin of org_id.

    is_org_admin already returns False for non-members (no matching row), so
    this alone is sufficient for admin-gated endpoints — mirrors
    teams.authz.require_team_admin exactly.
    """
    if not is_org_admin(db, user_id, org_id):
        raise HTTPException(status_code=403, detail="Admin access required")
