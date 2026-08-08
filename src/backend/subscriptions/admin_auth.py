"""Admin authorization. ADMIN_EMAILS env-var bootstraps "root" admins;
additional admins are managed via profiles.is_admin (toggled through the
/admin/users UI).

Both paths are equivalent at the auth layer: an admin is anyone in the env
allowlist OR with profiles.is_admin = true.
"""

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, HTTPException
from supabase import Client

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_email, get_current_user_id

logger = logging.getLogger(__name__)


def env_admin_emails() -> set[str]:
    """Lowercased, whitespace-stripped set of admin emails from ADMIN_EMAILS env."""
    raw = os.getenv("ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def env_tester_emails() -> set[str]:
    """Lowercased, whitespace-stripped set of tester emails from TESTER_EMAILS env.

    Used by /me/bootstrap-tester to auto-create a tier_overrides row for any
    user whose signup email matches the allowlist — avoids the manual admin
    grant for known beta testers.
    """
    raw = os.getenv("TESTER_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_env_tester(email: str | None) -> bool:
    """True if *email* is in the TESTER_EMAILS env-var allowlist."""
    if not email:
        return False
    return email.strip().lower() in env_tester_emails()


def is_active_tester_row(row: dict) -> bool:
    """True if `row` is an active tester grant — reason starts with `tester`
    (case-insensitive), is not the sticky `tester_revoked` marker, and the
    grant has not expired. Mirrors list_tester_grants' SQL filter.
    """
    reason = (row.get("reason") or "").lower()
    if not reason.startswith("tester"):
        return False
    if reason == "tester_revoked":
        return False
    expires_at = row.get("expires_at")
    if expires_at is None:
        return True
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return False
    return expiry > datetime.now(UTC)


def is_env_admin(email: str | None) -> bool:
    """True if *email* is in the ADMIN_EMAILS env-var allowlist."""
    if not email:
        return False
    return email.strip().lower() in env_admin_emails()


def is_db_admin(supabase: Client, user_id: str | None) -> bool:
    """True if profiles.is_admin = true for *user_id*. False on any error
    (transient DB issues must not block env-admins from logging in)."""
    if not user_id:
        return False
    try:
        res = supabase.table("profiles").select("is_admin").eq("id", user_id).limit(1).execute()
        rows = res.data or []
        return bool(rows and rows[0].get("is_admin") is True)
    except Exception as exc:
        logger.warning("is_db_admin lookup failed for %s: %s", user_id, exc)
        return False


def is_user_admin(supabase: Client, email: str | None, user_id: str | None) -> bool:
    """True if caller is admin via either path. Env check first (no DB hit
    when the caller is already an env admin)."""
    if is_env_admin(email):
        return True
    return is_db_admin(supabase, user_id)


def require_admin(
    user_email: str = Depends(get_current_user_email),
    user_id: str = Depends(get_current_user_id),
) -> str:
    """FastAPI dependency. Returns the caller's email if admin, else raises 403.

    ALWAYS 403 to the caller, never 500. An environment with no admins at all
    (empty ADMIN_EMAILS and no profiles.is_admin row) is an operator misconfig,
    but it is not the caller's fault and not a server fault they can act on —
    surfacing it as 500 made every /admin/* route look broken on a fresh
    environment, and put a red herring in front of anyone debugging it. The
    operator signal it existed for is preserved as an ERROR log, which is where
    an operator actually looks.
    """
    from main import get_supabase_client

    sb = get_supabase_client()

    if is_env_admin(user_email):
        return user_email
    if is_db_admin(sb, user_id):
        return user_email

    # Not admin via either path. Detect the "no admins exist anywhere" case so
    # a fresh deploy without a bootstrap admin is loud in the logs, then deny
    # exactly as any other non-admin would be denied.
    if not env_admin_emails():
        try:
            res = sb.table("profiles").select("id").eq("is_admin", True).limit(1).execute()
            if not (res.data or []):
                logger.error(
                    "No admins configured — every /admin/* route will deny. Set ADMIN_EMAILS "
                    "or grant at least one user profiles.is_admin=true to bootstrap."
                )
        except Exception as exc:
            # The bootstrap probe is diagnostics only; a failed read must never
            # change the answer we give the caller.
            logger.warning("admin bootstrap probe failed: %s", exc)

    raise HTTPException(status_code=403, detail="Admin access required")
