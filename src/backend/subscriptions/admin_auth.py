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


def is_active_tester_row(row: dict) -> bool:
    """Match AdminService.list_tester_grants() predicate:
    reason LIKE 'tester%' AND (expires_at IS NULL OR expires_at > now())."""
    reason = row.get("reason") or ""
    if not reason.startswith("tester"):
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
    """FastAPI dependency. Returns caller email if admin, else raises 403.

    Raises 500 only when BOTH conditions hold: env allowlist is empty AND
    no DB admin exists. (An empty env is still fine if at least one DB
    admin exists.)
    """
    from main import get_supabase_client

    sb = get_supabase_client()

    if is_env_admin(user_email):
        return user_email
    if is_db_admin(sb, user_id):
        return user_email

    # Not admin via either path. Distinguish operator misconfig (no admins
    # at all) from "not authorized" so operators get a loud signal on
    # fresh deploys with no bootstrap set.
    if not env_admin_emails():
        # Check if ANY DB admin exists. If not, this is fail-loud config error.
        try:
            res = sb.table("profiles").select("id").eq("is_admin", True).limit(1).execute()
            if not (res.data or []):
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "No admins configured — set ADMIN_EMAILS or grant at least one user is_admin=true to bootstrap."
                    ),
                )
        except HTTPException:
            raise
        except Exception:
            # If the bootstrap check itself fails, fall through to 403 —
            # we can't tell if there are DB admins, so don't lie with 500.
            pass

    raise HTTPException(status_code=403, detail="Admin access required")
