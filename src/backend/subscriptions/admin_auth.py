"""Admin authorization dependency. Gates /admin/* endpoints by env-var allowlist."""

import os
import sys
from pathlib import Path

from fastapi import Depends, HTTPException

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_email


def require_admin(user_email: str = Depends(get_current_user_email)) -> str:
    """Returns the caller's email if they're an admin; raises HTTPException otherwise.

    - 500 if ADMIN_EMAILS env var is unset/empty (operator misconfig)
    - 403 if caller's email is not in the allowlist
    """
    raw = os.getenv("ADMIN_EMAILS", "")
    admin_emails = {e.strip().lower() for e in raw.split(",") if e.strip()}
    if not admin_emails:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_EMAILS not configured — admin functions disabled",
        )
    if user_email.lower() not in admin_emails:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user_email
