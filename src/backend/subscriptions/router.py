"""Entitlements API: surfaces what the current user can do.

Single endpoint used by the frontend's useEntitlements hook (Sub-project 1)
and the Usage tab + paywall components (Sub-project 3).

No /refresh endpoint — frontend calls queryClient.invalidateQueries(['entitlements']).
"""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends

# Ensure backend dir is in path (matches the pattern in boards/router.py)
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from subscriptions.deps import _get_entitlements_service

router = APIRouter()


@router.get("/me/entitlements")
async def get_my_entitlements(user_id: str = Depends(get_current_user_id)):
    """Return the current user's merged entitlements (tier + caps + features + usage)."""
    ent = _get_entitlements_service().get_for_user_safe(user_id)
    return ent.to_dict()
