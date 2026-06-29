"""FastAPI router for OneClick royalties analytics endpoints.

Endpoints:
  GET /analytics/overview?base=USD          → OverviewOut
  GET /analytics/artist/{artist_id}?base=USD → ArtistAnalyticsOut
  GET /analytics/payee/{payee_id}?base=USD   → PayeeAnalyticsOut

All endpoints require authentication and are gated on Action.USE_ONECLICK (→ 402).
Artist ownership is enforced: a 404 is returned if the artist doesn't belong to the caller.
Payee ownership is enforced: a 404 is returned if the payee doesn't belong to the caller
(PermissionError raised by analytics_service.payee_analytics → 404).
"""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from oneclick.royalties import analytics_service
from subscriptions.enforcement import gated_feature
from subscriptions.models import Action

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


def _verify_owns_artist(user_id: str, artist_id: str) -> bool:
    from main import verify_user_owns_artist

    return verify_user_owns_artist(user_id, artist_id)


@router.get("/overview")
def get_overview(
    base: str = "USD",
    user_id: str = Depends(get_current_user_id),
):
    """Return portfolio-level royalty analytics for the caller."""
    gated_feature(user_id, Action.USE_ONECLICK)
    return analytics_service.overview(_get_supabase(), user_id, base)


@router.get("/artist/{artist_id}")
def get_artist_analytics(
    artist_id: str,
    base: str = "USD",
    user_id: str = Depends(get_current_user_id),
):
    """Return per-artist royalty analytics.  404 if artist not owned by caller."""
    gated_feature(user_id, Action.USE_ONECLICK)
    if not _verify_owns_artist(user_id, artist_id):
        raise HTTPException(status_code=404, detail="Artist not found")
    return analytics_service.artist_analytics(_get_supabase(), user_id, artist_id, base)


@router.get("/payee/{payee_id}")
def get_payee_analytics(
    payee_id: str,
    base: str = "USD",
    user_id: str = Depends(get_current_user_id),
):
    """Return per-payee royalty analytics.  404 if payee not owned by caller."""
    gated_feature(user_id, Action.USE_ONECLICK)
    try:
        return analytics_service.payee_analytics(_get_supabase(), user_id, payee_id, base)
    except PermissionError:
        raise HTTPException(status_code=404, detail="Payee not found")
