"""Endpoint to list a user's integration connections (no secrets exposed)."""

from fastapi import APIRouter, Depends
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client
    return get_supabase_client()


@router.get("/connections")
async def list_connections(user_id: str = Depends(get_current_user_id)):
    """Return all integration connections for the current user (tokens omitted)."""
    result = (
        _get_supabase()
        .table("integration_connections")
        .select("id, user_id, provider, status, provider_user_id, provider_workspace_id, scopes, created_at, updated_at")
        .eq("user_id", user_id)
        .execute()
    )
    return {"connections": result.data or []}
