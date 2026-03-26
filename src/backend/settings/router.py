"""FastAPI router for workspace settings."""

from fastapi import APIRouter, Query
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from settings import service
from settings.models import WorkspaceSettingsUpdate

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client
    return get_supabase_client()


@router.get("")
async def get_settings(user_id: str = Query(...)):
    """Get workspace settings for a user."""
    settings = await service.get_settings(_get_supabase(), user_id)
    return settings


@router.put("")
async def update_settings(body: WorkspaceSettingsUpdate, user_id: str = Query(...)):
    """Update workspace settings for a user."""
    data = body.model_dump(exclude_none=True)
    settings = await service.update_settings(_get_supabase(), user_id, data)
    return settings
