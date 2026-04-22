"""FastAPI router for workspace settings."""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from settings import service
from settings.models import WorkspaceSettingsUpdate

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


@router.get("")
async def get_settings(user_id: str = Depends(get_current_user_id)):
    """Get workspace settings for a user."""
    settings = await service.get_settings(_get_supabase(), user_id)
    return settings


@router.put("")
async def update_settings(body: WorkspaceSettingsUpdate, user_id: str = Depends(get_current_user_id)):
    """Update workspace settings for a user."""
    data = body.model_dump(exclude_none=True)
    settings = await service.update_settings(_get_supabase(), user_id, data)
    return settings
