"""FastAPI router for Notion integration."""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from integrations.oauth import (
    FRONTEND_URL,
    build_auth_url,
    exchange_code_for_tokens,
    get_valid_token,
    store_connection,
    verify_oauth_state,
)

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


class NotionSyncConfig(BaseModel):
    database_id: str
    sync_enabled: bool = True


@router.get("/auth")
async def initiate_auth(user_id: str = Depends(get_current_user_id)):
    """Start Notion OAuth flow."""
    auth_url = build_auth_url("notion", user_id)
    return {"auth_url": auth_url}


@router.get("/callback")
async def oauth_callback(code: str, state: str):
    """Handle Notion OAuth callback."""
    try:
        payload = verify_oauth_state(state)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    user_id = payload["user_id"]

    try:
        tokens = await exchange_code_for_tokens("notion", code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {str(e)}")

    await store_connection(_get_supabase(), user_id, "notion", tokens)
    return RedirectResponse(url=f"{FRONTEND_URL}/workspace?connected=notion")


@router.delete("/disconnect")
async def disconnect(user_id: str = Depends(get_current_user_id)):
    """Disconnect Notion integration."""
    _get_supabase().table("integration_connections").delete().eq("user_id", user_id).eq("provider", "notion").execute()
    return {"success": True}


@router.get("/databases")
async def list_databases(user_id: str = Depends(get_current_user_id)):
    """List available Notion databases."""
    token = await get_valid_token(_get_supabase(), user_id, "notion")
    if not token:
        raise HTTPException(status_code=401, detail="Notion not connected")

    from integrations.notion.service import get_databases

    databases = await get_databases(token)
    return {"databases": databases}


@router.post("/sync/tasks")
async def sync_tasks(body: NotionSyncConfig, user_id: str = Depends(get_current_user_id)):
    """Sync board tasks with a Notion database."""
    token = await get_valid_token(_get_supabase(), user_id, "notion")
    if not token:
        raise HTTPException(status_code=401, detail="Notion not connected")

    from integrations.notion.service import sync_tasks_with_notion

    result = await sync_tasks_with_notion(token, _get_supabase(), user_id, body.database_id)
    return result


@router.put("/settings")
async def update_settings(body: NotionSyncConfig, user_id: str = Depends(get_current_user_id)):
    """Update Notion sync settings."""
    supabase = _get_supabase()
    data = {
        "user_id": user_id,
        "provider": "notion",
        "event_type": "task_sync",
        "enabled": body.sync_enabled,
        "channel_id": body.database_id,  # Reusing channel_id field for database_id
    }

    existing = (
        supabase.table("notification_settings")
        .select("id")
        .eq("user_id", user_id)
        .eq("provider", "notion")
        .eq("event_type", "task_sync")
        .execute()
    )

    if existing.data:
        supabase.table("notification_settings").update(data).eq("id", existing.data[0]["id"]).execute()
    else:
        supabase.table("notification_settings").insert(data).execute()

    return {"success": True}
