"""FastAPI router for Monday.com integration."""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from integrations.oauth import (
    build_auth_url, verify_oauth_state, exchange_code_for_tokens,
    store_connection, get_valid_token, FRONTEND_URL,
)

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client
    return get_supabase_client()


class MondaySyncConfig(BaseModel):
    board_id: str
    sync_enabled: bool = True


@router.get("/auth")
async def initiate_auth(user_id: str = Query(...)):
    """Start Monday.com OAuth flow."""
    auth_url = build_auth_url("monday", user_id)
    return {"auth_url": auth_url}


@router.get("/callback")
async def oauth_callback(code: str, state: str):
    """Handle Monday.com OAuth callback."""
    try:
        payload = verify_oauth_state(state)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    user_id = payload["user_id"]

    try:
        tokens = await exchange_code_for_tokens("monday", code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {str(e)}")

    await store_connection(_get_supabase(), user_id, "monday", tokens)
    return RedirectResponse(url=f"{FRONTEND_URL}/workspace?connected=monday")


@router.delete("/disconnect")
async def disconnect(user_id: str = Query(...)):
    """Disconnect Monday.com integration."""
    _get_supabase().table("integration_connections").delete().eq(
        "user_id", user_id
    ).eq("provider", "monday").execute()
    return {"success": True}


@router.get("/boards")
async def list_boards(user_id: str = Query(...)):
    """List available Monday.com boards."""
    token = await get_valid_token(_get_supabase(), user_id, "monday")
    if not token:
        raise HTTPException(status_code=401, detail="Monday.com not connected")

    from integrations.monday.service import get_boards
    boards = await get_boards(token)
    return {"boards": boards}


@router.post("/sync/tasks")
async def sync_tasks(body: MondaySyncConfig, user_id: str = Query(...)):
    """Sync board tasks with a Monday.com board."""
    token = await get_valid_token(_get_supabase(), user_id, "monday")
    if not token:
        raise HTTPException(status_code=401, detail="Monday.com not connected")

    from integrations.monday.service import sync_tasks_with_monday
    result = await sync_tasks_with_monday(token, _get_supabase(), user_id, body.board_id)
    return result


@router.post("/webhook")
async def handle_webhook(request: Request):
    """Handle incoming Monday.com webhooks."""
    body = await request.json()

    # Monday.com webhook verification challenge
    if body.get("challenge"):
        return {"challenge": body["challenge"]}

    # TODO: Process Monday.com webhook events
    return {"ok": True}


@router.put("/settings")
async def update_settings(body: MondaySyncConfig, user_id: str = Query(...)):
    """Update Monday.com sync settings."""
    supabase = _get_supabase()
    data = {
        "user_id": user_id,
        "provider": "monday",
        "event_type": "task_sync",
        "enabled": body.sync_enabled,
        "channel_id": body.board_id,
    }

    existing = (
        supabase.table("notification_settings")
        .select("id")
        .eq("user_id", user_id)
        .eq("provider", "monday")
        .eq("event_type", "task_sync")
        .execute()
    )

    if existing.data:
        supabase.table("notification_settings").update(data).eq("id", existing.data[0]["id"]).execute()
    else:
        supabase.table("notification_settings").insert(data).execute()

    return {"success": True}
