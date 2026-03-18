"""FastAPI router for Slack integration."""

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


class NotificationSettingsUpdate(BaseModel):
    event_type: str
    enabled: bool
    channel_id: Optional[str] = None


@router.get("/auth")
async def initiate_auth(user_id: str = Query(...)):
    """Start Slack OAuth flow."""
    auth_url = build_auth_url("slack", user_id)
    return {"auth_url": auth_url}


@router.get("/callback")
async def oauth_callback(code: str, state: str):
    """Handle Slack OAuth callback."""
    try:
        payload = verify_oauth_state(state)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    user_id = payload["user_id"]

    try:
        tokens = await exchange_code_for_tokens("slack", code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {str(e)}")

    await store_connection(_get_supabase(), user_id, "slack", tokens)
    return RedirectResponse(url=f"{FRONTEND_URL}/workspace?connected=slack")


@router.delete("/disconnect")
async def disconnect(user_id: str = Query(...)):
    """Disconnect Slack integration."""
    _get_supabase().table("integration_connections").delete().eq(
        "user_id", user_id
    ).eq("provider", "slack").execute()
    # Also clean up notification settings
    _get_supabase().table("notification_settings").delete().eq(
        "user_id", user_id
    ).eq("provider", "slack").execute()
    return {"success": True}


@router.get("/channels")
async def list_channels(user_id: str = Query(...)):
    """List available Slack channels."""
    token = await get_valid_token(_get_supabase(), user_id, "slack")
    if not token:
        raise HTTPException(status_code=401, detail="Slack not connected")

    from integrations.slack.service import get_channels
    channels = await get_channels(token)
    return {"channels": channels}


@router.put("/settings")
async def update_settings(body: NotificationSettingsUpdate, user_id: str = Query(...)):
    """Update notification settings for Slack."""
    supabase = _get_supabase()

    existing = (
        supabase.table("notification_settings")
        .select("id")
        .eq("user_id", user_id)
        .eq("provider", "slack")
        .eq("event_type", body.event_type)
        .execute()
    )

    data = {
        "user_id": user_id,
        "provider": "slack",
        "event_type": body.event_type,
        "enabled": body.enabled,
        "channel_id": body.channel_id,
    }

    if existing.data:
        supabase.table("notification_settings").update(data).eq("id", existing.data[0]["id"]).execute()
    else:
        supabase.table("notification_settings").insert(data).execute()

    return {"success": True}


@router.get("/settings")
async def get_settings(user_id: str = Query(...)):
    """Get all Slack notification settings."""
    result = (
        _get_supabase()
        .table("notification_settings")
        .select("*")
        .eq("user_id", user_id)
        .eq("provider", "slack")
        .execute()
    )
    return {"settings": result.data or []}


@router.post("/webhook")
async def handle_webhook(request: Request):
    """Handle incoming Slack events (slash commands, interactivity)."""
    body = await request.json()

    # Slack URL verification challenge
    if body.get("type") == "url_verification":
        return {"challenge": body["challenge"]}

    # TODO: Process Slack events (slash commands, interactive messages)
    return {"ok": True}
