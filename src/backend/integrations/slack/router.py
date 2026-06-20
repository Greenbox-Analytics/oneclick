"""FastAPI router for Slack integration."""

import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_current_user_id
from integrations.oauth import (
    FRONTEND_URL,
    build_auth_url,
    exchange_code_for_tokens,
    get_valid_token,
    store_connection,
    verify_oauth_state,
)
from subscriptions.enforcement import gated_feature
from subscriptions.models import Action

router = APIRouter()


def _verify_slack_signature(timestamp: str, signature: str, raw_body: bytes) -> bool:
    secret = os.getenv("SLACK_SIGNING_SECRET")
    if not secret or not timestamp or not signature:
        return False
    try:
        if abs(time.time() - int(timestamp)) > 60 * 5:  # replay window
            return False
    except ValueError:
        return False
    base = f"v0:{timestamp}:".encode() + raw_body
    expected = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


class NotificationSettingsUpdate(BaseModel):
    event_type: str
    enabled: bool
    channel_id: str | None = None


@router.get("/auth")
async def initiate_auth(user_id: str = Depends(get_current_user_id)):
    """Start Slack OAuth flow."""
    gated_feature(user_id, Action.USE_INTEGRATION, name="slack")
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
        analytics_capture(
            user_id,
            "integration_connect_failed",
            {"tool": "slack", "error_code": type(e).__name__},
        )
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {str(e)}")

    try:
        await store_connection(_get_supabase(), user_id, "slack", tokens)
    except Exception as e:
        analytics_capture(
            user_id,
            "integration_connect_failed",
            {"tool": "slack", "error_code": type(e).__name__},
        )
        raise

    analytics_capture(user_id, "integration_connected", {"tool": "slack"})
    return RedirectResponse(url=f"{FRONTEND_URL}/workspace?connected=slack")


@router.delete("/disconnect")
async def disconnect(user_id: str = Depends(get_current_user_id)):
    """Disconnect Slack integration."""
    _get_supabase().table("integration_connections").delete().eq("user_id", user_id).eq("provider", "slack").execute()
    # Also clean up notification settings
    _get_supabase().table("notification_settings").delete().eq("user_id", user_id).eq("provider", "slack").execute()
    return {"success": True}


@router.get("/channels")
async def list_channels(user_id: str = Depends(get_current_user_id)):
    """List available Slack channels."""
    token = await get_valid_token(_get_supabase(), user_id, "slack")
    if not token:
        raise HTTPException(status_code=401, detail="Slack not connected")

    from integrations.slack.service import get_channels

    channels = await get_channels(token)
    return {"channels": channels}


@router.put("/settings")
async def update_settings(body: NotificationSettingsUpdate, user_id: str = Depends(get_current_user_id)):
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
async def get_settings(user_id: str = Depends(get_current_user_id)):
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
    """Handle incoming Slack events (app_mention). Verifies the Slack signing secret."""
    raw_body = await request.body()
    if not _verify_slack_signature(
        request.headers.get("X-Slack-Request-Timestamp", ""),
        request.headers.get("X-Slack-Signature", ""),
        raw_body,
    ):
        raise HTTPException(status_code=403, detail="Invalid Slack signature")
    body = json.loads(raw_body)

    # Slack URL verification challenge
    if body.get("type") == "url_verification":
        return {"challenge": body["challenge"]}

    if body.get("type") == "event_callback":
        event = body.get("event", {})
        if event.get("type") == "app_mention":
            await _process_app_mention(event)

    return {"ok": True}


async def _process_app_mention(event: dict):
    """Store an @mention as a notification for the project owner."""
    channel_id = event.get("channel", "")
    sender_id = event.get("user", "")
    message_text = event.get("text", "")
    slack_ts = event.get("ts", "")

    supabase = _get_supabase()

    # Find which project this channel is linked to
    project = supabase.table("projects").select("id, user_id").eq("slack_channel_id", channel_id).execute()
    if not project.data:
        return

    # Resolve sender name from Slack
    sender_name = sender_id  # fallback

    for proj in project.data:
        supabase.table("slack_notifications").insert(
            {
                "user_id": proj["user_id"],
                "project_id": proj["id"],
                "channel_id": channel_id,
                "sender_name": sender_name,
                "sender_avatar_url": None,
                "message_text": message_text,
                "slack_ts": slack_ts,
            }
        ).execute()
