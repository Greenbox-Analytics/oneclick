"""Slack business logic - sending notifications and listing channels."""

import httpx
from supabase import Client

SLACK_API = "https://slack.com/api"


async def get_channels(token: str) -> list:
    """List Slack channels the bot has access to."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SLACK_API}/conversations.list",
            headers={"Authorization": f"Bearer {token}"},
            params={"types": "public_channel,private_channel", "limit": 200},
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            return []
        return [
            {"id": ch["id"], "name": ch["name"], "is_private": ch.get("is_private", False)}
            for ch in data.get("channels", [])
        ]


async def send_notification(token: str, channel_id: str, text: str, blocks: list = None) -> dict:
    """Send a notification message to a Slack channel."""
    payload = {"channel": channel_id, "text": text}
    if blocks:
        payload["blocks"] = blocks

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SLACK_API}/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def upload_file_to_channel(token: str, channel_id: str, file_content: bytes, filename: str, title: str) -> dict:
    """Upload a file to a Slack channel."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SLACK_API}/files.upload",
            headers={"Authorization": f"Bearer {token}"},
            data={"channels": channel_id, "title": title, "filename": filename},
            files={"file": (filename, file_content)},
        )
        response.raise_for_status()
        return response.json()


async def notify_for_event(supabase: Client, user_id: str, event_name: str, event_data: dict):
    """
    Send Slack notification for an event.
    Routes to project-linked channel first, falls back to workspace settings.
    """
    from integrations.oauth import get_valid_token
    from integrations.slack.blocks import build_blocks

    token = await get_valid_token(supabase, user_id, "slack")
    if not token:
        return

    text, blocks = build_blocks(event_name, event_data)

    # 1. Try project-level channel
    project_id = event_data.get("project_id") or _extract_project_id(event_data)
    if project_id:
        sent = await _try_project_channel(supabase, project_id, event_name, token, text, blocks)
        if sent:
            return

    # 2. Fall back to workspace-level settings
    settings = (
        supabase.table("notification_settings")
        .select("*")
        .eq("user_id", user_id)
        .eq("provider", "slack")
        .eq("event_type", event_name)
        .eq("enabled", True)
        .execute()
    )

    if not settings.data:
        return

    channel_id = settings.data[0].get("channel_id")
    if not channel_id:
        return

    await send_notification(token, channel_id, text, blocks)


async def _try_project_channel(
    supabase: Client,
    project_id: str,
    event_name: str,
    token: str,
    text: str,
    blocks: list | None,
) -> bool:
    """Try to send to a project's linked Slack channel. Returns True if sent."""
    project = supabase.table("projects").select("slack_channel_id").eq("id", project_id).single().execute()
    channel_id = project.data.get("slack_channel_id") if project.data else None
    if not channel_id:
        return False

    # Check if this event type is enabled for this project
    setting = (
        supabase.table("project_notification_settings")
        .select("enabled")
        .eq("project_id", project_id)
        .eq("event_type", event_name)
        .execute()
    )
    if setting.data and not setting.data[0].get("enabled", True):
        return False

    await send_notification(token, channel_id, text, blocks)
    return True


def _extract_project_id(event_data: dict) -> str | None:
    """Extract project_id from event data (tasks store it in project_ids array)."""
    task = event_data.get("task", {})
    project_ids = task.get("project_ids", [])
    return project_ids[0] if project_ids else None
