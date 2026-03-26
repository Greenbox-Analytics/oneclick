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
    payload = {
        "channel": channel_id,
        "text": text,
    }
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


async def notify_for_event(
    supabase: Client, user_id: str, event_name: str, event_data: dict
):
    """Check if user has Slack notifications enabled for this event, and send if so."""
    from integrations.oauth import get_valid_token

    # Check notification settings
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

    token = await get_valid_token(supabase, user_id, "slack")
    if not token:
        return

    setting = settings.data[0]
    channel_id = setting.get("channel_id")
    if not channel_id:
        return

    # Format message based on event type
    message = _format_event_message(event_name, event_data)
    await send_notification(token, channel_id, message)


def _format_event_message(event_name: str, data: dict) -> str:
    """Format an event into a human-readable Slack message."""
    messages = {
        "contract_uploaded": f"New contract uploaded: {data.get('file_name', 'Unknown')}",
        "contract_deleted": f"Contract deleted: {data.get('file_name', 'Unknown')}",
        "royalty_calculated": f"Royalty calculation complete for {data.get('artist_name', 'Unknown')}",
        "task_created": f"New task created: {data.get('task', {}).get('title', 'Unknown')}",
        "task_updated": f"Task updated: {data.get('task', {}).get('title', 'Unknown')}",
        "task_completed": f"Task completed: {data.get('task', {}).get('title', 'Unknown')}",
    }
    return messages.get(event_name, f"Event: {event_name}")
