"""Slack Block Kit message builders for Msanii notifications."""

import os

FRONTEND_URL = os.getenv("VITE_FRONTEND_URL", "http://localhost:8080")


def _header(text: str) -> dict:
    return {"type": "header", "text": {"type": "plain_text", "text": text, "emoji": True}}


def _section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _context(elements: list[str]) -> dict:
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": e} for e in elements],
    }


def _actions(button_text: str, url: str) -> dict:
    return {
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": button_text, "emoji": True},
                "url": url,
                "style": "primary",
            }
        ],
    }


def _divider() -> dict:
    return {"type": "divider"}


def task_created(data: dict) -> tuple[str, list[dict]]:
    task = data.get("task", {})
    title = task.get("title", "Unknown")
    priority = task.get("priority", "")

    text = f"New task created: {title}"
    blocks = [
        _header("\U0001f3b5 New task created"),
        _section(f"*{title}*"),
        _context(
            [
                f"Priority: {priority}" if priority else "No priority set",
            ]
        ),
        _divider(),
        _actions("View in Msanii", f"{FRONTEND_URL}/workspace?tab=boards"),
    ]
    return text, blocks


def task_updated(data: dict) -> tuple[str, list[dict]]:
    task = data.get("task", {})
    title = task.get("title", "Unknown")

    text = f"Task updated: {title}"
    blocks = [
        _header("\u270f\ufe0f Task updated"),
        _section(f"*{title}*"),
        _divider(),
        _actions("View in Msanii", f"{FRONTEND_URL}/workspace?tab=boards"),
    ]
    return text, blocks


def task_completed(data: dict) -> tuple[str, list[dict]]:
    task = data.get("task", {})
    title = task.get("title", "Unknown")

    text = f"Task completed: {title}"
    blocks = [
        _header("\u2705 Task completed"),
        _section(f"*{title}*"),
        _divider(),
        _actions("View in Msanii", f"{FRONTEND_URL}/workspace?tab=boards"),
    ]
    return text, blocks


def contract_uploaded(data: dict) -> tuple[str, list[dict]]:
    file_name = data.get("file_name", "Unknown")
    project_name = data.get("project_name", "")
    project_id = data.get("project_id", "")

    text = f"New contract uploaded: {file_name}"
    context_parts = []
    if project_name:
        context_parts.append(f"Project: {project_name}")

    blocks = [
        _header("\U0001f4c4 Contract uploaded"),
        _section(f"*{file_name}*"),
    ]
    if context_parts:
        blocks.append(_context(context_parts))
    blocks.append(_divider())
    if project_id:
        blocks.append(_actions("View Project", f"{FRONTEND_URL}/projects/{project_id}"))
    return text, blocks


def contract_deleted(data: dict) -> tuple[str, list[dict]]:
    file_name = data.get("file_name", "Unknown")
    project_name = data.get("project_name", "")

    text = f"Contract deleted: {file_name}"
    context_parts = []
    if project_name:
        context_parts.append(f"Project: {project_name}")

    blocks = [
        _header("\U0001f5d1\ufe0f Contract deleted"),
        _section(f"*{file_name}*"),
    ]
    if context_parts:
        blocks.append(_context(context_parts))
    return text, blocks


def royalty_calculated(data: dict) -> tuple[str, list[dict]]:
    artist_name = data.get("artist_name", "Unknown")

    text = f"Royalty calculation complete for {artist_name}"
    blocks = [
        _header("\U0001f4b0 Royalty calculation complete"),
        _section(f"Artist: *{artist_name}*"),
        _divider(),
        _actions("View Results", f"{FRONTEND_URL}/tools/oneclick"),
    ]
    return text, blocks


BLOCK_BUILDERS: dict[str, object] = {
    "task_created": task_created,
    "task_updated": task_updated,
    "task_completed": task_completed,
    "contract_uploaded": contract_uploaded,
    "contract_deleted": contract_deleted,
    "royalty_calculated": royalty_calculated,
}


def build_blocks(event_name: str, data: dict) -> tuple[str, list[dict] | None]:
    """Build Block Kit blocks for an event. Returns (fallback_text, blocks)."""
    builder = BLOCK_BUILDERS.get(event_name)
    if builder:
        return builder(data)
    return f"Event: {event_name}", None
