"""Notion business logic - database listing and task sync."""

import hashlib

import httpx
from supabase import Client

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


async def get_databases(token: str) -> list:
    """List Notion databases the integration has access to."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{NOTION_API}/search",
            headers=_headers(token),
            json={"filter": {"value": "database", "property": "object"}},
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        return [
            {
                "id": db["id"],
                "title": db.get("title", [{}])[0].get("plain_text", "Untitled") if db.get("title") else "Untitled",
                "url": db.get("url", ""),
            }
            for db in results
        ]


async def sync_tasks_with_notion(token: str, supabase: Client, user_id: str, database_id: str) -> dict:
    """Bidirectional sync of board tasks with a Notion database."""
    # Get local tasks
    local_tasks = (supabase.table("board_tasks").select("*").eq("user_id", user_id).execute()).data or []

    # Get Notion pages
    notion_pages = await _get_notion_pages(token, database_id)

    pushed = 0
    pulled = 0

    # Push local tasks that don't exist in Notion
    _local_by_external = {
        t["external_id"]: t for t in local_tasks if t.get("external_id") and t.get("external_provider") == "notion"
    }
    local_without_external = [t for t in local_tasks if not t.get("external_id")]

    for task in local_without_external:
        page = await _create_notion_page(token, database_id, task)
        if page:
            supabase.table("board_tasks").update(
                {
                    "external_id": page["id"],
                    "external_provider": "notion",
                    "external_url": page.get("url", ""),
                    "sync_hash": _task_hash(task),
                    "last_synced_at": "now()",
                }
            ).eq("id", task["id"]).execute()
            pushed += 1

    # Pull Notion pages that don't exist locally
    existing_external_ids = {t.get("external_id") for t in local_tasks if t.get("external_id")}
    for page in notion_pages:
        if page["id"] not in existing_external_ids:
            task_data = _notion_page_to_task(page, user_id)
            if task_data:
                supabase.table("board_tasks").insert(task_data).execute()
                pulled += 1

    # Log sync
    supabase.table("sync_log").insert(
        {
            "user_id": user_id,
            "provider": "notion",
            "direction": "bidirectional",
            "entity_type": "task",
            "status": "success",
            "metadata": {"pushed": pushed, "pulled": pulled},
        }
    ).execute()

    return {"pushed": pushed, "pulled": pulled}


async def _get_notion_pages(token: str, database_id: str) -> list:
    """Query all pages from a Notion database."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{NOTION_API}/databases/{database_id}/query",
            headers=_headers(token),
            json={},
        )
        response.raise_for_status()
        return response.json().get("results", [])


async def _create_notion_page(token: str, database_id: str, task: dict) -> dict:
    """Create a Notion page from a board task."""
    properties = {
        "Name": {"title": [{"text": {"content": task.get("title", "Untitled")}}]},
    }
    if task.get("description"):
        properties["Description"] = {"rich_text": [{"text": {"content": task["description"][:2000]}}]}
    if task.get("priority"):
        properties["Priority"] = {"select": {"name": task["priority"].capitalize()}}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{NOTION_API}/pages",
            headers=_headers(token),
            json={"parent": {"database_id": database_id}, "properties": properties},
        )
        if response.status_code == 200:
            return response.json()
        return {}


def _notion_page_to_task(page: dict, user_id: str) -> dict:
    """Convert a Notion page to a board task dict."""
    props = page.get("properties", {})
    title = ""
    if "Name" in props and props["Name"].get("title"):
        title = props["Name"]["title"][0].get("plain_text", "") if props["Name"]["title"] else ""

    if not title:
        return {}

    return {
        "user_id": user_id,
        "title": title,
        "external_id": page["id"],
        "external_provider": "notion",
        "external_url": page.get("url", ""),
        "sync_hash": hashlib.sha256(title.encode()).hexdigest()[:16],
        "position": 0,
    }


def _task_hash(task: dict) -> str:
    """Generate a hash of task content for sync conflict detection."""
    content = f"{task.get('title', '')}{task.get('description', '')}{task.get('priority', '')}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
