"""Monday.com business logic - board listing and task sync via GraphQL API."""

import hashlib

import httpx
from supabase import Client

MONDAY_API = "https://api.monday.com/v2"


async def _graphql(token: str, query: str, variables: dict = None) -> dict:
    """Execute a Monday.com GraphQL query."""
    async with httpx.AsyncClient() as client:
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        response = await client.post(
            MONDAY_API,
            headers={
                "Authorization": token,
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def get_boards(token: str) -> list:
    """List Monday.com boards."""
    query = """
    query {
        boards(limit: 50) {
            id
            name
            state
            board_kind
            columns { id title type }
        }
    }
    """
    result = await _graphql(token, query)
    boards = result.get("data", {}).get("boards", [])
    return [
        {
            "id": b["id"],
            "name": b["name"],
            "state": b["state"],
            "kind": b["board_kind"],
            "columns": b.get("columns", []),
        }
        for b in boards
        if b["state"] == "active"
    ]


async def get_board_items(token: str, board_id: str) -> list:
    """Get all items from a Monday.com board."""
    query = """
    query($boardId: [ID!]) {
        boards(ids: $boardId) {
            items_page(limit: 500) {
                items {
                    id
                    name
                    state
                    column_values {
                        id
                        text
                        type
                    }
                    url
                }
            }
        }
    }
    """
    result = await _graphql(token, query, {"boardId": [board_id]})
    boards = result.get("data", {}).get("boards", [])
    if not boards:
        return []
    return boards[0].get("items_page", {}).get("items", [])


async def create_monday_item(token: str, board_id: str, item_name: str, column_values: dict = None) -> dict:
    """Create an item on a Monday.com board."""
    import json

    col_values_str = json.dumps(column_values) if column_values else "{}"
    query = """
    mutation($boardId: ID!, $itemName: String!, $columnValues: JSON!) {
        create_item(board_id: $boardId, item_name: $itemName, column_values: $columnValues) {
            id
            name
            url
        }
    }
    """
    result = await _graphql(
        token,
        query,
        {
            "boardId": board_id,
            "itemName": item_name,
            "columnValues": col_values_str,
        },
    )
    return result.get("data", {}).get("create_item", {})


async def sync_tasks_with_monday(token: str, supabase: Client, user_id: str, board_id: str) -> dict:
    """Bidirectional sync of board tasks with Monday.com."""
    # Get local tasks
    local_tasks = (supabase.table("board_tasks").select("*").eq("user_id", user_id).execute()).data or []

    # Get Monday items
    monday_items = await get_board_items(token, board_id)

    pushed = 0
    pulled = 0

    # Push local tasks without external ID to Monday
    for task in local_tasks:
        if not task.get("external_id") or task.get("external_provider") != "monday":
            if not task.get("external_id"):
                item = await create_monday_item(token, board_id, task["title"])
                if item.get("id"):
                    supabase.table("board_tasks").update(
                        {
                            "external_id": str(item["id"]),
                            "external_provider": "monday",
                            "external_url": item.get("url", ""),
                            "sync_hash": _task_hash(task),
                            "last_synced_at": "now()",
                        }
                    ).eq("id", task["id"]).execute()
                    pushed += 1

    # Pull Monday items not in local tasks
    existing_external_ids = {t.get("external_id") for t in local_tasks if t.get("external_id")}
    for item in monday_items:
        if str(item["id"]) not in existing_external_ids:
            task_data = {
                "user_id": user_id,
                "title": item["name"],
                "external_id": str(item["id"]),
                "external_provider": "monday",
                "external_url": item.get("url", ""),
                "sync_hash": hashlib.sha256(item["name"].encode()).hexdigest()[:16],
                "position": 0,
            }
            supabase.table("board_tasks").insert(task_data).execute()
            pulled += 1

    # Log sync
    supabase.table("sync_log").insert(
        {
            "user_id": user_id,
            "provider": "monday",
            "direction": "bidirectional",
            "entity_type": "task",
            "status": "success",
            "metadata": {"pushed": pushed, "pulled": pulled},
        }
    ).execute()

    return {"pushed": pushed, "pulled": pulled}


def _task_hash(task: dict) -> str:
    content = f"{task.get('title', '')}{task.get('description', '')}{task.get('priority', '')}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
