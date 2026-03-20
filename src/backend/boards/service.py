"""Business logic for the Kanban board feature."""

from typing import Optional, List
from datetime import date, datetime, timezone
from supabase import Client
from integrations import events


# --- Junction table helpers ---

def _set_junction(supabase: Client, table: str, task_id: str, fk_column: str, ids: List[str]):
    """Replace all junction rows for a task. Delete existing, insert new."""
    supabase.table(table).delete().eq("task_id", task_id).execute()
    if ids:
        rows = [{"task_id": task_id, fk_column: fk_id} for fk_id in ids]
        supabase.table(table).insert(rows).execute()


def _get_junction_ids(supabase: Client, table: str, fk_column: str, task_ids: List[str]) -> dict:
    """Batch-fetch junction rows for multiple tasks. Returns {task_id: [fk_ids]}."""
    if not task_ids:
        return {}
    result = supabase.table(table).select(f"task_id, {fk_column}").in_("task_id", task_ids).execute()
    mapping = {}
    for row in (result.data or []):
        tid = row["task_id"]
        if tid not in mapping:
            mapping[tid] = []
        mapping[tid].append(row[fk_column])
    return mapping


def _enrich_tasks(supabase: Client, tasks: list) -> list:
    """Add artist_ids, project_ids, contract_ids, artist names, and parent_title to tasks."""
    if not tasks:
        return tasks
    task_ids = [t["id"] for t in tasks]

    artist_map = _get_junction_ids(supabase, "board_task_artists", "artist_id", task_ids)
    project_map = _get_junction_ids(supabase, "board_task_projects", "project_id", task_ids)
    contract_map = _get_junction_ids(supabase, "board_task_contracts", "project_file_id", task_ids)

    # Fetch artist names for display
    all_artist_ids = list({aid for ids in artist_map.values() for aid in ids})
    artist_names = {}
    if all_artist_ids:
        result = supabase.table("artists").select("id, name").in_("id", all_artist_ids).execute()
        artist_names = {a["id"]: a["name"] for a in (result.data or [])}

    # Fetch parent titles for child tasks
    parent_ids = list({t["parent_task_id"] for t in tasks if t.get("parent_task_id")})
    parent_titles = {}
    if parent_ids:
        result = supabase.table("board_tasks").select("id, title").in_("id", parent_ids).execute()
        parent_titles = {p["id"]: p["title"] for p in (result.data or [])}

    for task in tasks:
        tid = task["id"]
        task["artist_ids"] = artist_map.get(tid, [])
        task["project_ids"] = project_map.get(tid, [])
        task["contract_ids"] = contract_map.get(tid, [])
        task["artists"] = [
            {"id": aid, "name": artist_names.get(aid, "Unknown")}
            for aid in task["artist_ids"]
        ]
        if task.get("parent_task_id"):
            task["parent_title"] = parent_titles.get(task["parent_task_id"], "")

    return tasks


# --- Columns ---

async def get_columns(supabase: Client, user_id: str, artist_id: Optional[str] = None) -> list:
    """Get board columns for a user, optionally filtered by artist."""
    query = supabase.table("board_columns").select("*").eq("user_id", user_id).order("position")
    if artist_id:
        query = query.eq("artist_id", artist_id)
    result = query.execute()
    return result.data or []


async def create_column(supabase: Client, user_id: str, data: dict) -> dict:
    """Create a new board column."""
    data["user_id"] = user_id
    result = supabase.table("board_columns").insert(data).execute()
    return result.data[0] if result.data else {}


async def update_column(supabase: Client, user_id: str, column_id: str, data: dict) -> dict:
    """Update a board column."""
    clean = {k: v for k, v in data.items() if v is not None}
    result = (
        supabase.table("board_columns")
        .update(clean)
        .eq("id", column_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else {}


async def delete_column(supabase: Client, user_id: str, column_id: str) -> bool:
    """Delete a board column and all its tasks."""
    result = (
        supabase.table("board_columns")
        .delete()
        .eq("id", column_id)
        .eq("user_id", user_id)
        .execute()
    )
    return bool(result.data)


# --- Tasks (board tasks only — excludes parent tasks) ---

async def get_tasks(supabase: Client, user_id: str, column_id: Optional[str] = None) -> list:
    """Get board tasks (non-parent) with junction data, optionally filtered by column."""
    query = (
        supabase.table("board_tasks")
        .select("*")
        .eq("user_id", user_id)
        .or_("is_parent.eq.false,is_parent.is.null")
        .order("position")
    )
    if column_id:
        query = query.eq("column_id", column_id)
    result = query.execute()
    tasks = result.data or []
    return _enrich_tasks(supabase, tasks)


async def create_task(supabase: Client, user_id: str, data: dict) -> dict:
    """Create a new task with junction relations."""
    artist_ids = data.pop("artist_ids", [])
    project_ids = data.pop("project_ids", [])
    contract_ids = data.pop("contract_ids", [])

    if not data.get("start_date"):
        data["start_date"] = str(date.today())

    data["user_id"] = user_id
    result = supabase.table("board_tasks").insert(data).execute()
    task = result.data[0] if result.data else {}

    if task:
        task_id = task["id"]
        if artist_ids:
            _set_junction(supabase, "board_task_artists", task_id, "artist_id", artist_ids)
        if project_ids:
            _set_junction(supabase, "board_task_projects", task_id, "project_id", project_ids)
        if contract_ids:
            _set_junction(supabase, "board_task_contracts", task_id, "project_file_id", contract_ids)

        task["artist_ids"] = artist_ids
        task["project_ids"] = project_ids
        task["contract_ids"] = contract_ids
        await events.emit(events.TASK_CREATED, {"user_id": user_id, "task": task})

    return task


async def update_task(supabase: Client, user_id: str, task_id: str, data: dict) -> dict:
    """Update a task and its junction relations."""
    artist_ids = data.pop("artist_ids", None)
    project_ids = data.pop("project_ids", None)
    contract_ids = data.pop("contract_ids", None)

    # Empty string column_id means "clear it" (backlog)
    if "column_id" in data and data["column_id"] == "":
        data["column_id"] = None

    # Handle completed_at based on column change
    if "column_id" in data and data["column_id"] is not None:
        col_result = (
            supabase.table("board_columns")
            .select("title")
            .eq("id", data["column_id"])
            .single()
            .execute()
        )
        if col_result.data:
            col_title = col_result.data.get("title", "").lower()
            if col_title == "done":
                data["completed_at"] = datetime.now(timezone.utc).isoformat()
            else:
                data["completed_at"] = None

    # Build update dict — include explicit None for column_id to clear it
    clean = {}
    for k, v in data.items():
        if k in ("column_id", "completed_at"):
            clean[k] = v  # Allow None to clear these fields
        elif v is not None:
            clean[k] = v
    if clean:
        result = (
            supabase.table("board_tasks")
            .update(clean)
            .eq("id", task_id)
            .eq("user_id", user_id)
            .execute()
        )
        task = result.data[0] if result.data else {}
    else:
        result = (
            supabase.table("board_tasks")
            .select("*")
            .eq("id", task_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        task = result.data or {}

    if task:
        if artist_ids is not None:
            _set_junction(supabase, "board_task_artists", task_id, "artist_id", artist_ids)
        if project_ids is not None:
            _set_junction(supabase, "board_task_projects", task_id, "project_id", project_ids)
        if contract_ids is not None:
            _set_junction(supabase, "board_task_contracts", task_id, "project_file_id", contract_ids)

        await events.emit(events.TASK_UPDATED, {"user_id": user_id, "task": task})

    return task


async def delete_task(supabase: Client, user_id: str, task_id: str) -> bool:
    """Delete a task (junction rows cascade). Children get parent_task_id set to NULL."""
    result = (
        supabase.table("board_tasks")
        .delete()
        .eq("id", task_id)
        .eq("user_id", user_id)
        .execute()
    )
    return bool(result.data)


async def get_task_detail(supabase: Client, user_id: str, task_id: str) -> dict:
    """Get a single task with full related data including children and parent."""
    result = (
        supabase.table("board_tasks")
        .select("*")
        .eq("id", task_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    task = result.data
    if not task:
        return {}

    # Junction data
    artists_result = supabase.table("board_task_artists").select("artist_id").eq("task_id", task_id).execute()
    artist_ids = [r["artist_id"] for r in (artists_result.data or [])]

    projects_result = supabase.table("board_task_projects").select("project_id").eq("task_id", task_id).execute()
    project_ids = [r["project_id"] for r in (projects_result.data or [])]

    contracts_result = supabase.table("board_task_contracts").select("project_file_id").eq("task_id", task_id).execute()
    contract_ids = [r["project_file_id"] for r in (contracts_result.data or [])]

    artists = []
    if artist_ids:
        r = supabase.table("artists").select("id, name, avatar_url").in_("id", artist_ids).execute()
        artists = r.data or []

    projects = []
    if project_ids:
        r = supabase.table("projects").select("id, name").in_("id", project_ids).execute()
        projects = r.data or []

    contracts = []
    if contract_ids:
        r = supabase.table("project_files").select("id, file_name").in_("id", contract_ids).execute()
        contracts = r.data or []

    comments_result = (
        supabase.table("board_task_comments")
        .select("*")
        .eq("task_id", task_id)
        .order("created_at")
        .execute()
    )

    # Fetch children if this is a parent task
    children = []
    if task.get("is_parent"):
        children_result = (
            supabase.table("board_tasks")
            .select("*")
            .eq("parent_task_id", task_id)
            .eq("user_id", user_id)
            .order("position")
            .execute()
        )
        children = _enrich_tasks(supabase, children_result.data or [])

    # Fetch parent info if this is a child task
    parent = None
    if task.get("parent_task_id"):
        parent_result = (
            supabase.table("board_tasks")
            .select("id, title")
            .eq("id", task["parent_task_id"])
            .single()
            .execute()
        )
        parent = parent_result.data

    task["artist_ids"] = artist_ids
    task["project_ids"] = project_ids
    task["contract_ids"] = contract_ids
    task["artists"] = artists
    task["projects"] = projects
    task["contracts"] = contracts
    task["comments"] = comments_result.data or []
    task["children"] = children
    task["parent"] = parent

    return task


async def get_tasks_by_date_range(supabase: Client, user_id: str, start: str, end: str) -> list:
    """Get non-parent tasks that have due_date or start_date within a date range."""
    due_result = (
        supabase.table("board_tasks")
        .select("*")
        .eq("user_id", user_id)
        .or_("is_parent.eq.false,is_parent.is.null")
        .gte("due_date", start)
        .lte("due_date", end)
        .execute()
    )
    start_result = (
        supabase.table("board_tasks")
        .select("*")
        .eq("user_id", user_id)
        .or_("is_parent.eq.false,is_parent.is.null")
        .gte("start_date", start)
        .lte("start_date", end)
        .execute()
    )

    seen = set()
    tasks = []
    for task in (due_result.data or []) + (start_result.data or []):
        if task["id"] not in seen:
            seen.add(task["id"])
            tasks.append(task)

    return _enrich_tasks(supabase, tasks)


# --- Period-based Tasks ---

async def get_tasks_by_period(
    supabase: Client, user_id: str, period_start: str, period_end: str, is_current: bool = True
) -> list:
    """Get tasks filtered by period for date-based board views."""
    try:
        if is_current:
            # Current period: single query — all tasks except those completed before this period
            result = (
                supabase.table("board_tasks")
                .select("*")
                .eq("user_id", user_id)
                .or_("is_parent.eq.false,is_parent.is.null")
                .or_(f"completed_at.is.null,completed_at.gte.{period_start}")
                .order("position")
                .execute()
            )
            tasks = result.data or []
        else:
            # Past period: done tasks completed in period + tasks created in period
            done_result = (
                supabase.table("board_tasks")
                .select("*")
                .eq("user_id", user_id)
                .or_("is_parent.eq.false,is_parent.is.null")
                .filter("completed_at", "not.is", "null")
                .gte("completed_at", period_start)
                .lte("completed_at", period_end)
                .execute()
            )
            created_result = (
                supabase.table("board_tasks")
                .select("*")
                .eq("user_id", user_id)
                .or_("is_parent.eq.false,is_parent.is.null")
                .gte("created_at", period_start)
                .lte("created_at", period_end)
                .execute()
            )
            seen = set()
            tasks = []
            for task in (done_result.data or []) + (created_result.data or []):
                if task["id"] not in seen:
                    seen.add(task["id"])
                    tasks.append(task)
    except Exception:
        # Fallback: if completed_at column missing or other error, return all tasks
        return await get_tasks(supabase, user_id)

    return _enrich_tasks(supabase, tasks)


# --- Parent Tasks ---

async def create_parent_task(supabase: Client, user_id: str, data: dict) -> dict:
    """Create a parent task (no column_id, is_parent=True)."""
    artist_ids = data.pop("artist_ids", [])
    project_ids = data.pop("project_ids", [])

    if not data.get("start_date"):
        data["start_date"] = str(date.today())

    data["user_id"] = user_id
    data["is_parent"] = True
    result = supabase.table("board_tasks").insert(data).execute()
    task = result.data[0] if result.data else {}

    if task:
        task_id = task["id"]
        if artist_ids:
            _set_junction(supabase, "board_task_artists", task_id, "artist_id", artist_ids)
        if project_ids:
            _set_junction(supabase, "board_task_projects", task_id, "project_id", project_ids)
        task["artist_ids"] = artist_ids
        task["project_ids"] = project_ids

    return task


async def get_all_parents_with_children(
    supabase: Client, user_id: str, search: Optional[str] = None, artist_id: Optional[str] = None
) -> list:
    """Get all parent tasks with nested children for the overview tab."""
    # Fetch parent tasks
    query = (
        supabase.table("board_tasks")
        .select("*")
        .eq("user_id", user_id)
        .eq("is_parent", True)
        .order("created_at", desc=True)
    )
    parents_result = query.execute()
    parents = parents_result.data or []

    # Fetch ALL child tasks for this user (we'll group them)
    children_result = (
        supabase.table("board_tasks")
        .select("*")
        .eq("user_id", user_id)
        .or_("is_parent.eq.false,is_parent.is.null")
        .order("position")
        .execute()
    )
    all_children = children_result.data or []

    # Enrich both sets
    all_tasks = parents + all_children
    enriched = _enrich_tasks(supabase, all_tasks)

    # Split back into parents and children
    parent_map = {}
    enriched_children = []
    for t in enriched:
        if t.get("is_parent"):
            parent_map[t["id"]] = t
        else:
            enriched_children.append(t)

    # Get column names for child task display
    columns_result = supabase.table("board_columns").select("id, title").eq("user_id", user_id).execute()
    column_names = {c["id"]: c["title"] for c in (columns_result.data or [])}

    # Add column_title to parents (for status display) and children
    for parent in parent_map.values():
        parent["column_title"] = column_names.get(parent.get("column_id"), "")
    for child in enriched_children:
        child["column_title"] = column_names.get(child.get("column_id"), "")

    children_by_parent = {}
    ungrouped = []
    for child in enriched_children:
        pid = child.get("parent_task_id")
        if pid and pid in parent_map:
            if pid not in children_by_parent:
                children_by_parent[pid] = []
            children_by_parent[pid].append(child)
        elif not pid:
            ungrouped.append(child)

    # Build result
    result = []
    for pid, parent in parent_map.items():
        children = children_by_parent.get(pid, [])
        parent["children"] = children
        parent["child_count"] = len(children)
        result.append(parent)

    # Apply filters
    if artist_id:
        result = [
            p for p in result
            if artist_id in p.get("artist_ids", [])
            or any(artist_id in c.get("artist_ids", []) for c in p.get("children", []))
        ]
        ungrouped = [c for c in ungrouped if artist_id in c.get("artist_ids", [])]

    if search:
        search_lower = search.lower()
        filtered = []
        for p in result:
            # Match parent title or any child title
            if search_lower in p.get("title", "").lower():
                filtered.append(p)
            else:
                matching_children = [
                    c for c in p.get("children", [])
                    if search_lower in c.get("title", "").lower()
                ]
                if matching_children:
                    p["children"] = matching_children
                    p["child_count"] = len(matching_children)
                    filtered.append(p)
        result = filtered
        ungrouped = [c for c in ungrouped if search_lower in c.get("title", "").lower()]

    return {"parents": result, "ungrouped": ungrouped}


# --- Comments ---

async def create_comment(supabase: Client, user_id: str, task_id: str, content: str) -> dict:
    """Add a comment to a task."""
    result = supabase.table("board_task_comments").insert({
        "task_id": task_id,
        "user_id": user_id,
        "content": content,
    }).execute()
    return result.data[0] if result.data else {}


async def delete_comment(supabase: Client, user_id: str, comment_id: str) -> bool:
    """Delete a comment (only the author can delete)."""
    result = (
        supabase.table("board_task_comments")
        .delete()
        .eq("id", comment_id)
        .eq("user_id", user_id)
        .execute()
    )
    return bool(result.data)


# --- Reorder + Defaults ---

async def batch_reorder(supabase: Client, user_id: str, reorders: List[dict]) -> bool:
    """Batch reorder tasks (used for drag-and-drop)."""
    for reorder in reorders:
        supabase.table("board_tasks").update({
            "column_id": reorder["target_column_id"],
            "position": reorder["position"],
        }).eq("id", reorder["task_id"]).eq("user_id", user_id).execute()
    return True


async def create_default_columns(supabase: Client, user_id: str, artist_id: Optional[str] = None) -> list:
    """Create default Kanban columns for a new board."""
    defaults = [
        {"title": "To Do", "position": 0, "color": "#6366f1"},
        {"title": "In Progress", "position": 1, "color": "#f59e0b"},
        {"title": "Review", "position": 2, "color": "#3b82f6"},
        {"title": "Done", "position": 3, "color": "#10b981"},
    ]
    columns = []
    for col in defaults:
        col["user_id"] = user_id
        if artist_id:
            col["artist_id"] = artist_id
        result = supabase.table("board_columns").insert(col).execute()
        if result.data:
            columns.append(result.data[0])
    return columns
