"""FastAPI router for the Kanban board feature."""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

# Ensure backend dir is in path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_current_user_id
from boards import service
from boards.models import (
    BatchReorder,
    ColumnCreate,
    ColumnUpdate,
    CommentCreate,
    ParentTaskCreate,
    TaskCreate,
    TaskUpdate,
)
from subscriptions.enforcement import gated_create

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


# --- Columns ---


@router.get("/columns")
async def list_columns(
    user_id: str = Depends(get_current_user_id),
    artist_id: str | None = Query(None),
):
    """Get all board columns for a user."""
    columns = await service.get_columns(_get_supabase(), user_id, artist_id)
    return {"columns": columns}


@router.post("/columns")
async def create_column(body: ColumnCreate, user_id: str = Depends(get_current_user_id)):
    """Create a new board column."""
    data = body.model_dump(exclude_none=True)
    column = await service.create_column(_get_supabase(), user_id, data)
    if not column:
        raise HTTPException(status_code=500, detail="Failed to create column")
    return column


@router.put("/columns/{column_id}")
async def update_column(column_id: str, body: ColumnUpdate, user_id: str = Depends(get_current_user_id)):
    """Update a board column."""
    data = body.model_dump(exclude_none=True)
    column = await service.update_column(_get_supabase(), user_id, column_id, data)
    if not column:
        raise HTTPException(status_code=404, detail="Column not found")
    return column


@router.delete("/columns/{column_id}")
async def delete_column(column_id: str, user_id: str = Depends(get_current_user_id)):
    """Delete a board column and all its tasks."""
    success = await service.delete_column(_get_supabase(), user_id, column_id)
    if not success:
        raise HTTPException(status_code=404, detail="Column not found")
    return {"success": True}


@router.post("/columns/defaults")
async def create_defaults(
    user_id: str = Depends(get_current_user_id),
    artist_id: str | None = Query(None),
):
    """Create default columns (To Do, In Progress, Review, Done)."""
    columns = await service.create_default_columns(_get_supabase(), user_id, artist_id)
    return {"columns": columns}


# --- Parent Tasks (must come before /tasks/{task_id} routes) ---


@router.get("/parents")
async def list_parents(
    user_id: str = Depends(get_current_user_id),
    search: str | None = Query(None),
    artist_id: str | None = Query(None),
):
    """Get all parent tasks with nested children for the overview tab."""
    result = await service.get_all_parents_with_children(_get_supabase(), user_id, search, artist_id)
    return result


@router.post("/parents")
async def create_parent(body: ParentTaskCreate, user_id: str = Depends(get_current_user_id)):
    """Create a parent task (no column, is_parent=True)."""
    data = body.model_dump(exclude_none=True)
    if data.get("due_date"):
        data["due_date"] = str(data["due_date"])
    if data.get("start_date"):
        data["start_date"] = str(data["start_date"])
    task = await service.create_parent_task(_get_supabase(), user_id, data)
    if not task:
        raise HTTPException(status_code=500, detail="Failed to create parent task")
    analytics_capture(user_id, "board_created", {"tool": "boards"})
    return task


# --- Calendar (must come before /tasks/{task_id} routes) ---


@router.get("/calendar")
async def calendar_tasks(
    user_id: str = Depends(get_current_user_id),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
):
    """Get tasks within a date range for the calendar view."""
    tasks = await service.get_tasks_by_date_range(_get_supabase(), user_id, start, end)
    return {"tasks": tasks}


# --- Period-based Tasks (must come before /tasks/{task_id} routes) ---


@router.get("/tasks/period")
async def period_tasks(
    user_id: str = Depends(get_current_user_id),
    period_start: str = Query(..., description="Period start date YYYY-MM-DD"),
    period_end: str = Query(..., description="Period end date YYYY-MM-DD"),
    is_current: bool = Query(True, description="Whether this is the current period"),
):
    """Get tasks within a period for date-based board views."""
    tasks = await service.get_tasks_by_period(_get_supabase(), user_id, period_start, period_end, is_current)
    return {"tasks": tasks}


# --- Reorder (must come before /tasks/{task_id} routes) ---


@router.put("/tasks/reorder")
async def reorder_tasks(body: BatchReorder, user_id: str = Depends(get_current_user_id)):
    """Batch reorder tasks (drag-and-drop)."""
    reorders = [r.model_dump() for r in body.reorders]
    await service.batch_reorder(_get_supabase(), user_id, reorders)
    return {"success": True}


# --- Tasks ---


@router.get("/tasks")
async def list_tasks(
    user_id: str = Depends(get_current_user_id),
    column_id: str | None = Query(None),
    page: int | None = Query(None, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    """Get all tasks for a user, optionally filtered by column."""
    result = await service.get_tasks(_get_supabase(), user_id, column_id, page, page_size)
    if isinstance(result, list):
        return {"tasks": result}
    return result


@router.post("/tasks")
async def create_task(body: TaskCreate, user_id: str = Depends(get_current_user_id)):
    """Create a new task."""
    # Gate: count user's existing board tasks
    count_res = _get_supabase().table("board_tasks").select("id", count="exact").eq("user_id", user_id).execute()
    task_count = count_res.count or 0
    gated_create(user_id, "task", current_count=task_count)

    data = body.model_dump(exclude_none=True)
    if data.get("due_date"):
        data["due_date"] = str(data["due_date"])
    if data.get("start_date"):
        data["start_date"] = str(data["start_date"])
    task = await service.create_task(_get_supabase(), user_id, data)
    if not task:
        raise HTTPException(status_code=500, detail="Failed to create task")
    analytics_capture(user_id, "task_created", {"tool": "boards", "source": "manual"})
    return task


@router.get("/tasks/{task_id}/detail")
async def get_task_detail(task_id: str, user_id: str = Depends(get_current_user_id)):
    """Get a single task with full detail (artists, projects, contracts, comments)."""
    task = await service.get_task_detail(_get_supabase(), user_id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.put("/tasks/{task_id}")
async def update_task(task_id: str, body: TaskUpdate, user_id: str = Depends(get_current_user_id)):
    """Update a task."""
    data = body.model_dump(exclude_none=True)
    if data.get("due_date"):
        data["due_date"] = str(data["due_date"])
    if data.get("start_date"):
        data["start_date"] = str(data["start_date"])

    # Read existing column_id BEFORE the update so we can detect column changes
    # for analytics (task_status_changed / task_completed).
    supabase = _get_supabase()
    old_column_id = None
    try:
        existing = supabase.table("board_tasks").select("column_id").eq("id", task_id).single().execute()
        if existing and existing.data:
            old_column_id = existing.data.get("column_id")
    except Exception:
        # If pre-read fails we still try the update; analytics is best-effort.
        old_column_id = None

    task = await service.update_task(supabase, user_id, task_id, data)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Detect column transition for analytics.
    new_column_id = data.get("column_id") if "column_id" in data else old_column_id
    if new_column_id and new_column_id != old_column_id:
        analytics_capture(
            user_id,
            "task_status_changed",
            {
                "tool": "boards",
                "from_column_id": old_column_id,
                "to_column_id": new_column_id,
            },
        )
        # Check if the new column means the task is "done".
        try:
            col_result = supabase.table("board_columns").select("title").eq("id", new_column_id).single().execute()
            new_title = (col_result.data or {}).get("title") or ""
            if new_title.strip().lower() == "done":
                analytics_capture(user_id, "task_completed", {"tool": "boards"})
        except Exception:
            pass

    return task


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, user_id: str = Depends(get_current_user_id)):
    """Delete a task."""
    success = await service.delete_task(_get_supabase(), user_id, task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True}


# --- Comments ---


@router.post("/tasks/{task_id}/comments")
async def add_comment(task_id: str, body: CommentCreate, user_id: str = Depends(get_current_user_id)):
    """Add a comment to a task."""
    try:
        comment = await service.create_comment(_get_supabase(), user_id, task_id, body.content)
    except PermissionError:
        raise HTTPException(status_code=404, detail="Task not found")
    if not comment:
        raise HTTPException(status_code=500, detail="Failed to add comment")
    return comment


@router.delete("/comments/{comment_id}")
async def remove_comment(comment_id: str, user_id: str = Depends(get_current_user_id)):
    """Delete a comment."""
    success = await service.delete_comment(_get_supabase(), user_id, comment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Comment not found")
    return {"success": True}
