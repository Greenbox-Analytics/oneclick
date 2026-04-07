"""FastAPI router for the Kanban board feature."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import sys
from pathlib import Path

# Ensure backend dir is in path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from boards import service
from boards.models import (
    ColumnCreate, ColumnUpdate,
    TaskCreate, TaskUpdate, BatchReorder,
    CommentCreate, ParentTaskCreate,
)

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client
    return get_supabase_client()


# --- Columns ---

@router.get("/columns")
async def list_columns(
    user_id: str = Query(...),
    artist_id: Optional[str] = Query(None),
):
    """Get all board columns for a user."""
    columns = await service.get_columns(_get_supabase(), user_id, artist_id)
    return {"columns": columns}


@router.post("/columns")
async def create_column(body: ColumnCreate, user_id: str = Query(...)):
    """Create a new board column."""
    data = body.model_dump(exclude_none=True)
    column = await service.create_column(_get_supabase(), user_id, data)
    if not column:
        raise HTTPException(status_code=500, detail="Failed to create column")
    return column


@router.put("/columns/{column_id}")
async def update_column(column_id: str, body: ColumnUpdate, user_id: str = Query(...)):
    """Update a board column."""
    data = body.model_dump(exclude_none=True)
    column = await service.update_column(_get_supabase(), user_id, column_id, data)
    if not column:
        raise HTTPException(status_code=404, detail="Column not found")
    return column


@router.delete("/columns/{column_id}")
async def delete_column(column_id: str, user_id: str = Query(...)):
    """Delete a board column and all its tasks."""
    success = await service.delete_column(_get_supabase(), user_id, column_id)
    if not success:
        raise HTTPException(status_code=404, detail="Column not found")
    return {"success": True}


@router.post("/columns/defaults")
async def create_defaults(
    user_id: str = Query(...),
    artist_id: Optional[str] = Query(None),
):
    """Create default columns (To Do, In Progress, Review, Done)."""
    columns = await service.create_default_columns(_get_supabase(), user_id, artist_id)
    return {"columns": columns}


# --- Parent Tasks (must come before /tasks/{task_id} routes) ---

@router.get("/parents")
async def list_parents(
    user_id: str = Query(...),
    search: Optional[str] = Query(None),
    artist_id: Optional[str] = Query(None),
):
    """Get all parent tasks with nested children for the overview tab."""
    result = await service.get_all_parents_with_children(
        _get_supabase(), user_id, search, artist_id
    )
    return result


@router.post("/parents")
async def create_parent(body: ParentTaskCreate, user_id: str = Query(...)):
    """Create a parent task (no column, is_parent=True)."""
    data = body.model_dump(exclude_none=True)
    if data.get("due_date"):
        data["due_date"] = str(data["due_date"])
    if data.get("start_date"):
        data["start_date"] = str(data["start_date"])
    task = await service.create_parent_task(_get_supabase(), user_id, data)
    if not task:
        raise HTTPException(status_code=500, detail="Failed to create parent task")
    return task


# --- Calendar (must come before /tasks/{task_id} routes) ---

@router.get("/calendar")
async def calendar_tasks(
    user_id: str = Query(...),
    start: str = Query(..., description="Start date YYYY-MM-DD"),
    end: str = Query(..., description="End date YYYY-MM-DD"),
):
    """Get tasks within a date range for the calendar view."""
    tasks = await service.get_tasks_by_date_range(_get_supabase(), user_id, start, end)
    return {"tasks": tasks}


# --- Period-based Tasks (must come before /tasks/{task_id} routes) ---

@router.get("/tasks/period")
async def period_tasks(
    user_id: str = Query(...),
    period_start: str = Query(..., description="Period start date YYYY-MM-DD"),
    period_end: str = Query(..., description="Period end date YYYY-MM-DD"),
    is_current: bool = Query(True, description="Whether this is the current period"),
):
    """Get tasks within a period for date-based board views."""
    tasks = await service.get_tasks_by_period(
        _get_supabase(), user_id, period_start, period_end, is_current
    )
    return {"tasks": tasks}


# --- Reorder (must come before /tasks/{task_id} routes) ---

@router.put("/tasks/reorder")
async def reorder_tasks(body: BatchReorder, user_id: str = Query(...)):
    """Batch reorder tasks (drag-and-drop)."""
    reorders = [r.model_dump() for r in body.reorders]
    await service.batch_reorder(_get_supabase(), user_id, reorders)
    return {"success": True}


# --- Tasks ---

@router.get("/tasks")
async def list_tasks(
    user_id: str = Query(...),
    column_id: Optional[str] = Query(None),
    page: Optional[int] = Query(None, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    """Get all tasks for a user, optionally filtered by column."""
    result = await service.get_tasks(_get_supabase(), user_id, column_id, page, page_size)
    if isinstance(result, list):
        return {"tasks": result}
    return result


@router.post("/tasks")
async def create_task(body: TaskCreate, user_id: str = Query(...)):
    """Create a new task."""
    data = body.model_dump(exclude_none=True)
    if data.get("due_date"):
        data["due_date"] = str(data["due_date"])
    if data.get("start_date"):
        data["start_date"] = str(data["start_date"])
    task = await service.create_task(_get_supabase(), user_id, data)
    if not task:
        raise HTTPException(status_code=500, detail="Failed to create task")
    return task


@router.get("/tasks/{task_id}/detail")
async def get_task_detail(task_id: str, user_id: str = Query(...)):
    """Get a single task with full detail (artists, projects, contracts, comments)."""
    task = await service.get_task_detail(_get_supabase(), user_id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.put("/tasks/{task_id}")
async def update_task(task_id: str, body: TaskUpdate, user_id: str = Query(...)):
    """Update a task."""
    data = body.model_dump(exclude_none=True)
    if data.get("due_date"):
        data["due_date"] = str(data["due_date"])
    if data.get("start_date"):
        data["start_date"] = str(data["start_date"])
    task = await service.update_task(_get_supabase(), user_id, task_id, data)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, user_id: str = Query(...)):
    """Delete a task."""
    success = await service.delete_task(_get_supabase(), user_id, task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"success": True}


# --- Comments ---

@router.post("/tasks/{task_id}/comments")
async def add_comment(task_id: str, body: CommentCreate, user_id: str = Query(...)):
    """Add a comment to a task."""
    comment = await service.create_comment(_get_supabase(), user_id, task_id, body.content)
    if not comment:
        raise HTTPException(status_code=500, detail="Failed to add comment")
    return comment


@router.delete("/comments/{comment_id}")
async def remove_comment(comment_id: str, user_id: str = Query(...)):
    """Delete a comment."""
    success = await service.delete_comment(_get_supabase(), user_id, comment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Comment not found")
    return {"success": True}
