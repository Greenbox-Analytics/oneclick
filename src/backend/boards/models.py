"""Pydantic models for the Kanban board feature."""

from datetime import date

from pydantic import BaseModel


class ColumnCreate(BaseModel):
    title: str
    artist_id: str | None = None
    board_id: str | None = None
    color: str | None = None
    position: int | None = 0


class ColumnUpdate(BaseModel):
    title: str | None = None
    color: str | None = None
    position: int | None = None


class TaskCreate(BaseModel):
    column_id: str | None = None  # nullable for parent tasks
    board_id: str | None = None
    title: str
    description: str | None = None
    priority: str | None = None  # low, medium, high, urgent
    start_date: date | None = None
    due_date: date | None = None
    color: str | None = None
    parent_task_id: str | None = None
    is_parent: bool = False
    artist_ids: list[str] | None = []
    project_ids: list[str] | None = []
    contract_ids: list[str] | None = []
    assignee_name: str | None = None
    labels: list[str] | None = []


class TaskUpdate(BaseModel):
    column_id: str | None = None
    title: str | None = None
    description: str | None = None
    priority: str | None = None
    start_date: date | None = None
    due_date: date | None = None
    color: str | None = None
    parent_task_id: str | None = None
    is_parent: bool | None = None
    artist_ids: list[str] | None = None
    project_ids: list[str] | None = None
    contract_ids: list[str] | None = None
    assignee_name: str | None = None
    labels: list[str] | None = None
    position: int | None = None


class ParentTaskCreate(BaseModel):
    board_id: str | None = None
    title: str
    description: str | None = None
    priority: str | None = None
    start_date: date | None = None
    due_date: date | None = None
    color: str | None = None
    artist_ids: list[str] | None = []
    project_ids: list[str] | None = []
    labels: list[str] | None = []


class TaskReorder(BaseModel):
    task_id: str
    target_column_id: str
    position: int


class BatchReorder(BaseModel):
    reorders: list[TaskReorder]


class CommentCreate(BaseModel):
    content: str


class BoardCreate(BaseModel):
    name: str
    team_id: str | None = None
    artist_id: str | None = None
    description: str | None = None


class BoardUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class AssigneeAdd(BaseModel):
    user_id: str


class DeleteConfirm(BaseModel):
    confirm_name: str
