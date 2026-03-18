"""Pydantic models for the Kanban board feature."""

from pydantic import BaseModel
from typing import Optional, List
from datetime import date


class ColumnCreate(BaseModel):
    title: str
    artist_id: Optional[str] = None
    color: Optional[str] = None
    position: Optional[int] = 0


class ColumnUpdate(BaseModel):
    title: Optional[str] = None
    color: Optional[str] = None
    position: Optional[int] = None


class TaskCreate(BaseModel):
    column_id: Optional[str] = None  # nullable for parent tasks
    title: str
    description: Optional[str] = None
    priority: Optional[str] = None  # low, medium, high, urgent
    start_date: Optional[date] = None
    due_date: Optional[date] = None
    color: Optional[str] = None
    parent_task_id: Optional[str] = None
    is_parent: bool = False
    artist_ids: Optional[List[str]] = []
    project_ids: Optional[List[str]] = []
    contract_ids: Optional[List[str]] = []
    assignee_name: Optional[str] = None
    labels: Optional[List[str]] = []


class TaskUpdate(BaseModel):
    column_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[str] = None
    start_date: Optional[date] = None
    due_date: Optional[date] = None
    color: Optional[str] = None
    parent_task_id: Optional[str] = None
    is_parent: Optional[bool] = None
    artist_ids: Optional[List[str]] = None
    project_ids: Optional[List[str]] = None
    contract_ids: Optional[List[str]] = None
    assignee_name: Optional[str] = None
    labels: Optional[List[str]] = None
    position: Optional[int] = None


class ParentTaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    priority: Optional[str] = None
    start_date: Optional[date] = None
    due_date: Optional[date] = None
    color: Optional[str] = None
    artist_ids: Optional[List[str]] = []
    project_ids: Optional[List[str]] = []
    labels: Optional[List[str]] = []


class TaskReorder(BaseModel):
    task_id: str
    target_column_id: str
    position: int


class BatchReorder(BaseModel):
    reorders: List[TaskReorder]


class CommentCreate(BaseModel):
    content: str
