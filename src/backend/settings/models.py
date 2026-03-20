"""Pydantic models for workspace settings."""

from pydantic import BaseModel
from typing import Optional


class WorkspaceSettingsUpdate(BaseModel):
    board_period: Optional[str] = None
    custom_period_days: Optional[int] = None
    calendar_view: Optional[str] = None
    timezone: Optional[str] = None
    use_24h_time: Optional[bool] = None
