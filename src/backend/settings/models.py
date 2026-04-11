"""Pydantic models for workspace settings."""

from pydantic import BaseModel


class WorkspaceSettingsUpdate(BaseModel):
    board_period: str | None = None
    custom_period_days: int | None = None
    calendar_view: str | None = None
    timezone: str | None = None
    use_24h_time: bool | None = None
