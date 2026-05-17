"""Pydantic models for /admin/analytics endpoints."""

from pydantic import BaseModel


class ToolRow(BaseModel):
    tool: str
    opens: int
    completions: int
    completion_rate: float  # 0.0 - 1.0
    last_used: str | None  # ISO timestamp


class SparklinePoint(BaseModel):
    date: str  # YYYY-MM-DD
    value: int


class AnalyticsSummary(BaseModel):
    available: bool
    window: str
    cohort: str
    active_users: int
    total_users: int
    tool_actions: int
    top_tool: str | None
    top_tool_share: float  # 0.0 - 1.0
    funnel_completion_avg: float  # 0.0 - 1.0
    per_tool: list[ToolRow]
    sparkline: list[SparklinePoint]
    reason: str | None = None  # set when available=False
