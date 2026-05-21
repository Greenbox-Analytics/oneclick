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


class PageRow(BaseModel):
    path: str
    views: int
    unique_visitors: int
    avg_dwell_ms: int
    bounce_rate: float  # fraction of views with dwell < 10s


class PageFlowEdge(BaseModel):
    from_path: str
    to_path: str
    count: int


class DailyVisitorPoint(BaseModel):
    date: str
    views: int
    unique_visitors: int


class BehaviorSummary(BaseModel):
    available: bool
    window: str
    cohort: str
    total_pageviews: int = 0
    unique_visitors: int = 0
    pageviews_per_visitor: float = 0.0  # window-level avg, NOT per-session (we don't track sessions)
    top_pages: list[PageRow] = []
    daily_visitors: list[DailyVisitorPoint] = []
    top_flows: list[PageFlowEdge] = []
    reason: str | None = None
