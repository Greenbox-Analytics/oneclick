"""Admin analytics summary endpoint."""

import asyncio

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, Depends, Query

from admin.analytics_models import AnalyticsSummary, SparklinePoint, ToolRow
from admin.posthog_client import PostHogClient
from subscriptions.admin_auth import require_admin

router = APIRouter()
_cache: TTLCache = TTLCache(maxsize=8, ttl=300)

_COMPLETION_EVENTS = (
    "oneclick_calc_completed",
    "zoe_response_received",
    "splitsheet_generated",
    "registry_work_registered",
    "task_completed",
    "integration_used",
)

_COMPLETION_EVENTS_SQL = "(" + ", ".join(f"'{e}'" for e in _COMPLETION_EVENTS) + ")"
_ALL_TOOL_EVENTS_SQL = "(" + ", ".join(f"'{e}'" for e in ("tool_opened", "tool_used", *_COMPLETION_EVENTS)) + ")"

_TOOL_FUNNELS = {
    "oneclick": ("tool_opened", "oneclick_calc_completed"),
    "zoe": ("tool_opened", "zoe_response_received"),
    "splitsheet": ("tool_opened", "splitsheet_generated"),
    "registry": ("tool_opened", "registry_work_registered"),
}


def _window_to_days(window: str) -> int | None:
    if window == "7d":
        return 7
    if window == "30d":
        return 30
    return None  # "all"


def _build_summary_query(days: int | None, cohort: str) -> str:
    """Per-tool aggregation: opens, completions, last_used."""
    date_filter = ""
    if days is not None:
        date_filter = f"AND timestamp >= now() - INTERVAL {days} DAY"
    cohort_filter = ""
    if cohort == "testers":
        cohort_filter = "AND person.properties.is_tester = true"
    return f"""
    SELECT
        properties.tool AS tool,
        countIf(event = 'tool_opened') AS opens,
        countIf(event IN {_COMPLETION_EVENTS_SQL}) AS completions,
        max(timestamp) AS last_used
    FROM events
    WHERE properties.tool IS NOT NULL
        {date_filter}
        {cohort_filter}
    GROUP BY properties.tool
    ORDER BY opens DESC
    """


def _build_active_users_query(days: int | None, cohort: str) -> str:
    """Distinct users with any tool event in window."""
    date_filter = f"AND timestamp >= now() - INTERVAL {days} DAY" if days is not None else ""
    cohort_filter = "AND person.properties.is_tester = true" if cohort == "testers" else ""
    return f"""
    SELECT uniqExact(person_id) AS active_users
    FROM events
    WHERE event IN {_ALL_TOOL_EVENTS_SQL}
        {date_filter}
        {cohort_filter}
    """


def _build_total_users_query(cohort: str) -> str:
    """Total persons in cohort."""
    if cohort == "testers":
        return "SELECT count() FROM persons WHERE properties.is_tester = true"
    return "SELECT count() FROM persons"


def _build_sparkline_query(days: int | None, cohort: str) -> str:
    """Per-day distinct user count for the activity sparkline."""
    days_val = days or 30
    cohort_filter = "AND person.properties.is_tester = true" if cohort == "testers" else ""
    return f"""
    SELECT
        toDate(timestamp) AS day,
        uniqExact(person_id) AS users
    FROM events
    WHERE event = 'tool_opened'
        AND timestamp >= now() - INTERVAL {days_val} DAY
        {cohort_filter}
    GROUP BY day
    ORDER BY day
    """


def _parse_results(
    per_tool_rows: list,
    active_users: int,
    total_users: int,
    sparkline_rows: list,
    window: str,
    cohort: str,
) -> AnalyticsSummary:
    per_tool: list[ToolRow] = []
    total_opens = 0
    total_completions = 0
    for row in per_tool_rows:
        tool, opens, completions, last_used = row[0], row[1], row[2], row[3]
        rate = (completions / opens) if opens else 0.0
        per_tool.append(
            ToolRow(
                tool=tool,
                opens=opens,
                completions=completions,
                completion_rate=rate,
                last_used=last_used,
            )
        )
        total_opens += opens
        total_completions += completions

    top_tool = max(per_tool, key=lambda r: r.opens, default=None)
    top_tool_share = (top_tool.opens / total_opens) if (top_tool and total_opens) else 0.0

    named = [r for r in per_tool if r.tool in _TOOL_FUNNELS]
    rates = [r.completion_rate for r in named if r.opens > 0]
    funnel_avg = (sum(rates) / len(rates)) if rates else 0.0

    sparkline = [SparklinePoint(date=str(r[0]), value=int(r[1])) for r in sparkline_rows]

    return AnalyticsSummary(
        available=True,
        window=window,
        cohort=cohort,
        active_users=active_users,
        total_users=total_users,
        tool_actions=total_completions,
        top_tool=top_tool.tool if top_tool else None,
        top_tool_share=top_tool_share,
        funnel_completion_avg=funnel_avg,
        per_tool=per_tool,
        sparkline=sparkline,
    )


@router.get("/summary", response_model=AnalyticsSummary)
async def get_summary(
    window: str = Query("7d", pattern="^(7d|30d|all)$"),
    cohort: str = Query("testers", pattern="^(testers|all)$"),
    _admin: str = Depends(require_admin),
):
    cache_key = (window, cohort)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    client = PostHogClient()
    if not client.configured:
        return AnalyticsSummary(
            available=False,
            window=window,
            cohort=cohort,
            active_users=0,
            total_users=0,
            tool_actions=0,
            top_tool=None,
            top_tool_share=0.0,
            funnel_completion_avg=0.0,
            per_tool=[],
            sparkline=[],
            reason="POSTHOG_PERSONAL_API_KEY or POSTHOG_PROJECT_ID not configured",
        )

    days = _window_to_days(window)
    try:
        per_tool_raw, active_raw, total_raw, spark_raw = await asyncio.gather(
            client.query(_build_summary_query(days, cohort)),
            client.query(_build_active_users_query(days, cohort)),
            client.query(_build_total_users_query(cohort)),
            client.query(_build_sparkline_query(days, cohort)),
        )
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        return AnalyticsSummary(
            available=False,
            window=window,
            cohort=cohort,
            active_users=0,
            total_users=0,
            tool_actions=0,
            top_tool=None,
            top_tool_share=0.0,
            funnel_completion_avg=0.0,
            per_tool=[],
            sparkline=[],
            reason=f"posthog_query_failed: {type(e).__name__}",
        )

    active_users = int((active_raw.get("results") or [[0]])[0][0])
    total_users = int((total_raw.get("results") or [[0]])[0][0])
    result = _parse_results(
        per_tool_raw.get("results", []),
        active_users,
        total_users,
        spark_raw.get("results", []),
        window,
        cohort,
    )
    _cache[cache_key] = result
    return result
