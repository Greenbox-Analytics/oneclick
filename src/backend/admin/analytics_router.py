"""Admin analytics summary endpoint."""

import asyncio

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, Depends, Query

from admin.analytics_models import (
    AnalyticsSummary,
    BehaviorSummary,
    DailyVisitorPoint,
    FunnelStep,
    PageFlowEdge,
    PageRow,
    RegistryLifecycle,
    SparklinePoint,
    ToolFunnel,
    ToolRow,
)
from admin.behavior_queries import (
    build_daily_visitors_query,
    build_pageview_totals_query,
    build_top_flows_query,
    build_top_pages_query,
)
from admin.posthog_client import PostHogClient
from analytics import ENV_FILTER_VALUES, POSTHOG_DATA_CUTOFF
from subscriptions.admin_auth import require_admin

router = APIRouter()
_cache: TTLCache = TTLCache(maxsize=8, ttl=300)
_behavior_cache: TTLCache = TTLCache(maxsize=8, ttl=300)

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

# Session-funnel event sets. tool_opened is the shared open step (keyed by properties.tool);
# all others are tool-specific by event name. splitsheet has no *_started event.
_FUNNEL_EVENTS = (
    "tool_opened",
    "oneclick_calc_started",
    "oneclick_calc_completed",
    "oneclick_calc_failed",
    "zoe_query_submitted",
    "zoe_response_received",
    "zoe_query_failed",
    "splitsheet_generated",
    "splitsheet_generation_failed",
)
_FUNNEL_EVENTS_SQL = "(" + ", ".join(f"'{e}'" for e in _FUNNEL_EVENTS) + ")"
_FUNNEL_START_EVENTS_SQL = "('oneclick_calc_started', 'zoe_query_submitted')"
_FUNNEL_DONE_EVENTS_SQL = "('oneclick_calc_completed', 'zoe_response_received', 'splitsheet_generated')"
_FUNNEL_FAIL_EVENTS_SQL = "('oneclick_calc_failed', 'zoe_query_failed', 'splitsheet_generation_failed')"
_REGISTRY_LIFECYCLE_EVENTS_SQL = "('work_created', 'work_submitted_for_registration', 'registry_work_registered')"

_ENV_VALUES_SQL = "(" + ", ".join(f"'{e}'" for e in ENV_FILTER_VALUES) + ")"
_ENV_DATE_FILTER = f"AND properties.environment IN {_ENV_VALUES_SQL} AND timestamp >= '{POSTHOG_DATA_CUTOFF}'"

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
    """Per-tool aggregation: raw volume (opens, completions) plus a per-user
    funnel (openers, converters) so completion_rate is bounded 0-100%.

    Inner query collapses to one row per (tool, person): did they open, did they
    complete, and how many raw events each fired. Outer query then sums the raw
    volume and counts distinct openers vs. distinct converters (openers who also
    completed >= 1 action). converters is a subset of openers by construction, so
    converters / openers can never exceed 100% — unlike raw completions / opens,
    which inflates whenever a user completes multiple actions per open.
    """
    date_filter = ""
    if days is not None:
        date_filter = f"AND timestamp >= now() - INTERVAL {days} DAY"
    cohort_filter = ""
    if cohort == "testers":
        cohort_filter = "AND person.properties.is_tester = true"
    return f"""
    SELECT
        tool,
        sum(opens) AS opens,
        sum(completions) AS completions,
        countIf(opened) AS openers,
        countIf(opened AND completed) AS converters,
        max(last_ts) AS last_used
    FROM (
        SELECT
            properties.tool AS tool,
            person_id,
            countIf(event = 'tool_opened') AS opens,
            countIf(event IN {_COMPLETION_EVENTS_SQL}) AS completions,
            countIf(event = 'tool_opened') > 0 AS opened,
            countIf(event IN {_COMPLETION_EVENTS_SQL}) > 0 AS completed,
            max(timestamp) AS last_ts
        FROM events
        WHERE properties.tool IS NOT NULL
            {date_filter}
            {cohort_filter}
            {_ENV_DATE_FILTER}
        GROUP BY properties.tool, person_id
    )
    GROUP BY tool
    ORDER BY opens DESC
    """


def _build_session_funnel_query(days: int | None, cohort: str) -> str:
    """Per-tool session funnel: nested distinct-user steps + raw completed/failed counts.

    Inner query collapses to one row per (tool, person) carrying open/start/complete flags
    and raw completion/failure counts, over an explicit 9-event allowlist. `started_raw` is
    always false for splitsheet (no start event); the outer query forces the started gate
    open for splitsheet so `opened AND started AND completed` collapses to a correct 2-step
    funnel, while oneclick/zoe keep the real 3-step gate.
    """
    date_filter = f"AND timestamp >= now() - INTERVAL {days} DAY" if days is not None else ""
    cohort_filter = "AND person.properties.is_tester = true" if cohort == "testers" else ""
    # Group the inner query by the multiIf EXPRESSION (not the alias) — this mirrors the
    # proven `_build_summary_query`, which groups by `properties.tool` directly. We define
    # the expression once and reuse it in SELECT and GROUP BY so the two never drift.
    tool_expr = """multiIf(
                event = 'tool_opened', properties.tool,
                event IN ('oneclick_calc_started', 'oneclick_calc_completed', 'oneclick_calc_failed'), 'oneclick',
                event IN ('zoe_query_submitted', 'zoe_response_received', 'zoe_query_failed'), 'zoe',
                event IN ('splitsheet_generated', 'splitsheet_generation_failed'), 'splitsheet',
                NULL
            )"""
    return f"""
    SELECT
        tool,
        countIf(opened) AS opened,
        countIf(opened AND (started_raw OR tool = 'splitsheet')) AS started,
        countIf(opened AND (started_raw OR tool = 'splitsheet') AND completed) AS completed,
        sum(completed_ct) AS completed_events,
        sum(failed_ct) AS failed_events
    FROM (
        SELECT
            {tool_expr} AS tool,
            person_id,
            countIf(event = 'tool_opened') > 0 AS opened,
            countIf(event IN {_FUNNEL_START_EVENTS_SQL}) > 0 AS started_raw,
            countIf(event IN {_FUNNEL_DONE_EVENTS_SQL}) > 0 AS completed,
            countIf(event IN {_FUNNEL_DONE_EVENTS_SQL}) AS completed_ct,
            countIf(event IN {_FUNNEL_FAIL_EVENTS_SQL}) AS failed_ct
        FROM events
        WHERE event IN {_FUNNEL_EVENTS_SQL}
            {date_filter}
            {cohort_filter}
            {_ENV_DATE_FILTER}
        GROUP BY {tool_expr}, person_id
    )
    WHERE tool IN ('oneclick', 'zoe', 'splitsheet')
    GROUP BY tool
    ORDER BY opened DESC
    """


def _build_registry_lifecycle_query(days: int | None, cohort: str) -> str:
    """Registry stage volumes in-window (works): created -> submitted -> registered.

    NOT a nested cohort funnel — works move through stages over days, so a nested % would
    read a misleading ~0% for short windows. Stages are also attributed to different persons
    (created/submitted = acting user, registered = work owner), so per-person nesting would be
    wrong. We report independent stage volumes only.
    """
    date_filter = f"AND timestamp >= now() - INTERVAL {days} DAY" if days is not None else ""
    cohort_filter = "AND person.properties.is_tester = true" if cohort == "testers" else ""
    return f"""
    SELECT
        countIf(event = 'work_created') AS created,
        countIf(event = 'work_submitted_for_registration') AS submitted,
        countIf(event = 'registry_work_registered') AS registered
    FROM events
    WHERE event IN {_REGISTRY_LIFECYCLE_EVENTS_SQL}
        {date_filter}
        {cohort_filter}
        {_ENV_DATE_FILTER}
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
        {_ENV_DATE_FILTER}
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
    # Two timestamp lower bounds are AND-ed here: the sparkline window (now - N days)
    # and the env-rollout cutoff in _ENV_DATE_FILTER. Both are intentional; the tighter
    # one wins at runtime. As POSTHOG_DATA_CUTOFF ages, the INTERVAL becomes binding again.
    return f"""
    SELECT
        toDate(timestamp) AS day,
        uniqExact(person_id) AS users
    FROM events
    WHERE event = 'tool_opened'
        AND timestamp >= now() - INTERVAL {days_val} DAY
        {cohort_filter}
        {_ENV_DATE_FILTER}
    GROUP BY day
    ORDER BY day
    """


_FUNNEL_STEP_LABELS = {
    "oneclick": ("opened", "started", "completed"),
    "zoe": ("opened", "submitted", "answered"),
    "splitsheet": ("opened", "generated"),
}


def _parse_funnels(rows: list) -> list[ToolFunnel]:
    """Rows: [tool, opened, started, completed, completed_events, failed_events]."""
    funnels: list[ToolFunnel] = []
    for row in rows:
        tool, opened, started, completed, completed_events, failed_events = (
            row[0],
            int(row[1]),
            int(row[2]),
            int(row[3]),
            int(row[4]),
            int(row[5]),
        )
        labels = _FUNNEL_STEP_LABELS.get(tool)
        if labels is None or opened == 0:
            continue  # unknown tool, or zero opens -> no empty strip
        # splitsheet is a 2-step funnel (opened -> generated); the SQL forced its
        # `started` gate open, so `completed` already equals opened&completed.
        step_users = [opened, completed] if tool == "splitsheet" else [opened, started, completed]
        steps = [FunnelStep(label=lbl, users=u) for lbl, u in zip(labels, step_users, strict=True)]
        denom = completed_events + failed_events
        error_rate = (failed_events / denom) if denom else 0.0
        funnels.append(
            ToolFunnel(
                tool=tool,
                steps=steps,
                error_rate=error_rate,
                completed_events=completed_events,
                failed_events=failed_events,
            )
        )
    return funnels


def _parse_registry_lifecycle(rows: list) -> RegistryLifecycle | None:
    # A countIf(...) query ALWAYS returns exactly one row — [[0, 0, 0]] when there is no
    # registry activity. Treat the all-zero case as "no data" so the UI hides the strip
    # instead of rendering "created 0 -> submitted 0 -> registered 0".
    if not rows:
        return None
    r = rows[0]
    created, submitted, registered = int(r[0]), int(r[1]), int(r[2])
    if created + submitted + registered == 0:
        return None
    return RegistryLifecycle(created=created, submitted=submitted, registered=registered)


def _parse_results(
    per_tool_rows: list,
    active_users: int,
    total_users: int,
    sparkline_rows: list,
    window: str,
    cohort: str,
    funnel_rows: list,
    lifecycle_rows: list,
) -> AnalyticsSummary:
    per_tool: list[ToolRow] = []
    total_opens = 0
    total_completions = 0
    for row in per_tool_rows:
        tool, opens, completions, openers, converters, last_used = (
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
        )
        # Per-user funnel: of the testers who opened, how many completed >= 1 action.
        # converters is a subset of openers, so this is bounded 0-100%.
        rate = (converters / openers) if openers else 0.0
        per_tool.append(
            ToolRow(
                tool=tool,
                opens=opens,
                completions=completions,
                openers=openers,
                converters=converters,
                completion_rate=rate,
                last_used=last_used,
            )
        )
        total_opens += opens
        total_completions += completions

    top_tool = max(per_tool, key=lambda r: r.opens, default=None)
    top_tool_share = (top_tool.opens / total_opens) if (top_tool and total_opens) else 0.0

    named = [r for r in per_tool if r.tool in _TOOL_FUNNELS]
    rates = [r.completion_rate for r in named if r.openers > 0]
    funnel_avg = (sum(rates) / len(rates)) if rates else 0.0

    sparkline = [SparklinePoint(date=str(r[0]), value=int(r[1])) for r in sparkline_rows]
    funnels = _parse_funnels(funnel_rows)
    registry_lifecycle = _parse_registry_lifecycle(lifecycle_rows)

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
        funnels=funnels,
        registry_lifecycle=registry_lifecycle,
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

    # The two new queries are the most complex and least proven (the multiIf-grouped funnel
    # has no in-repo precedent). Run them in an ISOLATED gather with return_exceptions=True so
    # a malformed-query / PostHog-4xx failure degrades to an empty funnel section instead of
    # blanking the whole card (which the core try/except above would otherwise do).
    funnel_raw, lifecycle_raw = await asyncio.gather(
        client.query(_build_session_funnel_query(days, cohort)),
        client.query(_build_registry_lifecycle_query(days, cohort)),
        return_exceptions=True,
    )

    def _section_rows(res, label: str) -> list:
        if isinstance(res, dict):
            return res.get("results", [])
        print(f"[admin-analytics] {label} query failed; omitting funnel section: {type(res).__name__}: {res}")
        return []

    funnel_rows = _section_rows(funnel_raw, "session-funnel")
    lifecycle_rows = _section_rows(lifecycle_raw, "registry-lifecycle")

    active_users = int((active_raw.get("results") or [[0]])[0][0])
    total_users = int((total_raw.get("results") or [[0]])[0][0])
    result = _parse_results(
        per_tool_raw.get("results", []),
        active_users,
        total_users,
        spark_raw.get("results", []),
        window,
        cohort,
        funnel_rows,
        lifecycle_rows,
    )
    _cache[cache_key] = result
    return result


def _parse_behavior(
    totals_rows: list,
    pages_rows: list,
    daily_rows: list,
    flows_rows: list,
    window: str,
    cohort: str,
) -> BehaviorSummary:
    total = int((totals_rows or [[0, 0, 0.0]])[0][0]) if totals_rows else 0
    unique = int((totals_rows or [[0, 0, 0.0]])[0][1]) if totals_rows else 0
    avg_pv = float((totals_rows or [[0, 0, 0.0]])[0][2] or 0.0) if totals_rows else 0.0

    top_pages = [
        PageRow(
            path=r[0],
            views=int(r[1]),
            unique_visitors=int(r[2]),
            avg_dwell_ms=int(r[3] or 0),
            bounce_rate=float(r[4] or 0.0),
        )
        for r in pages_rows
        if r and r[0] is not None
    ]
    daily = [DailyVisitorPoint(date=str(r[0]), views=int(r[1]), unique_visitors=int(r[2])) for r in daily_rows]
    flows = [
        PageFlowEdge(from_path=r[0], to_path=r[1], count=int(r[2]))
        for r in flows_rows
        if r and r[0] is not None and r[1] is not None
    ]

    return BehaviorSummary(
        available=True,
        window=window,
        cohort=cohort,
        total_pageviews=total,
        unique_visitors=unique,
        pageviews_per_visitor=round(avg_pv, 2),
        top_pages=top_pages,
        daily_visitors=daily,
        top_flows=flows,
    )


@router.get("/behavior", response_model=BehaviorSummary)
async def get_behavior(
    window: str = Query("7d", pattern="^(7d|30d|all)$"),
    cohort: str = Query("testers", pattern="^(testers|all)$"),
    _admin: str = Depends(require_admin),
):
    cache_key = (window, cohort)
    cached = _behavior_cache.get(cache_key)
    if cached is not None:
        return cached

    client = PostHogClient()
    if not client.configured:
        return BehaviorSummary(
            available=False,
            window=window,
            cohort=cohort,
            reason="POSTHOG_PERSONAL_API_KEY or POSTHOG_PROJECT_ID not configured",
        )

    days = _window_to_days(window)
    try:
        totals_raw, pages_raw, daily_raw, flows_raw = await asyncio.gather(
            client.query(build_pageview_totals_query(days, cohort)),
            client.query(build_top_pages_query(days, cohort)),
            client.query(build_daily_visitors_query(days, cohort)),
            client.query(build_top_flows_query(days, cohort)),
        )
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        return BehaviorSummary(
            available=False,
            window=window,
            cohort=cohort,
            reason=f"posthog_query_failed: {type(e).__name__}",
        )

    result = _parse_behavior(
        totals_raw.get("results", []),
        pages_raw.get("results", []),
        daily_raw.get("results", []),
        flows_raw.get("results", []),
        window,
        cohort,
    )
    _behavior_cache[cache_key] = result
    return result
