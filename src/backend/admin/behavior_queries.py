"""HogQL query builders for behavioral analytics (pages, dwell time, flow).

Inputs come from two event streams:
- `$pageview` (autocaptured by posthog-js) — what page, who, when
- `page_time_spent` (custom; see src/hooks/usePageTimer.ts) — dwell duration

Both events use `properties.$pathname` so joins are direct equality.
Pre-normalization events emitted property `path` and are orphaned by the
join — expected.

The bounce-ish rate is computed as: fraction of page_time_spent events where
duration_ms < 10s, grouped by pathname. It's an *approximation* — `usePageTimer`
fires on route change + page hide, so a user who lingers 9s and navigates is
a "bounce" here but probably wasn't conceptually. Document this in the UI.

Note on HogQL: `person.properties.*` is accessed directly on the event row
WITHOUT a table alias — `t.person.properties.is_tester` is invalid HogQL
even when the events table is aliased as `t`.
"""


def _date_filter(days: int | None) -> str:
    if days is None:
        return ""
    return f"AND timestamp >= now() - INTERVAL {days} DAY"


def _cohort_filter(cohort: str) -> str:
    return "AND person.properties.is_tester = true" if cohort == "testers" else ""


def build_top_pages_query(days: int | None, cohort: str) -> str:
    """Top pages by views, with unique visitors, avg dwell, and bounce rate.

    LEFT JOIN page_time_spent CTE so pages with no dwell events still appear
    with avg_dwell_ms=0.
    """
    date_f = _date_filter(days)
    cohort_f = _cohort_filter(cohort)

    return f"""
    WITH dwell AS (
        SELECT
            properties.$pathname AS path,
            avg(toInt(properties.duration_ms)) AS avg_ms,
            countIf(toInt(properties.duration_ms) < 10000) / count() AS bounce_rate
        FROM events
        WHERE event = 'page_time_spent'
            {date_f}
            {cohort_f}
        GROUP BY properties.$pathname
    )
    SELECT
        properties.$pathname AS path,
        count() AS views,
        uniqExact(person_id) AS unique_visitors,
        coalesce(dwell.avg_ms, 0) AS avg_dwell_ms,
        coalesce(dwell.bounce_rate, 0.0) AS bounce_rate
    FROM events
    LEFT JOIN dwell ON dwell.path = properties.$pathname
    WHERE event = '$pageview'
        {date_f}
        {cohort_f}
    GROUP BY properties.$pathname, dwell.avg_ms, dwell.bounce_rate
    ORDER BY views DESC
    LIMIT 25
    """


def build_daily_visitors_query(days: int | None, cohort: str) -> str:
    """Per-day pageviews + unique visitors for the sparkline."""
    date_f = _date_filter(days)
    cohort_f = _cohort_filter(cohort)
    return f"""
    SELECT
        toDate(timestamp) AS day,
        count() AS views,
        uniqExact(person_id) AS unique_visitors
    FROM events
    WHERE event = '$pageview'
        {date_f}
        {cohort_f}
    GROUP BY day
    ORDER BY day
    """


def build_top_flows_query(days: int | None, cohort: str) -> str:
    """Top page-to-page transitions in the window.

    Uses `lagInFrame` over per-person ordered pageviews to derive (prev, curr).
    Counts each ordered pair (from, to) over the window.
    """
    date_f = _date_filter(days)
    cohort_f = _cohort_filter(cohort)
    return f"""
    SELECT
        from_path,
        to_path,
        count() AS count
    FROM (
        SELECT
            person_id,
            properties.$pathname AS to_path,
            lagInFrame(properties.$pathname) OVER (
                PARTITION BY person_id ORDER BY timestamp
            ) AS from_path
        FROM events
        WHERE event = '$pageview'
            {date_f}
            {cohort_f}
    )
    WHERE from_path IS NOT NULL AND from_path != to_path
    GROUP BY from_path, to_path
    ORDER BY count DESC
    LIMIT 20
    """


def build_pageview_totals_query(days: int | None, cohort: str) -> str:
    """Top-line totals: total pageviews + unique visitors + avg pageviews per visitor."""
    date_f = _date_filter(days)
    cohort_f = _cohort_filter(cohort)
    return f"""
    SELECT
        count() AS total_pageviews,
        uniqExact(person_id) AS unique_visitors,
        (count() / nullIf(uniqExact(person_id), 0)) AS avg_pv_per_visitor
    FROM events
    WHERE event = '$pageview'
        {date_f}
        {cohort_f}
    """
