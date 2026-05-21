"""Tests for admin/behavior_queries.py — HogQL builders for behavior analytics."""


class TestTopPagesQuery:
    def test_includes_tester_cohort_filter(self):
        from admin.behavior_queries import build_top_pages_query

        q = build_top_pages_query(days=7, cohort="testers")
        assert "person.properties.is_tester = true" in q
        assert "$pageview" in q
        assert "page_time_spent" in q
        assert "INTERVAL 7 DAY" in q

    def test_omits_cohort_filter_for_all(self):
        from admin.behavior_queries import build_top_pages_query

        q = build_top_pages_query(days=30, cohort="all")
        assert "person.properties.is_tester" not in q

    def test_omits_date_filter_for_all_time(self):
        from admin.behavior_queries import build_top_pages_query

        q = build_top_pages_query(days=None, cohort="all")
        assert "INTERVAL" not in q


class TestDailyVisitorsQuery:
    def test_groups_by_day(self):
        from admin.behavior_queries import build_daily_visitors_query

        q = build_daily_visitors_query(days=30, cohort="all")
        assert "toDate(timestamp)" in q
        assert "GROUP BY" in q
        assert "ORDER BY" in q


class TestTopFlowsQuery:
    def test_self_joins_for_sequential_pageviews(self):
        from admin.behavior_queries import build_top_flows_query

        q = build_top_flows_query(days=7, cohort="testers")
        assert "$pageview" in q
        assert q.lower().count("pathname") >= 1 or "$current_url" in q


class TestPageviewTotalsQuery:
    def test_returns_totals_query(self):
        from admin.behavior_queries import build_pageview_totals_query

        q = build_pageview_totals_query(days=7, cohort="all")
        assert "uniqExact" in q
        assert "count" in q.lower()
