"""Unit tests for the analytics module."""

from unittest.mock import patch

from tests.conftest import TEST_USER_ID


class TestInitAnalytics:
    def test_initializes_when_enabled(self, monkeypatch):
        monkeypatch.setenv("POSTHOG_ENABLED", "true")
        monkeypatch.setenv("POSTHOG_PROJECT_TOKEN", "phc_test_dummy")
        monkeypatch.setenv("POSTHOG_HOST", "https://us.i.posthog.com")

        import analytics

        monkeypatch.setattr(analytics, "_initialized", False)

        with patch.object(analytics, "_set_posthog_config") as m:
            analytics.init_analytics()
            assert analytics._initialized is True
            m.assert_called_once()

    def test_no_op_when_disabled(self, monkeypatch):
        monkeypatch.setenv("POSTHOG_ENABLED", "false")
        monkeypatch.setenv("POSTHOG_PROJECT_TOKEN", "phc_test_dummy")

        import analytics

        monkeypatch.setattr(analytics, "_initialized", False)

        with patch.object(analytics, "_set_posthog_config") as m:
            analytics.init_analytics()
            assert analytics._initialized is False
            m.assert_not_called()

    def test_init_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("POSTHOG_ENABLED", "true")
        monkeypatch.setenv("POSTHOG_PROJECT_TOKEN", "phc_test_dummy")

        import analytics

        monkeypatch.setattr(analytics, "_initialized", False)

        with patch.object(analytics, "_set_posthog_config") as m:
            analytics.init_analytics()
            analytics.init_analytics()
            m.assert_called_once()  # only first call configures


class TestCapture:
    def test_calls_posthog_capture_when_enabled(self, monkeypatch):
        import analytics

        monkeypatch.setattr(analytics, "_initialized", True)

        with patch("posthog.capture") as m:
            analytics.capture(TEST_USER_ID, "tool_used", {"tool": "zoe"})
            m.assert_called_once_with(
                distinct_id=TEST_USER_ID,
                event="tool_used",
                properties={"tool": "zoe"},
            )

    def test_no_op_when_disabled(self, monkeypatch):
        import analytics

        monkeypatch.setattr(analytics, "_initialized", False)

        with patch("posthog.capture") as m:
            analytics.capture(TEST_USER_ID, "tool_used", {"tool": "zoe"})
            m.assert_not_called()


class TestIdentify:
    def test_calls_posthog_identify_when_enabled(self, monkeypatch):
        import analytics

        monkeypatch.setattr(analytics, "_initialized", True)

        with patch("posthog.identify") as m:
            analytics.identify(TEST_USER_ID, {"email": "user@example.com", "tier": "pro"})
            m.assert_called_once_with(
                distinct_id=TEST_USER_ID,
                properties={"email": "user@example.com", "tier": "pro"},
            )

    def test_no_op_when_disabled(self, monkeypatch):
        import analytics

        monkeypatch.setattr(analytics, "_initialized", False)

        with patch("posthog.identify") as m:
            analytics.identify(TEST_USER_ID, {"email": "x@y.z"})
            m.assert_not_called()
