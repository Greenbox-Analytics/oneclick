"""Unit tests for AnalyticsMiddleware."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app_with_middleware():
    from middleware import analytics_middleware
    from middleware.analytics_middleware import AnalyticsMiddleware

    app = FastAPI()
    app.add_middleware(AnalyticsMiddleware)

    @app.get("/echo")
    def echo():
        return {"ok": True}

    @app.get("/typeerror")
    def typeerror():
        raise TypeError("nope")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app, analytics_middleware


class TestAnalyticsMiddleware:
    def test_captures_request_completed_on_success(self, monkeypatch):
        import analytics

        monkeypatch.setattr(analytics, "_initialized", True)

        app, analytics_middleware = _app_with_middleware()
        client = TestClient(app)

        with patch.object(analytics_middleware, "analytics_capture") as m:
            r = client.get("/echo")
            assert r.status_code == 200

        event_args = [(call.args[1] if len(call.args) >= 2 else call.kwargs.get("event")) for call in m.call_args_list]
        assert "request_completed" in event_args

    def test_captures_request_failed_on_handler_exception(self, monkeypatch):
        import analytics

        monkeypatch.setattr(analytics, "_initialized", True)

        app, analytics_middleware = _app_with_middleware()
        client = TestClient(app, raise_server_exceptions=False)

        with patch.object(analytics_middleware, "analytics_capture") as m:
            r = client.get("/typeerror")
            assert r.status_code == 500

        event_args = [(call.args[1] if len(call.args) >= 2 else call.kwargs.get("event")) for call in m.call_args_list]
        assert "request_failed" in event_args

    def test_skips_excluded_paths(self, monkeypatch):
        import analytics

        monkeypatch.setattr(analytics, "_initialized", True)

        app, analytics_middleware = _app_with_middleware()
        client = TestClient(app)

        with patch.object(analytics_middleware, "analytics_capture") as m:
            r = client.get("/health")
            assert r.status_code == 200
            # No analytics captures for /health
            m.assert_not_called()
