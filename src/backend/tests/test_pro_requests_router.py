"""Endpoint tests for POST /pro-requests."""

from unittest.mock import MagicMock, patch

from tests.conftest import TEST_USER_ID, MockQueryBuilder


class TestSubmitProRequest:
    def test_logged_in_records_user_id(self, client, mock_supabase):
        import main
        from auth import get_optional_user_id

        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "pro_requests":
                original = b.insert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original(payload, *a, **kw)

                b.insert = _capture
                b.execute.return_value = MagicMock(data=[{"id": "p1"}], count=1)
            return b

        mock_supabase.table.side_effect = _table

        # client fixture overrides get_current_user_id; also override get_optional_user_id
        async def _override_optional():
            return TEST_USER_ID

        main.app.dependency_overrides[get_optional_user_id] = _override_optional

        try:
            with patch("subscriptions.pro_requests_router._send_ops_notification") as mock_send:
                resp = client.post(
                    "/pro-requests",
                    json={"email": "alice@example.com", "message": "interested"},
                )
        finally:
            main.app.dependency_overrides.pop(get_optional_user_id, None)

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
        assert captured["payload"]["email"] == "alice@example.com"
        assert captured["payload"]["message"] == "interested"
        assert captured["payload"]["user_id"] == TEST_USER_ID  # client fixture sets this
        assert captured["payload"]["status"] == "new"
        mock_send.assert_called_once()

    def test_logged_out_records_null_user_id(self, mock_supabase):
        """Without the auth override fixture, get_optional_user_id returns None."""
        from fastapi.testclient import TestClient

        import main

        main.get_supabase_client = lambda: mock_supabase
        main.supabase = mock_supabase

        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "pro_requests":
                original = b.insert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original(payload, *a, **kw)

                b.insert = _capture
                b.execute.return_value = MagicMock(data=[{"id": "p1"}], count=1)
            return b

        mock_supabase.table.side_effect = _table
        original_overrides = dict(main.app.dependency_overrides)
        main.app.dependency_overrides.clear()  # No auth override → unauth

        try:
            with (
                patch("subscriptions.pro_requests_router._send_ops_notification"),
                TestClient(main.app) as tc,
            ):
                resp = tc.post(
                    "/pro-requests",
                    json={"email": "bob@example.com"},
                )
            assert resp.status_code == 200
            assert captured["payload"]["user_id"] is None
        finally:
            main.app.dependency_overrides.update(original_overrides)

    def test_invalid_email_returns_422(self, client, mock_supabase):
        resp = client.post("/pro-requests", json={"email": "not-an-email"})
        assert resp.status_code == 422

    def test_message_is_optional(self, client, mock_supabase):
        captured = {}

        def _table(name):
            b = MockQueryBuilder()
            if name == "pro_requests":
                original = b.insert

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original(payload, *a, **kw)

                b.insert = _capture
                b.execute.return_value = MagicMock(data=[{"id": "p1"}], count=1)
            return b

        mock_supabase.table.side_effect = _table

        with patch("subscriptions.pro_requests_router._send_ops_notification"):
            resp = client.post("/pro-requests", json={"email": "a@b.com"})

        assert resp.status_code == 200
        assert captured["payload"]["message"] is None

    def test_resend_failure_returns_200(self, client, mock_supabase):
        def _table(name):
            b = MockQueryBuilder()
            if name == "pro_requests":
                b.execute.return_value = MagicMock(data=[{"id": "p1"}], count=1)
            return b

        mock_supabase.table.side_effect = _table

        with patch("subscriptions.pro_requests_router._send_ops_notification", side_effect=RuntimeError("Resend down")):
            resp = client.post("/pro-requests", json={"email": "a@b.com"})

        assert resp.status_code == 200  # DB insert succeeded; Resend failure swallowed

    def test_db_failure_returns_500(self, client, mock_supabase):
        def _table(name):
            b = MockQueryBuilder()
            if name == "pro_requests":
                b.execute.side_effect = RuntimeError("DB down")
            return b

        mock_supabase.table.side_effect = _table

        with patch("subscriptions.pro_requests_router._send_ops_notification"):
            resp = client.post("/pro-requests", json={"email": "a@b.com"})

        assert resp.status_code == 500
