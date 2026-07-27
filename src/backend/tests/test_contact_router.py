"""Endpoint tests for POST /contact-submissions."""

from unittest.mock import MagicMock, patch

from tests.conftest import TEST_USER_ID, MockQueryBuilder


def _table_factory(captured: dict, recent_count: int = 0):
    """Mock `sb.table(...)` capturing the inserted contact_submissions payload.

    `recent_count` drives the rate-limit lookup, which reads `res.count` from
    the same builder the insert goes through.
    """

    def _table(name):
        b = MockQueryBuilder()
        if name == "contact_submissions":
            original = b.insert

            def _capture(payload, *a, **kw):
                captured["payload"] = payload
                return original(payload, *a, **kw)

            b.insert = _capture
            b.execute.return_value = MagicMock(data=[{"id": "c1"}], count=recent_count)
        return b

    return _table


def _ticket_form(**overrides) -> dict:
    form = {
        "mode": "ticket",
        "name": "Jane Doe",
        "email": "jane@studio.com",
        "subject": "Royalty split looks wrong",
        "message": "The master split totals 90% but should be 100%.",
        "product": "OneClick",
        "account_email": "jane@workspace.com",
    }
    form.update(overrides)
    return form


def _message_form(**overrides) -> dict:
    form = {
        "mode": "message",
        "name": "Sam Ray",
        "email": "sam@label.com",
        "subject": "Partnership",
        "message": "We would like to discuss a partnership.",
        "company": "Indie Label",
        "topic": "Partnership",
    }
    form.update(overrides)
    return form


class TestSubmitContact:
    def test_ticket_records_row_and_returns_reference(self, client, mock_supabase):
        import main
        from auth import get_optional_user_id

        captured = {}
        mock_supabase.table.side_effect = _table_factory(captured)

        # The client fixture overrides get_current_user_id, not the optional
        # dependency this endpoint uses — override it to get an attributed submit.
        async def _override_optional():
            return TEST_USER_ID

        main.app.dependency_overrides[get_optional_user_id] = _override_optional

        try:
            with patch("contact.router._send_ops_notification") as mock_send:
                resp = client.post("/contact-submissions", data=_ticket_form())
        finally:
            main.app.dependency_overrides.pop(get_optional_user_id, None)

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        # MSN-XXXX-NNNN, as shown on the design's success screen
        assert body["reference_id"].startswith("MSN-")
        assert len(body["reference_id"]) == len("MSN-ABCD-1234")

        payload = captured["payload"]
        assert payload["mode"] == "ticket"
        assert payload["email"] == "jane@studio.com"
        assert payload["product"] == "OneClick"
        assert payload["account_email"] == "jane@workspace.com"
        # message-mode fields stay NULL on a ticket
        assert payload["company"] is None
        assert payload["topic"] is None
        assert payload["status"] == "new"
        assert payload["attachment_count"] == 0
        assert payload["user_id"] == TEST_USER_ID
        assert payload["reference_id"] == body["reference_id"]
        mock_send.assert_called_once()

    def test_logged_out_records_null_user_id(self, client, mock_supabase):
        """Anonymous submits are allowed and recorded with no user attribution."""
        captured = {}
        mock_supabase.table.side_effect = _table_factory(captured)

        with patch("contact.router._send_ops_notification"):
            resp = client.post("/contact-submissions", data=_ticket_form())

        assert resp.status_code == 200
        assert captured["payload"]["user_id"] is None

    def test_message_mode_stores_only_its_own_fields(self, client, mock_supabase):
        captured = {}
        mock_supabase.table.side_effect = _table_factory(captured)

        with patch("contact.router._send_ops_notification"):
            resp = client.post("/contact-submissions", data=_message_form())

        assert resp.status_code == 200
        payload = captured["payload"]
        assert payload["mode"] == "message"
        assert payload["company"] == "Indie Label"
        assert payload["topic"] == "Partnership"
        assert payload["product"] is None
        assert payload["account_email"] is None

    def test_ticket_ignores_message_mode_fields(self, client, mock_supabase):
        """A ticket submitting company/topic must not have them persisted."""
        captured = {}
        mock_supabase.table.side_effect = _table_factory(captured)

        with patch("contact.router._send_ops_notification"):
            resp = client.post(
                "/contact-submissions",
                data=_ticket_form(company="Sneaky Co", topic="Partnership"),
            )

        assert resp.status_code == 200
        assert captured["payload"]["company"] is None
        assert captured["payload"]["topic"] is None

    def test_invalid_mode_returns_400(self, client, mock_supabase):
        mock_supabase.table.side_effect = _table_factory({})
        resp = client.post("/contact-submissions", data=_ticket_form(mode="bogus"))
        assert resp.status_code == 400

    def test_invalid_email_returns_422(self, client, mock_supabase):
        mock_supabase.table.side_effect = _table_factory({})
        resp = client.post("/contact-submissions", data=_ticket_form(email="not-an-email"))
        assert resp.status_code == 422

    def test_missing_field_returns_422(self, client, mock_supabase):
        mock_supabase.table.side_effect = _table_factory({})
        form = _ticket_form()
        del form["subject"]
        resp = client.post("/contact-submissions", data=form)
        assert resp.status_code == 422

    def test_whitespace_only_message_returns_400(self, client, mock_supabase):
        """Present but blank must be rejected — Form(...) alone allows empty strings."""
        mock_supabase.table.side_effect = _table_factory({})
        resp = client.post("/contact-submissions", data=_ticket_form(message="   "))
        assert resp.status_code == 400

    def test_long_message_is_truncated(self, client, mock_supabase):
        captured = {}
        mock_supabase.table.side_effect = _table_factory(captured)

        with patch("contact.router._send_ops_notification"):
            resp = client.post("/contact-submissions", data=_ticket_form(message="x" * 9000))

        assert resp.status_code == 200
        assert len(captured["payload"]["message"]) == 5000

    def test_honeypot_returns_success_without_recording(self, client, mock_supabase):
        """Bots must not be able to tell they were filtered."""
        captured = {}
        mock_supabase.table.side_effect = _table_factory(captured)

        with patch("contact.router._send_ops_notification") as mock_send:
            resp = client.post("/contact-submissions", data=_ticket_form(website="http://spam.example"))

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert "payload" not in captured  # nothing written
        mock_send.assert_not_called()

    def test_rate_limit_returns_429(self, client, mock_supabase):
        captured = {}
        mock_supabase.table.side_effect = _table_factory(captured, recent_count=5)

        with patch("contact.router._send_ops_notification") as mock_send:
            resp = client.post("/contact-submissions", data=_ticket_form())

        assert resp.status_code == 429
        assert "payload" not in captured
        mock_send.assert_not_called()

    def test_under_rate_limit_is_allowed(self, client, mock_supabase):
        mock_supabase.table.side_effect = _table_factory({}, recent_count=4)

        with patch("contact.router._send_ops_notification"):
            resp = client.post("/contact-submissions", data=_ticket_form())

        assert resp.status_code == 200

    def test_rate_limit_check_failure_allows_request(self, client, mock_supabase):
        """A broken counter must not take the contact form down."""
        mock_supabase.table.side_effect = _table_factory({})

        with (
            patch("contact.router._count_recent", side_effect=RuntimeError("count failed")),
            patch("contact.router._send_ops_notification"),
        ):
            resp = client.post("/contact-submissions", data=_ticket_form())

        assert resp.status_code == 200

    def test_resend_failure_returns_200(self, client, mock_supabase):
        mock_supabase.table.side_effect = _table_factory({})

        with patch("contact.router._send_ops_notification", side_effect=RuntimeError("Resend down")):
            resp = client.post("/contact-submissions", data=_ticket_form())

        assert resp.status_code == 200  # DB insert succeeded; Resend failure swallowed

    def test_db_failure_returns_500(self, client, mock_supabase):
        def _table(name):
            b = MockQueryBuilder()
            if name == "contact_submissions":
                b.execute.side_effect = RuntimeError("DB down")
            return b

        mock_supabase.table.side_effect = _table

        with patch("contact.router._send_ops_notification"):
            resp = client.post("/contact-submissions", data=_ticket_form())

        assert resp.status_code == 500


class TestAttachments:
    def test_accepts_allowed_file(self, client, mock_supabase):
        captured = {}
        mock_supabase.table.side_effect = _table_factory(captured)

        with patch("contact.router._send_ops_notification") as mock_send:
            resp = client.post(
                "/contact-submissions",
                data=_ticket_form(),
                files=[("attachments", ("screenshot.png", b"fake-image-bytes", "image/png"))],
            )

        assert resp.status_code == 200
        assert captured["payload"]["attachment_count"] == 1
        # the file itself is forwarded, not stored
        forwarded_files = mock_send.call_args[0][2]
        assert forwarded_files == [("screenshot.png", b"fake-image-bytes")]

    def test_rejects_disallowed_extension(self, client, mock_supabase):
        mock_supabase.table.side_effect = _table_factory({})

        resp = client.post(
            "/contact-submissions",
            data=_ticket_form(),
            files=[("attachments", ("payload.exe", b"MZ", "application/octet-stream"))],
        )

        assert resp.status_code == 400

    def test_rejects_oversized_file(self, client, mock_supabase):
        mock_supabase.table.side_effect = _table_factory({})

        oversized = b"x" * (int(2.5 * 1024 * 1024) + 1)
        resp = client.post(
            "/contact-submissions",
            data=_ticket_form(),
            files=[("attachments", ("big.pdf", oversized, "application/pdf"))],
        )

        assert resp.status_code == 400
        assert "2.5 MB" in resp.json()["detail"]

    def test_rejects_too_many_files(self, client, mock_supabase):
        mock_supabase.table.side_effect = _table_factory({})

        resp = client.post(
            "/contact-submissions",
            data=_ticket_form(),
            files=[("attachments", (f"shot{i}.png", b"bytes", "image/png")) for i in range(4)],
        )

        assert resp.status_code == 400

    def test_strips_directory_from_filename(self, client, mock_supabase):
        """A path-y filename must not survive into the attachment name."""
        mock_supabase.table.side_effect = _table_factory({})

        with patch("contact.router._send_ops_notification") as mock_send:
            resp = client.post(
                "/contact-submissions",
                data=_ticket_form(),
                files=[("attachments", ("../../etc/passwd.png", b"bytes", "image/png"))],
            )

        assert resp.status_code == 200
        assert mock_send.call_args[0][2][0][0] == "passwd.png"
