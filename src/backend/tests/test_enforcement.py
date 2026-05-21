"""Unit tests for subscriptions.enforcement wrapper functions."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from tests.conftest import TEST_USER_ID

HOST_USER_ID = "00000000-0000-0000-0000-000000000099"


def _mock_service(can_return):
    """Mock _get_entitlements_service() to return a service whose can() returns can_return."""
    svc = MagicMock()
    svc.can.return_value = can_return
    return svc


class TestGatedCreate:
    def test_allows_under_cap(self, monkeypatch):
        from subscriptions import enforcement
        from subscriptions.models import CheckResult

        svc = _mock_service(CheckResult(allowed=True, reason=None, upgrade_required=False))
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        # Should not raise
        enforcement.gated_create(TEST_USER_ID, "artist", current_count=2)

    def test_raises_402_at_cap(self, monkeypatch):
        from subscriptions import enforcement
        from subscriptions.models import CheckResult

        svc = _mock_service(
            CheckResult(
                allowed=False,
                reason="You've reached your limit of 3 artists.",
                upgrade_required=True,
            )
        )
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        with pytest.raises(HTTPException) as exc:
            enforcement.gated_create(TEST_USER_ID, "artist", current_count=3)
        assert exc.value.status_code == 402
        assert "artists" in exc.value.detail

    def test_unknown_resource_no_op(self, monkeypatch, caplog):
        import logging

        from subscriptions import enforcement

        svc = MagicMock()
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        with caplog.at_level(logging.WARNING):
            # Should not raise; should not call can()
            enforcement.gated_create(TEST_USER_ID, "bogus", current_count=999)

        svc.can.assert_not_called()
        # Warning logged
        assert any("bogus" in record.message and "ungated" in record.message for record in caplog.records)


class TestGatedFeature:
    def test_allows_when_pro(self, monkeypatch):
        from subscriptions import enforcement
        from subscriptions.models import Action, CheckResult

        svc = _mock_service(CheckResult(allowed=True, reason=None, upgrade_required=False))
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        enforcement.gated_feature(TEST_USER_ID, Action.USE_ZOE)

    def test_raises_402_when_free(self, monkeypatch):
        from subscriptions import enforcement
        from subscriptions.models import Action, CheckResult

        svc = _mock_service(
            CheckResult(
                allowed=False,
                reason="Zoe is a Pro feature.",
                upgrade_required=True,
            )
        )
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        with pytest.raises(HTTPException) as exc:
            enforcement.gated_feature(TEST_USER_ID, Action.USE_ZOE)
        assert exc.value.status_code == 402
        assert "Zoe" in exc.value.detail

    def test_passes_host_user_id_to_service(self, monkeypatch):
        from subscriptions import enforcement
        from subscriptions.models import Action, CheckResult

        svc = _mock_service(CheckResult(allowed=True, reason=None, upgrade_required=False))
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        enforcement.gated_feature(TEST_USER_ID, Action.USE_ZOE, host_user_id=HOST_USER_ID)

        svc.can.assert_called_once_with(
            TEST_USER_ID,
            Action.USE_ZOE,
            host_user_id=HOST_USER_ID,
        )


class TestGatedUpload:
    def test_allows_under_cap(self, monkeypatch):
        from subscriptions import enforcement
        from subscriptions.models import CheckResult

        svc = _mock_service(CheckResult(allowed=True, reason=None, upgrade_required=False))
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        enforcement.gated_upload(TEST_USER_ID, size=100_000)

    def test_raises_402_at_cap(self, monkeypatch):
        from subscriptions import enforcement
        from subscriptions.models import CheckResult

        svc = _mock_service(
            CheckResult(
                allowed=False,
                reason="Upload would exceed your storage cap.",
                upgrade_required=True,
            )
        )
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        with pytest.raises(HTTPException) as exc:
            enforcement.gated_upload(TEST_USER_ID, size=100_000)
        assert exc.value.status_code == 402

    def test_passes_host_user_id_to_service(self, monkeypatch):
        from subscriptions import enforcement
        from subscriptions.models import Action, CheckResult

        svc = _mock_service(CheckResult(allowed=True, reason=None, upgrade_required=False))
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        enforcement.gated_upload(TEST_USER_ID, size=100, host_user_id=HOST_USER_ID)

        svc.can.assert_called_once_with(
            TEST_USER_ID,
            Action.UPLOAD_BYTES,
            size=100,
            host_user_id=HOST_USER_ID,
        )


class TestGatedSplitSheet:
    def test_allows_when_under_cap(self, monkeypatch):
        from subscriptions import enforcement
        from subscriptions.models import CheckResult

        svc = _mock_service(CheckResult(allowed=True, reason=None, upgrade_required=False))
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        # Should not raise
        enforcement.gated_split_sheet(TEST_USER_ID)

    def test_raises_402_at_cap(self, monkeypatch):
        from subscriptions import enforcement
        from subscriptions.models import CheckResult

        svc = _mock_service(
            CheckResult(
                allowed=False,
                reason="You've used your 5 split sheet(s) for this period.",
                upgrade_required=True,
            )
        )
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        with pytest.raises(HTTPException) as exc:
            enforcement.gated_split_sheet(TEST_USER_ID)
        assert exc.value.status_code == 402


class TestServiceErrorPropagation:
    def test_propagates_service_500(self, monkeypatch):
        """If service.can() raises a non-HTTPException, it propagates as 500."""
        from subscriptions import enforcement

        svc = MagicMock()
        svc.can.side_effect = RuntimeError("DB unreachable")
        monkeypatch.setattr(enforcement, "_service", lambda: svc)

        with pytest.raises(RuntimeError):
            enforcement.gated_create(TEST_USER_ID, "artist", current_count=0)
