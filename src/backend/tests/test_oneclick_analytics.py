"""Tests for OneClick calc event instrumentation (streaming + non-streaming).

Verifies that the OneClick calc endpoints emit the right analytics events
(`oneclick_calc_started`, `oneclick_calc_completed`, `oneclick_calc_failed`)
in the right paths (cache hit, cache miss success, exception).

Uses the existing TestClient fixture pattern (client + mock_supabase from
conftest.py) and intercepts `analytics_capture` by patching the symbol
imported into `main` at module scope.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import MockQueryBuilder, _default_table_side_effect


@pytest.fixture(autouse=True)
def _bypass_oneclick_ownership(monkeypatch):
    """Bypass the OneClick ownership guard for all tests in this module.

    These tests focus on analytics instrumentation, not access control.
    Ownership guard correctness is covered by test_oneclick_ownership.py.
    """
    monkeypatch.setattr(
        "main._assert_can_access_oneclick_inputs",
        AsyncMock(return_value=None),
    )


PROJECT_ID = "proj-0000-0000-0000-0000-000000000001"
CONTRACT_ID = "cont-0000-0000-0000-0000-000000000001"
CONTRACT_ID_2 = "cont-0000-0000-0000-0000-000000000002"
STATEMENT_FILE_ID = "file-0000-0000-0000-0000-000000000001"
CALCULATION_ID = "calc-0000-0000-0000-0000-000000000001"

SAMPLE_STATEMENT_FILE = {
    "id": STATEMENT_FILE_ID,
    "file_path": "project-files/statement.xlsx",
    "file_name": "royalty_statement.xlsx",
}

SAMPLE_PAYMENTS = [
    {
        "song_title": "Blue Sky",
        "party_name": "Jane Doe",
        "role": "Producer",
        "royalty_type": "master",
        "percentage": 50.0,
        "total_royalty": 1000.0,
        "amount_to_pay": 500.0,
        "terms": "Net 30 days",
    },
]

_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})


def _builder(data: list):
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=data, count=len(data))
    return b


def _sub_table(name: str, data_for_others: list):
    if name in _SUBSCRIPTION_TABLES:
        return _default_table_side_effect(name)
    return _builder(data_for_others)


def _sse_events(raw: bytes) -> list[dict]:
    events = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _capture_events():
    """Returns (sink_list, fake_capture) — sink collects (event, props) tuples."""
    sink: list[tuple[str, dict]] = []

    def _fake(uid, event, props=None):
        sink.append((event, dict(props or {})))

    return sink, _fake


# ---------------------------------------------------------------------------
# Streaming endpoint
# ---------------------------------------------------------------------------


class TestOneclickStreamAnalytics:
    def test_streaming_success_fires_started_and_completed(self, client, mock_supabase):
        """Cache miss + success path emits both `_started` and `_completed`."""
        mock_supabase.table.side_effect = lambda name: _sub_table(
            name, [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        sink, fake = _capture_events()
        with (
            patch("main.analytics_capture", side_effect=fake),
            patch("main.calculate_royalty_payments", return_value=(SAMPLE_PAYMENTS, None)),
        ):
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID, CONTRACT_ID_2],
                    "force_recalculate": "true",  # skip cache lookup
                },
            )

        assert response.status_code == 200

        started = [(e, p) for e, p in sink if e == "oneclick_calc_started"]
        completed = [(e, p) for e, p in sink if e == "oneclick_calc_completed"]

        assert len(started) == 1, f"expected one _started, got events={sink}"
        assert started[0][1]["tool"] == "oneclick"
        assert started[0][1]["contract_count"] == 2
        assert started[0][1].get("cached") is False

        assert len(completed) == 1, f"expected one _completed, got events={sink}"
        assert completed[0][1]["tool"] == "oneclick"
        assert completed[0][1]["contract_count"] == 2
        assert completed[0][1].get("cached") is False
        assert "duration_ms" in completed[0][1]

        # Existing `tool_used` capture must still fire for backward compat.
        tool_used = [e for e, _ in sink if e == "tool_used"]
        assert len(tool_used) == 1

    def test_streaming_review_fires_split_verification_event(self, client, mock_supabase):
        """A review from the calc pipeline emits `oneclick_split_verification_run` with its counts."""
        mock_supabase.table.side_effect = lambda name: _sub_table(
            name, [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        review = {"overall": "needs_review", "checked": 2, "flagged": 1, "findings": []}
        sink, fake = _capture_events()
        with (
            patch("main.analytics_capture", side_effect=fake),
            patch("main.calculate_royalty_payments", return_value=(SAMPLE_PAYMENTS, review)),
        ):
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID],
                    "force_recalculate": "true",
                },
            )

        assert response.status_code == 200

        fired = [(e, p) for e, p in sink if e == "oneclick_split_verification_run"]
        assert len(fired) == 1, f"expected one split-verification event, got events={sink}"
        props = fired[0][1]
        assert props["tool"] == "oneclick"
        assert props["checked"] == 2
        assert props["flagged"] == 1
        assert props["overall"] == "needs_review"

    def test_streaming_no_review_skips_split_verification_event(self, client, mock_supabase):
        """When the verification pass didn't run (review=None), no event is emitted."""
        mock_supabase.table.side_effect = lambda name: _sub_table(
            name, [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        sink, fake = _capture_events()
        with (
            patch("main.analytics_capture", side_effect=fake),
            patch("main.calculate_royalty_payments", return_value=(SAMPLE_PAYMENTS, None)),
        ):
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID],
                    "force_recalculate": "true",
                },
            )

        assert response.status_code == 200
        assert not any(e == "oneclick_split_verification_run" for e, _ in sink), f"unexpected event in {sink}"

    def test_streaming_cache_hit_fires_started_and_completed_paired(self, client, mock_supabase):
        """Cache hit fires BOTH `_started` and `_completed` (paired) with cached=True."""
        cached_result = {
            "status": "success",
            "total_payments": 1,
            "payments": [SAMPLE_PAYMENTS[0]],
            "message": "1 payment",
        }
        cached_calc = {"id": CALCULATION_ID, "results": cached_result}
        cached_contract_link = {"calculation_id": CALCULATION_ID, "contract_id": CONTRACT_ID}

        call_idx = [0]
        sequences = [
            [cached_calc],
            [cached_contract_link],
        ]

        def _side_effect(name):
            if name in _SUBSCRIPTION_TABLES:
                return _default_table_side_effect(name)
            data = sequences[call_idx[0]] if call_idx[0] < len(sequences) else []
            call_idx[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        sink, fake = _capture_events()
        with (
            patch("main.analytics_capture", side_effect=fake),
            patch("main.calculate_royalty_payments") as mock_calc,
        ):
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID],
                },
            )
            mock_calc.assert_not_called()

        assert response.status_code == 200

        started = [(e, p) for e, p in sink if e == "oneclick_calc_started"]
        completed = [(e, p) for e, p in sink if e == "oneclick_calc_completed"]

        assert len(started) == 1, f"expected one _started on cache hit, got events={sink}"
        assert started[0][1].get("cached") is True
        assert len(completed) == 1, f"expected one _completed on cache hit, got events={sink}"
        assert completed[0][1].get("cached") is True
        assert completed[0][1].get("duration_ms") == 0

    def test_streaming_exception_fires_failed(self, client, mock_supabase):
        """Exception inside generator triggers `_failed` with error_code+stage."""
        # Have project_files lookup succeed (statement file found), but make
        # `calculate_royalty_payments` raise — that exception bubbles into the
        # outer try/except in `generate_progress`, which fires `_failed`.
        mock_supabase.table.side_effect = lambda name: _sub_table(
            name, [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        sink, fake = _capture_events()
        with (
            patch("main.analytics_capture", side_effect=fake),
            patch(
                "main.calculate_royalty_payments",
                side_effect=RuntimeError("calc blew up"),
            ),
        ):
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID],
                    "force_recalculate": "true",
                },
            )

        assert response.status_code == 200

        failed = [(e, p) for e, p in sink if e == "oneclick_calc_failed"]
        assert len(failed) >= 1, f"expected _failed, got events={sink}"
        assert failed[0][1]["tool"] == "oneclick"
        assert failed[0][1]["error_code"] == "RuntimeError"
        assert failed[0][1]["stage"] == "calc"

    def test_streaming_fires_failed_on_validation_error(self, client, mock_supabase):
        """Empty contract_ids list should fire _failed with stage=validation."""
        # Mock supabase but don't configure any tables — we'll never get there
        # because the validation check fires first
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])

        sink, fake = _capture_events()
        with patch("main.analytics_capture", side_effect=fake):
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    # omit contract_id and contract_ids entirely to trigger validation
                    "force_recalculate": "false",
                },
            )

        assert response.status_code == 200

        failed = [(e, p) for e, p in sink if e == "oneclick_calc_failed"]
        assert len(failed) == 1, f"expected validation _failed, got {sink}"
        assert failed[0][1]["stage"] == "validation"
        assert failed[0][1]["error_code"] == "ValidationError"


# ---------------------------------------------------------------------------
# Non-streaming endpoint
# ---------------------------------------------------------------------------


class TestOneclickPostAnalytics:
    def test_post_success_fires_started_and_completed(self, client, mock_supabase):
        """POST success path emits `_started` + `_completed`."""
        mock_supabase.table.side_effect = lambda name: _sub_table(
            name, [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        sink, fake = _capture_events()
        with (
            patch("main.analytics_capture", side_effect=fake),
            patch("main.calculate_royalty_payments", return_value=(SAMPLE_PAYMENTS, None)),
        ):
            response = client.post(
                "/oneclick/calculate-royalties",
                json={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID],
                },
            )

        assert response.status_code == 200

        started = [(e, p) for e, p in sink if e == "oneclick_calc_started"]
        completed = [(e, p) for e, p in sink if e == "oneclick_calc_completed"]

        assert len(started) == 1
        assert started[0][1]["tool"] == "oneclick"
        assert started[0][1]["contract_count"] == 1
        assert started[0][1].get("cached") is False
        assert len(completed) == 1
        assert completed[0][1]["tool"] == "oneclick"
        assert "duration_ms" in completed[0][1]

    def test_post_exception_fires_failed(self, client, mock_supabase):
        """POST exception path emits `_failed`."""

        # Make the statement file lookup raise.
        def _side_effect(name):
            if name in _SUBSCRIPTION_TABLES:
                return _default_table_side_effect(name)
            raise RuntimeError("db kaboom")

        mock_supabase.table.side_effect = _side_effect

        sink, fake = _capture_events()
        with patch("main.analytics_capture", side_effect=fake):
            response = client.post(
                "/oneclick/calculate-royalties",
                json={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID],
                },
            )

        assert response.status_code == 500

        failed = [(e, p) for e, p in sink if e == "oneclick_calc_failed"]
        assert len(failed) >= 1, f"expected _failed, got events={sink}"
        assert failed[0][1]["tool"] == "oneclick"
        assert failed[0][1]["error_code"] == "RuntimeError"
        assert failed[0][1]["stage"] == "calc"
