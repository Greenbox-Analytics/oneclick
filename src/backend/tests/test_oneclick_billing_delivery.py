"""No delivered output, no charge (owner decision 2026-09-04).

The OneClick stream's whole deliverable rides in ONE `complete` frame, so a
debit taken before that frame lands bills a user for an answer they never got.
Both branches (fresh calc and cache hit) must charge only after the yield.

Simulating a dropped client: TestClient drains the response, so instead the
StreamingResponse is wrapped in a generator that closes the inner one right
after the `complete` frame — which is exactly what Starlette does on disconnect.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import main
from tests.test_oneclick_analytics import (
    CONTRACT_ID,
    CONTRACT_ID_2,
    PROJECT_ID,
    SAMPLE_PAYMENTS,
    SAMPLE_STATEMENT_FILE,
    STATEMENT_FILE_ID,
    _sub_table,
)


@pytest.fixture(autouse=True)
def _bypass_ownership(monkeypatch):
    monkeypatch.setattr("main._assert_can_access_oneclick_inputs", AsyncMock(return_value=None))


@pytest.fixture
def debit_spy(monkeypatch):
    ent = MagicMock()
    monkeypatch.setattr("main._get_entitlements_service", lambda: ent)
    return ent


def _truncate_at_complete(monkeypatch):
    """Patch main.StreamingResponse so the stream is cut at the `complete`
    frame, as a vanished client would cut it."""
    real = main.StreamingResponse

    def _wrap(content, **kw):
        async def cut():
            async for chunk in content:
                yield chunk
                if '"type": "complete"' in chunk:
                    await content.aclose()
                    return

        return real(cut(), **kw)

    monkeypatch.setattr("main.StreamingResponse", _wrap)


def _run(client, force_recalculate: str):
    return client.get(
        "/oneclick/calculate-royalties-stream",
        params={
            "project_id": PROJECT_ID,
            "royalty_statement_file_id": STATEMENT_FILE_ID,
            "contract_ids": [CONTRACT_ID, CONTRACT_ID_2],
            "force_recalculate": force_recalculate,
        },
    )


@pytest.fixture
def calc_ok(mock_supabase):
    mock_supabase.table.side_effect = lambda name: _sub_table(
        name, [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
    )
    mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"


def test_delivered_result_is_charged(client, calc_ok, debit_spy):
    """Positive control — without this the disconnect test proves nothing."""
    with patch("main.calculate_royalty_payments", return_value=(SAMPLE_PAYMENTS, None)):
        assert _run(client, "true").status_code == 200
    debit_spy.debit_for_action.assert_called_once()


def test_client_gone_at_the_result_frame_is_not_charged(client, calc_ok, debit_spy, monkeypatch):
    _truncate_at_complete(monkeypatch)
    with patch("main.calculate_royalty_payments", return_value=(SAMPLE_PAYMENTS, None)):
        assert _run(client, "true").status_code == 200
    debit_spy.debit_for_action.assert_not_called()
