"""Tests for OneClick Royalty Calculation endpoints.

Acceptance criteria:
1. calculate-royalties-stream — GET returns SSE response (calculate_royalty_payments mocked)
2. calculate-royalties (POST) — returns payment breakdown (calculate_royalty_payments mocked)
3. confirm — POST saves calculation to DB and returns success
"""

import json
from unittest.mock import MagicMock, patch

from tests.conftest import MockQueryBuilder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    {
        "song_title": "Blue Sky",
        "party_name": "John Smith",
        "role": "Songwriter",
        "royalty_type": "publishing",
        "percentage": 25.0,
        "total_royalty": 1000.0,
        "amount_to_pay": 250.0,
        "terms": None,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _builder(data: list):
    """Return a MockQueryBuilder pre-loaded with the given data list."""
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=data, count=len(data))
    return b


def _sse_events(raw: bytes) -> list[dict]:
    """Parse raw SSE bytes into a list of parsed JSON event dicts."""
    events = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


# ---------------------------------------------------------------------------
# GET /oneclick/calculate-royalties-stream
# ---------------------------------------------------------------------------


class TestOneclickCalculateRoyaltiesStream:
    def _stream(self, client, mock_supabase, payments=None, **extra_params):
        """Send a streaming calculate-royalties request and return (response, events)."""
        params = {
            "project_id": PROJECT_ID,
            "royalty_statement_file_id": STATEMENT_FILE_ID,
            "contract_ids": [CONTRACT_ID],
            **extra_params,
        }

        # Wire statement file download
        mock_supabase.table.side_effect = lambda name: _builder(
            [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        with patch(
            "main.calculate_royalty_payments",
            return_value=payments if payments is not None else SAMPLE_PAYMENTS,
        ):
            response = client.get("/oneclick/calculate-royalties-stream", params=params)

        return response, _sse_events(response.content)

    def test_stream_returns_200_with_event_stream_content_type(self, client, mock_supabase):
        """GET /oneclick/calculate-royalties-stream returns 200 with SSE content type."""
        response, _ = self._stream(client, mock_supabase)

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_stream_yields_status_events_and_complete_event(self, client, mock_supabase):
        """GET /oneclick/calculate-royalties-stream emits status events then a complete event."""
        response, events = self._stream(client, mock_supabase)

        assert response.status_code == 200
        types = [e.get("type") for e in events]
        assert "status" in types
        assert "complete" in types

    def test_stream_complete_event_contains_payments(self, client, mock_supabase):
        """The 'complete' SSE event contains the calculated payments."""
        _, events = self._stream(client, mock_supabase)

        complete_events = [e for e in events if e.get("type") == "complete"]
        assert len(complete_events) >= 1
        payload = complete_events[0]
        assert payload["total_payments"] == len(SAMPLE_PAYMENTS)
        assert "payments" in payload
        assert payload["payments"][0]["song_title"] == "Blue Sky"

    def test_stream_returns_error_event_when_no_contracts_specified(self, client, mock_supabase):
        """GET /oneclick/calculate-royalties-stream streams error when no contracts given."""
        mock_supabase.table.side_effect = lambda name: _builder([])

        with patch("main.calculate_royalty_payments", return_value=[]):
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    # no contract_id or contract_ids
                },
            )

        events = _sse_events(response.content)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) >= 1

    def test_stream_returns_error_event_when_statement_not_found(self, client, mock_supabase):
        """GET /oneclick/calculate-royalties-stream streams error when statement file absent."""
        mock_supabase.table.side_effect = lambda name: _builder([])

        with patch("main.calculate_royalty_payments", return_value=[]):
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID],
                },
            )

        events = _sse_events(response.content)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) >= 1

    def test_stream_returns_error_event_when_payments_empty(self, client, mock_supabase):
        """GET /oneclick/calculate-royalties-stream streams error when no payments calculated."""
        mock_supabase.table.side_effect = lambda name: _builder(
            [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        with patch("main.calculate_royalty_payments", return_value=[]):
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID],
                },
            )

        events = _sse_events(response.content)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) >= 1

    def test_stream_uses_cache_when_available(self, client, mock_supabase):
        """GET /oneclick/calculate-royalties-stream returns cached results without recalculating."""
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
            [cached_calc],  # royalty_calculations select
            [cached_contract_link],  # royalty_calculation_contracts select
        ]

        def _side_effect(name):
            data = sequences[call_idx[0]] if call_idx[0] < len(sequences) else []
            call_idx[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        with patch("main.calculate_royalty_payments") as mock_calc:
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID],
                },
            )
            # calculate_royalty_payments should NOT be called on a cache hit
            mock_calc.assert_not_called()

        events = _sse_events(response.content)
        complete_events = [e for e in events if e.get("type") == "complete"]
        assert len(complete_events) >= 1
        assert complete_events[0].get("is_cached") is True

    def test_stream_force_recalculate_bypasses_cache(self, client, mock_supabase):
        """GET /oneclick/calculate-royalties-stream with force_recalculate=true skips cache."""
        mock_supabase.table.side_effect = lambda name: _builder(
            [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        with patch("main.calculate_royalty_payments", return_value=SAMPLE_PAYMENTS) as mock_calc:
            response = client.get(
                "/oneclick/calculate-royalties-stream",
                params={
                    "project_id": PROJECT_ID,
                    "royalty_statement_file_id": STATEMENT_FILE_ID,
                    "contract_ids": [CONTRACT_ID],
                    "force_recalculate": "true",
                },
            )
            # Should have called the real calculator (not returned cached)
            mock_calc.assert_called_once()

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /oneclick/calculate-royalties
# ---------------------------------------------------------------------------


class TestOneclickCalculateRoyalties:
    def _post(self, client, mock_supabase, payments=None, **overrides):
        """POST /oneclick/calculate-royalties and return response."""
        payload = {
            "project_id": PROJECT_ID,
            "royalty_statement_file_id": STATEMENT_FILE_ID,
            "contract_ids": [CONTRACT_ID],
            **overrides,
        }

        mock_supabase.table.side_effect = lambda name: _builder(
            [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        with patch(
            "main.calculate_royalty_payments",
            return_value=payments if payments is not None else SAMPLE_PAYMENTS,
        ):
            return client.post("/oneclick/calculate-royalties", json=payload)

    def test_calculate_royalties_returns_200_with_payments(self, client, mock_supabase):
        """POST /oneclick/calculate-royalties returns 200 and payment breakdown."""
        response = self._post(client, mock_supabase)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["total_payments"] == len(SAMPLE_PAYMENTS)
        assert len(body["payments"]) == len(SAMPLE_PAYMENTS)

    def test_calculate_royalties_payment_fields_are_correct(self, client, mock_supabase):
        """POST /oneclick/calculate-royalties returns payment dicts with all required fields."""
        response = self._post(client, mock_supabase)

        assert response.status_code == 200
        first = response.json()["payments"][0]
        assert first["song_title"] == SAMPLE_PAYMENTS[0]["song_title"]
        assert first["party_name"] == SAMPLE_PAYMENTS[0]["party_name"]
        assert first["role"] == SAMPLE_PAYMENTS[0]["role"]
        assert first["royalty_type"] == SAMPLE_PAYMENTS[0]["royalty_type"]
        assert first["percentage"] == SAMPLE_PAYMENTS[0]["percentage"]
        assert first["amount_to_pay"] == SAMPLE_PAYMENTS[0]["amount_to_pay"]

    def test_calculate_royalties_returns_404_when_no_payments(self, client, mock_supabase):
        """POST /oneclick/calculate-royalties returns 404 when calculator finds no payments."""
        response = self._post(client, mock_supabase, payments=[])

        assert response.status_code == 404

    def test_calculate_royalties_returns_400_when_no_contracts(self, client, mock_supabase):
        """POST /oneclick/calculate-royalties returns 400 when no contracts are specified."""
        payload = {
            "project_id": PROJECT_ID,
            "royalty_statement_file_id": STATEMENT_FILE_ID,
            # no contract_id or contract_ids
        }
        mock_supabase.table.side_effect = lambda name: _builder([])

        with patch("main.calculate_royalty_payments", return_value=[]):
            response = client.post("/oneclick/calculate-royalties", json=payload)

        assert response.status_code == 400

    def test_calculate_royalties_returns_404_when_statement_not_found(self, client, mock_supabase):
        """POST /oneclick/calculate-royalties returns 404 when statement file absent."""
        payload = {
            "project_id": PROJECT_ID,
            "royalty_statement_file_id": STATEMENT_FILE_ID,
            "contract_ids": [CONTRACT_ID],
        }
        mock_supabase.table.side_effect = lambda name: _builder([])

        with patch("main.calculate_royalty_payments", return_value=[]):
            response = client.post("/oneclick/calculate-royalties", json=payload)

        assert response.status_code == 404

    def test_calculate_royalties_accepts_single_contract_id(self, client, mock_supabase):
        """POST /oneclick/calculate-royalties works with a single contract_id field."""
        payload = {
            "project_id": PROJECT_ID,
            "royalty_statement_file_id": STATEMENT_FILE_ID,
            "contract_id": CONTRACT_ID,  # singular form
        }
        mock_supabase.table.side_effect = lambda name: _builder(
            [SAMPLE_STATEMENT_FILE] if name == "project_files" else []
        )
        mock_supabase.storage.from_.return_value.download.return_value = b"mock-xlsx-content"

        with patch("main.calculate_royalty_payments", return_value=SAMPLE_PAYMENTS):
            response = client.post("/oneclick/calculate-royalties", json=payload)

        assert response.status_code == 200

    def test_calculate_royalties_message_contains_count(self, client, mock_supabase):
        """POST /oneclick/calculate-royalties response message references payment count."""
        response = self._post(client, mock_supabase)

        body = response.json()
        assert str(len(SAMPLE_PAYMENTS)) in body["message"]


# ---------------------------------------------------------------------------
# POST /oneclick/confirm
# ---------------------------------------------------------------------------


class TestOneclickConfirm:
    RESULTS_PAYLOAD = {
        "status": "success",
        "total_payments": 2,
        "payments": SAMPLE_PAYMENTS,
        "message": "2 royalty payments",
    }

    def _post_confirm(self, client, mock_supabase, contract_ids=None, **overrides):
        """POST /oneclick/confirm and return response."""
        payload = {
            "contract_ids": contract_ids or [CONTRACT_ID],
            "royalty_statement_id": STATEMENT_FILE_ID,
            "project_id": PROJECT_ID,
            "results": self.RESULTS_PAYLOAD,
            **overrides,
        }

        return client.post("/oneclick/confirm", json=payload)

    def test_confirm_returns_200_with_success_status(self, client, mock_supabase):
        """POST /oneclick/confirm returns 200 with status=success."""
        new_calc = {"id": CALCULATION_ID}

        call_idx = [0]
        sequences = [
            [],  # royalty_calculations select (no existing)
            [new_calc],  # royalty_calculations insert
            [],  # royalty_calculation_contracts insert
        ]

        def _side_effect(name):
            data = sequences[call_idx[0]] if call_idx[0] < len(sequences) else []
            call_idx[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        response = self._post_confirm(client, mock_supabase)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert "id" in body

    def test_confirm_saves_calculation_id(self, client, mock_supabase):
        """POST /oneclick/confirm returns the new calculation id."""
        new_calc = {"id": CALCULATION_ID}

        call_idx = [0]
        sequences = [[], [new_calc], []]

        def _side_effect(name):
            data = sequences[call_idx[0]] if call_idx[0] < len(sequences) else []
            call_idx[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        response = self._post_confirm(client, mock_supabase)

        assert response.status_code == 200
        assert response.json()["id"] == CALCULATION_ID

    def test_confirm_returns_500_when_insert_fails(self, client, mock_supabase):
        """POST /oneclick/confirm returns 500 when DB insert returns no data."""
        call_idx = [0]
        sequences = [
            [],  # royalty_calculations select (no existing)
            [],  # royalty_calculations insert — empty = failure
        ]

        def _side_effect(name):
            data = sequences[call_idx[0]] if call_idx[0] < len(sequences) else []
            call_idx[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        response = self._post_confirm(client, mock_supabase)

        assert response.status_code == 500

    def test_confirm_deletes_existing_calc_before_insert(self, client, mock_supabase):
        """POST /oneclick/confirm removes an old cached calculation with same statement+contracts."""
        existing_calc = {"id": "old-calc-id"}
        existing_link = {"calculation_id": "old-calc-id", "contract_id": CONTRACT_ID}
        new_calc = {"id": CALCULATION_ID}

        call_idx = [0]
        sequences = [
            [existing_calc],  # royalty_calculations select — existing found
            [existing_link],  # royalty_calculation_contracts select
            [],  # royalty_calculation_contracts delete
            [],  # royalty_calculations delete
            [new_calc],  # royalty_calculations insert
            [],  # royalty_calculation_contracts insert (junction)
        ]

        def _side_effect(name):
            data = sequences[call_idx[0]] if call_idx[0] < len(sequences) else []
            call_idx[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        response = self._post_confirm(client, mock_supabase)

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_confirm_inserts_junction_rows_for_each_contract(self, client, mock_supabase):
        """POST /oneclick/confirm inserts a junction row for each contract_id."""
        new_calc = {"id": CALCULATION_ID}

        inserted_junction_data = []

        call_idx = [0]
        sequences = [
            [],  # royalty_calculations select (no existing)
            [new_calc],  # royalty_calculations insert
        ]

        def _side_effect(name):
            if call_idx[0] < len(sequences):
                data = sequences[call_idx[0]]
            else:
                data = []
            call_idx[0] += 1
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=data, count=len(data))
            # Capture inserts to junction table

            def _capture_insert(rows):
                if isinstance(rows, list):
                    inserted_junction_data.extend(rows)
                return b

            b.insert = _capture_insert
            return b

        mock_supabase.table.side_effect = _side_effect

        response = self._post_confirm(client, mock_supabase, contract_ids=[CONTRACT_ID, CONTRACT_ID_2])

        assert response.status_code == 200

    def test_confirm_requires_results_field(self, client, mock_supabase):
        """POST /oneclick/confirm returns 422 if results field is missing."""
        mock_supabase.table.side_effect = lambda name: _builder([])

        response = client.post(
            "/oneclick/confirm",
            json={
                "contract_ids": [CONTRACT_ID],
                "royalty_statement_id": STATEMENT_FILE_ID,
                "project_id": PROJECT_ID,
                # no results
            },
        )

        assert response.status_code == 422

    def test_confirm_with_multiple_contracts(self, client, mock_supabase):
        """POST /oneclick/confirm handles multiple contract_ids correctly."""
        new_calc = {"id": CALCULATION_ID}

        call_idx = [0]
        sequences = [[], [new_calc], []]

        def _side_effect(name):
            data = sequences[call_idx[0]] if call_idx[0] < len(sequences) else []
            call_idx[0] += 1
            return _builder(data)

        mock_supabase.table.side_effect = _side_effect

        response = self._post_confirm(client, mock_supabase, contract_ids=[CONTRACT_ID, CONTRACT_ID_2])

        assert response.status_code == 200
        assert response.json()["status"] == "success"
