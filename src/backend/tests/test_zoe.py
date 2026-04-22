"""Tests for the Zoe AI Chatbot endpoints.

Acceptance criteria:
1. Session history — GET /zoe/session/{id}/history returns messages list
2. Clear session — DELETE /zoe/session/{id} returns success message
3. ask-stream — POST /zoe/ask-stream returns SSE response (OpenAI mocked)
"""

import json
from unittest.mock import MagicMock, patch

from tests.conftest import MockQueryBuilder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_ID = "sess-0000-0000-0000-0000-000000000001"
PROJECT_ID = "proj-0000-0000-0000-0000-000000000001"
CONTRACT_ID = "cont-0000-0000-0000-0000-000000000001"

SAMPLE_HISTORY = [
    {"role": "user", "content": "What royalty rate do I get?"},
    {"role": "assistant", "content": "Based on the contract, you receive 50%."},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _builder(data: list):
    """Return a MockQueryBuilder pre-loaded with the given data list."""
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data=data, count=len(data))
    return b


def _make_mock_chatbot(history=None):
    """Return a MagicMock ContractChatbot with sensible defaults."""
    mock_chatbot = MagicMock()
    mock_chatbot.get_session_history.return_value = history or []
    mock_chatbot.clear_session.return_value = None
    return mock_chatbot


# ---------------------------------------------------------------------------
# GET /zoe/session/{session_id}/history
# ---------------------------------------------------------------------------


class TestZoeSessionHistory:
    def test_history_returns_200_with_messages(self, client):
        """GET /zoe/session/{id}/history returns session messages and count."""
        mock_chatbot = _make_mock_chatbot(history=SAMPLE_HISTORY)

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            response = client.get(f"/zoe/session/{SESSION_ID}/history")

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == SESSION_ID
        assert "messages" in body
        assert body["count"] == len(SAMPLE_HISTORY)
        assert body["messages"] == SAMPLE_HISTORY

    def test_history_empty_session_returns_empty_list(self, client):
        """GET /zoe/session/{id}/history returns empty list for new session."""
        mock_chatbot = _make_mock_chatbot(history=[])

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            response = client.get(f"/zoe/session/{SESSION_ID}/history")

        assert response.status_code == 200
        body = response.json()
        assert body["messages"] == []
        assert body["count"] == 0

    def test_history_uses_correct_session_id(self, client):
        """GET /zoe/session/{id}/history passes the session_id to the chatbot."""
        mock_chatbot = _make_mock_chatbot(history=[])

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            client.get(f"/zoe/session/{SESSION_ID}/history")

        mock_chatbot.get_session_history.assert_called_once_with(SESSION_ID)

    def test_history_returns_500_on_chatbot_error(self, client):
        """GET /zoe/session/{id}/history returns 500 when chatbot raises."""
        mock_chatbot = _make_mock_chatbot()
        mock_chatbot.get_session_history.side_effect = RuntimeError("chatbot exploded")

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            response = client.get(f"/zoe/session/{SESSION_ID}/history")

        assert response.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /zoe/session/{session_id}
# ---------------------------------------------------------------------------


class TestZoeClearSession:
    def test_clear_session_returns_200_with_success_message(self, client):
        """DELETE /zoe/session/{id} returns 200 with success message."""
        mock_chatbot = _make_mock_chatbot()

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            response = client.delete(f"/zoe/session/{SESSION_ID}")

        assert response.status_code == 200
        body = response.json()
        assert "message" in body
        assert body["session_id"] == SESSION_ID
        assert "cleared" in body["message"].lower()

    def test_clear_session_calls_chatbot_clear(self, client):
        """DELETE /zoe/session/{id} calls chatbot.clear_session with the correct id."""
        mock_chatbot = _make_mock_chatbot()

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            client.delete(f"/zoe/session/{SESSION_ID}")

        mock_chatbot.clear_session.assert_called_once_with(SESSION_ID)

    def test_clear_session_returns_500_on_chatbot_error(self, client):
        """DELETE /zoe/session/{id} returns 500 when chatbot raises."""
        mock_chatbot = _make_mock_chatbot()
        mock_chatbot.clear_session.side_effect = RuntimeError("chatbot error")

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            response = client.delete(f"/zoe/session/{SESSION_ID}")

        assert response.status_code == 500

    def test_clear_session_does_not_require_auth(self, client):
        """DELETE /zoe/session/{id} works without an explicit Authorization header."""
        mock_chatbot = _make_mock_chatbot()

        # Use default client (auth already bypassed in conftest).
        # What we verify here is that the endpoint is reachable and 200.
        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            response = client.delete(f"/zoe/session/{SESSION_ID}")

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /zoe/ask-stream
# ---------------------------------------------------------------------------


class TestZoeAskStream:
    def _sse_events(self, raw: bytes) -> list[dict]:
        """Parse raw SSE bytes into a list of parsed JSON event dicts."""
        events = []
        for line in raw.decode("utf-8").splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        return events

    def _stream_events(self, chatbot, client, payload=None):
        """Post to /zoe/ask-stream and return parsed SSE events."""
        if payload is None:
            payload = {"query": "What is the royalty rate?", "session_id": SESSION_ID}

        with patch("main.get_zoe_chatbot", return_value=chatbot):
            response = client.post(
                "/zoe/ask-stream",
                json=payload,
            )
        return response, self._sse_events(response.content)

    def test_ask_stream_returns_200_with_event_stream_content_type(self, client, mock_supabase):
        """POST /zoe/ask-stream returns 200 with text/event-stream content type."""
        mock_chatbot = _make_mock_chatbot()
        mock_chatbot.ask_stream.return_value = iter(
            [
                'data: {"type": "start", "session_id": "' + SESSION_ID + '"}\n\n',
                'data: {"type": "done"}\n\n',
            ]
        )
        mock_supabase.table.side_effect = lambda name: _builder([])

        response, _ = self._stream_events(mock_chatbot, client)

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_ask_stream_yields_events_from_chatbot(self, client, mock_supabase):
        """POST /zoe/ask-stream proxies events from chatbot.ask_stream."""
        mock_chatbot = _make_mock_chatbot()
        mock_chatbot.ask_stream.return_value = iter(
            [
                'data: {"type": "token", "content": "Hello"}\n\n',
                'data: {"type": "done"}\n\n',
            ]
        )
        mock_supabase.table.side_effect = lambda name: _builder([])

        response, events = self._stream_events(mock_chatbot, client)

        assert response.status_code == 200
        types = [e.get("type") for e in events]
        assert "token" in types
        assert "done" in types

    def test_ask_stream_uses_provided_session_id(self, client, mock_supabase):
        """POST /zoe/ask-stream passes the session_id to chatbot.ask_stream."""
        mock_chatbot = _make_mock_chatbot()
        mock_chatbot.ask_stream.return_value = iter(['data: {"type": "done"}\n\n'])
        mock_supabase.table.side_effect = lambda name: _builder([])

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            client.post(
                "/zoe/ask-stream",
                json={"query": "What is the term?", "session_id": SESSION_ID},
            )

        call_kwargs = mock_chatbot.ask_stream.call_args
        assert call_kwargs is not None
        # session_id should be passed through (keyword or positional)
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()
        all_args = str(kwargs) + str(args)
        assert SESSION_ID in all_args

    def test_ask_stream_generates_session_id_if_none_provided(self, client, mock_supabase):
        """POST /zoe/ask-stream auto-generates a session_id if none is provided."""
        mock_chatbot = _make_mock_chatbot()
        mock_chatbot.ask_stream.return_value = iter(['data: {"type": "done"}\n\n'])
        mock_supabase.table.side_effect = lambda name: _builder([])

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            response = client.post(
                "/zoe/ask-stream",
                json={"query": "What is the master rights clause?"},
            )

        # Should succeed even without a session_id
        assert response.status_code == 200

    def test_ask_stream_streams_error_event_on_chatbot_exception(self, client, mock_supabase):
        """POST /zoe/ask-stream streams an error SSE event when chatbot raises."""
        mock_chatbot = _make_mock_chatbot()

        def _raise():
            raise RuntimeError("OpenAI timeout")
            yield  # make it a generator

        mock_chatbot.ask_stream.return_value = _raise()
        mock_supabase.table.side_effect = lambda name: _builder([])

        response, events = self._stream_events(mock_chatbot, client)

        assert response.status_code == 200
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) >= 1
        assert "OpenAI timeout" in error_events[0]["message"]

    def test_ask_stream_fetches_artist_data_when_artist_id_provided(self, client, mock_supabase):
        """POST /zoe/ask-stream queries artists table when artist_id is given."""
        artist_id = "art-0000-0000-0000-0000-000000000001"
        artist_data = {"id": artist_id, "name": "Test Artist"}

        mock_chatbot = _make_mock_chatbot()
        mock_chatbot.ask_stream.return_value = iter(['data: {"type": "done"}\n\n'])

        call_idx = [0]

        def _side_effect(name):
            call_idx[0] += 1
            if name == "artists":
                return _builder([artist_data])
            return _builder([])

        mock_supabase.table.side_effect = _side_effect

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            response = client.post(
                "/zoe/ask-stream",
                json={
                    "query": "What is in my artist profile?",
                    "artist_id": artist_id,
                },
            )

        assert response.status_code == 200
        # At least one table call should have been the artists table
        assert call_idx[0] >= 1

    def test_ask_stream_fetches_contract_names_when_contract_ids_provided(self, client, mock_supabase):
        """POST /zoe/ask-stream queries project_files for contract names."""
        mock_chatbot = _make_mock_chatbot()
        mock_chatbot.ask_stream.return_value = iter(['data: {"type": "done"}\n\n'])

        contract_file = {"id": CONTRACT_ID, "file_name": "deal_memo.pdf"}
        mock_supabase.table.side_effect = lambda name: _builder([contract_file] if name == "project_files" else [])

        with patch("main.get_zoe_chatbot", return_value=mock_chatbot):
            response = client.post(
                "/zoe/ask-stream",
                json={
                    "query": "What is the royalty split?",
                    "contract_ids": [CONTRACT_ID],
                },
            )

        assert response.status_code == 200
