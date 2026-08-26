"""Tests for Zoe step event instrumentation on the streaming endpoint.

Verifies that POST /zoe/ask-stream emits:
  - zoe_query_submitted at handler entry
  - zoe_response_received after a successful stream
  - zoe_query_failed when the chatbot raises mid-stream

Mirrors the patching pattern from test_oneclick_analytics.py: uses the shared
`client` + `mock_supabase` fixtures and patches `main.analytics_capture` plus
`main.gated_credits` (to bypass paywall) and `main.get_zoe_chatbot` (to return
a fake whose `ask_stream` is a controllable generator).
"""

import json
from unittest.mock import MagicMock, patch


def _sse_events(raw: bytes) -> list[dict]:
    """Parse SSE wire bytes into a list of decoded event dicts."""
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


def _sse(event_type: str, data: dict) -> str:
    """Format an SSE wire line the way the real chatbot._sse_event does."""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


class TestZoeStreamAnalytics:
    def test_zoe_fires_query_submitted_and_response_received(self, client):
        """Happy path: query_submitted at entry + response_received after success."""

        fake_bot = MagicMock()

        def fake_stream(**_kwargs):
            yield _sse("start", {"session_id": "sess-1"})
            yield _sse("sources", {"sources": [{"id": "s1"}, {"id": "s2"}]})
            yield _sse("token", {"content": "hello"})
            yield _sse("done", {"answered_from": "document"})

        fake_bot.ask_stream = fake_stream

        sink, fake = _capture_events()
        with (
            patch("main.analytics_capture", side_effect=fake),
            patch("main.gated_credits", return_value=None),
            patch("main.get_zoe_chatbot", return_value=fake_bot),
        ):
            response = client.post(
                "/zoe/ask-stream",
                json={"query": "What does my contract say about royalties?"},
            )

        assert response.status_code == 200
        # Drain the SSE body to ensure generate() runs to completion.
        _sse_events(response.content)

        submitted = [(e, p) for e, p in sink if e == "zoe_query_submitted"]
        received = [(e, p) for e, p in sink if e == "zoe_response_received"]
        tool_used = [(e, p) for e, p in sink if e == "tool_used"]

        assert len(submitted) == 1, f"expected one zoe_query_submitted, got {sink}"
        assert submitted[0][1]["tool"] == "zoe"
        assert submitted[0][1]["query_length"] == len("What does my contract say about royalties?")
        assert submitted[0][1]["has_attachment"] is False

        assert len(received) == 1, f"expected one zoe_response_received, got {sink}"
        assert received[0][1]["tool"] == "zoe"
        assert received[0][1]["source_count"] == 2
        assert "duration_ms" in received[0][1]
        assert received[0][1]["duration_ms"] >= 0

        # Existing `tool_used` capture must still fire (backward compat).
        assert len(tool_used) == 1, f"expected one tool_used, got {sink}"
        assert tool_used[0][1]["tool"] == "zoe"

    def test_zoe_fires_failed_on_chatbot_exception(self, client):
        """Stream-time exception inside the generator emits zoe_query_failed."""

        fake_bot = MagicMock()

        def fake_stream(**_kwargs):
            yield _sse("start", {"session_id": "sess-1"})
            raise RuntimeError("model down")

        fake_bot.ask_stream = fake_stream

        sink, fake = _capture_events()
        with (
            patch("main.analytics_capture", side_effect=fake),
            patch("main.gated_credits", return_value=None),
            patch("main.get_zoe_chatbot", return_value=fake_bot),
        ):
            response = client.post(
                "/zoe/ask-stream",
                json={"query": "x"},
            )

        assert response.status_code == 200
        # Drain so the inner generator actually runs and hits the exception path.
        _sse_events(response.content)

        failed = [(e, p) for e, p in sink if e == "zoe_query_failed"]
        submitted = [(e, p) for e, p in sink if e == "zoe_query_submitted"]

        # Submitted should fire regardless (it's before any work).
        assert len(submitted) == 1, f"expected zoe_query_submitted, got {sink}"
        assert len(failed) == 1, f"expected one zoe_query_failed, got {sink}"
        assert failed[0][1]["tool"] == "zoe"
        assert failed[0][1]["error_code"] == "RuntimeError"

    def test_zoe_fires_failed_on_setup_exception(self, client):
        """Setup-phase failure (e.g. chatbot init raises) fires zoe_query_failed
        from the outer except, BEFORE the stream begins."""

        sink, fake = _capture_events()

        def explode():
            raise RuntimeError("chatbot init failed")

        with (
            patch("main.analytics_capture", side_effect=fake),
            patch("main.gated_credits", return_value=None),
            patch("main.get_zoe_chatbot", side_effect=explode),
        ):
            response = client.post(
                "/zoe/ask-stream",
                json={"query": "x"},
            )

        # Outer except should raise HTTPException -> 500.
        assert response.status_code == 500

        failed = [(e, p) for e, p in sink if e == "zoe_query_failed"]
        submitted = [(e, p) for e, p in sink if e == "zoe_query_submitted"]
        received = [(e, p) for e, p in sink if e == "zoe_response_received"]

        # Submitted should fire (it's before the try block).
        assert len(submitted) == 1, f"expected zoe_query_submitted, got {sink}"
        # Outer except should fire zoe_query_failed once.
        assert len(failed) == 1, f"expected one zoe_query_failed from outer except, got {sink}"
        assert failed[0][1]["tool"] == "zoe"
        assert failed[0][1]["error_code"] == "RuntimeError"
        # Response_received should NOT fire on setup failure.
        assert len(received) == 0, f"zoe_response_received should not fire on setup failure, got {sink}"
