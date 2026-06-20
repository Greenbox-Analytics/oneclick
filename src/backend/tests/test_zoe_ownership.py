from unittest.mock import MagicMock, patch

from tests.conftest import MockQueryBuilder, _default_table_side_effect


def test_ask_stream_drops_unowned_artist_context(client, mock_supabase):
    """Per-contract access model: artist_id is optional CONTEXT, not a gate.

    An unowned artist_id no longer 403s — it is dropped (never fetched/leaked) and
    the request proceeds. This replaces the old hard-403 behavior so a collaborator
    with a granted contract can still run Zoe even when an artist_id they don't own
    is passed. The security guarantee is that the unowned artist's data is NOT read.
    """

    def _router(name):
        if name == "artists":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)  # not owned
            return b
        # Subscription tables must return Pro so gated_feature passes once the
        # request proceeds past the (now-dropped) artist check.
        return _default_table_side_effect(name)

    mock_supabase.table.side_effect = _router

    fake_chatbot = MagicMock()
    fake_chatbot.ask_stream.return_value = iter([])

    with patch("main.get_zoe_chatbot", return_value=fake_chatbot):
        resp = client.post("/zoe/ask-stream", json={"query": "hi", "artist_id": "victim-artist"})

    # Not blocked, and the unowned artist's profile must NOT have been fetched.
    assert resp.status_code != 403
    # artist_data fetch uses .single(); a select on artists for the dropped id would
    # be a "*" select with .single(). We assert ask_stream got artist_data=None.
    _, kwargs = fake_chatbot.ask_stream.call_args
    assert kwargs.get("artist_data") is None


def test_ask_stream_rejects_foreign_session(client):
    import main

    main._zoe_session_owners["sess-A"] = "some-other-user"
    try:
        resp = client.post("/zoe/ask-stream", json={"query": "hi", "session_id": "sess-A"})
        assert resp.status_code == 403
    finally:
        del main._zoe_session_owners["sess-A"]


def test_session_history_requires_auth_and_ownership(client):
    resp = client.get("/zoe/session/someone-elses-session/history")
    assert resp.status_code == 404
