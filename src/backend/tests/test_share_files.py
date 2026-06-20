from unittest.mock import MagicMock

from tests.conftest import MockQueryBuilder

PAYLOAD = {
    "recipient_email": "friend@example.com",
    "files": [
        {
            "file_name": "deal.pdf",
            "file_path": "victim-artist/victim-proj/contract/deal.pdf",
            "file_source": "project_file",
            "file_id": "file-victim",
        }
    ],
}


def test_share_rejects_unowned_project_file(client, mock_supabase, monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "test-key")

    def _router(name):
        b = MockQueryBuilder()
        if name == "project_files":
            b.execute.return_value = MagicMock(
                data=[{"id": "file-victim", "project_id": "victim-proj", "file_path": "victim/path.pdf"}], count=1
            )
        elif name == "project_members":
            b.execute.return_value = MagicMock(data=None)
        elif name == "artists":
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    mock_supabase.table.side_effect = _router
    resp = client.post("/share/files", json=PAYLOAD)
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Access denied"


def test_share_uses_db_path_not_client_path(client, mock_supabase, monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "test-key")
    monkeypatch.setattr("main.resend.Emails.send", lambda *a, **k: {"id": "email-1"}, raising=False)

    def _router(name):
        b = MockQueryBuilder()
        if name == "project_files":
            b.execute.return_value = MagicMock(
                data=[{"id": "file-1", "project_id": "p1", "file_path": "real/server/path.pdf"}], count=1
            )
        elif name == "project_members":
            b.execute.return_value = MagicMock(data={"role": "editor"})
        return b

    mock_supabase.table.side_effect = _router
    payload = {**PAYLOAD, "files": [{**PAYLOAD["files"][0], "file_id": "file-1"}]}
    client.post("/share/files", json=payload)
    bucket = mock_supabase.storage.from_.return_value
    called_path = bucket.create_signed_url.call_args[0][0]
    assert called_path == "real/server/path.pdf"


def test_share_audio_uses_audio_bucket_and_artist_ownership(client, mock_supabase, monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "test-key")
    monkeypatch.setattr("main.resend.Emails.send", lambda *a, **k: {"id": "email-1"}, raising=False)

    def _router(name):
        b = MockQueryBuilder()
        if name == "audio_files":
            b.execute.return_value = MagicMock(data=[{"id": "au-1", "file_path": "a/b.mp3", "folder_id": "fold-1"}])
        elif name == "audio_folders":
            b.execute.return_value = MagicMock(data=[{"artist_id": "art-1"}])
        elif name == "artists":
            b.execute.return_value = MagicMock(data=[{"id": "art-1"}], count=1)
        return b

    mock_supabase.table.side_effect = _router
    payload = {
        **PAYLOAD,
        "files": [{"file_name": "t.mp3", "file_path": "x", "file_source": "audio_file", "file_id": "au-1"}],
    }
    client.post("/share/files", json=payload)
    assert mock_supabase.storage.from_.call_args_list[-1][0][0] == "audio-files"
