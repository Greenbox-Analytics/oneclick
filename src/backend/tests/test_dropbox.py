"""Tests for the Dropbox integration (OAuth config, service, router)."""

from integrations import oauth


class TestDropboxAuthUrl:
    def _url(self, monkeypatch):
        monkeypatch.setenv("DROPBOX_APP_KEY", "dbx-client")
        # OAUTH_STATE_SECRET is read at module import; patch the module attr.
        monkeypatch.setattr(oauth, "OAUTH_STATE_SECRET", "test-secret")
        return oauth.build_auth_url("dropbox", "user-1")

    def test_auth_url_base_and_offline_access(self, monkeypatch):
        url = self._url(monkeypatch)
        assert url.startswith("https://www.dropbox.com/oauth2/authorize?")
        assert "token_access_type=offline" in url
        assert "client_id=dbx-client" in url

    def test_auth_url_scopes(self, monkeypatch):
        url = self._url(monkeypatch)
        for scope in [
            "account_info.read",
            "files.metadata.read",
            "files.content.read",
            "files.content.write",
            "sharing.read",
            "sharing.write",
        ]:
            assert scope in url

    def test_auth_url_redirect_uri(self, monkeypatch):
        url = self._url(monkeypatch)
        assert "/integrations/dropbox/callback" in url


import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from integrations import storage_import
from integrations.dropbox import service as dbx
from integrations.dropbox.service import FileTooLargeError
from integrations.google_drive import service as gdrive
from integrations.storage_import import StorageCapExceededError
from tests.conftest import TEST_USER_ID, MockQueryBuilder


class RecordingBuilder(MockQueryBuilder):
    """MockQueryBuilder that records .eq() filters and .update() payloads so
    tests can assert on them."""

    def __init__(self):
        super().__init__()
        self.eq_calls = []
        self.update_calls = []

    def eq(self, column, value):
        self.eq_calls.append((column, value))
        return self

    def update(self, payload):
        self.update_calls.append(payload)
        return self


class TestNormalizeEntry:
    def test_folder_gets_sentinel_and_null_meta(self):
        out = dbx._normalize_entry({".tag": "folder", "id": "id:f1", "name": "Contracts"})
        assert out == {
            "id": "id:f1",
            "name": "Contracts",
            "mimeType": dbx.FOLDER_MIME,
            "modifiedTime": None,
            "size": None,
        }

    def test_file_infers_mime_and_stringifies_size(self):
        out = dbx._normalize_entry(
            {
                ".tag": "file",
                "id": "id:a1",
                "name": "deal.pdf",
                "server_modified": "2026-08-01T00:00:00Z",
                "size": 1234,
            }
        )
        assert out["mimeType"] == "application/pdf"
        assert out["size"] == "1234"
        assert out["modifiedTime"] == "2026-08-01T00:00:00Z"

    def test_unknown_extension_falls_back_to_octet_stream(self):
        out = dbx._normalize_entry({".tag": "file", "id": "id:a2", "name": "weird.xyz123", "size": 1})
        assert out["mimeType"] == "application/octet-stream"


class TestExportIdempotency:
    def _supabase_with_existing(self, rows):
        sb = MagicMock()
        self.mappings = RecordingBuilder()
        self.mappings.execute.return_value = MagicMock(data=rows)

        def _router(name):
            if name == "drive_sync_mappings":
                return self.mappings
            return MockQueryBuilder()

        sb.table.side_effect = _router
        return sb

    def test_existing_export_row_returns_stored_link_without_upload(self):
        sb = self._supabase_with_existing([{"id": "m1", "share_url": "https://dbx.link/x", "drive_file_id": "id:a1"}])
        result = asyncio.run(dbx.export_to_dropbox("tok", sb, TEST_USER_ID, {"project_file_id": "pf1"}))
        assert result["share_url"] == "https://dbx.link/x"
        assert result["already_saved"] is True
        sb.storage.from_.assert_not_called()  # no download, no upload

    def test_idempotency_filter_is_fully_scoped(self):
        """Regression guard: the lookup must filter on user_id AND sync_direction,
        so an import-provenance row or another user's export can never match."""
        sb = self._supabase_with_existing([{"id": "m1", "share_url": "https://dbx.link/x", "drive_file_id": "id:a1"}])
        asyncio.run(dbx.export_to_dropbox("tok", sb, TEST_USER_ID, {"project_file_id": "pf1"}))
        assert ("user_id", TEST_USER_ID) in self.mappings.eq_calls
        assert ("project_file_id", "pf1") in self.mappings.eq_calls
        assert ("provider", "dropbox") in self.mappings.eq_calls
        assert ("sync_direction", "to_drive") in self.mappings.eq_calls

    def test_partial_row_without_link_creates_link_without_reupload(self):
        sb = self._supabase_with_existing([{"id": "m1", "share_url": None, "drive_file_id": "id:a1"}])
        with patch.object(dbx, "create_share_link", new=AsyncMock(return_value="https://dbx.link/new")):
            result = asyncio.run(dbx.export_to_dropbox("tok", sb, TEST_USER_ID, {"project_file_id": "pf1"}))
        assert result["share_url"] == "https://dbx.link/new"
        sb.storage.from_.assert_not_called()
        # Regression guard: without this assertion, the update() call could be
        # deleted entirely and the test above would still pass.
        assert {"share_url": "https://dbx.link/new"} in self.mappings.update_calls

    def test_oversized_file_raises_file_too_large_error(self, monkeypatch):
        monkeypatch.setattr(dbx, "MAX_UPLOAD_BYTES", 10)
        pf = MockQueryBuilder()
        pf.execute.return_value = MagicMock(
            data={"id": "pf1", "project_id": "p1", "file_name": "big.pdf", "file_path": "u/p/big.pdf"}
        )
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[])
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings, "project_files": pf}.get(
            name, MockQueryBuilder()
        )
        sb.storage.from_.return_value.download.return_value = b"x" * 11
        with (
            patch("projects.service.get_user_role", new=AsyncMock(return_value="owner")),
            pytest.raises(FileTooLargeError, match="too large"),
        ):
            asyncio.run(dbx.export_to_dropbox("tok", sb, TEST_USER_ID, {"project_file_id": "pf1"}))

    def test_file_size_column_short_circuits_before_download(self, monkeypatch):
        """When project_files.file_size is already known and over the cap, reject
        without ever touching Storage — the post-download len(content) check
        exists only as the authority for rows where file_size is NULL."""
        monkeypatch.setattr(dbx, "MAX_UPLOAD_BYTES", 10)
        pf = MockQueryBuilder()
        pf.execute.return_value = MagicMock(
            data={
                "id": "pf1",
                "project_id": "p1",
                "file_name": "big.pdf",
                "file_path": "u/p/big.pdf",
                "file_size": 999,
            }
        )
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[])
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings, "project_files": pf}.get(
            name, MockQueryBuilder()
        )
        with (
            patch("projects.service.get_user_role", new=AsyncMock(return_value="owner")),
            pytest.raises(FileTooLargeError, match="too large"),
        ):
            asyncio.run(dbx.export_to_dropbox("tok", sb, TEST_USER_ID, {"project_file_id": "pf1"}))
        sb.storage.from_.assert_not_called()

    def test_happy_path_inserts_row_with_null_link_then_updates(self):
        """Pins the shape of the two writes: the insert carries share_url=None
        and the update carries the real link. If link creation throws between
        them, this leaves a repairable row instead of an orphan upload that a
        retry would re-upload (and Dropbox would autorename)."""
        pf = MockQueryBuilder()
        pf.execute.return_value = MagicMock(
            data={"id": "pf1", "project_id": "p1", "file_name": "big.pdf", "file_path": "u/p/big.pdf"}
        )
        mappings = RecordingBuilder()
        mappings.execute.side_effect = [
            MagicMock(data=[]),  # idempotency lookup: nothing yet
            MagicMock(data=[{"id": "m-new"}]),  # insert result (share_url=None)
            MagicMock(data=[{"id": "m-new"}]),  # update result
        ]
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings, "project_files": pf}.get(
            name, MockQueryBuilder()
        )
        sb.storage.from_.return_value.download.return_value = b"small content"

        upload_response = MagicMock()
        upload_response.status_code = 200
        upload_response.json.return_value = {"id": "id:new1", "name": "big.pdf"}
        upload_response.raise_for_status = MagicMock()
        client = AsyncMock()
        client.post.side_effect = [upload_response]

        with (
            patch("projects.service.get_user_role", new=AsyncMock(return_value="owner")),
            patch.object(dbx, "create_share_link", new=AsyncMock(return_value="https://dbx.link/final")),
            patch("integrations.dropbox.service.httpx.AsyncClient") as mock_cls,
        ):
            mock_cls.return_value.__aenter__.return_value = client
            result = asyncio.run(dbx.export_to_dropbox("tok", sb, TEST_USER_ID, {"project_file_id": "pf1"}))

        assert result["share_url"] == "https://dbx.link/final"
        insert_payload = mappings.insert.call_args_list[0][0][0]
        assert insert_payload["share_url"] is None
        assert {"share_url": "https://dbx.link/final"} in mappings.update_calls
        # The upload is the only real httpx.AsyncClient() call in this test
        # (create_share_link is mocked out) — it must use the long transfer
        # timeout, not httpx's 5s default, or a real upload would ReadTimeout.
        assert mock_cls.call_args.kwargs.get("timeout") == dbx.TRANSFER_TIMEOUT

    def test_concurrent_export_insert_race_resolves_to_winners_link(self):
        """The partial unique index (drive_sync_mappings_dropbox_export_uniq)
        turns a lost race into a resolved lookup instead of a 500: this racer
        already uploaded (Dropbox autorenames its copy — an accepted cost of
        the race), but its insert loses to the unique index, so we hand back
        the winner's row instead of propagating the DB error."""
        pf = MockQueryBuilder()
        pf.execute.return_value = MagicMock(
            data={"id": "pf1", "project_id": "p1", "file_name": "big.pdf", "file_path": "u/p/big.pdf"}
        )
        mappings = RecordingBuilder()
        mappings.execute.side_effect = [
            MagicMock(data=[]),  # pre-check: no row yet, both racers pass
            Exception('duplicate key value violates unique constraint "drive_sync_mappings_dropbox_export_uniq"'),
            MagicMock(
                data=[{"id": "winner", "share_url": "https://dbx.link/winner", "drive_file_id": "id:winner"}]
            ),  # after-race lookup: the winner's row
        ]
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings, "project_files": pf}.get(
            name, MockQueryBuilder()
        )
        sb.storage.from_.return_value.download.return_value = b"small content"

        upload_response = MagicMock()
        upload_response.status_code = 200
        upload_response.json.return_value = {"id": "id:loser1", "name": "big (1).pdf"}
        upload_response.raise_for_status = MagicMock()
        client = AsyncMock()
        client.post.side_effect = [upload_response]

        with (
            patch("projects.service.get_user_role", new=AsyncMock(return_value="owner")),
            patch.object(dbx, "create_share_link", new=AsyncMock()) as mock_create_link,
            patch("integrations.dropbox.service.httpx.AsyncClient") as mock_cls,
        ):
            mock_cls.return_value.__aenter__.return_value = client
            result = asyncio.run(dbx.export_to_dropbox("tok", sb, TEST_USER_ID, {"project_file_id": "pf1"}))

        assert result == {"dropbox_file": None, "share_url": "https://dbx.link/winner", "already_saved": True}
        mock_create_link.assert_not_called()  # winner's row already had a link


class TestTransferTimeout:
    def test_download_uses_transfer_timeout(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.content = b"data"
        client = AsyncMock()
        client.post.return_value = resp
        with patch("integrations.dropbox.service.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = client
            asyncio.run(dbx.download_dropbox_file("tok", "id:a1"))
        assert mock_cls.call_args.kwargs.get("timeout") == dbx.TRANSFER_TIMEOUT


class TestImportDedup:
    def test_dropbox_import_dedup_raises_and_filters_by_provider(self):
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[{"id": "m1"}])
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings}.get(name, MockQueryBuilder())

        with pytest.raises(ValueError, match="already been imported"):
            asyncio.run(
                dbx.import_dropbox_file("tok", sb, TEST_USER_ID, {"project_id": "p1", "dropbox_file_id": "id:a1"})
            )
        assert ("provider", "dropbox") in mappings.eq_calls

    def test_drive_import_dedup_raises_on_duplicate(self):
        """No provider filter assertion here on purpose: google_drive/service.py
        deliberately does NOT filter on provider (that column only exists after
        a migration this query must keep working without — see the comment at
        its call site). Dropbox ids are always "id:"-prefixed and Drive ids
        never are, so a cross-provider collision can't happen regardless."""
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[{"id": "m1"}])
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings}.get(name, MockQueryBuilder())

        with pytest.raises(ValueError, match="already been imported"):
            asyncio.run(
                gdrive.import_drive_file("tok", sb, TEST_USER_ID, {"project_id": "p1", "drive_file_id": "gfile1"})
            )

    def test_dropbox_import_writes_provider_and_sync_direction(self):
        """The migration declares provider NOT NULL DEFAULT 'google_drive' — if
        this insert ever dropped the key, every Dropbox import would silently
        mislabel as google_drive and permanently break dedup for that row."""
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[])  # no existing dedup row
        pf = MockQueryBuilder()
        pf.execute.return_value = MagicMock(data=[{"id": "pf-new"}])
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings, "project_files": pf}.get(
            name, MockQueryBuilder()
        )
        sb.storage.from_.return_value.get_public_url = MagicMock(return_value="https://example.com/public")

        with (
            patch.object(dbx, "get_dropbox_metadata", new=AsyncMock(return_value={"name": "deal.pdf", "size": 100})),
            patch.object(dbx, "download_dropbox_file", new=AsyncMock(return_value=b"content")),
        ):
            asyncio.run(
                dbx.import_dropbox_file("tok", sb, TEST_USER_ID, {"project_id": "p1", "dropbox_file_id": "id:a1"})
            )

        mapping_insert_payload = mappings.insert.call_args_list[0][0][0]
        assert mapping_insert_payload["provider"] == "dropbox"
        assert mapping_insert_payload["sync_direction"] == "from_drive"

    def test_dropbox_import_does_not_precheck_gate_relies_on_trigger(self):
        """The router only verifies project-member role, not that user_id is
        the project's storage-counter owner (that's an ARTIST-ownership
        question a role check can't answer) — so import_dropbox_file must NOT
        call gated_upload at all. Gating the acting user's own wallet here
        would false-402 an editor importing into someone else's project.
        The DB trigger (-> StorageCapExceededError) is the real enforcement;
        see TestStorageImportGate in test_storage_import.py for the gate's
        own owner_user_id contract."""
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[])
        pf = MockQueryBuilder()
        pf.execute.return_value = MagicMock(data=[{"id": "pf-new"}])
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings, "project_files": pf}.get(
            name, MockQueryBuilder()
        )
        sb.storage.from_.return_value.get_public_url = MagicMock(return_value="https://example.com/public")

        with (
            patch.object(dbx, "get_dropbox_metadata", new=AsyncMock(return_value={"name": "deal.pdf", "size": 100})),
            patch.object(dbx, "download_dropbox_file", new=AsyncMock(return_value=b"content")),
            patch.object(storage_import, "gated_upload") as mock_gate,
        ):
            asyncio.run(
                dbx.import_dropbox_file("tok", sb, TEST_USER_ID, {"project_id": "p1", "dropbox_file_id": "id:a1"})
            )

        mock_gate.assert_not_called()

    def test_dropbox_import_oversized_metadata_raises_before_download(self, monkeypatch):
        """No ceiling previously existed on import at all — Dropbox metadata
        already reports size, so reject before download_dropbox_file ever
        loads an arbitrary-sized file into the container's memory."""
        monkeypatch.setattr(dbx, "MAX_UPLOAD_BYTES", 10)
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[])
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings}.get(name, MockQueryBuilder())

        with (
            patch.object(dbx, "get_dropbox_metadata", new=AsyncMock(return_value={"name": "big.pdf", "size": 11})),
            patch.object(dbx, "download_dropbox_file", new=AsyncMock()) as mock_download,
            pytest.raises(FileTooLargeError, match="too large"),
        ):
            asyncio.run(
                dbx.import_dropbox_file("tok", sb, TEST_USER_ID, {"project_id": "p1", "dropbox_file_id": "id:a1"})
            )
        mock_download.assert_not_called()

    def test_dropbox_import_db_error_cleans_up_orphan_and_maps_storage_cap(self):
        """The blob lands in Storage before the project_files row exists. Since
        the pre-check is skipped for imports (see the "does not precheck" test
        above), the trigger is the ONLY enforcement here, not a race backstop —
        if it rejects, the orphan object must be removed and the error must map
        to a typed exception the router turns into 402, not a bare 500 with a
        leaked blob nothing references."""
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[])
        pf = MockQueryBuilder()
        pf.execute.side_effect = Exception("new row violates check constraint - Storage cap exceeded (23514)")
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings, "project_files": pf}.get(
            name, MockQueryBuilder()
        )
        sb.storage.from_.return_value.get_public_url = MagicMock(return_value="https://example.com/public")

        with (
            patch.object(dbx, "get_dropbox_metadata", new=AsyncMock(return_value={"name": "deal.pdf", "size": 100})),
            patch.object(dbx, "download_dropbox_file", new=AsyncMock(return_value=b"content")),
            pytest.raises(StorageCapExceededError),
        ):
            asyncio.run(
                dbx.import_dropbox_file("tok", sb, TEST_USER_ID, {"project_id": "p1", "dropbox_file_id": "id:a1"})
            )

        sb.storage.from_.return_value.remove.assert_called_once()
        removed_path = sb.storage.from_.return_value.remove.call_args[0][0][0]
        assert TEST_USER_ID in removed_path and "p1" in removed_path


class TestShareLinkFallback:
    def _resp(self, status_code, payload):
        r = MagicMock()
        r.status_code = status_code
        r.json.return_value = payload
        return r

    def test_409_falls_back_to_list_shared_links(self):
        conflict = self._resp(409, {"error": {".tag": "shared_link_already_exists"}})
        listing = self._resp(200, {"links": [{"url": "https://dbx.link/existing"}]})
        client = AsyncMock()
        client.post.side_effect = [conflict, listing]
        with patch("integrations.dropbox.service.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = client
            url = asyncio.run(dbx.create_share_link("tok", "id:a1"))
        assert url == "https://dbx.link/existing"

    def test_success_returns_url(self):
        ok = self._resp(200, {"url": "https://dbx.link/fresh"})
        client = AsyncMock()
        client.post.side_effect = [ok]
        with patch("integrations.dropbox.service.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = client
            url = asyncio.run(dbx.create_share_link("tok", "id:a1"))
        assert url == "https://dbx.link/fresh"


class TestDropboxRouter:
    @patch("integrations.dropbox.router.get_valid_token", new=AsyncMock(return_value=None))
    def test_browse_returns_401_when_not_connected(self, client):
        resp = client.get("/integrations/dropbox/browse")
        assert resp.status_code == 401

    def test_import_denied_for_non_member(self, client, mock_supabase):
        def _router(name):
            b = MockQueryBuilder()
            if name == "project_members":
                b.execute.return_value = MagicMock(data=None)  # get_user_role -> None
            return b

        mock_supabase.table.side_effect = _router
        resp = client.post(
            "/integrations/dropbox/import",
            json={"project_id": "victim-proj", "dropbox_file_id": "id:f1"},
        )
        assert resp.status_code == 403

    @patch("integrations.dropbox.router.get_valid_token", new=AsyncMock(return_value="tok"))
    def test_import_duplicate_returns_409(self, client, mock_supabase):
        def _router(name):
            b = MockQueryBuilder()
            if name == "project_members":
                b.execute.return_value = MagicMock(data={"role": "editor"})
            if name == "drive_sync_mappings":
                b.execute.return_value = MagicMock(data=[{"id": "m1"}])  # already imported
            return b

        mock_supabase.table.side_effect = _router
        resp = client.post(
            "/integrations/dropbox/import",
            json={"project_id": "p1", "dropbox_file_id": "id:f1"},
        )
        assert resp.status_code == 409

    def test_export_status_unsaved_when_no_row(self, client, mock_supabase):
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=[])
        mock_supabase.table.side_effect = lambda name: b
        resp = client.get("/integrations/dropbox/export-status", params={"project_file_id": "pf1"})
        assert resp.status_code == 200
        assert resp.json() == {"saved": False, "share_url": None}

    def test_export_status_returns_stored_link(self, client, mock_supabase):
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=[{"share_url": "https://dbx.link/x"}])
        mock_supabase.table.side_effect = lambda name: b
        resp = client.get("/integrations/dropbox/export-status", params={"project_file_id": "pf1"})
        assert resp.status_code == 200
        assert resp.json() == {"saved": True, "share_url": "https://dbx.link/x"}

    def test_export_status_saved_but_link_pending(self, client, mock_supabase):
        """A row with share_url NULL means uploaded-but-link-pending, NOT unsaved —
        reporting it as unsaved would offer a folder picker for a file already in Dropbox."""
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=[{"share_url": None}])
        mock_supabase.table.side_effect = lambda name: b
        resp = client.get("/integrations/dropbox/export-status", params={"project_file_id": "pf1"})
        assert resp.status_code == 200
        assert resp.json() == {"saved": True, "share_url": None}

    def test_export_status_filters_on_all_four_columns(self, client, mock_supabase):
        """Regression guard: deleting .eq("user_id", user_id) here turns the
        endpoint into a cross-user existence oracle — same failure mode
        test_idempotency_filter_is_fully_scoped guards at the service layer."""
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[])
        mock_supabase.table.side_effect = lambda name: mappings
        resp = client.get("/integrations/dropbox/export-status", params={"project_file_id": "pf1"})
        assert resp.status_code == 200
        assert ("user_id", TEST_USER_ID) in mappings.eq_calls
        assert ("project_file_id", "pf1") in mappings.eq_calls
        assert ("provider", "dropbox") in mappings.eq_calls
        assert ("sync_direction", "to_drive") in mappings.eq_calls

    def test_disconnect_returns_success(self, client, mock_supabase):
        mock_supabase.table.side_effect = lambda name: MockQueryBuilder()
        resp = client.delete("/integrations/dropbox/disconnect")
        assert resp.status_code == 200
        assert resp.json() == {"success": True}
