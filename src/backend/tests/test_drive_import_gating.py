"""Storage-cap gating tests for the Google Drive import path.

Mirrors the coverage added for Dropbox in test_dropbox.py::TestImportDedup
when import_drive_file was routed through the shared
integrations.storage_import.store_imported_file helper (previously it had
no gate, no orphan cleanup, and no storage-cap error mapping at all).

Kept out of test_drive_idor.py (403/IDOR-only, per its own docstring) and
test_integrations.py (connections/auth/OneClick-share) since this is a
distinct concern — storage-cap gating and cleanup on the import write path —
and neither existing file's scope fits it cleanly.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from integrations.google_drive import service as gdrive
from integrations.storage_import import StorageCapExceededError
from tests.conftest import TEST_USER_ID, MockQueryBuilder


class RecordingBuilder(MockQueryBuilder):
    """MockQueryBuilder that records .eq() filters so tests can assert on them.
    Mirrors test_dropbox.py's RecordingBuilder."""

    def __init__(self):
        super().__init__()
        self.eq_calls = []

    def eq(self, column, value):
        self.eq_calls.append((column, value))
        return self


def _drive_metadata(size=100):
    return {"id": "gfile1", "name": "deal.pdf", "mimeType": "application/pdf", "size": size}


def _mock_meta_response(metadata):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = metadata
    return resp


class TestDriveImportGating:
    def test_db_error_with_storage_cap_signature_cleans_up_orphan_and_raises(self):
        """Before this fix, a rejected insert here 500'd and leaked the Storage
        object — no cleanup, no cap mapping. After routing through
        store_imported_file, the same cleanup + typed-exception behavior
        Dropbox has now applies to Drive too."""
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[])
        pf = MockQueryBuilder()
        pf.execute.side_effect = Exception("new row violates check constraint - Storage cap exceeded (23514)")
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings, "project_files": pf}.get(
            name, MockQueryBuilder()
        )
        sb.storage.from_.return_value.get_public_url = MagicMock(return_value="https://example.com/public")

        client = AsyncMock()
        client.get.return_value = _mock_meta_response(_drive_metadata(size=100))

        with (
            patch("integrations.google_drive.service.httpx.AsyncClient") as mock_cls,
            patch.object(gdrive, "download_drive_file", new=AsyncMock(return_value=b"content")),
            pytest.raises(StorageCapExceededError),
        ):
            mock_cls.return_value.__aenter__.return_value = client
            asyncio.run(
                gdrive.import_drive_file("tok", sb, TEST_USER_ID, {"project_id": "p1", "drive_file_id": "gfile1"})
            )

        sb.storage.from_.return_value.remove.assert_called_once()
        removed_path = sb.storage.from_.return_value.remove.call_args[0][0][0]
        assert TEST_USER_ID in removed_path and "p1" in removed_path

    def test_router_maps_storage_cap_error_to_402(self, client, mock_supabase):
        """End-to-end: the router must map StorageCapExceededError to 402,
        same status main.py's direct-upload endpoint uses for the same trigger."""

        def _router(name):
            b = MockQueryBuilder()
            if name == "project_members":
                b.execute.return_value = MagicMock(data={"role": "editor"})
            if name == "drive_sync_mappings":
                b.execute.return_value = MagicMock(data=[])  # no existing dedup row
            if name == "project_files":
                b.execute.side_effect = Exception("new row violates check constraint - Storage cap exceeded (23514)")
            return b

        mock_supabase.table.side_effect = _router

        drive_client = AsyncMock()
        drive_client.get.return_value = _mock_meta_response(_drive_metadata(size=100))

        with (
            patch("integrations.google_drive.router.get_valid_token", new=AsyncMock(return_value="tok")),
            patch("integrations.google_drive.service.httpx.AsyncClient") as mock_cls,
            patch.object(gdrive, "download_drive_file", new=AsyncMock(return_value=b"content")),
        ):
            mock_cls.return_value.__aenter__.return_value = drive_client
            resp = client.post(
                "/integrations/google-drive/import",
                json={"project_id": "p1", "drive_file_id": "gfile1"},
            )

        assert resp.status_code == 402

    def test_provenance_insert_shape_unchanged(self):
        """The drive_sync_mappings provenance insert never carried an explicit
        'provider' key — the column DEFAULT already supplies 'google_drive'.
        The storage_import refactor must not change that shape.

        The dedup pre-check does NOT filter on provider — deliberately: that
        column only exists after a migration this query must keep working
        without (see the comment at its call site in google_drive/service.py),
        and Dropbox ids are always "id:"-prefixed so a cross-provider dedup
        collision can't happen regardless."""
        mappings = RecordingBuilder()
        mappings.execute.return_value = MagicMock(data=[])
        pf = MockQueryBuilder()
        pf.execute.return_value = MagicMock(data=[{"id": "pf-new"}])
        sb = MagicMock()
        sb.table.side_effect = lambda name: {"drive_sync_mappings": mappings, "project_files": pf}.get(
            name, MockQueryBuilder()
        )
        sb.storage.from_.return_value.get_public_url = MagicMock(return_value="https://example.com/public")

        client = AsyncMock()
        client.get.return_value = _mock_meta_response(_drive_metadata(size=100))

        with (
            patch("integrations.google_drive.service.httpx.AsyncClient") as mock_cls,
            patch.object(gdrive, "download_drive_file", new=AsyncMock(return_value=b"content")),
        ):
            mock_cls.return_value.__aenter__.return_value = client
            asyncio.run(
                gdrive.import_drive_file("tok", sb, TEST_USER_ID, {"project_id": "p1", "drive_file_id": "gfile1"})
            )

        provenance_payload = mappings.insert.call_args_list[0][0][0]
        assert "provider" not in provenance_payload
        assert provenance_payload["sync_direction"] == "from_drive"
        assert provenance_payload["project_file_id"] == "pf-new"
