"""Tests for the shared storage-cap gate used by both the Google Drive and
Dropbox import paths (integrations/storage_import.py).

Provider-specific behavior (dedup, metadata mapping, ceilings) is covered in
each provider's own test file; this file owns store_imported_file's own
contract: when the pre-check runs, what size it gates on, the cleanup/error
mapping on a rejected insert, and the keyword-only call shape.
"""

from unittest.mock import MagicMock, patch

import pytest

from integrations import storage_import
from integrations.storage_import import StorageCapExceededError, store_imported_file
from tests.conftest import TEST_USER_ID


def _sb(pf_execute_side_effect=None):
    sb = MagicMock()
    pf = MagicMock()
    if pf_execute_side_effect is not None:
        pf.insert.return_value.execute.side_effect = pf_execute_side_effect
    else:
        pf.insert.return_value.execute.return_value = MagicMock(data=[{"id": "pf-new"}])
    sb.table.return_value = pf
    sb.storage.from_.return_value.get_public_url = MagicMock(return_value="https://example.com/public")
    return sb


class TestOwnerGate:
    def test_skips_gate_when_owner_unknown(self):
        """The default (no owner_user_id) must never call gated_upload — this
        is the path both import services use today, since neither can cheaply
        prove the acting user is the project's storage-counter owner."""
        sb = _sb()
        with patch.object(storage_import, "gated_upload") as mock_gate:
            result = store_imported_file(
                sb, TEST_USER_ID, "p1", "deal.pdf", b"content", mime="application/pdf", folder_category="contract"
            )
        mock_gate.assert_not_called()
        assert result == {"id": "pf-new"}

    def test_gates_against_owner_when_known(self):
        """When the caller passes owner_user_id, gate using THAT identity for
        both the acting-user and host_user_id args — never the uploader's own
        id if it could differ from the actual owner."""
        sb = _sb()
        with patch.object(storage_import, "gated_upload") as mock_gate:
            store_imported_file(
                sb,
                TEST_USER_ID,
                "p1",
                "deal.pdf",
                b"content",
                mime="application/pdf",
                folder_category="contract",
                file_size=100,
                owner_user_id="owner-1",
            )
        mock_gate.assert_called_once_with("owner-1", size=100, host_user_id="owner-1", resource_project_id="p1")

    def test_size_falls_back_to_content_length_not_zero(self):
        """Regression guard: a provider that omits file size (e.g. Drive
        metadata without a size field) must gate on the real byte count, not
        0 — a 0-byte gate can only ever reject a wallet already at exactly
        100% of cap, which defeats the point of the check."""
        sb = _sb()
        with patch.object(storage_import, "gated_upload") as mock_gate:
            store_imported_file(
                sb,
                TEST_USER_ID,
                "p1",
                "deal.pdf",
                b"12345",
                mime="application/pdf",
                folder_category="contract",
                file_size=None,
                owner_user_id="owner-1",
            )
        mock_gate.assert_called_once_with("owner-1", size=5, host_user_id="owner-1", resource_project_id="p1")


class TestCapTriggerBackstop:
    def test_rejected_insert_cleans_up_orphan_and_raises_plain_message(self):
        sb = _sb(pf_execute_side_effect=Exception("new row violates check constraint - Storage cap exceeded (23514)"))
        with pytest.raises(StorageCapExceededError) as exc_info:
            store_imported_file(
                sb, TEST_USER_ID, "p1", "deal.pdf", b"content", mime="application/pdf", folder_category="contract"
            )
        message = str(exc_info.value)
        # Plainly true for a non-technical user now that the pre-check isn't
        # always run first — this is sometimes the ONLY enforcement, not just
        # a race loser, so the message must not claim otherwise.
        assert "concurrent" not in message.lower()
        assert "storage limit" in message.lower()
        sb.storage.from_.return_value.remove.assert_called_once()


class TestKeywordOnlyArgs:
    def test_positional_call_after_content_raises_type_error(self):
        """mime, folder_category, and file_name are three adjacent strings —
        without keyword-only enforcement, transposing mime and folder_category
        type-checks fine and silently writes a garbage file_type/folder_category."""
        sb = _sb()
        with pytest.raises(TypeError):
            store_imported_file(sb, TEST_USER_ID, "p1", "deal.pdf", b"content", "application/pdf", "contract")
