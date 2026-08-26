"""Tests for the shared storage-cap gate used by both the Google Drive and
Dropbox import paths (integrations/storage_import.py).

Provider-specific behavior (dedup, metadata mapping, ceilings) is covered in
each provider's own test file; this file owns store_imported_file's own
contract: the cleanup/error mapping on a rejected insert, and the keyword-only
call shape.
"""

from unittest.mock import MagicMock

import pytest

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


class TestCapTriggerBackstop:
    def test_rejected_insert_cleans_up_orphan_and_raises_plain_message(self):
        sb = _sb(pf_execute_side_effect=Exception("new row violates check constraint - Storage cap exceeded (23514)"))
        with pytest.raises(StorageCapExceededError) as exc_info:
            store_imported_file(
                sb, TEST_USER_ID, "p1", "deal.pdf", b"content", mime="application/pdf", folder_category="contract"
            )
        message = str(exc_info.value)
        # The trigger is the ONLY enforcement, not a race loser, so the
        # message must not claim otherwise.
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
