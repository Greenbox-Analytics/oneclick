"""Tests for access-checked signed download URLs (Task 10).

These exercise ``service.get_file_download_url`` / ``service.get_audio_download_url``
directly, patching ``registry.service.get_work_access`` so each test pins a
specific WorkAccess shape, and mocking the supabase client (table read + storage
``create_signed_url``). Mirrors the pattern in test_registry_read_filtering.py.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from registry import service
from registry.access import WorkAccess


def _run(coro):
    return asyncio.run(coro)


def _patch_access(wa):
    """Patch registry.service.get_work_access to return *wa* (awaitable)."""
    return patch.object(service, "get_work_access", AsyncMock(return_value=wa))


def _db_with_file(file_path="works/abc.pdf", file_name="abc.pdf", signed_key="signedURL"):
    """MagicMock db where a single-row table read returns a file path and
    storage.create_signed_url returns {signed_key: url}."""
    db = MagicMock()

    row = MagicMock(data={"file_path": file_path, "file_name": file_name} if file_path else None)

    def table(_name):
        q = MagicMock()
        q.select.return_value = q
        q.eq.return_value = q
        maybe = MagicMock()
        maybe.execute.return_value = row
        q.maybe_single.return_value = maybe
        return q

    db.table.side_effect = table

    bucket = MagicMock()
    bucket.create_signed_url.return_value = {signed_key: "https://signed"}
    db.storage.from_.return_value = bucket
    return db


# ============================================================
# Files
# ============================================================


def test_ungranted_viewer_file_forbidden():
    # file_id not in visible_file_ids and not all_visible -> 403, never touches storage.
    wa = WorkAccess(work_role="viewer", my_collaborator_id="c1", visible_file_ids={"other-file"})
    db = _db_with_file()
    with _patch_access(wa), pytest.raises(HTTPException) as exc:
        _run(service.get_file_download_url(db, "u1", "w1", "f1"))
    assert exc.value.status_code == 403
    db.storage.from_.assert_not_called()


def test_granted_viewer_file_returns_url():
    wa = WorkAccess(work_role="viewer", my_collaborator_id="c1", visible_file_ids={"f1"})
    db = _db_with_file()
    with _patch_access(wa):
        url = _run(service.get_file_download_url(db, "u1", "w1", "f1"))
    assert url == "https://signed"


def test_all_visible_file_returns_url_with_empty_grant_set():
    # Owner / project member: all_visible True even though visible_file_ids is empty.
    wa = WorkAccess(work_role="owner")
    wa._all_visible = True
    db = _db_with_file()
    with _patch_access(wa):
        url = _run(service.get_file_download_url(db, "u1", "w1", "f1"))
    assert url == "https://signed"


def test_file_signed_url_camelcase_variant():
    # Some supabase-py versions key it "signedUrl" instead of "signedURL".
    wa = WorkAccess(work_role="owner")
    wa._all_visible = True
    db = _db_with_file(signed_key="signedUrl")
    with _patch_access(wa):
        url = _run(service.get_file_download_url(db, "u1", "w1", "f1"))
    assert url == "https://signed"


def test_file_missing_path_returns_404():
    wa = WorkAccess(work_role="owner")
    wa._all_visible = True
    db = _db_with_file(file_path=None)
    with _patch_access(wa), pytest.raises(HTTPException) as exc:
        _run(service.get_file_download_url(db, "u1", "w1", "f1"))
    assert exc.value.status_code == 404


def test_file_unsignable_returns_502():
    wa = WorkAccess(work_role="owner")
    wa._all_visible = True
    db = _db_with_file()
    db.storage.from_.return_value.create_signed_url.return_value = {}  # neither key present
    with _patch_access(wa), pytest.raises(HTTPException) as exc:
        _run(service.get_file_download_url(db, "u1", "w1", "f1"))
    assert exc.value.status_code == 502


# ============================================================
# Audio
# ============================================================


def test_ungranted_viewer_audio_forbidden():
    wa = WorkAccess(work_role="viewer", my_collaborator_id="c1", visible_audio_ids={"other-audio"})
    db = _db_with_file()
    with _patch_access(wa), pytest.raises(HTTPException) as exc:
        _run(service.get_audio_download_url(db, "u1", "w1", "a1"))
    assert exc.value.status_code == 403
    db.storage.from_.assert_not_called()


def test_granted_viewer_audio_returns_url():
    wa = WorkAccess(work_role="viewer", my_collaborator_id="c1", visible_audio_ids={"a1"})
    db = _db_with_file()
    with _patch_access(wa):
        url = _run(service.get_audio_download_url(db, "u1", "w1", "a1"))
    assert url == "https://signed"


def test_all_visible_audio_returns_url_with_empty_grant_set():
    wa = WorkAccess(work_role="owner")
    wa._all_visible = True
    db = _db_with_file()
    with _patch_access(wa):
        url = _run(service.get_audio_download_url(db, "u1", "w1", "a1"))
    assert url == "https://signed"


def test_audio_uses_audio_bucket():
    wa = WorkAccess(work_role="owner")
    wa._all_visible = True
    db = _db_with_file()
    with _patch_access(wa):
        _run(service.get_audio_download_url(db, "u1", "w1", "a1"))
    db.storage.from_.assert_called_once_with("audio-files")
