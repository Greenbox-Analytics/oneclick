"""Tests for registry notes and folders CRUD endpoints.

Acceptance criteria:
1. Notes: list, create, get, update, delete
2. Folders: list, create, update, delete
"""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder

NOTE_ID = "eeeeeeee-0000-0000-0000-000000000001"
FOLDER_ID = "ffffffff-0000-0000-0000-000000000001"
ARTIST_ID = "bbbbbbbb-0000-0000-0000-000000000001"
PROJECT_ID = "cccccccc-0000-0000-0000-000000000001"

SAMPLE_NOTE = {
    "id": NOTE_ID,
    "user_id": TEST_USER_ID,
    "title": "My Note",
    "content": [{"type": "paragraph", "children": [{"text": "Hello"}]}],
    "artist_id": ARTIST_ID,
    "project_id": None,
    "folder_id": None,
    "pinned": False,
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
}

SAMPLE_FOLDER = {
    "id": FOLDER_ID,
    "user_id": TEST_USER_ID,
    "name": "My Folder",
    "artist_id": ARTIST_ID,
    "project_id": None,
    "parent_folder_id": None,
    "sort_order": 0,
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
}


# ============================================================
# Notes — List
# ============================================================


def test_list_notes_returns_notes_key(client, mock_supabase):
    """GET /registry/notes returns {"notes": [...]}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_NOTE], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get("/registry/notes")

    assert response.status_code == 200
    body = response.json()
    assert "notes" in body
    assert isinstance(body["notes"], list)


def test_list_notes_empty(client, mock_supabase):
    """GET /registry/notes returns empty list when no notes exist."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[], count=0)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get("/registry/notes")

    assert response.status_code == 200
    assert response.json()["notes"] == []


def test_list_notes_with_notes(client, mock_supabase):
    """GET /registry/notes returns notes from Supabase."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_NOTE], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get("/registry/notes")

    assert response.status_code == 200
    body = response.json()
    assert len(body["notes"]) == 1
    assert body["notes"][0]["id"] == NOTE_ID
    assert body["notes"][0]["title"] == "My Note"


def test_list_notes_filtered_by_artist_id(client, mock_supabase):
    """GET /registry/notes?artist_id=... filters by artist."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_NOTE], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/notes?artist_id={ARTIST_ID}")

    assert response.status_code == 200
    assert "notes" in response.json()


def test_list_notes_filtered_by_project_id(client, mock_supabase):
    """GET /registry/notes?project_id=... filters by project."""
    project_note = {**SAMPLE_NOTE, "artist_id": None, "project_id": PROJECT_ID}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[project_note], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/notes?project_id={PROJECT_ID}")

    assert response.status_code == 200
    assert "notes" in response.json()


def test_list_notes_filtered_by_folder_id(client, mock_supabase):
    """GET /registry/notes?folder_id=... filters by folder."""
    foldered_note = {**SAMPLE_NOTE, "folder_id": FOLDER_ID}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[foldered_note], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/notes?folder_id={FOLDER_ID}")

    assert response.status_code == 200
    assert "notes" in response.json()


# ============================================================
# Notes — Get
# ============================================================


def test_get_note_by_id_returns_note(client, mock_supabase):
    """GET /registry/notes/{note_id} returns the note when found."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=SAMPLE_NOTE)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/notes/{NOTE_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == NOTE_ID
    assert body["title"] == "My Note"


def test_get_note_by_id_not_found(client, mock_supabase):
    """GET /registry/notes/{note_id} returns 404 when note not found."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=None)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/notes/{NOTE_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Note not found"


# ============================================================
# Notes — Create
# ============================================================


def test_create_note_success(client, mock_supabase):
    """POST /registry/notes creates and returns the new note."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_NOTE])
    mock_supabase.table.side_effect = lambda name: builder

    payload = {
        "title": "My Note",
        "content": [{"type": "paragraph", "children": [{"text": "Hello"}]}],
        "artist_id": ARTIST_ID,
    }
    response = client.post("/registry/notes", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == NOTE_ID
    assert body["title"] == "My Note"


def test_create_note_minimal_payload(client, mock_supabase):
    """POST /registry/notes works with minimal payload (title/content have defaults)."""
    minimal_note = {**SAMPLE_NOTE, "title": "Untitled", "content": []}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[minimal_note])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.post("/registry/notes", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Untitled"


def test_create_note_db_failure_returns_500(client, mock_supabase):
    """POST /registry/notes returns 500 when Supabase insert returns no data."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.post("/registry/notes", json={"title": "Bad Note"})

    assert response.status_code == 500
    assert "Failed to create note" in response.json()["detail"]


def test_create_note_pinned_flag(client, mock_supabase):
    """POST /registry/notes accepts pinned=True."""
    pinned_note = {**SAMPLE_NOTE, "pinned": True}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[pinned_note])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.post("/registry/notes", json={"title": "Pinned Note", "pinned": True})

    assert response.status_code == 200
    assert response.json()["pinned"] is True


# ============================================================
# Notes — Update
# ============================================================


def test_update_note_success(client, mock_supabase):
    """PUT /registry/notes/{note_id} updates and returns the note."""
    updated_note = {**SAMPLE_NOTE, "title": "Updated Title"}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[updated_note])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.put(f"/registry/notes/{NOTE_ID}", json={"title": "Updated Title"})

    assert response.status_code == 200
    assert response.json()["title"] == "Updated Title"


def test_update_note_not_found(client, mock_supabase):
    """PUT /registry/notes/{note_id} returns 404 when note not found."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=None)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.put(f"/registry/notes/{NOTE_ID}", json={"title": "New Title"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Note not found"


def test_update_note_pin_toggle(client, mock_supabase):
    """PUT /registry/notes/{note_id} can toggle pinned status."""
    pinned_note = {**SAMPLE_NOTE, "pinned": True}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[pinned_note])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.put(f"/registry/notes/{NOTE_ID}", json={"pinned": True})

    assert response.status_code == 200
    assert response.json()["pinned"] is True


def test_update_note_move_to_folder(client, mock_supabase):
    """PUT /registry/notes/{note_id} can move note to a folder."""
    foldered_note = {**SAMPLE_NOTE, "folder_id": FOLDER_ID}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[foldered_note])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.put(f"/registry/notes/{NOTE_ID}", json={"folder_id": FOLDER_ID})

    assert response.status_code == 200
    assert response.json()["folder_id"] == FOLDER_ID


# ============================================================
# Notes — Delete
# ============================================================


def test_delete_note_success(client, mock_supabase):
    """DELETE /registry/notes/{note_id} returns {"ok": True}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_NOTE])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.delete(f"/registry/notes/{NOTE_ID}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_delete_note_always_returns_ok(client, mock_supabase):
    """DELETE /registry/notes/{note_id} returns {"ok": True} even if note not found."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.delete(f"/registry/notes/{NOTE_ID}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


# ============================================================
# Folders — List
# ============================================================


def test_list_folders_returns_folders_key(client, mock_supabase):
    """GET /registry/folders returns {"folders": [...]}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_FOLDER], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get("/registry/folders")

    assert response.status_code == 200
    body = response.json()
    assert "folders" in body
    assert isinstance(body["folders"], list)


def test_list_folders_empty(client, mock_supabase):
    """GET /registry/folders returns empty list when no folders exist."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[], count=0)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get("/registry/folders")

    assert response.status_code == 200
    assert response.json()["folders"] == []


def test_list_folders_with_folders(client, mock_supabase):
    """GET /registry/folders returns folders from Supabase."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_FOLDER], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get("/registry/folders")

    assert response.status_code == 200
    body = response.json()
    assert len(body["folders"]) == 1
    assert body["folders"][0]["id"] == FOLDER_ID
    assert body["folders"][0]["name"] == "My Folder"


def test_list_folders_filtered_by_artist_id(client, mock_supabase):
    """GET /registry/folders?artist_id=... filters by artist."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_FOLDER], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/folders?artist_id={ARTIST_ID}")

    assert response.status_code == 200
    assert "folders" in response.json()


def test_list_folders_filtered_by_project_id(client, mock_supabase):
    """GET /registry/folders?project_id=... filters by project."""
    project_folder = {**SAMPLE_FOLDER, "artist_id": None, "project_id": PROJECT_ID}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[project_folder], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/folders?project_id={PROJECT_ID}")

    assert response.status_code == 200
    assert "folders" in response.json()


# ============================================================
# Folders — Create
# ============================================================


def test_create_folder_success(client, mock_supabase):
    """POST /registry/folders creates and returns the new folder."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_FOLDER])
    mock_supabase.table.side_effect = lambda name: builder

    payload = {"name": "My Folder", "artist_id": ARTIST_ID}
    response = client.post("/registry/folders", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == FOLDER_ID
    assert body["name"] == "My Folder"


def test_create_folder_with_sort_order(client, mock_supabase):
    """POST /registry/folders accepts sort_order."""
    sorted_folder = {**SAMPLE_FOLDER, "sort_order": 5}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[sorted_folder])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.post("/registry/folders", json={"name": "Sorted Folder", "sort_order": 5})

    assert response.status_code == 200
    assert response.json()["sort_order"] == 5


def test_create_folder_db_failure_returns_500(client, mock_supabase):
    """POST /registry/folders returns 500 when Supabase insert returns no data."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.post("/registry/folders", json={"name": "Bad Folder"})

    assert response.status_code == 500
    assert "Failed to create folder" in response.json()["detail"]


def test_create_folder_with_parent(client, mock_supabase):
    """POST /registry/folders accepts a parent_folder_id for nested folders."""
    PARENT_FOLDER_ID = "11111111-0000-0000-0000-000000000001"
    nested_folder = {**SAMPLE_FOLDER, "parent_folder_id": PARENT_FOLDER_ID}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[nested_folder])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.post(
        "/registry/folders",
        json={"name": "Sub Folder", "parent_folder_id": PARENT_FOLDER_ID},
    )

    assert response.status_code == 200
    assert response.json()["parent_folder_id"] == PARENT_FOLDER_ID


# ============================================================
# Folders — Update
# ============================================================


def test_update_folder_success(client, mock_supabase):
    """PUT /registry/folders/{folder_id} updates and returns the folder."""
    updated_folder = {**SAMPLE_FOLDER, "name": "Renamed Folder"}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[updated_folder])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.put(f"/registry/folders/{FOLDER_ID}", json={"name": "Renamed Folder"})

    assert response.status_code == 200
    assert response.json()["name"] == "Renamed Folder"


def test_update_folder_not_found(client, mock_supabase):
    """PUT /registry/folders/{folder_id} returns 404 when folder not found."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=None)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.put(f"/registry/folders/{FOLDER_ID}", json={"name": "New Name"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Folder not found"


def test_update_folder_sort_order(client, mock_supabase):
    """PUT /registry/folders/{folder_id} can update sort_order."""
    reordered_folder = {**SAMPLE_FOLDER, "sort_order": 10}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[reordered_folder])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.put(f"/registry/folders/{FOLDER_ID}", json={"sort_order": 10})

    assert response.status_code == 200
    assert response.json()["sort_order"] == 10


# ============================================================
# Folders — Delete
# ============================================================


def test_delete_folder_success(client, mock_supabase):
    """DELETE /registry/folders/{folder_id} returns {"ok": True}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_FOLDER])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.delete(f"/registry/folders/{FOLDER_ID}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_delete_folder_always_returns_ok(client, mock_supabase):
    """DELETE /registry/folders/{folder_id} returns {"ok": True} even if not found."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.delete(f"/registry/folders/{FOLDER_ID}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
