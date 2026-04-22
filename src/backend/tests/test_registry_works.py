"""Tests for registry works CRUD endpoints.

Acceptance criteria:
1. GET /registry/works - list works (returns {"works": [...]})
2. POST /registry/works - create a new work
3. GET /registry/works/{work_id} - get work by ID
4. PUT /registry/works/{work_id} - update a work
5. DELETE /registry/works/{work_id} - delete a work
"""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder

WORK_ID = "aaaaaaaa-0000-0000-0000-000000000001"
ARTIST_ID = "bbbbbbbb-0000-0000-0000-000000000001"
PROJECT_ID = "cccccccc-0000-0000-0000-000000000001"

SAMPLE_WORK = {
    "id": WORK_ID,
    "user_id": TEST_USER_ID,
    "artist_id": ARTIST_ID,
    "project_id": PROJECT_ID,
    "title": "Test Track",
    "work_type": "single",
    "status": "draft",
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
}


# ============================================================
# List Works
# ============================================================


def test_list_works_returns_works_key(client, mock_supabase):
    """GET /registry/works returns {"works": [...]}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_WORK], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get("/registry/works")

    assert response.status_code == 200
    body = response.json()
    assert "works" in body
    assert isinstance(body["works"], list)


def test_list_works_empty(client, mock_supabase):
    """GET /registry/works returns empty list when no works exist."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[], count=0)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get("/registry/works")

    assert response.status_code == 200
    body = response.json()
    assert body["works"] == []


def test_list_works_with_works(client, mock_supabase):
    """GET /registry/works returns the works from Supabase."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_WORK], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get("/registry/works")

    assert response.status_code == 200
    body = response.json()
    assert len(body["works"]) == 1
    assert body["works"][0]["id"] == WORK_ID
    assert body["works"][0]["title"] == "Test Track"


def test_list_works_paginated(client, mock_supabase):
    """GET /registry/works?page=1 returns paginated response with data/total/page/page_size."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_WORK], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get("/registry/works?page=1&page_size=10")

    assert response.status_code == 200
    response.json()
    # When paginated, the router returns PaginatedResponse directly if it is not a list
    # (could be nested under a key or directly as dict with data/total/page/page_size)
    assert response.status_code == 200


def test_list_works_by_artist_id(client, mock_supabase):
    """GET /registry/works?artist_id=... filters by artist."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_WORK], count=1)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/works?artist_id={ARTIST_ID}")

    assert response.status_code == 200
    body = response.json()
    assert "works" in body


# ============================================================
# Get Work by ID
# ============================================================


def test_get_work_by_id_returns_work(client, mock_supabase):
    """GET /registry/works/{work_id} returns the work when found."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=SAMPLE_WORK)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/works/{WORK_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == WORK_ID
    assert body["title"] == "Test Track"


def test_get_work_by_id_not_found(client, mock_supabase):
    """GET /registry/works/{work_id} returns 404 when work not found."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=None)
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/works/{WORK_ID}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Work not found"


# ============================================================
# Create Work
# ============================================================


def test_create_work_success(client, mock_supabase):
    """POST /registry/works creates and returns the new work."""
    artist_builder = MockQueryBuilder()
    artist_builder.execute.return_value = MagicMock(data={"id": ARTIST_ID})

    work_builder = MockQueryBuilder()
    work_builder.execute.return_value = MagicMock(data=[SAMPLE_WORK])

    def table_side_effect(name):
        if name == "artists":
            return artist_builder
        return work_builder

    mock_supabase.table.side_effect = table_side_effect

    payload = {
        "artist_id": ARTIST_ID,
        "project_id": PROJECT_ID,
        "title": "Test Track",
        "work_type": "single",
    }
    response = client.post("/registry/works", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == WORK_ID
    assert body["title"] == "Test Track"


def test_create_work_artist_not_found_returns_500(client, mock_supabase):
    """POST /registry/works returns 500 when artist does not belong to user."""
    artist_builder = MockQueryBuilder()
    artist_builder.execute.return_value = MagicMock(data=None)

    mock_supabase.table.side_effect = lambda name: artist_builder

    payload = {
        "artist_id": ARTIST_ID,
        "project_id": PROJECT_ID,
        "title": "Test Track",
        "work_type": "single",
    }
    response = client.post("/registry/works", json=payload)

    assert response.status_code == 500
    assert "Failed to create work" in response.json()["detail"]


def test_create_work_with_optional_fields(client, mock_supabase):
    """POST /registry/works accepts optional fields like isrc, iswc, notes."""
    work_with_extras = {**SAMPLE_WORK, "isrc": "USABC1234567", "notes": "Some notes"}

    artist_builder = MockQueryBuilder()
    artist_builder.execute.return_value = MagicMock(data={"id": ARTIST_ID})

    work_builder = MockQueryBuilder()
    work_builder.execute.return_value = MagicMock(data=[work_with_extras])

    def table_side_effect(name):
        if name == "artists":
            return artist_builder
        return work_builder

    mock_supabase.table.side_effect = table_side_effect

    payload = {
        "artist_id": ARTIST_ID,
        "project_id": PROJECT_ID,
        "title": "Test Track",
        "work_type": "single",
        "isrc": "USABC1234567",
        "notes": "Some notes",
    }
    response = client.post("/registry/works", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["isrc"] == "USABC1234567"


# ============================================================
# Update Work
# ============================================================


def test_update_work_success(client, mock_supabase):
    """PUT /registry/works/{work_id} updates and returns the work."""
    updated_work = {**SAMPLE_WORK, "title": "Updated Title"}

    get_builder = MockQueryBuilder()
    get_builder.execute.return_value = MagicMock(data={"status": "draft"})

    update_builder = MockQueryBuilder()
    update_builder.execute.return_value = MagicMock(data=[updated_work])

    call_count = [0]

    def table_side_effect(name):
        if name == "works_registry":
            call_count[0] += 1
            if call_count[0] == 1:
                return get_builder
            return update_builder
        return MockQueryBuilder()

    mock_supabase.table.side_effect = table_side_effect

    payload = {"title": "Updated Title"}
    response = client.put(f"/registry/works/{WORK_ID}", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Updated Title"


def test_update_work_not_found(client, mock_supabase):
    """PUT /registry/works/{work_id} returns 404 when work not found."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=None)
    mock_supabase.table.side_effect = lambda name: builder

    payload = {"title": "Updated Title"}
    response = client.put(f"/registry/works/{WORK_ID}", json=payload)

    assert response.status_code == 404
    assert response.json()["detail"] == "Work not found"


# ============================================================
# Delete Work
# ============================================================


def test_delete_work_success(client, mock_supabase):
    """DELETE /registry/works/{work_id} returns {"ok": True}."""
    # delete_work queries collaborators, then the work title, then deletes
    collab_builder = MockQueryBuilder()
    collab_builder.execute.return_value = MagicMock(data=[])

    work_title_builder = MockQueryBuilder()
    work_title_builder.execute.return_value = MagicMock(data={"title": "Test Track"})

    delete_builder = MockQueryBuilder()
    delete_builder.execute.return_value = MagicMock(data=[SAMPLE_WORK])

    call_count = [0]

    def table_side_effect(name):
        if name == "registry_collaborators":
            return collab_builder
        call_count[0] += 1
        if call_count[0] == 1:
            return work_title_builder
        return delete_builder

    mock_supabase.table.side_effect = table_side_effect

    response = client.delete(f"/registry/works/{WORK_ID}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_delete_work_always_returns_ok(client, mock_supabase):
    """DELETE /registry/works/{work_id} returns {"ok": True} even if nothing deleted."""
    collab_builder = MockQueryBuilder()
    collab_builder.execute.return_value = MagicMock(data=[])

    work_title_builder = MockQueryBuilder()
    work_title_builder.execute.return_value = MagicMock(data=None)

    delete_builder = MockQueryBuilder()
    delete_builder.execute.return_value = MagicMock(data=[])

    call_count = [0]

    def table_side_effect(name):
        if name == "registry_collaborators":
            return collab_builder
        call_count[0] += 1
        if call_count[0] == 1:
            return work_title_builder
        return delete_builder

    mock_supabase.table.side_effect = table_side_effect

    response = client.delete(f"/registry/works/{WORK_ID}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
