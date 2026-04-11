"""Tests for registry ownership stakes CRUD endpoints.

Acceptance criteria:
1. GET /registry/stakes?work_id=... - list stakes by work
2. POST /registry/stakes - create a new stake
3. PUT /registry/stakes/{stake_id} - update a stake
4. DELETE /registry/stakes/{stake_id} - delete a stake
"""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder

WORK_ID = "aaaaaaaa-0000-0000-0000-000000000001"
STAKE_ID = "dddddddd-0000-0000-0000-000000000001"

SAMPLE_STAKE = {
    "id": STAKE_ID,
    "user_id": TEST_USER_ID,
    "work_id": WORK_ID,
    "stake_type": "master",
    "holder_name": "Jane Doe",
    "holder_role": "producer",
    "percentage": 50.0,
    "created_at": "2026-01-01T00:00:00+00:00",
}


# ============================================================
# List Stakes
# ============================================================


def test_list_stakes_returns_stakes_key(client, mock_supabase):
    """GET /registry/stakes?work_id=... returns {"stakes": [...]}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_STAKE])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/stakes?work_id={WORK_ID}")

    assert response.status_code == 200
    body = response.json()
    assert "stakes" in body
    assert isinstance(body["stakes"], list)


def test_list_stakes_empty(client, mock_supabase):
    """GET /registry/stakes?work_id=... returns empty list when no stakes exist."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/stakes?work_id={WORK_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["stakes"] == []


def test_list_stakes_with_stakes(client, mock_supabase):
    """GET /registry/stakes?work_id=... returns the stakes from Supabase."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_STAKE])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.get(f"/registry/stakes?work_id={WORK_ID}")

    assert response.status_code == 200
    body = response.json()
    assert len(body["stakes"]) == 1
    assert body["stakes"][0]["id"] == STAKE_ID
    assert body["stakes"][0]["holder_name"] == "Jane Doe"
    assert body["stakes"][0]["percentage"] == 50.0


def test_list_stakes_requires_work_id(client, mock_supabase):
    """GET /registry/stakes without work_id returns 422."""
    response = client.get("/registry/stakes")
    assert response.status_code == 422


# ============================================================
# Create Stake
# ============================================================


def test_create_stake_success(client, mock_supabase):
    """POST /registry/stakes creates and returns the new stake."""
    # validate_stake_percentage queries existing stakes, then create_stake inserts
    validate_builder = MockQueryBuilder()
    validate_builder.execute.return_value = MagicMock(data=[])  # no existing stakes

    insert_builder = MockQueryBuilder()
    insert_builder.execute.return_value = MagicMock(data=[SAMPLE_STAKE])

    call_count = [0]

    def table_side_effect(name):
        if name == "ownership_stakes":
            call_count[0] += 1
            if call_count[0] == 1:
                return validate_builder
            return insert_builder
        return MockQueryBuilder()

    mock_supabase.table.side_effect = table_side_effect

    payload = {
        "work_id": WORK_ID,
        "stake_type": "master",
        "holder_name": "Jane Doe",
        "holder_role": "producer",
        "percentage": 50.0,
    }
    response = client.post("/registry/stakes", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == STAKE_ID
    assert body["percentage"] == 50.0


def test_create_stake_exceeds_100_percent(client, mock_supabase):
    """POST /registry/stakes returns 400 when percentage would exceed 100%."""
    # simulate existing stake at 80% so adding 30% would exceed 100
    existing_stake = {"id": "other-id", "percentage": 80.0}
    validate_builder = MockQueryBuilder()
    validate_builder.execute.return_value = MagicMock(data=[existing_stake])
    mock_supabase.table.side_effect = lambda name: validate_builder

    payload = {
        "work_id": WORK_ID,
        "stake_type": "master",
        "holder_name": "Bob Smith",
        "holder_role": "artist",
        "percentage": 30.0,
    }
    response = client.post("/registry/stakes", json=payload)

    assert response.status_code == 400
    assert "exceed 100%" in response.json()["detail"]


def test_create_stake_insert_fails_returns_500(client, mock_supabase):
    """POST /registry/stakes returns 500 when insert returns no data."""
    validate_builder = MockQueryBuilder()
    validate_builder.execute.return_value = MagicMock(data=[])

    insert_builder = MockQueryBuilder()
    insert_builder.execute.return_value = MagicMock(data=[])

    call_count = [0]

    def table_side_effect(name):
        if name == "ownership_stakes":
            call_count[0] += 1
            if call_count[0] == 1:
                return validate_builder
            return insert_builder
        return MockQueryBuilder()

    mock_supabase.table.side_effect = table_side_effect

    payload = {
        "work_id": WORK_ID,
        "stake_type": "master",
        "holder_name": "Jane Doe",
        "holder_role": "producer",
        "percentage": 50.0,
    }
    response = client.post("/registry/stakes", json=payload)

    assert response.status_code == 500
    assert "Failed to create stake" in response.json()["detail"]


def test_create_stake_exactly_at_100_percent(client, mock_supabase):
    """POST /registry/stakes succeeds when total equals exactly 100%."""
    existing_stake = {"id": "other-id", "percentage": 50.0}
    validate_builder = MockQueryBuilder()
    validate_builder.execute.return_value = MagicMock(data=[existing_stake])

    insert_builder = MockQueryBuilder()
    insert_builder.execute.return_value = MagicMock(data=[SAMPLE_STAKE])

    call_count = [0]

    def table_side_effect(name):
        if name == "ownership_stakes":
            call_count[0] += 1
            if call_count[0] == 1:
                return validate_builder
            return insert_builder
        return MockQueryBuilder()

    mock_supabase.table.side_effect = table_side_effect

    payload = {
        "work_id": WORK_ID,
        "stake_type": "master",
        "holder_name": "Jane Doe",
        "holder_role": "producer",
        "percentage": 50.0,
    }
    response = client.post("/registry/stakes", json=payload)

    assert response.status_code == 200


# ============================================================
# Update Stake
# ============================================================


def test_update_stake_success(client, mock_supabase):
    """PUT /registry/stakes/{stake_id} updates and returns the stake."""
    updated_stake = {**SAMPLE_STAKE, "holder_name": "Updated Name"}

    lookup_builder = MockQueryBuilder()
    lookup_builder.execute.return_value = MagicMock(data={"work_id": WORK_ID, "stake_type": "master"})

    validate_builder = MockQueryBuilder()
    validate_builder.execute.return_value = MagicMock(data=[])

    update_builder = MockQueryBuilder()
    update_builder.execute.return_value = MagicMock(data=[updated_stake])

    # update_stake router: first queries existing stake (ownership_stakes for lookup),
    # then validate_stake_percentage (ownership_stakes again), then update (ownership_stakes)
    call_count = [0]

    def table_side_effect(name):
        if name == "ownership_stakes":
            call_count[0] += 1
            if call_count[0] == 1:
                return lookup_builder
            if call_count[0] == 2:
                return validate_builder
            return update_builder
        return MockQueryBuilder()

    mock_supabase.table.side_effect = table_side_effect

    payload = {"holder_name": "Updated Name", "percentage": 40.0}
    response = client.put(f"/registry/stakes/{STAKE_ID}", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["holder_name"] == "Updated Name"


def test_update_stake_not_found(client, mock_supabase):
    """PUT /registry/stakes/{stake_id} returns 404 when stake not found for lookup."""
    lookup_builder = MockQueryBuilder()
    lookup_builder.execute.return_value = MagicMock(data=None)
    mock_supabase.table.side_effect = lambda name: lookup_builder

    payload = {"holder_name": "Updated Name", "percentage": 40.0}
    response = client.put(f"/registry/stakes/{STAKE_ID}", json=payload)

    assert response.status_code == 404
    assert "Stake not found" in response.json()["detail"]


def test_update_stake_without_percentage_skips_validation(client, mock_supabase):
    """PUT /registry/stakes/{stake_id} with no percentage change skips validation."""
    updated_stake = {**SAMPLE_STAKE, "holder_name": "New Name"}

    update_builder = MockQueryBuilder()
    update_builder.execute.return_value = MagicMock(data=[updated_stake])

    mock_supabase.table.side_effect = lambda name: update_builder

    payload = {"holder_name": "New Name"}
    response = client.put(f"/registry/stakes/{STAKE_ID}", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["holder_name"] == "New Name"


def test_update_stake_not_found_after_update(client, mock_supabase):
    """PUT /registry/stakes/{stake_id} returns 404 when update returns no data."""
    update_builder = MockQueryBuilder()
    update_builder.execute.return_value = MagicMock(data=[])
    mock_supabase.table.side_effect = lambda name: update_builder

    # No percentage in body so validation is skipped
    payload = {"holder_name": "New Name"}
    response = client.put(f"/registry/stakes/{STAKE_ID}", json=payload)

    assert response.status_code == 404
    assert "Stake not found" in response.json()["detail"]


# ============================================================
# Delete Stake
# ============================================================


def test_delete_stake_success(client, mock_supabase):
    """DELETE /registry/stakes/{stake_id} returns {"ok": True}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_STAKE])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.delete(f"/registry/stakes/{STAKE_ID}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_delete_stake_always_returns_ok(client, mock_supabase):
    """DELETE /registry/stakes/{stake_id} returns {"ok": True} even if nothing deleted."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[])
    mock_supabase.table.side_effect = lambda name: builder

    response = client.delete(f"/registry/stakes/{STAKE_ID}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
