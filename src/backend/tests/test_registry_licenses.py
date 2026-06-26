"""Tests for registry licensing rights CRUD endpoints.

Acceptance criteria:
1. GET /registry/licenses?work_id=... - list licenses by work
2. POST /registry/licenses - create a new license
3. PUT /registry/licenses/{license_id} - update a license
4. DELETE /registry/licenses/{license_id} - delete a license
"""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder, _default_table_side_effect, grant_owner_access

_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})


def _sub_wrap(fn):
    """Wrap a table side-effect function to route subscription tables through the Pro default."""

    def _wrapped(name):
        if name in _SUBSCRIPTION_TABLES:
            return _default_table_side_effect(name)
        return fn(name)

    return _wrapped


WORK_ID = "aaaaaaaa-0000-0000-0000-000000000001"
LICENSE_ID = "eeeeeeee-0000-0000-0000-000000000001"

SAMPLE_LICENSE = {
    "id": LICENSE_ID,
    "user_id": TEST_USER_ID,
    "work_id": WORK_ID,
    "license_type": "sync",
    "licensee_name": "Big Film Studios",
    "licensee_email": "licensing@bigfilm.com",
    "territory": "worldwide",
    "start_date": "2026-01-01",
    "end_date": "2027-01-01",
    "terms": "Non-exclusive sync license",
    "status": "active",
    "created_at": "2026-01-01T00:00:00+00:00",
}


# ============================================================
# List Licenses
# ============================================================


def test_list_licenses_returns_licenses_key(client, mock_supabase):
    """GET /registry/licenses?work_id=... returns {"licenses": [...]}."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_LICENSE])
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    with grant_owner_access():
        response = client.get(f"/registry/licenses?work_id={WORK_ID}")

    assert response.status_code == 200
    body = response.json()
    assert "licenses" in body
    assert isinstance(body["licenses"], list)


def test_list_licenses_empty(client, mock_supabase):
    """GET /registry/licenses?work_id=... returns empty list when no licenses exist."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[])
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    with grant_owner_access():
        response = client.get(f"/registry/licenses?work_id={WORK_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["licenses"] == []


def test_list_licenses_with_licenses(client, mock_supabase):
    """GET /registry/licenses?work_id=... returns the licenses from Supabase."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_LICENSE])
    mock_supabase.table.side_effect = _sub_wrap(lambda name: builder)

    with grant_owner_access():
        response = client.get(f"/registry/licenses?work_id={WORK_ID}")

    assert response.status_code == 200
    body = response.json()
    assert len(body["licenses"]) == 1
    assert body["licenses"][0]["id"] == LICENSE_ID
    assert body["licenses"][0]["licensee_name"] == "Big Film Studios"
    assert body["licenses"][0]["territory"] == "worldwide"


def test_list_licenses_requires_work_id(client, mock_supabase):
    """GET /registry/licenses without work_id returns 422."""
    response = client.get("/registry/licenses")
    assert response.status_code == 422


# ============================================================
# Create License
# ============================================================


def _work_owner_builder():
    """Builder for the works_registry lookup create_license/create_agreement does to
    resolve the work owner for user_id stamping. Returns a dict, not a list."""
    b = MockQueryBuilder()
    b.execute.return_value = MagicMock(data={"user_id": TEST_USER_ID})
    return b


def test_create_license_success(client, mock_supabase):
    """POST /registry/licenses creates and returns the new license."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_LICENSE])

    def _side(name):
        return _work_owner_builder() if name == "works_registry" else builder

    mock_supabase.table.side_effect = _sub_wrap(_side)

    payload = {
        "work_id": WORK_ID,
        "license_type": "sync",
        "licensee_name": "Big Film Studios",
        "territory": "worldwide",
        "start_date": "2026-01-01",
    }
    with grant_owner_access():
        response = client.post("/registry/licenses", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == LICENSE_ID
    assert body["licensee_name"] == "Big Film Studios"


def test_create_license_with_all_fields(client, mock_supabase):
    """POST /registry/licenses accepts all optional fields."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[SAMPLE_LICENSE])

    def _side(name):
        return _work_owner_builder() if name == "works_registry" else builder

    mock_supabase.table.side_effect = _sub_wrap(_side)

    payload = {
        "work_id": WORK_ID,
        "license_type": "sync",
        "licensee_name": "Big Film Studios",
        "licensee_email": "licensing@bigfilm.com",
        "territory": "worldwide",
        "start_date": "2026-01-01",
        "end_date": "2027-01-01",
        "terms": "Non-exclusive sync license",
    }
    with grant_owner_access():
        response = client.post("/registry/licenses", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["licensee_email"] == "licensing@bigfilm.com"
    assert body["end_date"] == "2027-01-01"


def test_create_license_insert_fails_returns_500(client, mock_supabase):
    """POST /registry/licenses returns 500 when insert returns no data."""
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[])

    def _side(name):
        return _work_owner_builder() if name == "works_registry" else builder

    mock_supabase.table.side_effect = _sub_wrap(_side)

    payload = {
        "work_id": WORK_ID,
        "license_type": "sync",
        "licensee_name": "Big Film Studios",
        "territory": "worldwide",
        "start_date": "2026-01-01",
    }
    with grant_owner_access():
        response = client.post("/registry/licenses", json=payload)

    assert response.status_code == 500
    assert "Failed to create license" in response.json()["detail"]


def test_create_license_missing_required_fields(client, mock_supabase):
    """POST /registry/licenses returns 422 when required fields are missing."""
    # Missing start_date which is required
    payload = {
        "work_id": WORK_ID,
        "license_type": "sync",
        "licensee_name": "Big Film Studios",
    }
    response = client.post("/registry/licenses", json=payload)
    assert response.status_code == 422


def test_create_license_default_territory(client, mock_supabase):
    """POST /registry/licenses defaults territory to 'worldwide'."""
    license_worldwide = {**SAMPLE_LICENSE, "territory": "worldwide"}
    builder = MockQueryBuilder()
    builder.execute.return_value = MagicMock(data=[license_worldwide])

    def _side(name):
        return _work_owner_builder() if name == "works_registry" else builder

    mock_supabase.table.side_effect = _sub_wrap(_side)

    payload = {
        "work_id": WORK_ID,
        "license_type": "sync",
        "licensee_name": "Big Film Studios",
        "start_date": "2026-01-01",
        # territory omitted — should default to "worldwide"
    }
    with grant_owner_access():
        response = client.post("/registry/licenses", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["territory"] == "worldwide"


# ============================================================
# Update License
# ============================================================


def _license_update_side(update_data):
    """Side-effect: licensing_rights first returns the row lookup (dict with work_id),
    then the update result. update_license now resolves the work via the row before
    gating, so the first call must be a dict."""
    lookup = MockQueryBuilder()
    lookup.execute.return_value = MagicMock(data={"work_id": WORK_ID})
    update = MockQueryBuilder()
    update.execute.return_value = MagicMock(data=update_data)
    call_count = [0]

    def _side(name):
        if name == "licensing_rights":
            call_count[0] += 1
            return lookup if call_count[0] == 1 else update
        return MockQueryBuilder()

    return _side


def test_update_license_success(client, mock_supabase):
    """PUT /registry/licenses/{license_id} updates and returns the license."""
    updated_license = {**SAMPLE_LICENSE, "licensee_name": "New Studio LLC"}
    mock_supabase.table.side_effect = _sub_wrap(_license_update_side([updated_license]))

    payload = {"licensee_name": "New Studio LLC"}
    with grant_owner_access():
        response = client.put(f"/registry/licenses/{LICENSE_ID}", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["licensee_name"] == "New Studio LLC"


def test_update_license_not_found(client, mock_supabase):
    """PUT /registry/licenses/{license_id} returns 404 when license row not found."""
    lookup = MockQueryBuilder()
    lookup.execute.return_value = MagicMock(data=None)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: lookup)

    payload = {"licensee_name": "New Studio LLC"}
    with grant_owner_access():
        response = client.put(f"/registry/licenses/{LICENSE_ID}", json=payload)

    assert response.status_code == 404
    assert response.json()["detail"] == "License not found"


def test_update_license_status(client, mock_supabase):
    """PUT /registry/licenses/{license_id} can update status field."""
    updated_license = {**SAMPLE_LICENSE, "status": "expired"}
    mock_supabase.table.side_effect = _sub_wrap(_license_update_side([updated_license]))

    payload = {"status": "expired"}
    with grant_owner_access():
        response = client.put(f"/registry/licenses/{LICENSE_ID}", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "expired"


def test_update_license_with_dates(client, mock_supabase):
    """PUT /registry/licenses/{license_id} accepts date fields."""
    updated_license = {**SAMPLE_LICENSE, "end_date": "2028-01-01"}
    mock_supabase.table.side_effect = _sub_wrap(_license_update_side([updated_license]))

    payload = {"end_date": "2028-01-01"}
    with grant_owner_access():
        response = client.put(f"/registry/licenses/{LICENSE_ID}", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["end_date"] == "2028-01-01"


# ============================================================
# Delete License
# ============================================================


def test_delete_license_success(client, mock_supabase):
    """DELETE /registry/licenses/{license_id} returns {"ok": True}."""
    lookup = MockQueryBuilder()
    lookup.execute.return_value = MagicMock(data={"work_id": WORK_ID})
    delete_builder = MockQueryBuilder()
    delete_builder.delete.return_value.eq.return_value.execute.return_value = MagicMock(data=[SAMPLE_LICENSE])
    call_count = [0]

    def _side(name):
        if name == "licensing_rights":
            call_count[0] += 1
            return lookup if call_count[0] == 1 else delete_builder
        return MockQueryBuilder()

    mock_supabase.table.side_effect = _sub_wrap(_side)

    with grant_owner_access():
        response = client.delete(f"/registry/licenses/{LICENSE_ID}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_delete_license_row_not_found_returns_ok(client, mock_supabase):
    """DELETE /registry/licenses/{license_id} returns {"ok": True} when the row is gone
    (service returns None, endpoint still responds ok)."""
    lookup = MockQueryBuilder()
    lookup.execute.return_value = MagicMock(data=None)
    mock_supabase.table.side_effect = _sub_wrap(lambda name: lookup)

    with grant_owner_access():
        response = client.delete(f"/registry/licenses/{LICENSE_ID}")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
