"""Mock-based tests for registry grants_service.

Uses a small store-backed fake Supabase client: db.table(name) returns a query
object that filters/inserts/updates against an in-memory dict keyed by table name.
"""

import asyncio
from copy import deepcopy

import pytest

from registry import grants_service


class _Query:
    """Chainable fake query over a list of row dicts (a single table's store)."""

    def __init__(self, store, table_name):
        self._store = store  # dict: table_name -> list[dict]
        self._table = table_name
        self._filters = []  # list of (col, value) for == equality
        self._is_null = []  # list of cols required to be None
        self._op = "select"
        self._insert_payload = None
        self._update_payload = None

    # -- query builders --
    def select(self, *_args, **_kwargs):
        self._op = "select"
        return self

    def insert(self, payload):
        self._op = "insert"
        self._insert_payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._update_payload = payload
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, col, value):
        self._filters.append((col, value))
        return self

    def neq(self, col, value):
        self._filters.append(("__neq__" + col, value))
        return self

    def is_(self, col, _null):
        self._is_null.append(col)
        return self

    # -- matching --
    def _matches(self, row):
        for col, value in self._filters:
            if col.startswith("__neq__"):
                if row.get(col[len("__neq__") :]) == value:
                    return False
            elif row.get(col) != value:
                return False
        return all(row.get(col) is None for col in self._is_null)

    def execute(self):
        rows = self._store.setdefault(self._table, [])
        if self._op == "select":
            data = [deepcopy(r) for r in rows if self._matches(r)]
            return _Result(data)
        if self._op == "insert":
            payload = self._insert_payload
            new_rows = payload if isinstance(payload, list) else [payload]
            for r in new_rows:
                rows.append(deepcopy(r))
            return _Result([deepcopy(r) for r in new_rows])
        if self._op == "update":
            updated = []
            for r in rows:
                if self._matches(r):
                    r.update(self._update_payload)
                    updated.append(deepcopy(r))
            return _Result(updated)
        if self._op == "delete":
            kept, removed = [], []
            for r in rows:
                (removed if self._matches(r) else kept).append(r)
            self._store[self._table] = kept
            return _Result([deepcopy(r) for r in removed])
        raise AssertionError(f"unknown op {self._op}")


class _Result:
    def __init__(self, data):
        self.data = data


class _FakeDB:
    def __init__(self, store):
        self._store = store

    def table(self, name):
        return _Query(self._store, name)


def _make_db(**tables):
    return _FakeDB({name: list(rows) for name, rows in tables.items()})


# ---------------------------------------------------------------------------
# add_grant
# ---------------------------------------------------------------------------


def test_add_grant_cross_work_resource_raises():
    # A license row exists but belongs to a DIFFERENT work -> belongs check fails.
    db = _make_db(
        licensing_rights=[{"id": "lic1", "work_id": "OTHER_WORK"}],
        registry_access_grants=[],
    )
    with pytest.raises(ValueError, match="does not belong"):
        asyncio.run(
            grants_service.add_grant(
                db, collaborator_id="c1", work_id="w1", rtype="license", rid="lic1", granted_by="u1"
            )
        )
    # Nothing inserted.
    assert db._store["registry_access_grants"] == []


def test_add_grant_idempotent():
    db = _make_db(
        licensing_rights=[{"id": "lic1", "work_id": "w1"}],
        registry_access_grants=[],
    )
    first = asyncio.run(
        grants_service.add_grant(db, collaborator_id="c1", work_id="w1", rtype="license", rid="lic1", granted_by="u1")
    )
    assert first is True
    assert len(db._store["registry_access_grants"]) == 1

    # Second identical add is a no-op.
    second = asyncio.run(
        grants_service.add_grant(db, collaborator_id="c1", work_id="w1", rtype="license", rid="lic1", granted_by="u1")
    )
    assert second is False
    assert len(db._store["registry_access_grants"]) == 1  # no duplicate


def test_add_grant_invalid_resource_type_raises():
    db = _make_db(registry_access_grants=[])
    with pytest.raises(ValueError, match="bad resource_type"):
        asyncio.run(
            grants_service.add_grant(db, collaborator_id="c1", work_id="w1", rtype="bogus", rid="x", granted_by="u1")
        )
    assert db._store["registry_access_grants"] == []


def test_add_grant_ownership_breakdown_inserts_null_resource():
    db = _make_db(registry_access_grants=[])
    res = asyncio.run(
        grants_service.add_grant(
            db, collaborator_id="c1", work_id="w1", rtype="ownership_breakdown", rid=None, granted_by="u1"
        )
    )
    assert res is True
    row = db._store["registry_access_grants"][0]
    assert row["resource_type"] == "ownership_breakdown"
    assert row["resource_id"] is None


# ---------------------------------------------------------------------------
# remove_grant
# ---------------------------------------------------------------------------


def test_remove_grant_deletes_matching_null_resource():
    db = _make_db(
        registry_access_grants=[
            {"collaborator_id": "c1", "resource_type": "ownership_breakdown", "resource_id": None, "work_id": "w1"},
            {"collaborator_id": "c1", "resource_type": "license", "resource_id": "lic1", "work_id": "w1"},
        ],
    )
    asyncio.run(grants_service.remove_grant(db, collaborator_id="c1", rtype="ownership_breakdown", rid=None))
    remaining = db._store["registry_access_grants"]
    assert len(remaining) == 1
    assert remaining[0]["resource_type"] == "license"


# ---------------------------------------------------------------------------
# set_access_level
# ---------------------------------------------------------------------------


def test_set_access_level_invalid_raises():
    db = _make_db(registry_collaborators=[{"id": "c1", "access_level": "viewer"}])
    with pytest.raises(ValueError, match="bad access_level"):
        asyncio.run(grants_service.set_access_level(db, collaborator_id="c1", access_level="superadmin"))
    # Row unchanged.
    assert db._store["registry_collaborators"][0]["access_level"] == "viewer"


def test_set_access_level_admin_updates_row():
    db = _make_db(registry_collaborators=[{"id": "c1", "access_level": "viewer"}])
    asyncio.run(grants_service.set_access_level(db, collaborator_id="c1", access_level="admin"))
    assert db._store["registry_collaborators"][0]["access_level"] == "admin"


# ---------------------------------------------------------------------------
# set_work_role
# ---------------------------------------------------------------------------


def test_set_work_role_updates_row():
    db = _make_db(registry_collaborators=[{"id": "c1", "role": "Songwriter"}])
    asyncio.run(grants_service.set_work_role(db, collaborator_id="c1", role="Producer"))
    assert db._store["registry_collaborators"][0]["role"] == "Producer"


def test_set_work_role_trims_whitespace():
    db = _make_db(registry_collaborators=[{"id": "c1", "role": "Songwriter"}])
    asyncio.run(grants_service.set_work_role(db, collaborator_id="c1", role="  Producer  "))
    assert db._store["registry_collaborators"][0]["role"] == "Producer"


def test_set_work_role_blank_raises():
    db = _make_db(registry_collaborators=[{"id": "c1", "role": "Songwriter"}])
    with pytest.raises(ValueError, match="role required"):
        asyncio.run(grants_service.set_work_role(db, collaborator_id="c1", role="   "))
    # Row unchanged.
    assert db._store["registry_collaborators"][0]["role"] == "Songwriter"


# ---------------------------------------------------------------------------
# get_grant_matrix
# ---------------------------------------------------------------------------


def test_get_grant_matrix_groups_by_collaborator_and_excludes_revoked():
    db = _make_db(
        registry_collaborators=[
            {
                "id": "c1",
                "name": "A",
                "email": "a@x.com",
                "role": "writer",
                "access_level": "viewer",
                "status": "confirmed",
                "work_id": "w1",
            },
            {
                "id": "c2",
                "name": "B",
                "email": "b@x.com",
                "role": "writer",
                "access_level": "admin",
                "status": "revoked",
                "work_id": "w1",
            },
        ],
        registry_access_grants=[
            {"collaborator_id": "c1", "resource_type": "license", "resource_id": "lic1", "work_id": "w1"},
            {"collaborator_id": "c1", "resource_type": "agreement", "resource_id": "ag1", "work_id": "w1"},
        ],
    )
    matrix = asyncio.run(grants_service.get_grant_matrix(db, "w1"))
    # Revoked collaborator excluded.
    assert [c["id"] for c in matrix["collaborators"]] == ["c1"]
    assert len(matrix["grants_by_collaborator"]["c1"]) == 2
