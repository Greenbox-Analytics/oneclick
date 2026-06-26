import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from registry import service


class FakeTable:
    def __init__(self, store, name):
        self.store, self.name = store, name
        self._filters, self._payload, self._op = {}, None, None
        self._single = False

    def select(self, *a, **k):
        return self

    # Track the operation EXPLICITLY — do NOT guess insert-vs-update by whether the payload
    # contains "id" (real updates put the id in .eq(), not the payload).
    def insert(self, payload):
        self._payload, self._op = payload, "insert"
        return self

    def update(self, payload):
        self._payload, self._op = payload, "update"
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def neq(self, *a):
        return self

    def execute(self):
        if self._op == "insert":
            row = dict(self._payload)
            row.setdefault("id", f"{self.name}_{len(self.store[self.name])}")
            self.store[self.name].append(row)
            return SimpleNamespace(data=[row])
        if self._op == "update":
            updated = []
            for r in self.store[self.name]:
                if all(r.get(c) == v for c, v in self._filters.items()):
                    r.update(self._payload)
                    updated.append(r)
            return SimpleNamespace(data=updated)
        rows = [r for r in self.store[self.name] if all(r.get(c) == v for c, v in self._filters.items())]
        # single()/maybe_single() expect a single dict (or None), not a list — access.py does
        # work.data["user_id"], so the fake must mirror the real Supabase shape here.
        if self._single:
            return SimpleNamespace(data=rows[0] if rows else None)
        return SimpleNamespace(data=rows)

    def single(self):
        self._single = True
        return self

    def maybe_single(self):
        self._single = True
        return self


def _store_db(work_owner="owner1", stakes=None):
    store = {
        "works_registry": [{"id": "w1", "user_id": work_owner, "project_id": "p1", "status": "draft"}],
        "project_members": [],
        "registry_collaborators": [],
        "ownership_stakes": list(stakes or []),
        "registry_access_grants": [],
    }
    db = MagicMock()
    db.table.side_effect = lambda n: FakeTable(store, n)
    return db, store


def test_invite_with_stakes_links_every_stake():
    db, store = _store_db()
    body = SimpleNamespace(
        work_id="w1",
        email="m@x.com",
        name="Marcus",
        role="Producer",
        stakes=[
            SimpleNamespace(stake_type="master", percentage=30),
            SimpleNamespace(stake_type="publishing", percentage=10),
        ],
        notes=None,
    )
    asyncio.run(service.invite_with_stakes(db, "owner1", body))
    collab_id = store["registry_collaborators"][0]["id"]
    created = [s for s in store["ownership_stakes"] if s.get("collaborator_id") == collab_id]
    assert len(created) == 2  # BOTH stakes linked, not just the first
    assert all(s["user_id"] == "owner1" for s in created)  # user_id = work owner


def test_reinvite_after_revoke_does_not_duplicate_stakes():
    db, store = _store_db()
    body = SimpleNamespace(
        work_id="w1",
        email="m@x.com",
        name="Marcus",
        role="Producer",
        stakes=[SimpleNamespace(stake_type="master", percentage=30)],
        notes=None,
    )
    asyncio.run(service.invite_with_stakes(db, "owner1", body))
    # simulate revoke = keep stake, status revoked (Task 8 semantics)
    store["registry_collaborators"][0]["status"] = "revoked"
    # re-invite same email
    asyncio.run(service.invite_with_stakes(db, "owner1", body))
    masters = [s for s in store["ownership_stakes"] if s["stake_type"] == "master"]
    assert len(masters) == 1  # upserted, NOT duplicated
    assert len(store["registry_collaborators"]) == 1  # row reactivated, not re-inserted
    assert store["registry_collaborators"][0]["status"] == "invited"
