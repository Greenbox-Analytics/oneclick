"""Task 8: token-authorized claim-preview + revoke/decline lifecycle.

Covers:
- get_invite_preview returns ALL the claimant's stakes (via collaborator_id), enforces
  expiry, and enforces email-match (no split/terms leak to a wrong-email account).
- revoke_collaborator removes ACCESS but KEEPS ownership: stakes survive, grants are
  cleared, status -> 'revoked' (no registered->draft revert).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from registry import service
from registry.access import WorkAccess


class FakeTable:
    """Store-backed fake supabase table. Tracks the op EXPLICITLY (insert/update/delete)
    so we never guess by payload shape. single()/maybe_single() return a dict or None."""

    def __init__(self, store, name):
        self.store, self.name = store, name
        self._filters, self._payload, self._op = {}, None, None
        self._single = False

    def select(self, *a, **k):
        return self

    def insert(self, payload):
        self._payload, self._op = payload, "insert"
        return self

    def update(self, payload):
        self._payload, self._op = payload, "update"
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def neq(self, *a):
        return self

    def _matches(self, r):
        return all(r.get(c) == v for c, v in self._filters.items())

    def execute(self):
        if self._op == "insert":
            row = dict(self._payload)
            row.setdefault("id", f"{self.name}_{len(self.store[self.name])}")
            self.store[self.name].append(row)
            return SimpleNamespace(data=[row])
        if self._op == "update":
            updated = []
            for r in self.store[self.name]:
                if self._matches(r):
                    r.update(self._payload)
                    updated.append(r)
            return SimpleNamespace(data=updated)
        if self._op == "delete":
            removed = [r for r in self.store[self.name] if self._matches(r)]
            self.store[self.name] = [r for r in self.store[self.name] if not self._matches(r)]
            return SimpleNamespace(data=removed)
        rows = [r for r in self.store[self.name] if self._matches(r)]
        if self._single:
            return SimpleNamespace(data=rows[0] if rows else None)
        return SimpleNamespace(data=rows)

    def single(self):
        self._single = True
        return self

    def maybe_single(self):
        self._single = True
        return self


def _store_db(store):
    db = MagicMock()
    db.table.side_effect = lambda n: FakeTable(store, n)
    return db


# ============================================================
# Preview
# ============================================================


def test_preview_returns_all_stakes_of_two_stake_collaborator():
    db = MagicMock()
    collab = {
        "id": "c1",
        "work_id": "w1",
        "email": "Marcus@X.com",
        "name": "Marcus",
        "role": "Producer",
        "terms": ["term-a"],
    }
    stakes = [
        {"stake_type": "master", "percentage": 30, "holder_role": "Producer"},
        {"stake_type": "publishing", "percentage": 10, "holder_role": "Producer"},
    ]
    work = {"title": "Track 1", "project_id": "p1", "artist_id": "a1"}

    def table(name):
        t = MagicMock()
        if name == "registry_collaborators":
            t.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = SimpleNamespace(
                data=collab
            )
        elif name == "ownership_stakes":
            t.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(data=stakes)
        elif name == "works_registry":
            t.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = SimpleNamespace(
                data=work
            )
        return t

    db.table.side_effect = table

    with (
        patch.object(service, "is_invite_expired", AsyncMock(return_value=False)),
        patch.object(service, "_resolve_auth_email", return_value="marcus@x.com"),
    ):
        result = asyncio.run(service.get_invite_preview(db, "tok", "user-1"))

    assert len(result["stakes"]) == 2  # via collaborator_id, not first-only stake_id
    assert "collaborator" in result
    assert result["collaborator"]["name"] == "Marcus"
    assert result["work"] == work


def test_preview_email_mismatch_does_not_leak_stakes():
    db = MagicMock()
    collab = {"id": "c1", "work_id": "w1", "email": "invitee@x.com", "name": "Marcus", "role": "Producer"}
    work = {"title": "Track 1"}

    def table(name):
        t = MagicMock()
        if name == "registry_collaborators":
            t.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = SimpleNamespace(
                data=collab
            )
        elif name == "works_registry":
            t.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = SimpleNamespace(
                data=work
            )
        return t

    db.table.side_effect = table

    with (
        patch.object(service, "is_invite_expired", AsyncMock(return_value=False)),
        patch.object(service, "_resolve_auth_email", return_value="someone-else@x.com"),
    ):
        result = asyncio.run(service.get_invite_preview(db, "tok", "user-1"))

    assert result["email_mismatch"] is True
    assert "stakes" not in result  # no leak
    assert "collaborator" not in result  # no leak
    assert result["invite_email"] == "invitee@x.com"


def test_preview_expired_returns_expired_flag():
    db = MagicMock()
    collab = {"id": "c1", "work_id": "w1", "email": "invitee@x.com", "name": "Marcus", "role": "Producer"}

    def table(name):
        t = MagicMock()
        t.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = SimpleNamespace(
            data=collab
        )
        return t

    db.table.side_effect = table

    with patch.object(service, "is_invite_expired", AsyncMock(return_value=True)):
        result = asyncio.run(service.get_invite_preview(db, "tok", "user-1"))

    assert result == {"expired": True}


# ============================================================
# Revoke: remove access, keep ownership
# ============================================================


def test_revoke_keeps_stakes_clears_grants_sets_revoked():
    store = {
        "registry_collaborators": [
            {"id": "c1", "work_id": "w1", "email": "m@x.com", "name": "Marcus", "status": "confirmed"}
        ],
        "ownership_stakes": [
            {"id": "s1", "work_id": "w1", "collaborator_id": "c1", "stake_type": "master", "percentage": 30}
        ],
        "registry_access_grants": [{"id": "g1", "collaborator_id": "c1", "resource_type": "ownership_breakdown"}],
        "works_registry": [{"id": "w1", "user_id": "owner1", "project_id": "p1", "status": "registered"}],
    }
    db = _store_db(store)

    wa = WorkAccess(work_role="owner", can_see_full_ownership=True)
    wa._all_visible = True

    with patch.object(service, "get_work_access", AsyncMock(return_value=wa)):
        result = asyncio.run(service.revoke_collaborator(db, "owner1", "c1"))

    # stake KEPT (revoked producer may still legally own their %)
    assert any(s["id"] == "s1" for s in store["ownership_stakes"]), "stake must survive revoke"
    # grants CLEARED
    assert store["registry_access_grants"] == [], "grants must be cleared on revoke"
    # status revoked
    assert store["registry_collaborators"][0]["status"] == "revoked"
    # work NOT reverted to draft (registered->draft revert removed)
    assert store["works_registry"][0]["status"] == "registered"
    assert result["id"] == "c1"


def test_revoke_denied_without_can_manage():
    store = {
        "registry_collaborators": [
            {"id": "c1", "work_id": "w1", "email": "m@x.com", "name": "Marcus", "status": "confirmed"}
        ],
        "ownership_stakes": [],
        "registry_access_grants": [],
        "works_registry": [{"id": "w1", "user_id": "owner1", "project_id": "p1", "status": "registered"}],
    }
    db = _store_db(store)

    wa = WorkAccess(work_role="viewer")  # can_manage is False

    with patch.object(service, "get_work_access", AsyncMock(return_value=wa)):
        raised = False
        try:
            asyncio.run(service.revoke_collaborator(db, "viewer1", "c1"))
        except PermissionError:
            raised = True
    assert raised, "non-manager must not be able to revoke"
    assert store["registry_collaborators"][0]["status"] == "confirmed"  # untouched
