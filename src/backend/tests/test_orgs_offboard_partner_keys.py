"""Keys belong to the org: removal never revokes them, remaining admins are told.
Spec 2026-09-04-partner-portal-api-keys-design.md §3b."""

from unittest.mock import MagicMock, patch

import pytest

from orgs import service
from tests.conftest import MockQueryBuilder

ORG = "00000000-0000-0000-0000-0000000000aa"
MEMBER = "00000000-0000-0000-0000-0000000000ee"
U_ADMIN = "00000000-0000-0000-0000-0000000000d1"
U_OTHER_ADMIN = "00000000-0000-0000-0000-0000000000d3"
U_LEAVER = "00000000-0000-0000-0000-0000000000d2"

# asyncio_mode = "auto" (pyproject) collects the async tests here; an explicit
# pytestmark would also (wrongly) mark the sync ones.


def _db(tables: dict[str, list]):
    """Per-table FIFO of execute() results, shared across every db.table(name)
    call for that table (mirrors test_orgs_service._db_seq). Tables not listed
    return empty data. Each builder records its .eq() calls on `.filters` —
    MockQueryBuilder ignores filters, so without this the status/created_by
    narrowing the feature depends on is untestable."""
    iters = {name: iter(seq) for name, seq in tables.items()}
    builders: dict[str, MockQueryBuilder] = {}

    def _next(it):
        # A seeded Exception is RAISED, so a table can simulate a db failure.
        v = next(it)
        if isinstance(v, Exception):
            raise v
        return v

    def _side(name):
        b = builders.get(name)
        if b is None:
            b = MockQueryBuilder()
            it = iters.get(name)
            b.execute = MagicMock(side_effect=(lambda it=it: _next(it)) if it else (lambda: MagicMock(data=[])))
            # insert is already a MagicMock(return_value=self) on MockQueryBuilder,
            # so the notification test can read its call_args as-is.
            b.filters = []
            b.eq = lambda field, value, _b=b: (_b.filters.append((field, value)), _b)[1]
            builders[name] = b
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db, builders


def test_notifies_each_remaining_admin_once_when_leaver_has_active_keys(monkeypatch):
    monkeypatch.setattr(service, "_org_name", lambda db, org, default: "GNS Music")
    monkeypatch.setattr(service, "_resolve_user_email", lambda db, uid: "leaver@label.test")
    db, b = _db(
        {
            "partner_api_keys": [MagicMock(data=[{"id": "k1"}, {"id": "k2"}])],
            "org_members": [MagicMock(data=[{"user_id": U_ADMIN}, {"user_id": U_OTHER_ADMIN}, {"user_id": U_LEAVER}])],
            "notifications": [MagicMock(data=[])],
        }
    )
    n = service.notify_admins_of_removed_members_keys(db, ORG, U_LEAVER)
    assert n == 2
    rows = b["notifications"].insert.call_args[0][0]
    assert {r["user_id"] for r in rows} == {U_ADMIN, U_OTHER_ADMIN}
    assert all(r["type"] == "confirmation" and r["entity_type"] == "org" for r in rows)
    assert "leaver@label.test created 2 API keys" in rows[0]["message"]
    assert "Teams → API keys" in rows[0]["message"]
    assert rows[0]["metadata"] == {"org_id": ORG, "removed_user_id": U_LEAVER, "active_key_count": 2}
    # The two load-bearing narrowings: only ACTIVE admins are told, and only
    # the leaver's still-ACTIVE keys in THIS org are counted.
    assert ("status", "active") in b["org_members"].filters
    assert set(b["partner_api_keys"].filters) == {("org_id", ORG), ("created_by", U_LEAVER), ("status", "active")}


def test_no_active_keys_writes_nothing():
    db, b = _db({"partner_api_keys": [MagicMock(data=[])]})
    assert service.notify_admins_of_removed_members_keys(db, ORG, U_LEAVER) == 0
    assert "notifications" not in b


def test_never_raises():
    db = MagicMock()
    db.table.side_effect = RuntimeError("db down")
    assert service.notify_admins_of_removed_members_keys(db, ORG, U_LEAVER) == 0
    assert service.notify_admins_of_removed_members_keys(db, ORG, None) == 0


async def _offboard(db, final_status):
    fn = service.remove_member if final_status == "removed" else service.suspend_member
    with patch("orgs.service._revoke_offboarded_member_access"), patch("orgs.service._cancel_topup_if_purchaser"):
        return await fn(db, U_ADMIN, ORG, MEMBER)


LIVE_ORG_ROW = {
    "kind": "enterprise",
    "status": "active",
    "covered_by": None,
    "covered_at": None,
    "archived_at": None,
    "dissolved_at": None,
}


def _member_db(status="active", revoked_at=None, extra_tables=None):
    member = {"id": MEMBER, "org_id": ORG, "user_id": U_LEAVER, "status": status, "revoked_at": revoked_at}
    seq = [MagicMock(data=member)]  # initial maybe_single read
    if not (status in ("suspended", "removed") and revoked_at):
        seq.append(MagicMock(data=[{**member, "status": "removed", "revoked_at": "2026-09-04T00:00:00+00:00"}]))
    # _first_org uses maybe_single(), so .data is a DICT. Seeding a list would
    # make _require_live_org fail OPEN (it returns on any non-dict) and the
    # guard would never run in these tests.
    return _db({"org_members": seq, "organizations": [MagicMock(data=dict(LIVE_ORG_ROW))], **(extra_tables or {})})


async def test_removal_calls_the_notifier(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    calls = []
    monkeypatch.setattr(
        service, "notify_admins_of_removed_members_keys", lambda db, org, uid: calls.append((org, uid)) or 1
    )
    db, _ = _member_db()
    row = await _offboard(db, "removed")
    assert row["status"] == "removed"
    assert calls == [(ORG, U_LEAVER)]


async def test_suspension_never_touches_keys(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    notifier = MagicMock()
    monkeypatch.setattr(service, "notify_admins_of_removed_members_keys", notifier)
    db, b = _member_db()
    await _offboard(db, "suspended")
    notifier.assert_not_called()
    assert "partner_api_keys" not in b


async def test_retried_removal_notifies_nobody(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    notifier = MagicMock()
    monkeypatch.setattr(service, "notify_admins_of_removed_members_keys", notifier)
    db, _ = _member_db(status="removed", revoked_at="2026-09-01T00:00:00+00:00")
    await _offboard(db, "removed")
    notifier.assert_not_called()


async def test_notifier_db_failure_does_not_fail_removal(monkeypatch):
    """End-to-end, through the REAL notifier: _offboard wraps it in no
    try/except, because the notifier swallows its own failures. This drives a
    db failure through it rather than stubbing a raise the caller can't see."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db, _ = _member_db(extra_tables={"partner_api_keys": [RuntimeError("db down")]})
    row = await _offboard(db, "removed")
    assert row["status"] == "removed"


async def test_archived_org_409s_before_any_offboarding(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    from fastapi import HTTPException

    db, _ = _db({"organizations": [MagicMock(data={**LIVE_ORG_ROW, "archived_at": "2026-08-01T00:00:00+00:00"})]})
    with pytest.raises(HTTPException) as exc:
        await _offboard(db, "removed")
    assert exc.value.status_code == 409
