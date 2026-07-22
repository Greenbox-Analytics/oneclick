"""Mock-based tests for orgs.service core CRUD + archive (Licensing Phase B,
Task 2) and invites/roles/offboarding (Task 3). Mirrors
tests/test_teams_service.py + tests/test_teams_invites.py's idioms."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from orgs import projects as org_projects
from orgs import service
from tests.conftest import MockQueryBuilder

U1 = "00000000-0000-0000-0000-000000000001"
U2 = "00000000-0000-0000-0000-000000000002"
ORG = "20000000-0000-0000-0000-000000000001"
EXISTING = "00000000-0000-0000-0000-000000000099"
MEMBER = "40000000-0000-0000-0000-000000000001"
TOKEN = "30000000-0000-0000-0000-000000000001"
SEAT_WALLET = "50000000-0000-0000-0000-000000000001"
POOL_WALLET = "50000000-0000-0000-0000-000000000002"


def _db_seq(seqs):
    """seqs: dict table_name -> list of execute() return values, consumed in
    call order (mirrors tests/test_teams_invites.py's helper of the same
    name). rpc() always returns a fresh MagicMock configurable via
    db.rpc.return_value.execute.return_value / .side_effect."""
    counters = {k: 0 for k in seqs}

    def _side(name):
        b = MockQueryBuilder()
        if name in seqs:
            i = min(counters[name], len(seqs[name]) - 1)
            counters[name] += 1
            b.execute.return_value = seqs[name][i]
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


# ---------------------------------------------------------------------------
# create_org
# ---------------------------------------------------------------------------


async def test_create_org_returns_pending_org_with_admin_role():
    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(
                data=[{"id": ORG, "name": "Acme", "created_by": U1, "status": "pending"}], count=1
            )
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.create_org(db, U1, "Acme")
    assert result["id"] == ORG
    assert result["status"] == "pending"
    assert result["my_role"] == "admin"


async def test_create_org_does_not_insert_org_members():
    """The auto_create_org_admin DB trigger adds the creator's admin row
    atomically with the org insert — create_org must NEVER write org_members
    itself (a hand-written insert here would duplicate/race the trigger and
    break the "an org can never exist without an admin, in ONE write"
    invariant the migration relies on)."""
    tables_inserted = []

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=[{"id": ORG, "name": "Acme", "created_by": U1}], count=1)
        original_insert = b.insert

        def _capture_insert(payload, *a, **kw):
            tables_inserted.append(name)
            return original_insert(payload, *a, **kw)

        b.insert = _capture_insert
        return b

    db = MagicMock()
    db.table.side_effect = _side
    await service.create_org(db, U1, "Acme")
    assert tables_inserted == ["organizations"]
    assert "org_members" not in tables_inserted


async def test_create_org_raises_when_insert_returns_nothing():
    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    with pytest.raises(RuntimeError):
        await service.create_org(db, U1, "Acme")


# ---------------------------------------------------------------------------
# list_my_orgs
# ---------------------------------------------------------------------------


async def test_list_my_orgs_attaches_role_and_status():
    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=[{"org_id": ORG, "role": "admin", "status": "active"}], count=1)
        elif name == "organizations":
            b.execute.return_value = MagicMock(data=[{"id": ORG, "name": "Acme"}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    orgs = await service.list_my_orgs(db, U1)
    assert orgs[0]["my_role"] == "admin"
    assert orgs[0]["my_status"] == "active"


async def test_list_my_orgs_excludes_removed_memberships():
    """Query must filter status != 'removed' (spec §4: a removed seat is not
    membership); asserted directly against the neq() call the service makes."""
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            original_neq = b.neq

            def _capture_neq(field, value):
                captured["neq"] = (field, value)
                return original_neq(field, value)

            b.neq = _capture_neq
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    await service.list_my_orgs(db, U1)
    assert captured["neq"] == ("status", "removed")


async def test_list_my_orgs_empty_when_no_memberships():
    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    assert await service.list_my_orgs(db, U1) == []


# ---------------------------------------------------------------------------
# get_org
# ---------------------------------------------------------------------------


async def test_get_org_requires_membership_404(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: False)
    db = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        await service.get_org(db, U2, ORG)
    assert exc_info.value.status_code == 404


async def test_get_org_computes_remaining_to_activate_with_partial_purchase(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    monkeypatch.delenv("ENTERPRISE_MIN_INITIAL_CREDITS", raising=False)

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(
                data={"id": ORG, "status": "pending", "min_initial_purchase_credits": None}, count=1
            )
        elif name == "org_members":
            b.execute.return_value = MagicMock(data={"role": "admin"}, count=1)
        elif name == "credit_wallets":
            b.execute.return_value = MagicMock(
                data=[{"id": "w1", "bundle_balance": 0, "reserve_balance": 4000}], count=1
            )
        elif name == "credit_ledger":
            b.execute.return_value = MagicMock(data=[{"delta": 4000}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.get_org(db, U1, ORG)
    assert result["cumulative_purchased"] == 4000
    assert result["pool_balance"] == 4000
    # No env override in this test → platform default of 10,000.
    assert result["remaining_to_activate"] == 6000


async def test_get_org_respects_custom_env_default_minimum(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    monkeypatch.setenv("ENTERPRISE_MIN_INITIAL_CREDITS", "500")

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(
                data={"id": ORG, "status": "pending", "min_initial_purchase_credits": None}, count=1
            )
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=None, count=0)
        elif name == "credit_wallets":
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.get_org(db, U1, ORG)
    assert result["remaining_to_activate"] == 500


async def test_get_org_uses_org_specific_minimum_over_env_default(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    monkeypatch.setenv("ENTERPRISE_MIN_INITIAL_CREDITS", "10000")

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(
                data={"id": ORG, "status": "pending", "min_initial_purchase_credits": 2000}, count=1
            )
        elif name == "org_members":
            b.execute.return_value = MagicMock(data={"role": "member"}, count=1)
        elif name == "credit_wallets":
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.get_org(db, U1, ORG)
    assert result["pool_balance"] == 0
    assert result["cumulative_purchased"] == 0
    assert result["remaining_to_activate"] == 2000


async def test_get_org_zero_balance_when_no_wallet(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(
                data={"id": ORG, "status": "pending", "min_initial_purchase_credits": None}, count=1
            )
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=None, count=0)
        elif name == "credit_wallets":
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.get_org(db, U1, ORG)
    assert result["pool_balance"] == 0
    assert result["cumulative_purchased"] == 0
    assert result["my_role"] is None


async def test_get_org_raises_when_row_missing_after_membership_check(monkeypatch):
    """Defensive: authz passed (membership row exists) but the org row itself
    is somehow gone. Should surface as 404, not a crash."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=None, count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    with pytest.raises(ValueError):
        await service.get_org(db, U1, ORG)


# ---------------------------------------------------------------------------
# update_org
# ---------------------------------------------------------------------------


async def test_update_org_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    db = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        await service.update_org(db, U2, ORG, {"name": "New"})
    assert exc_info.value.status_code == 403


async def test_update_org_updates_fields(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            original_update = b.update

            def _capture(payload, *a, **kw):
                captured["payload"] = payload
                return original_update(payload, *a, **kw)

            b.update = _capture
            b.execute.return_value = MagicMock(data=[{"id": ORG, "name": "New"}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.update_org(db, U1, ORG, {"name": "New"})
    assert captured["payload"] == {"name": "New"}
    assert result["name"] == "New"


async def test_update_org_clears_default_seat_allowance_with_explicit_null(monkeypatch):
    """An explicit null in the request must WRITE null (manual-only), distinct
    from an omitted field (which the router never forwards, via
    model_dump(exclude_unset=True))."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            original_update = b.update

            def _capture(payload, *a, **kw):
                captured["payload"] = payload
                return original_update(payload, *a, **kw)

            b.update = _capture
            b.execute.return_value = MagicMock(data=[{"id": ORG, "default_seat_allowance": None}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.update_org(db, U1, ORG, {"default_seat_allowance": None})
    assert captured["payload"] == {"default_seat_allowance": None}
    assert result["default_seat_allowance"] is None


async def test_update_org_noop_returns_current_row_when_no_fields(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data={"id": ORG, "name": "Acme"}, count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.update_org(db, U1, ORG, {})
    assert result == {"id": ORG, "name": "Acme"}


# ---------------------------------------------------------------------------
# archive_org
# ---------------------------------------------------------------------------


async def test_archive_org_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    db = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        await service.archive_org(db, U2, ORG)
    assert exc_info.value.status_code == 403


async def test_archive_org_409_when_a_seat_balance_is_nonzero(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=[{"id": "m1"}, {"id": "m2"}], count=2)
        elif name == "credit_wallets":
            b.execute.return_value = MagicMock(
                data=[{"id": "w1", "owner_id": "m1", "bundle_balance": 0, "reserve_balance": 50}], count=1
            )
        return b

    db = MagicMock()
    db.table.side_effect = _side
    with pytest.raises(service.SeatBalanceNotZeroError):
        await service.archive_org(db, U1, ORG)


async def test_archive_org_success_when_all_seat_balances_zero(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=[{"id": "m1"}], count=1)
        elif name == "credit_wallets":
            b.execute.return_value = MagicMock(
                data=[{"id": "w1", "owner_id": "m1", "bundle_balance": 0, "reserve_balance": 0}], count=1
            )
        elif name == "organizations":
            b.execute.return_value = MagicMock(data=[{"id": ORG, "archived_at": "2026-07-20T00:00:00Z"}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.archive_org(db, U1, ORG)
    assert result["archived_at"] is not None


async def test_archive_org_success_when_no_members(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=[], count=0)
        elif name == "organizations":
            b.execute.return_value = MagicMock(data=[{"id": ORG, "archived_at": "2026-07-20T00:00:00Z"}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.archive_org(db, U1, ORG)
    assert result["archived_at"] is not None


# ---------------------------------------------------------------------------
# archive_org — Task 4 teardown (rule 12): revoke org-granted memberships AND
# delete this org's org_project_links rows, after the archive lands.
# ---------------------------------------------------------------------------


def _archive_success_side(name):
    b = MockQueryBuilder()
    if name == "org_members":
        b.execute.return_value = MagicMock(data=[], count=0)
    elif name == "organizations":
        b.execute.return_value = MagicMock(data=[{"id": ORG, "archived_at": "2026-07-20T00:00:00Z"}], count=1)
    return b


async def test_archive_org_revokes_org_granted_memberships_org_scoped(monkeypatch):
    """Rule 12: after archiving, ALL of the org's granted memberships are
    revoked — org-scoped only (no user_id/project_id narrowing), since an
    archived org loses every grant it ever made, on every project."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = MagicMock()
    db.table.side_effect = _archive_success_side
    fake_revoke = MagicMock(return_value=3)
    monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", fake_revoke)

    result = await service.archive_org(db, U1, ORG)

    assert result["archived_at"] is not None
    fake_revoke.assert_called_once_with(db, ORG)


async def test_archive_org_deletes_org_project_links_rows(monkeypatch):
    """Rule 12 — archive is an UPDATE, so the ON DELETE CASCADE on
    org_project_links.org_id never fires; archive_org must explicitly DELETE
    this org's link row(s), scoped by org_id only, so a stranded link never
    blocks re-linking the project to a different (live) org afterward."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", lambda *a, **k: 0)

    links_builder = MockQueryBuilder()

    def _side(name):
        if name == "org_project_links":
            return links_builder
        return _archive_success_side(name)

    db = MagicMock()
    db.table.side_effect = _side

    result = await service.archive_org(db, U1, ORG)

    assert result["archived_at"] is not None
    links_builder.delete.assert_called_once()
    links_builder.delete.return_value.eq.assert_called_once_with("org_id", ORG)


async def test_archive_org_teardown_failures_do_not_block_archive(monkeypatch):
    """Both cleanup steps are best-effort — a failure in the revoke call OR
    the link delete must not prevent archive_org from returning the archived
    org row; the archived_at write has already landed by the time either
    runs."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(
        org_projects, "revoke_org_granted_memberships", MagicMock(side_effect=RuntimeError("revoke boom"))
    )

    def _side(name):
        if name == "org_project_links":
            b = MockQueryBuilder()
            b.delete.side_effect = RuntimeError("link delete boom")
            return b
        return _archive_success_side(name)

    db = MagicMock()
    db.table.side_effect = _side

    result = await service.archive_org(db, U1, ORG)
    assert result["archived_at"] is not None


async def test_archive_org_organic_and_other_org_rows_survive(monkeypatch):
    """End-to-end through the REAL revoke_org_granted_memberships (not
    mocked): the delete is filtered on org_id only, so organic rows
    (org_id NULL) and rows granted by a DIFFERENT org are never targeted by
    archive_org's teardown — asserted the same way
    test_org_projects.py::TestUnlinkProject does, by inspecting the exact
    .eq() filter args passed to project_members.delete()."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

    pm_builder = MockQueryBuilder()
    pm_builder.delete.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[{"id": "m1", "org_id": ORG}, {"id": "m2", "org_id": ORG}], count=2
    )

    def _side(name):
        if name == "project_members":
            return pm_builder
        return _archive_success_side(name)

    db = MagicMock()
    db.table.side_effect = _side

    result = await service.archive_org(db, U1, ORG)

    assert result["archived_at"] is not None
    pm_builder.delete.return_value.eq.assert_called_once_with("org_id", ORG)


# ---------------------------------------------------------------------------
# invite_member (Task 3)
# ---------------------------------------------------------------------------


async def test_invite_member_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.invite_member(MagicMock(), U2, ORG, "x@example.com", "member")
    assert exc_info.value.status_code == 403


async def test_invite_member_invalid_role_raises(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: None)
    with pytest.raises(ValueError):
        await service.invite_member(MagicMock(), U1, ORG, "x@example.com", "owner")


async def test_invite_existing_active_member_raises_duplicate(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: EXISTING)
    with pytest.raises(service.DuplicateInviteError):
        await service.invite_member(_db_seq({}), U1, ORG, "x@example.com", "member")


async def test_invite_previously_removed_member_is_allowed_not_duplicate(monkeypatch):
    """is_org_member only counts ACTIVE seats — re-inviting a suspended/removed
    member must NOT be flagged as a duplicate (rule 13's re-invite path);
    accept_invite is what reactivates the row."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: False)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: EXISTING)
    db = _db_seq(
        {
            "pending_org_invites": [
                MagicMock(data=None, count=0),
                MagicMock(
                    data=[{"id": "i1", "token": TOKEN, "email": "removed@example.com", "role": "member"}], count=1
                ),
            ]
        }
    )
    result = await service.invite_member(db, U1, ORG, "removed@example.com", "member")
    assert result["type"] == "invited"


async def test_invite_fresh_email_inserts_pending(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: None)
    db = _db_seq(
        {
            "pending_org_invites": [
                MagicMock(data=None, count=0),
                MagicMock(data=[{"id": "i1", "token": TOKEN, "email": "new@example.com", "role": "member"}], count=1),
            ]
        }
    )
    result = await service.invite_member(db, U1, ORG, "new@example.com", "member")
    assert result["type"] == "invited"
    assert result["notify_user_id"] is None
    assert result["invite"]["token"] == TOKEN


async def test_invite_existing_invite_updates_row(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: None)
    db = _db_seq(
        {
            "pending_org_invites": [
                MagicMock(data={"id": "i1"}, count=1),
                MagicMock(data=[{"id": "i1", "token": TOKEN, "status": "pending", "role": "admin"}], count=1),
            ]
        }
    )
    result = await service.invite_member(db, U1, ORG, "back@example.com", "admin")
    assert result["invite"]["status"] == "pending"
    assert result["invite"]["role"] == "admin"


# ---------------------------------------------------------------------------
# get_pending_invites / cancel_invite
# ---------------------------------------------------------------------------


async def test_get_pending_invites_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.get_pending_invites(MagicMock(), U2, ORG)
    assert exc_info.value.status_code == 403


async def test_get_pending_invites_filters_status_pending(monkeypatch):
    """Unlike teams' get_pending_invites (unfiltered), the org version filters
    status='pending' explicitly — asserted directly against the eq() call."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "pending_org_invites":
            original_eq = b.eq

            def _capture_eq(field, value):
                captured.setdefault("eq_calls", []).append((field, value))
                return original_eq(field, value)

            b.eq = _capture_eq
            b.execute.return_value = MagicMock(data=[{"id": "i1", "status": "pending"}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.get_pending_invites(db, U1, ORG)
    assert result == [{"id": "i1", "status": "pending"}]
    assert ("status", "pending") in captured["eq_calls"]


async def test_cancel_invite_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.cancel_invite(MagicMock(), U2, ORG, "i1")
    assert exc_info.value.status_code == 403


async def test_cancel_invite_deletes_row(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    result = await service.cancel_invite(MagicMock(), U1, ORG, "i1")
    assert result == {"deleted": "i1"}


# ---------------------------------------------------------------------------
# accept_invite / decline_invite
# ---------------------------------------------------------------------------


def _pending_invite(**overrides):
    base = {
        "id": "i1",
        "org_id": ORG,
        "email": "u@example.com",
        "role": "member",
        "status": "pending",
        "expires_at": "2999-01-01T00:00:00+00:00",
        "invited_by": U1,
    }
    base.update(overrides)
    return base


async def test_accept_invite_not_found_raises():
    db = _db_seq({"pending_org_invites": [MagicMock(data=None, count=0)]})
    with pytest.raises(ValueError):
        await service.accept_invite(db, U2, "u@example.com", TOKEN)


async def test_accept_invite_email_mismatch_raises():
    db = _db_seq({"pending_org_invites": [MagicMock(data=_pending_invite(), count=1)]})
    with pytest.raises(PermissionError):
        await service.accept_invite(db, U2, "intruder@example.com", TOKEN)


async def test_accept_invite_already_accepted_short_circuits():
    db = _db_seq({"pending_org_invites": [MagicMock(data=_pending_invite(status="accepted"), count=1)]})
    result = await service.accept_invite(db, U2, "u@example.com", TOKEN)
    assert result == {"type": "already_accepted", "org_id": ORG}


async def test_accept_invite_declined_raises_invalid():
    db = _db_seq({"pending_org_invites": [MagicMock(data=_pending_invite(status="declined"), count=1)]})
    with pytest.raises(service.InviteInvalidError):
        await service.accept_invite(db, U2, "u@example.com", TOKEN)


async def test_accept_invite_expired_raises():
    db = _db_seq(
        {"pending_org_invites": [MagicMock(data=_pending_invite(expires_at="2000-01-01T00:00:00+00:00"), count=1)]}
    )
    with pytest.raises(service.InviteInvalidError):
        await service.accept_invite(db, U2, "u@example.com", TOKEN)


async def test_accept_invite_fresh_inserts_member_and_sets_billing_context():
    """KEY TEST: accepting a fresh invite (1) inserts an active org_members
    row, carrying the invite's email (licensing follow-ups Task 4 — powers
    get_org_usage's rollup without a per-row auth lookup), and (2) sets the
    accepter's billing_context_org_id to this org (spec §5 default-context
    rule)."""
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "pending_org_invites":
            b.execute.return_value = MagicMock(data=_pending_invite(), count=1)
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=None, count=0)  # no existing row
            original_insert = b.insert

            def _capture_insert(payload, *a, **kw):
                captured["org_members_insert"] = payload
                return original_insert(payload, *a, **kw)

            b.insert = _capture_insert
        elif name == "profiles":
            original_update = b.update

            def _capture_update(payload, *a, **kw):
                captured["profiles_update"] = payload
                return original_update(payload, *a, **kw)

            b.update = _capture_update
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.accept_invite(db, U2, "u@example.com", TOKEN)
    assert result == {"type": "accepted", "org_id": ORG}
    assert captured["org_members_insert"] == {
        "org_id": ORG,
        "user_id": U2,
        "role": "member",
        "status": "active",
        "invited_by": U1,
        "email": "u@example.com",
    }
    assert captured["profiles_update"] == {"billing_context_org_id": ORG}


async def test_accept_invite_reactivates_removed_row_instead_of_inserting():
    """KEY TEST: a REMOVED org_members row for (org, user) must be
    REACTIVATED (UPDATE), never re-inserted — UNIQUE(org_id, user_id) makes a
    fresh INSERT impossible, and reactivation IS the designed re-invite path
    (rule 13). The reactivation also (re)writes the invite's email onto the
    row (licensing follow-ups Task 4)."""
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "pending_org_invites":
            b.execute.return_value = MagicMock(data=_pending_invite(role="admin"), count=1)
        elif name == "org_members":
            b.execute.return_value = MagicMock(
                data={
                    "id": "m1",
                    "org_id": ORG,
                    "user_id": U2,
                    "status": "removed",
                    "revoked_at": "2026-07-01T00:00:00+00:00",
                },
                count=1,
            )
            original_update = b.update
            original_insert = b.insert

            def _capture_update(payload, *a, **kw):
                captured.setdefault("org_members_updates", []).append(payload)
                return original_update(payload, *a, **kw)

            def _capture_insert(payload, *a, **kw):
                captured.setdefault("org_members_inserts", []).append(payload)
                return original_insert(payload, *a, **kw)

            b.update = _capture_update
            b.insert = _capture_insert
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.accept_invite(db, U2, "u@example.com", TOKEN)
    assert result == {"type": "accepted", "org_id": ORG}
    assert "org_members_inserts" not in captured
    update_payload = captured["org_members_updates"][0]
    assert update_payload["status"] == "active"
    assert update_payload["revoked_at"] is None
    assert update_payload["role"] == "admin"
    assert update_payload["email"] == "u@example.com"


async def test_accept_invite_reactivates_suspended_row_instead_of_inserting():
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "pending_org_invites":
            b.execute.return_value = MagicMock(data=_pending_invite(), count=1)
        elif name == "org_members":
            b.execute.return_value = MagicMock(
                data={
                    "id": "m1",
                    "org_id": ORG,
                    "user_id": U2,
                    "status": "suspended",
                    "revoked_at": "2026-07-01T00:00:00+00:00",
                },
                count=1,
            )
            original_update = b.update
            original_insert = b.insert

            def _capture_update(payload, *a, **kw):
                captured.setdefault("org_members_updates", []).append(payload)
                return original_update(payload, *a, **kw)

            def _capture_insert(payload, *a, **kw):
                captured.setdefault("org_members_inserts", []).append(payload)
                return original_insert(payload, *a, **kw)

            b.update = _capture_update
            b.insert = _capture_insert
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.accept_invite(db, U2, "u@example.com", TOKEN)
    assert result == {"type": "accepted", "org_id": ORG}
    assert "org_members_inserts" not in captured
    assert captured["org_members_updates"][0]["status"] == "active"
    assert captured["org_members_updates"][0]["email"] == "u@example.com"


async def test_accept_invite_leaves_already_active_row_untouched():
    captured = {"writes": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "pending_org_invites":
            b.execute.return_value = MagicMock(data=_pending_invite(), count=1)
        elif name == "org_members":
            b.execute.return_value = MagicMock(data={"id": "m1", "status": "active"}, count=1)
            original_update = b.update
            original_insert = b.insert

            def _capture_update(payload, *a, **kw):
                captured["writes"] += 1
                return original_update(payload, *a, **kw)

            def _capture_insert(payload, *a, **kw):
                captured["writes"] += 1
                return original_insert(payload, *a, **kw)

            b.update = _capture_update
            b.insert = _capture_insert
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.accept_invite(db, U2, "u@example.com", TOKEN)
    assert result == {"type": "accepted", "org_id": ORG}
    assert captured["writes"] == 0


async def test_decline_invite_not_found_raises():
    db = _db_seq({"pending_org_invites": [MagicMock(data=None, count=0)]})
    with pytest.raises(ValueError):
        await service.decline_invite(db, U2, "u@example.com", TOKEN)


async def test_decline_invite_email_mismatch_raises():
    db = _db_seq({"pending_org_invites": [MagicMock(data=_pending_invite(), count=1)]})
    with pytest.raises(PermissionError):
        await service.decline_invite(db, U2, "intruder@example.com", TOKEN)


async def test_decline_invite_success():
    db = _db_seq({"pending_org_invites": [MagicMock(data=_pending_invite(), count=1)]})
    result = await service.decline_invite(db, U2, "u@example.com", TOKEN)
    assert result == {"type": "declined", "org_id": ORG}


# ---------------------------------------------------------------------------
# update_member_role
# ---------------------------------------------------------------------------


async def test_update_member_role_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.update_member_role(MagicMock(), U2, ORG, MEMBER, "admin")
    assert exc_info.value.status_code == 403


async def test_update_member_role_invalid_role_raises(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    with pytest.raises(ValueError):
        await service.update_member_role(MagicMock(), U1, ORG, MEMBER, "owner")


async def test_update_member_role_maps_last_admin_db_error(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.side_effect = Exception("You are the only admin of this organization")
        return b

    db = MagicMock()
    db.table.side_effect = _side
    with pytest.raises(service.LastAdminError):
        await service.update_member_role(db, U1, ORG, MEMBER, "member")


async def test_update_member_role_success(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=[{"id": MEMBER, "role": "admin"}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.update_member_role(db, U1, ORG, MEMBER, "admin")
    assert result["role"] == "admin"


# ---------------------------------------------------------------------------
# Offboarding: suspend_member / remove_member (_offboard) — spec rule 5 + 13
# ---------------------------------------------------------------------------


def _member_row(**overrides):
    base = {"id": MEMBER, "org_id": ORG, "user_id": U2, "role": "member", "status": "active", "revoked_at": None}
    base.update(overrides)
    return base


async def test_offboard_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.suspend_member(MagicMock(), U2, ORG, MEMBER)
    assert exc_info.value.status_code == 403


async def test_offboard_member_not_found_raises(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"org_members": [MagicMock(data=None, count=0)]})
    with pytest.raises(ValueError):
        await service.suspend_member(db, U1, ORG, MEMBER)


async def test_offboard_maps_last_admin_db_error(monkeypatch):
    """Covers BOTH "removing another member who's the last admin" and
    "self-offboarding as the last admin" — the DB guard fires identically
    either way; the service layer doesn't distinguish."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    calls = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            calls["n"] += 1
            if calls["n"] == 1:
                b.execute.return_value = MagicMock(data=_member_row(role="admin"), count=1)
            else:
                b.execute.side_effect = Exception("You are the only admin of this organization")
        return b

    db = MagicMock()
    db.table.side_effect = _side
    with pytest.raises(service.LastAdminError):
        await service.remove_member(db, U1, ORG, MEMBER)


async def test_offboard_zero_balance_skips_rpc_entirely(monkeypatch):
    """KEY TEST: a zero (or missing) seat wallet balance must skip the
    transfer_credits RPC entirely — nothing to reclaim."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="suspended", revoked_at=revoked_at)], count=1),
                MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at), count=1),
            ],
            "credit_wallets": [MagicMock(data=[], count=0)],  # no seat wallet
        }
    )
    result = await service.suspend_member(db, U1, ORG, MEMBER)
    assert result["status"] == "suspended"
    db.rpc.assert_not_called()


async def test_offboard_negative_balance_skips_rpc_entirely(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="suspended", revoked_at=revoked_at)], count=1),
                MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at), count=1),
            ],
            "credit_wallets": [
                MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": -50, "reserve_balance": 0}], count=1)
            ],
        }
    )
    result = await service.suspend_member(db, U1, ORG, MEMBER)
    assert result["status"] == "suspended"
    db.rpc.assert_not_called()


async def test_offboard_nonzero_balance_calls_transfer_with_stored_epoch_key(monkeypatch):
    """KEY TEST: the reclaim's request_id is derived from the STORED
    (reread) revoked_at, exactly `offboard:{member_id}:{epoch}`."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"
    expected_epoch = int(datetime.fromisoformat(revoked_at).timestamp())
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="suspended", revoked_at=revoked_at)], count=1),
                MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at), count=1),
            ],
            "credit_wallets": [
                MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 500}], count=1),
                MagicMock(data=[{"id": POOL_WALLET}], count=1),
            ],
        }
    )
    db.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False})
    result = await service.suspend_member(db, U1, ORG, MEMBER)
    assert result["status"] == "suspended"
    name, params = db.rpc.call_args[0]
    assert name == "transfer_credits"
    assert params["p_from_wallet"] == SEAT_WALLET
    assert params["p_to_wallet"] == POOL_WALLET
    assert params["p_amount"] == 500
    assert params["p_kind"] == "reclaim"
    assert params["p_request_id"] == f"offboard:{MEMBER}:{expected_epoch}"


async def test_offboard_retry_single_call_reuses_stored_key():
    """KEY TEST: if org_members is ALREADY at final_status with revoked_at
    set (simulating a retry landing after a prior attempt's status write
    succeeded), the UPDATE/reread branch is skipped entirely and the STORED
    revoked_at from the initial SELECT is reused for the request_id."""
    from unittest.mock import patch

    with patch.object(service.authz, "is_org_admin", lambda *a: True):
        revoked_at = "2026-07-20T12:00:00+00:00"
        expected_epoch = int(datetime.fromisoformat(revoked_at).timestamp())
        db = _db_seq(
            {
                "org_members": [MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at), count=1)],
                "credit_wallets": [
                    MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 500}], count=1),
                    MagicMock(data=[{"id": POOL_WALLET}], count=1),
                ],
            }
        )
        db.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": True})
        await service.suspend_member(db, U1, ORG, MEMBER)
        params = db.rpc.call_args[0][1]
        assert params["p_request_id"] == f"offboard:{MEMBER}:{expected_epoch}"
        # Only the initial current-state SELECT — no UPDATE/reread.
        org_members_calls = [c for c in db.table.call_args_list if c.args[0] == "org_members"]
        assert len(org_members_calls) == 1


async def test_offboard_full_retry_produces_identical_request_id(monkeypatch):
    """KEY TEST: a fresh offboard attempt (transfer fails downstream) and a
    subsequent retry (org_members already reflects the transitioned status)
    must derive the IDENTICAL request_id."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"

    db1 = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="suspended", revoked_at=revoked_at)], count=1),
                MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at), count=1),
            ],
            "credit_wallets": [
                MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 500}], count=1),
                MagicMock(data=[{"id": POOL_WALLET}], count=1),
            ],
        }
    )
    db1.rpc.return_value.execute.side_effect = RuntimeError("insufficient balance on source wallet")
    with pytest.raises(service.ReclaimFailedError):
        await service.suspend_member(db1, U1, ORG, MEMBER)
    first_request_id = db1.rpc.call_args[0][1]["p_request_id"]

    db2 = _db_seq(
        {
            "org_members": [MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at), count=1)],
            "credit_wallets": [
                MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 500}], count=1),
                MagicMock(data=[{"id": POOL_WALLET}], count=1),
            ],
        }
    )
    db2.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": True})
    await service.suspend_member(db2, U1, ORG, MEMBER)
    second_request_id = db2.rpc.call_args[0][1]["p_request_id"]

    assert first_request_id == second_request_id


async def test_suspend_reactivate_suspend_produces_distinct_keys(monkeypatch):
    """KEY TEST: a second suspension cycle (after a reactivation cleared
    revoked_at) must mint a DIFFERENT request_id than the first cycle."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at_1 = "2026-07-20T12:00:00+00:00"
    revoked_at_2 = "2026-07-21T09:30:00+00:00"

    db1 = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="suspended", revoked_at=revoked_at_1)], count=1),
                MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at_1), count=1),
            ],
            "credit_wallets": [
                MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 500}], count=1),
                MagicMock(data=[{"id": POOL_WALLET}], count=1),
            ],
        }
    )
    db1.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False})
    await service.suspend_member(db1, U1, ORG, MEMBER)
    key1 = db1.rpc.call_args[0][1]["p_request_id"]

    # Second cycle: org_members' current state reflects a reactivation
    # (active, revoked_at cleared) that happened between the two suspends.
    db2 = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(status="active", revoked_at=None), count=1),
                MagicMock(data=[_member_row(status="suspended", revoked_at=revoked_at_2)], count=1),
                MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at_2), count=1),
            ],
            "credit_wallets": [
                MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 300}], count=1),
                MagicMock(data=[{"id": POOL_WALLET}], count=1),
            ],
        }
    )
    db2.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False})
    await service.suspend_member(db2, U1, ORG, MEMBER)
    key2 = db2.rpc.call_args[0][1]["p_request_id"]

    assert key1 != key2


async def test_offboard_transfer_failure_raises_and_leaves_status_transitioned(monkeypatch):
    """KEY TEST: a raising transfer_credits call surfaces as ReclaimFailedError
    (no silent success) and does NOT trigger any corrective/revert write —
    the status transition (money-first) stands."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="removed", revoked_at=revoked_at)], count=1),
                MagicMock(data=_member_row(status="removed", revoked_at=revoked_at), count=1),
            ],
            "credit_wallets": [
                MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 200}], count=1),
                MagicMock(data=[{"id": POOL_WALLET}], count=1),
            ],
        }
    )
    db.rpc.return_value.execute.side_effect = RuntimeError("boom")
    with pytest.raises(service.ReclaimFailedError):
        await service.remove_member(db, U1, ORG, MEMBER)
    org_members_calls = [c for c in db.table.call_args_list if c.args[0] == "org_members"]
    assert len(org_members_calls) == 3  # select, update, reread — no corrective revert


async def test_offboard_creates_pool_wallet_on_miss(monkeypatch):
    """Task 4 AC: _offboard's pool-wallet lookup now goes through
    wallets.read_or_create_org_wallet (create-on-miss) — the Task 3 stopgap
    that raised RuntimeError on a missing pool wallet is gone. A missing pool
    wallet no longer blocks the reclaim: it's created (NULL periods, zero
    balance) and the transfer proceeds against the freshly created wallet's
    id."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="suspended", revoked_at=revoked_at)], count=1),
                MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at), count=1),
            ],
            "credit_wallets": [
                MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 100}], count=1),
                MagicMock(data=[], count=0),  # pool wallet SELECT miss
                MagicMock(data=[{"id": POOL_WALLET}], count=1),  # pool wallet INSERT creates it
            ],
        }
    )
    db.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False})
    result = await service.suspend_member(db, U1, ORG, MEMBER)
    assert result["status"] == "suspended"
    name, params = db.rpc.call_args[0]
    assert name == "transfer_credits"
    assert params["p_from_wallet"] == SEAT_WALLET
    assert params["p_to_wallet"] == POOL_WALLET
    assert params["p_amount"] == 100


# ---------------------------------------------------------------------------
# _offboard also revokes org-granted project access (Task 4, rule 3 extended
# to seat offboarding) — best-effort, AFTER the reclaim step succeeds.
# ---------------------------------------------------------------------------


async def test_offboard_zero_balance_still_revokes_org_granted_access(monkeypatch):
    """Access revocation isn't gated on money having moved — a zero-balance
    seat being suspended must still lose its org-granted project access."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="suspended", revoked_at=revoked_at)], count=1),
                MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at), count=1),
            ],
            "credit_wallets": [MagicMock(data=[], count=0)],
        }
    )
    fake_revoke = MagicMock(return_value=0)
    monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", fake_revoke)

    result = await service.suspend_member(db, U1, ORG, MEMBER)

    assert result["status"] == "suspended"
    fake_revoke.assert_called_once_with(db, ORG, user_id=U2)


async def test_offboard_nonzero_balance_revokes_after_reclaim_succeeds(monkeypatch):
    """The revoke call happens AFTER the transfer_credits RPC succeeds — the
    money step still runs first."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="removed", revoked_at=revoked_at)], count=1),
                MagicMock(data=_member_row(status="removed", revoked_at=revoked_at), count=1),
            ],
            "credit_wallets": [
                MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 500}], count=1),
                MagicMock(data=[{"id": POOL_WALLET}], count=1),
            ],
        }
    )
    db.rpc.return_value.execute.return_value = MagicMock(data={"duplicate": False})
    order = []
    fake_revoke = MagicMock(side_effect=lambda *a, **k: order.append("revoke") or 2)
    monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", fake_revoke)
    original_rpc = db.rpc

    def _tracked_rpc(*a, **k):
        order.append("rpc")
        return original_rpc(*a, **k)

    db.rpc = MagicMock(side_effect=_tracked_rpc)

    result = await service.remove_member(db, U1, ORG, MEMBER)

    assert result["status"] == "removed"
    fake_revoke.assert_called_once_with(db, ORG, user_id=U2)
    assert order == ["rpc", "revoke"]


async def test_offboard_reclaim_failure_skips_revocation(monkeypatch):
    """Money-first: if the reclaim RPC raises, _offboard surfaces
    ReclaimFailedError and revocation is never attempted — a retry will
    reach it once the reclaim itself succeeds."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="removed", revoked_at=revoked_at)], count=1),
                MagicMock(data=_member_row(status="removed", revoked_at=revoked_at), count=1),
            ],
            "credit_wallets": [
                MagicMock(data=[{"id": SEAT_WALLET, "bundle_balance": 0, "reserve_balance": 200}], count=1),
                MagicMock(data=[{"id": POOL_WALLET}], count=1),
            ],
        }
    )
    db.rpc.return_value.execute.side_effect = RuntimeError("boom")
    fake_revoke = MagicMock(return_value=0)
    monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", fake_revoke)

    with pytest.raises(service.ReclaimFailedError):
        await service.remove_member(db, U1, ORG, MEMBER)

    fake_revoke.assert_not_called()


async def test_offboard_revocation_failure_does_not_undo_offboard(monkeypatch):
    """A revocation failure is logged and swallowed — the offboard's own
    result (the transitioned org_members row) still comes back successfully,
    matching the money-first "never undo" posture."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(), count=1),
                MagicMock(data=[_member_row(status="suspended", revoked_at=revoked_at)], count=1),
                MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at), count=1),
            ],
            "credit_wallets": [MagicMock(data=[], count=0)],
        }
    )
    monkeypatch.setattr(
        org_projects, "revoke_org_granted_memberships", MagicMock(side_effect=RuntimeError("db exploded"))
    )

    result = await service.suspend_member(db, U1, ORG, MEMBER)
    assert result["status"] == "suspended"


async def test_offboard_organic_and_other_org_rows_survive(monkeypatch):
    """End-to-end through the REAL revoke_org_granted_memberships: the
    delete is filtered on org_id AND user_id, so organic rows and rows
    granted by a DIFFERENT org for this same member (impossible in practice
    since org_id/user_id/project pairing is unique per org, but asserted
    here for the filter-args guarantee itself) are never touched."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoked_at = "2026-07-20T12:00:00+00:00"

    pm_builder = MockQueryBuilder()
    pm_builder.delete.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
        data=[{"id": "pm1", "org_id": ORG, "user_id": U2}], count=1
    )

    om_seq = [
        MagicMock(data=_member_row(), count=1),
        MagicMock(data=[_member_row(status="suspended", revoked_at=revoked_at)], count=1),
        MagicMock(data=_member_row(status="suspended", revoked_at=revoked_at), count=1),
    ]
    om_calls = {"n": 0}

    def _side(name):
        if name == "project_members":
            return pm_builder
        b = MockQueryBuilder()
        if name == "org_members":
            i = min(om_calls["n"], len(om_seq) - 1)
            om_calls["n"] += 1
            b.execute.return_value = om_seq[i]
        elif name == "credit_wallets":
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side

    result = await service.suspend_member(db, U1, ORG, MEMBER)

    assert result["status"] == "suspended"
    pm_builder.delete.return_value.eq.assert_called_once_with("org_id", ORG)
    pm_builder.delete.return_value.eq.return_value.eq.assert_called_once_with("user_id", U2)


# ---------------------------------------------------------------------------
# reactivate_member
# ---------------------------------------------------------------------------


async def test_reactivate_member_requires_admin_403(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc_info:
        await service.reactivate_member(MagicMock(), U2, ORG, MEMBER)
    assert exc_info.value.status_code == 403


async def test_reactivate_member_not_found_raises(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"org_members": [MagicMock(data=None, count=0)]})
    with pytest.raises(ValueError):
        await service.reactivate_member(db, U1, ORG, MEMBER)


async def test_reactivate_member_rejects_already_active(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"org_members": [MagicMock(data=_member_row(status="active"), count=1)]})
    with pytest.raises(ValueError):
        await service.reactivate_member(db, U1, ORG, MEMBER)


async def test_reactivate_member_from_suspended_clears_revoked_at(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(status="suspended", revoked_at="2026-07-20T00:00:00+00:00"), count=1),
                MagicMock(data=[_member_row(status="active", revoked_at=None)], count=1),
            ]
        }
    )
    result = await service.reactivate_member(db, U1, ORG, MEMBER)
    assert result["status"] == "active"
    assert result["revoked_at"] is None


async def test_reactivate_member_from_removed_clears_revoked_at(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=_member_row(status="removed", revoked_at="2026-07-20T00:00:00+00:00"), count=1),
                MagicMock(data=[_member_row(status="active", revoked_at=None)], count=1),
            ]
        }
    )
    result = await service.reactivate_member(db, U1, ORG, MEMBER)
    assert result["status"] == "active"
