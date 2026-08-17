"""Mock-based tests for orgs.service core CRUD + archive (Licensing Phase B,
Task 2) and invites/roles/offboarding (Task 3). Mirrors
tests/test_teams_service.py + tests/test_teams_invites.py's idioms."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from orgs import projects as org_projects
from orgs import service, standing, storage_guard
from orgs.standing import TeamDials
from tests.conftest import MockQueryBuilder

U1 = "00000000-0000-0000-0000-000000000001"
U2 = "00000000-0000-0000-0000-000000000002"
ORG = "20000000-0000-0000-0000-000000000001"
EXISTING = "00000000-0000-0000-0000-000000000099"
MEMBER = "40000000-0000-0000-0000-000000000001"
TOKEN = "30000000-0000-0000-0000-000000000001"
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

    # get_org reads org_members three times, in order: my_role, member_count,
    # then the member-visible admin contacts.
    db = _db_seq(
        {
            "organizations": [
                MagicMock(data={"id": ORG, "status": "pending", "min_initial_purchase_credits": None}, count=1)
            ],
            "org_members": [
                MagicMock(data={"role": "admin"}, count=1),
                MagicMock(data=[], count=1),
                MagicMock(data=[], count=0),
            ],
            "credit_wallets": [MagicMock(data=[{"id": "w1", "bundle_balance": 0, "reserve_balance": 4000}], count=1)],
            "credit_ledger": [MagicMock(data=[{"delta": 4000}], count=1)],
        }
    )
    result = await service.get_org(db, U1, ORG)
    assert result["cumulative_paid_in"] == 4000
    assert result["pool_balance"] == 4000
    # No env override in this test → platform default of 10,000.
    assert result["remaining_to_activate"] == 6000


async def test_get_org_respects_custom_env_default_minimum(monkeypatch):
    """Admin view — the activation figures are admin-only, so the math is
    asserted through an admin caller."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    monkeypatch.setenv("ENTERPRISE_MIN_INITIAL_CREDITS", "500")

    db = _db_seq(
        {
            "organizations": [
                MagicMock(data={"id": ORG, "status": "pending", "min_initial_purchase_credits": None}, count=1)
            ],
            "org_members": [
                MagicMock(data={"role": "admin"}, count=1),
                MagicMock(data=[], count=0),
                MagicMock(data=[], count=0),
            ],
            "credit_wallets": [MagicMock(data=[], count=0)],
        }
    )
    result = await service.get_org(db, U1, ORG)
    assert result["remaining_to_activate"] == 500


async def test_get_org_uses_org_specific_minimum_over_env_default(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    monkeypatch.setenv("ENTERPRISE_MIN_INITIAL_CREDITS", "10000")

    db = _db_seq(
        {
            "organizations": [
                MagicMock(data={"id": ORG, "status": "pending", "min_initial_purchase_credits": 2000}, count=1)
            ],
            "org_members": [
                MagicMock(data={"role": "admin"}, count=1),
                MagicMock(data=[], count=1),
                MagicMock(data=[], count=0),
            ],
            "credit_wallets": [MagicMock(data=[], count=0)],
        }
    )
    result = await service.get_org(db, U1, ORG)
    assert result["pool_balance"] == 0
    assert result["cumulative_paid_in"] == 0
    assert result["remaining_to_activate"] == 2000


async def test_get_org_returns_admin_contacts_to_a_plain_member(monkeypatch):
    """A member's only remedy for a reached cap or a dry pool is "ask an admin",
    so get_org (member-gated) names them. Admins only — the rest of the roster
    and every cap/spend figure stay in the admin-only /usage rollup."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)

    db = _db_seq(
        {
            "organizations": [
                MagicMock(data={"id": ORG, "status": "active", "min_initial_purchase_credits": None}, count=1)
            ],
            "org_members": [
                MagicMock(data={"role": "member"}, count=1),  # caller is NOT an admin
                MagicMock(data=[], count=8),  # member_count
                MagicMock(data=[{"id": MEMBER, "user_id": U2, "email": "boss@acme.com"}], count=1),
            ],
            "credit_wallets": [MagicMock(data=[], count=0)],
            "profiles": [MagicMock(data=[{"id": U2, "full_name": "Boss Person"}], count=1)],
        }
    )
    result = await service.get_org(db, U1, ORG)

    assert result["my_role"] == "member"
    assert result["member_count"] == 8
    assert result["admins"] == [{"userId": U2, "email": "boss@acme.com", "fullName": "Boss Person"}]
    # Email was on the row, so no auth-admin lookup was needed to resolve it.
    db.auth.admin.get_user_by_id.assert_not_called()


async def test_get_org_zero_balance_when_no_wallet(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)

    db = _db_seq(
        {
            "organizations": [
                MagicMock(data={"id": ORG, "status": "pending", "min_initial_purchase_credits": None}, count=1)
            ],
            "org_members": [
                MagicMock(data={"role": "admin"}, count=1),
                MagicMock(data=[], count=0),
                MagicMock(data=[], count=0),
            ],
            "credit_wallets": [MagicMock(data=[], count=0)],
        }
    )
    result = await service.get_org(db, U1, ORG)
    assert result["pool_balance"] == 0
    assert result["cumulative_paid_in"] == 0


async def test_get_org_no_active_seat_row_yields_null_role(monkeypatch):
    """my_role is read from ACTIVE seats only — no row means no role, and the
    redaction that keys off it therefore fails closed."""
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
    assert result["my_role"] is None
    assert "pool_balance" not in result


# ---------------------------------------------------------------------------
# Pool visibility: the shared pool is the ORG's money — admins only.
# ---------------------------------------------------------------------------


def _get_org_db(role):
    return _db_seq(
        {
            "organizations": [
                MagicMock(
                    data={
                        "id": ORG,
                        "name": "Acme",
                        "status": "active",
                        "min_initial_purchase_credits": 10000,
                        "monthly_dispersal_credits": 5000,
                        "default_member_cap": 300,
                        "storage_bytes": 1234,
                    },
                    count=1,
                )
            ],
            "org_members": [
                MagicMock(data={"role": role}, count=1),
                MagicMock(data=[], count=4),
                MagicMock(data=[], count=0),
            ],
            "credit_wallets": [MagicMock(data=[{"id": "w1", "bundle_balance": 900, "reserve_balance": 100}], count=1)],
            "credit_ledger": [MagicMock(data=[{"delta": 4000}], count=1)],
        }
    )


@pytest.mark.parametrize("field", sorted(service._ADMIN_ONLY_ORG_FIELDS))
async def test_get_org_hides_every_admin_only_field_from_a_member(monkeypatch, field):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    result = await service.get_org(_get_org_db("member"), U1, ORG)
    assert field not in result


async def test_get_org_shows_pool_to_an_admin(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    result = await service.get_org(_get_org_db("admin"), U1, ORG)
    assert result["pool_balance"] == 1000
    assert result["monthly_dispersal_credits"] == 5000
    assert result["cumulative_paid_in"] == 4000


async def test_get_org_still_gives_a_member_what_they_need(monkeypatch):
    """Redaction must not blind a member: identity, status, their own role and
    effective cap, the roster size and who to ask all survive."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    result = await service.get_org(_get_org_db("member"), U1, ORG)
    assert result["name"] == "Acme"
    assert result["status"] == "active"
    assert result["my_role"] == "member"
    assert result["member_count"] == 4
    # Their own ceiling when they hold no personal override — not an org secret.
    assert result["default_member_cap"] == 300


async def test_list_my_orgs_redacts_for_members_and_suspended_admins(monkeypatch):
    """list_my_orgs select("*")s the org row, so it needs the same redaction —
    and a SUSPENDED admin is not an admin (authz counts ACTIVE rows only)."""
    org_row = {"id": ORG, "name": "Acme", "monthly_dispersal_credits": 5000, "storage_bytes": 99}

    def _run(role, status):
        db = _db_seq(
            {
                "org_members": [MagicMock(data=[{"org_id": ORG, "role": role, "status": status}], count=1)],
                "organizations": [MagicMock(data=[dict(org_row)], count=1)],
            }
        )
        return db

    active_admin = await service.list_my_orgs(_run("admin", "active"), U1)
    assert active_admin[0]["monthly_dispersal_credits"] == 5000

    for role, status in (("member", "active"), ("admin", "suspended")):
        out = await service.list_my_orgs(_run(role, status), U1)
        assert "monthly_dispersal_credits" not in out[0], f"{role}/{status} saw the dispersal"
        assert "storage_bytes" not in out[0]
        assert out[0]["name"] == "Acme"


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
# graceDays + teamStorage (Task 15, spec §6) — the org billing console's
# storage meter. graceDays is a global constant (env-configured), so it
# carries nothing to redact and is asserted for both roles below. teamStorage
# IS admin-only (in _ADMIN_ONLY_ORG_FIELDS) AND only computed at all for a
# self_serve org with an active coverer — the two conditions are tested
# independently of the redaction sweep above, since `_get_org_db` carries no
# kind/covered_by and would never exercise the computation either way.
# ---------------------------------------------------------------------------


def _self_serve_org_db(role, *, kind="self_serve", covered_by=U1, pool_org_rows=None):
    """organizations is read TWICE when teamStorage is computed: once for
    get_org's own row, once inside storage_guard.pool_state's storage scan —
    hence two queued rows. profiles/subscriptions/tier_entitlements are
    pool_state's team_dials_for_user reads."""
    org_row = {
        "id": ORG,
        "name": "Acme",
        "status": "active",
        "kind": kind,
        "covered_by": covered_by,
        "min_initial_purchase_credits": None,
    }
    organizations_seq = [MagicMock(data=org_row, count=1)]
    if pool_org_rows is not None:
        organizations_seq.append(MagicMock(data=pool_org_rows, count=len(pool_org_rows)))
    return _db_seq(
        {
            "organizations": organizations_seq,
            "org_members": [
                MagicMock(data={"role": role}, count=1),
                MagicMock(data=[], count=1),
                MagicMock(data=[], count=0),
            ],
            "credit_wallets": [MagicMock(data=[], count=0)],
            "profiles": [MagicMock(data=[{"is_admin": False}], count=1)],
            "subscriptions": [MagicMock(data=[{"tier": "pro"}], count=1)],
            "tier_entitlements": [
                MagicMock(
                    data=[{"tier": "pro", "max_teams": 3, "max_team_members": 10, "team_storage_bytes": 10 * 2**30}],
                    count=1,
                )
            ],
        }
    )


async def test_get_org_includes_grace_days_for_admin_and_member(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    monkeypatch.setenv("ORG_GRACE_DAYS", "14")

    for role in ("admin", "member"):
        result = await service.get_org(_self_serve_org_db(role, pool_org_rows=[]), U1, ORG)
        assert result["graceDays"] == 14


async def test_get_org_admin_self_serve_includes_team_storage(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    monkeypatch.setenv("TEAM_STORAGE_OVERAGE_USD_PER_GB", "0.025")
    db = _self_serve_org_db("admin", pool_org_rows=[{"storage_bytes": 12 * 2**30}])

    result = await service.get_org(db, U1, ORG)

    assert result["teamStorage"] == {
        "usedBytes": 12 * 2**30,
        "poolBytes": 10 * 2**30,
        "overageGb": 2,
        "ratePerGb": 0.025,
    }


async def test_get_org_member_self_serve_hides_team_storage(monkeypatch):
    """Computed (organizations queried twice, same as the admin case) but
    stripped by redact_org_for_role — same posture as pool_balance etc."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    db = _self_serve_org_db("member", pool_org_rows=[{"storage_bytes": 12 * 2**30}])

    result = await service.get_org(db, U1, ORG)

    assert "teamStorage" not in result


async def test_get_org_enterprise_admin_never_computes_team_storage(monkeypatch):
    """kind != 'self_serve' -> the key is never even added, so pool_state must
    not be called at all (not just redacted away)."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)

    def _boom(*a, **kw):
        raise AssertionError("pool_state must not be called for an enterprise org")

    monkeypatch.setattr(storage_guard, "pool_state", _boom)
    db = _self_serve_org_db("admin", kind="enterprise", covered_by=U1, pool_org_rows=None)

    result = await service.get_org(db, U1, ORG)
    assert "teamStorage" not in result


async def test_get_org_released_self_serve_admin_never_computes_team_storage(monkeypatch):
    """self_serve but covered_by is None (released org, spec §2 rev 2) -> no
    owner to size or bill the pool against, so the key stays absent."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)

    def _boom(*a, **kw):
        raise AssertionError("pool_state must not be called with no coverer")

    monkeypatch.setattr(storage_guard, "pool_state", _boom)
    db = _self_serve_org_db("admin", kind="self_serve", covered_by=None, pool_org_rows=None)

    result = await service.get_org(db, U1, ORG)
    assert "teamStorage" not in result


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


async def test_update_org_clears_default_member_cap_with_explicit_null(monkeypatch):
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
            b.execute.return_value = MagicMock(data=[{"id": ORG, "default_member_cap": None}], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.update_org(db, U1, ORG, {"default_member_cap": None})
    assert captured["payload"] == {"default_member_cap": None}
    assert result["default_member_cap"] is None


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
# archive_org — Task 4 teardown (rule 12): revoke org-granted memberships,
# after the archive lands. (There is no link row to delete any more —
# org_project_links was retired in 20260804000001, and an archived org's
# artists.team_id rows are deliberately left in place: can_access_artist
# already denies on archived_at, so the roster is inert without being destroyed.)
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


async def test_archive_org_revokes_org_granted_memberships(monkeypatch):
    """Rule 12 — archive is an UPDATE, so nothing cascades; archive_org must
    explicitly drop every project_members row THIS org granted."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    revoke = MagicMock(return_value=0)
    monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", revoke)

    db = MagicMock()
    db.table.side_effect = _archive_success_side

    result = await service.archive_org(db, U1, ORG)

    assert result["archived_at"] is not None
    revoke.assert_called_once()
    assert revoke.call_args.args[1] == ORG


async def test_archive_org_leaves_team_owned_artists_attached(monkeypatch):
    """An archived org keeps its roster: can_access_artist denies on
    archived_at, so the artists go inert on their own. Detaching them here
    would destroy the ownership record support needs."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(org_projects, "revoke_org_granted_memberships", lambda *a, **k: 0)

    artists_builder = MockQueryBuilder()

    def _side(name):
        if name == "artists":
            return artists_builder
        return _archive_success_side(name)

    db = MagicMock()
    db.table.side_effect = _side

    result = await service.archive_org(db, U1, ORG)

    assert result["archived_at"] is not None
    artists_builder.delete.assert_not_called()


async def test_archive_org_teardown_failures_do_not_block_archive(monkeypatch):
    """The cleanup is best-effort — a failure in the revoke call must not
    prevent archive_org from returning the archived org row; the archived_at
    write has already landed by the time it runs."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(
        org_projects, "revoke_org_granted_memberships", MagicMock(side_effect=RuntimeError("revoke boom"))
    )

    db = MagicMock()
    db.table.side_effect = _archive_success_side

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
# _self_serve_seat_room / invite_member team-size gate (Task 7, spec §3)
# ---------------------------------------------------------------------------


def _org_row(**overrides):
    """Mirrors tests/test_org_standing.py's helper of the same name (kept
    local rather than imported — that file is Task 5's, edited in parallel).
    Matches _first_org's select list: kind, status, covered_by, covered_at,
    archived_at, dissolved_at."""
    row = {
        "kind": "self_serve",
        "status": "active",
        "covered_by": U1,
        "covered_at": "2026-08-01T00:00:00+00:00",
        "archived_at": None,
        "dissolved_at": None,
    }
    row.update(overrides)
    return row


async def test_invite_member_at_limit_raises_team_full(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(standing, "team_dials_for_user", lambda db, uid: TeamDials(max_team_members=3))
    db = _db_seq(
        {
            "organizations": [MagicMock(data=_org_row(covered_by=U1), count=1)],
            "org_members": [MagicMock(data=[], count=3)],  # 3 ACTIVE non-owner members == the limit
        }
    )
    with pytest.raises(service.TeamFullError) as exc_info:
        await service.invite_member(db, U1, ORG, "new@example.com", "member")
    assert "3" in str(exc_info.value)  # limit is in the message


PRO_MEMBER = "60000000-0000-0000-0000-000000000001"


def _active_members(n, pro_id=None):
    """n distinct active org_members rows (user_id only, matching the gate's
    .select("user_id")); pro_id, if given, replaces the first id so a test
    can point resolve_tier_for_user at exactly one Pro member."""
    ids = [f"70000000-0000-0000-0000-{i:012d}" for i in range(n)]
    if pro_id:
        ids[0] = pro_id
    return [{"user_id": uid} for uid in ids]


async def test_invite_member_pro_coverer_zero_pro_members_at_five_raises(monkeypatch):
    """Owner decision 2026-08-16 (SEATS_PER_PRO formula): a lone Pro coverer
    (no other Pro members) unlocks only the first 5-seat block. At 5 active
    members the wall hits — next_step points at another Pro member joining
    or Enterprise, never a per-org override."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(standing, "team_dials_for_user", lambda db, uid: TeamDials(max_team_members=10, tier="pro"))
    monkeypatch.setattr(standing, "resolve_tier_for_user", lambda db, uid: "basic")
    db = _db_seq(
        {
            "organizations": [MagicMock(data=_org_row(covered_by=U1), count=1)],
            "org_members": [MagicMock(data=_active_members(5), count=5)],
        }
    )
    with pytest.raises(service.TeamFullError) as exc_info:
        await service.invite_member(db, U1, ORG, "new@example.com", "member")
    exc = exc_info.value
    assert exc.limit == 5
    assert exc.next_step == "contact"
    assert "unlocks" in str(exc)
    assert "up to 10" in str(exc)


async def test_invite_member_pro_coverer_one_pro_member_under_ten_proceeds(monkeypatch):
    """One Pro member (besides the coverer) unlocks the second 5-seat block,
    up to 10 — 5 active members is comfortably under that, so the gate lets
    the invite through (no TeamFullError)."""
    monkeypatch.setattr(standing, "team_dials_for_user", lambda db, uid: TeamDials(max_team_members=10, tier="pro"))
    monkeypatch.setattr(standing, "resolve_tier_for_user", lambda db, uid: "pro" if uid == PRO_MEMBER else "basic")
    db = _db_seq(
        {
            "organizations": [MagicMock(data=_org_row(covered_by=U1), count=1)],
            "org_members": [MagicMock(data=_active_members(5, pro_id=PRO_MEMBER), count=5)],
        }
    )
    service._self_serve_seat_room(db, ORG)  # must not raise


async def test_invite_member_pro_coverer_one_pro_member_at_ten_raises(monkeypatch):
    """The same one-Pro-member team hits its OWN wall at 10 — the hard
    ceiling (tier_entitlements.pro.max_team_members) is never exceeded no
    matter how many Pro members join."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(standing, "team_dials_for_user", lambda db, uid: TeamDials(max_team_members=10, tier="pro"))
    monkeypatch.setattr(standing, "resolve_tier_for_user", lambda db, uid: "pro" if uid == PRO_MEMBER else "basic")
    db = _db_seq(
        {
            "organizations": [MagicMock(data=_org_row(covered_by=U1), count=1)],
            "org_members": [MagicMock(data=_active_members(10, pro_id=PRO_MEMBER), count=10)],
        }
    )
    with pytest.raises(service.TeamFullError) as exc_info:
        await service.invite_member(db, U1, ORG, "new@example.com", "member")
    exc = exc_info.value
    assert exc.limit == 10
    assert exc.next_step == "contact"
    assert "10-member limit" in str(exc)


async def test_invite_member_basic_coverer_pro_member_does_not_raise_cap(monkeypatch):
    """A Pro member sitting on a Basic-covered team does NOT raise the cap
    above Basic's 3 — the formula's min() always defers to
    dials.max_team_members for a non-Pro coverer, and next_step stays
    'upgrade' (not on Pro yet)."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(standing, "team_dials_for_user", lambda db, uid: TeamDials(max_team_members=3, tier="basic"))
    monkeypatch.setattr(standing, "resolve_tier_for_user", lambda db, uid: "pro" if uid == PRO_MEMBER else "basic")
    db = _db_seq(
        {
            "organizations": [MagicMock(data=_org_row(covered_by=U1), count=1)],
            "org_members": [MagicMock(data=_active_members(3, pro_id=PRO_MEMBER), count=3)],
        }
    )
    with pytest.raises(service.TeamFullError) as exc_info:
        await service.invite_member(db, U1, ORG, "new@example.com", "member")
    exc = exc_info.value
    assert exc.next_step == "upgrade"
    assert exc.limit == 3
    assert "3" in str(exc)
    assert "Basic" in str(exc)
    assert "Upgrade to Pro" in str(exc)


async def test_invite_member_under_limit_proceeds(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: None)
    monkeypatch.setattr(standing, "team_dials_for_user", lambda db, uid: TeamDials(max_team_members=3))
    db = _db_seq(
        {
            "organizations": [MagicMock(data=_org_row(covered_by=U1), count=1)],
            "org_members": [MagicMock(data=[], count=2)],  # under the limit of 3
            "pending_org_invites": [
                MagicMock(data=None, count=0),
                MagicMock(data=[{"id": "i1", "token": TOKEN, "email": "new@example.com", "role": "member"}], count=1),
            ],
        }
    )
    result = await service.invite_member(db, U1, ORG, "new@example.com", "member")
    assert result["type"] == "invited"


async def test_invite_member_lapsed_org_raises_team_lapsed(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"organizations": [MagicMock(data=_org_row(status="lapsed"), count=1)]})
    with pytest.raises(service.TeamLapsedError) as exc_info:
        await service.invite_member(db, U1, ORG, "new@example.com", "member")
    assert "reactivate" in str(exc_info.value).lower()


async def test_invite_member_enterprise_org_returns_immediately_no_tier_read(monkeypatch):
    """Enterprise: byte-identical behavior — the gate must not even read
    org_members (let alone tier_entitlements) once kind != 'self_serve'."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: None)
    tables_queried = []

    def _side(name):
        tables_queried.append(name)
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=_org_row(kind="enterprise"), count=1)
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=[{"id": f"m{i}"} for i in range(50)], count=50)
        elif name == "pending_org_invites":
            b.execute.return_value = MagicMock(data=None, count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    result = await service.invite_member(db, U1, ORG, "new@example.com", "member")
    assert result["type"] == "invited"
    assert "tier_entitlements" not in tables_queried
    assert "org_members" not in tables_queried


async def test_invite_member_refused_during_grace_when_coverer_is_free_tier(monkeypatch):
    """ACCEPTED consequence (review r2, plan task7.md AC): a covering owner
    who has downgraded to Free reads max_team_members=0 from
    team_dials_for_user even DURING their grace window — org.status stays
    'active' through grace; only the sweep flips it to 'lapsed' later. That
    yields 0 members >= 0 limit, so invites are refused mid-grace despite
    access otherwise continuing. This narrows spec §3's "invites still
    allowed during grace" for this one case; correct per plan."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(standing, "team_dials_for_user", lambda db, uid: TeamDials(max_team_members=0))
    db = _db_seq(
        {
            "organizations": [MagicMock(data=_org_row(covered_by=U1, status="active"), count=1)],
            "org_members": [MagicMock(data=[], count=0)],  # no other members at all
        }
    )
    with pytest.raises(service.TeamFullError) as exc_info:
        await service.invite_member(db, U1, ORG, "new@example.com", "member")
    assert "0" in str(exc_info.value)


async def test_seat_room_skips_gate_when_covered_by_is_none():
    """Interpretation taken for the released-org / pre-migration case
    (stated in the delivery report): covered_by is the dials owner, and with
    no coverer there is nothing to resolve a limit against, so the gate is
    skipped rather than guessed at. A released org is already headed to
    'lapsed' via the sweep, which is the real backstop."""
    db = _db_seq({"organizations": [MagicMock(data=_org_row(covered_by=None), count=1)]})
    service._self_serve_seat_room(db, ORG)  # must not raise


async def test_seat_room_query_excludes_coverer_and_filters_active():
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=_org_row(covered_by=U1), count=1)
        elif name == "org_members":
            original_eq = b.eq
            original_neq = b.neq

            def _capture_eq(field, value):
                captured.setdefault("eq", []).append((field, value))
                return original_eq(field, value)

            def _capture_neq(field, value):
                captured["neq"] = (field, value)
                return original_neq(field, value)

            b.eq = _capture_eq
            b.neq = _capture_neq
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    with patch.object(standing, "team_dials_for_user", lambda db, uid: TeamDials(max_team_members=5)):
        service._self_serve_seat_room(db, ORG)
    assert ("org_id", ORG) in captured["eq"]
    assert ("status", "active") in captured["eq"]
    assert captured["neq"] == ("user_id", U1)


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


# ---------------------------------------------------------------------------
# accept_invite team-size gate (Task 7, spec §3) — must raise InviteInvalidError
# (already mapped to 410) and leave the invite row 'pending' so the 48h expiry
# collects it, rather than flipping it to 'accepted'.
# ---------------------------------------------------------------------------


def _accept_gate_db(org_row, pending_updated: dict, *, org_members_count: int = 0):
    """pending_org_invites' update() is wrapped so tests can assert it was
    NEVER called — the row must stay pending on a gate refusal."""

    def _side(name):
        b = MockQueryBuilder()
        if name == "pending_org_invites":
            b.execute.return_value = MagicMock(data=_pending_invite(), count=1)
            original_update = b.update

            def _capture_update(payload, *a, **kw):
                pending_updated["called"] = True
                return original_update(payload, *a, **kw)

            b.update = _capture_update
        elif name == "organizations":
            b.execute.return_value = MagicMock(data=org_row, count=1)
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=[], count=org_members_count)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


async def test_accept_invite_at_limit_raises_invite_invalid_and_leaves_row_pending(monkeypatch):
    monkeypatch.setattr(standing, "team_dials_for_user", lambda db, uid: TeamDials(max_team_members=3))
    pending_updated = {"called": False}
    db = _accept_gate_db(_org_row(covered_by=U1), pending_updated, org_members_count=3)

    with pytest.raises(service.InviteInvalidError):
        await service.accept_invite(db, U2, "u@example.com", TOKEN)
    assert pending_updated["called"] is False


async def test_accept_invite_lapsed_org_raises_invite_invalid_and_leaves_row_pending():
    pending_updated = {"called": False}
    db = _accept_gate_db(_org_row(status="lapsed"), pending_updated)

    with pytest.raises(service.InviteInvalidError):
        await service.accept_invite(db, U2, "u@example.com", TOKEN)
    assert pending_updated["called"] is False


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


async def test_offboard_never_calls_a_money_rpc(monkeypatch):
    """KEY TEST: offboarding touches no money at all. A member only ever held a
    ceiling on the shared pool, so there is nothing to reclaim and no RPC to
    call — which is also why suspend/remove can no longer fail with a 502."""
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
        }
    )
    result = await service.suspend_member(db, U1, ORG, MEMBER)
    assert result["status"] == "suspended"
    db.rpc.assert_not_called()


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


# ---------------------------------------------------------------------------
# Offboarding: a status transition, nothing more. Members hold no credits, so
# there is no balance to reclaim and no money RPC to fail — which is exactly
# why suspend/remove can no longer 502.
# ---------------------------------------------------------------------------


async def test_offboard_transitions_status_and_moves_no_money(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    member = {"id": MEMBER, "org_id": ORG, "user_id": U1, "status": "active", "revoked_at": None}
    db = _db_seq(
        {
            "org_members": [
                MagicMock(data=member),  # initial maybe_single read
                MagicMock(data=[{**member, "status": "suspended"}]),  # UPDATE echo
                MagicMock(data={**member, "status": "suspended", "revoked_at": "2026-07-29T00:00:00+00:00"}),  # reread
            ]
        }
    )

    with patch("orgs.service._revoke_offboarded_member_access"):
        row = await service.suspend_member(db, U1, ORG, MEMBER)

    assert row["status"] == "suspended"
    assert row["revoked_at"]  # audit timestamp, not a reclaim key
    db.rpc.assert_not_called()  # nothing to move


async def test_offboard_already_at_final_status_reuses_the_row(monkeypatch):
    """A retry of a prior offboard: the row is already suspended with a
    revoked_at, so it is reused rather than re-stamped."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    member = {
        "id": MEMBER,
        "org_id": ORG,
        "user_id": U1,
        "status": "suspended",
        "revoked_at": "2026-07-20T00:00:00+00:00",
    }
    db = _db_seq({"org_members": [MagicMock(data=member)]})

    with patch("orgs.service._revoke_offboarded_member_access"):
        row = await service.suspend_member(db, U1, ORG, MEMBER)

    assert row["revoked_at"] == "2026-07-20T00:00:00+00:00"
    db.rpc.assert_not_called()


async def test_archive_org_needs_no_balance_precondition(monkeypatch):
    """Whatever the POOL holds survives archiving — disposing of it is a support
    decision (admin clawback), so archiving can no longer 409."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=[{"id": ORG, "archived_at": "2026-07-29T00:00:00+00:00"}], count=1)
            original_update = b.update

            def _update(payload):
                captured["payload"] = payload
                return original_update(payload)

            b.update = _update
        return b

    db = MagicMock()
    db.table.side_effect = _side

    with patch("orgs.service._teardown_archived_org_grants"):
        result = await service.archive_org(db, U1, ORG)

    assert "archived_at" in captured["payload"]
    assert result["archived_at"]
    # No credit_wallets read at all — there are no member balances to verify.
    assert not any(c.args[0] == "credit_wallets" for c in db.table.call_args_list)


# ---------------------------------------------------------------------------
# create_org_join_notifications
# ---------------------------------------------------------------------------


def test_join_notifications_go_to_the_member_and_every_other_admin():
    """Acceptance is the one org event both sides act on: the member learns
    their billing moved to the pool, admins learn the roster changed."""
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data={"name": "Acme"}, count=1)
        elif name == "org_members":
            b.execute.return_value = MagicMock(data=[{"user_id": U1}, {"user_id": U2}], count=2)
        elif name == "notifications":
            original = b.insert

            def _capture(payload, *a, **kw):
                captured["rows"] = payload
                return original(payload, *a, **kw)

            b.insert = _capture
        return b

    db = MagicMock()
    db.table.side_effect = _side
    service.create_org_join_notifications(db, ORG, U2, "new@acme.com")

    rows = captured["rows"]
    # U2 is BOTH the accepting member and an admin — they get the member notice
    # only, never a duplicate "someone joined" about themselves.
    assert [r["user_id"] for r in rows] == [U2, U1]
    assert rows[0]["title"] == "You joined Acme"
    assert "shared credit pool" in rows[0]["message"]
    assert rows[1]["message"] == "new@acme.com joined Acme."
    # 'confirmation', not 'invitation': NotificationRow renders Accept/Decline
    # for 'invitation', and a join notice has nothing to action.
    assert {r["type"] for r in rows} == {"confirmation"}
    assert {r["entity_type"] for r in rows} == {"org"}
    assert {r["entity_id"] for r in rows} == {ORG}


def test_join_notification_falls_back_when_org_name_missing():
    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=None, count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    service.create_org_join_notifications(db, ORG, U2, "new@acme.com")  # must not raise


# ---------------------------------------------------------------------------
# Invite delivery: the emailed link and the in-app row must both carry the token
# ---------------------------------------------------------------------------


def test_invite_email_links_to_the_tokened_claim_page(monkeypatch):
    """The token is the ONLY thing that carries the invite. /notifications and
    /auth both dead-ended (no in-app row existed at invite time, the pending
    list is admin-only, and no signup trigger converts a pending_org_invites
    row)."""
    from orgs import emails

    monkeypatch.setenv("VITE_FRONTEND_URL", "https://app.msanii.test")
    sent = {}
    monkeypatch.setattr(emails, "_send", lambda **kw: sent.update(kw))

    for existing_user in (True, False):
        sent.clear()
        emails.send_org_invite_email(
            recipient_email="new@acme.com",
            org_name="Acme",
            inviter_name="Alice",
            role="member",
            token=TOKEN,
            existing_user=existing_user,
        )
        assert f"https://app.msanii.test/orgs/invite/{TOKEN}" in sent["html_body"]
        assert "/notifications" not in sent["html_body"]


def test_invite_notification_is_an_org_invitation():
    """NotificationRow keys the Accept/Decline buttons off the PAIR
    invitation + entity_type='org'; either half alone renders no buttons."""
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data={"name": "Acme"}, count=1)
        elif name == "profiles":
            b.execute.return_value = MagicMock(data={"full_name": "Alice"}, count=1)
        elif name == "notifications":
            original = b.insert

            def _capture(payload, *a, **kw):
                captured["row"] = payload
                return original(payload, *a, **kw)

            b.insert = _capture
        return b

    db = MagicMock()
    db.table.side_effect = _side
    service.create_org_invite_notification(db, U2, ORG, U1, TOKEN)

    row = captured["row"]
    assert row["type"] == "invitation"
    assert row["entity_type"] == "org"
    assert row["user_id"] == U2
    assert row["metadata"]["token"] == TOKEN
    assert row["title"] == "Invited to Acme"
    assert "Alice" in row["message"]


# ---------------------------------------------------------------------------
# Invite lifetime: 48h, then a one-shot 'expired' transition that gives the
# sweep somewhere to hang "they never accepted" off.
# ---------------------------------------------------------------------------


async def test_reinvite_restarts_the_48h_window(monkeypatch):
    """Resending revives a lapsed invite AND resets its clock — that is what
    resending is for."""
    from datetime import UTC, datetime

    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(service, "_find_user_id_by_email", lambda *a: None)
    captured = {}

    db = _db_seq(
        {
            "pending_org_invites": [
                MagicMock(data={"id": "i1"}, count=1),  # existing-row lookup
                MagicMock(data=[{"id": "i1", "status": "pending"}], count=1),  # the update
            ]
        }
    )
    real_table = db.table.side_effect

    def _wrap(name):
        b = real_table(name)
        if name == "pending_org_invites":
            original_update = b.update

            def _capture_update(payload):
                captured["payload"] = payload
                return original_update(payload)

            b.update = _capture_update
        return b

    db.table.side_effect = _wrap
    await service.invite_member(db, U1, ORG, "late@example.com", "member")

    expires = datetime.fromisoformat(captured["payload"]["expires_at"])
    hours = (expires - datetime.now(UTC)).total_seconds() / 3600
    assert 47 < hours <= 48
    assert captured["payload"]["status"] == "pending"


async def test_get_pending_invites_hides_lapsed_rows(monkeypatch):
    """Between lapsing and being swept a row still reads status='pending' while
    accept_invite would reject it — so the list filters on expires_at too."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "pending_org_invites":
            original_gt = b.gt

            def _capture_gt(field, value):
                captured[field] = value
                return original_gt(field, value)

            b.gt = _capture_gt
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    await service.get_pending_invites(db, U1, ORG)
    assert "expires_at" in captured


def _expiry_db(stale, claimed=None):
    """Fake for expire_stale_invites: `stale` is the scan result, `claimed` the
    rows the UPDATE reports it actually transitioned."""
    claimed = stale if claimed is None else claimed
    db = MagicMock()
    scan = db.table.return_value.select.return_value.eq.return_value.lt.return_value.limit.return_value
    scan.execute.return_value = MagicMock(data=stale)
    upd = db.table.return_value.update.return_value.in_.return_value.eq.return_value
    upd.execute.return_value = MagicMock(data=claimed)
    db.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value = (
        MagicMock(data={"name": "Acme"})
    )
    return db


def _invite(i="i1", email="late@example.com", invited_by=U1):
    return {"id": i, "org_id": ORG, "email": email, "invited_by": invited_by}


async def test_expire_stale_invites_notifies_the_inviter():
    db = _expiry_db([_invite()])
    assert service.expire_stale_invites(db) == 1

    payload = db.table.return_value.insert.call_args.args[0]
    assert payload["user_id"] == U1  # the admin who sent it
    assert "late@example.com" in payload["message"]
    assert "48 hours" in payload["message"]
    # Must NOT be an actionable invite row — this is a report, and
    # invitation+org is what renders Accept/Decline.
    assert payload["type"] == "status_change"


async def test_expire_stale_invites_is_idempotent():
    """A concurrent sweep (or a retry) transitions nothing, so it notifies
    nothing — the status UPDATE is the claim, and it is filtered on pending."""
    db = _expiry_db([_invite()], claimed=[])
    assert service.expire_stale_invites(db) == 0
    db.table.return_value.insert.assert_not_called()


async def test_expire_stale_invites_notifies_per_invite():
    """One notification per lapsed invite, each naming its invitee — merging
    them would bury the action ("chase this person")."""
    db = _expiry_db([_invite("i1", "a@x.com"), _invite("i2", "b@x.com")])
    assert service.expire_stale_invites(db) == 2
    assert db.table.return_value.insert.call_count == 2


async def test_expire_stale_invites_no_op_when_nothing_lapsed():
    db = _expiry_db([])
    assert service.expire_stale_invites(db) == 0
    db.table.return_value.update.assert_not_called()


async def test_expire_stale_invites_survives_a_notification_failure():
    """The status transition has already committed — a failed courtesy message
    must not take the whole sweep step down."""
    db = _expiry_db([_invite("i1"), _invite("i2")])
    db.table.return_value.insert.side_effect = [RuntimeError("smtp down"), MagicMock()]
    assert service.expire_stale_invites(db) == 2


async def test_expire_stale_invites_skips_rows_with_no_inviter():
    """invited_by is NOT NULL in schema, but a deleted admin cascades — no
    recipient means no notification, not a crash."""
    db = _expiry_db([_invite(invited_by=None)])
    assert service.expire_stale_invites(db) == 1
    db.table.return_value.insert.assert_not_called()


# ---------------------------------------------------------------------------
# unarchive_org (Task 8, spec §3): self-serve reactivation — needs a free
# slot AND the reactivation storage guard, since an archived team holds no
# slot but its bytes still count.
# ---------------------------------------------------------------------------


def _org_update_capture(first_read: dict, updated_row: dict):
    """organizations table: the FIRST execute() serves _first_org's read; the
    SECOND serves the update, with its payload captured for assertion.
    Mirrors tests/test_org_standing.py's `_capture_update_db` helper of the
    same shape (kept local — that file is Task 5's, edited in parallel)."""
    captured = {}
    calls = {"n": 0}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            calls["n"] += 1
            if calls["n"] == 1:
                b.execute.return_value = MagicMock(data=first_read, count=1)
            else:
                original_update = b.update

                def _capture(payload, *a, **kw):
                    captured["payload"] = payload
                    return original_update(payload, *a, **kw)

                b.update = _capture
                b.execute.return_value = MagicMock(data=[updated_row], count=1)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db, captured


class TestUnarchiveOrg:
    async def test_unarchive_happy_path_asserts_update_payload(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        monkeypatch.setattr(standing, "require_free_slot", lambda *a, **kw: None)
        monkeypatch.setattr(storage_guard, "reactivation_allowed", lambda *a, **kw: True)
        row = _org_row(archived_at="2026-08-01T00:00:00+00:00", status="lapsed")
        db, captured = _org_update_capture(row, {"id": ORG, "archived_at": None, "status": "active"})

        result = await service.unarchive_org(db, U1, ORG)

        payload = captured["payload"]
        assert payload["archived_at"] is None
        assert payload["covered_by"] == U1
        assert payload["covered_at"] is not None
        assert payload["status"] == "active"
        assert payload["grace_started_at"] is None
        assert set(payload.keys()) == {"archived_at", "covered_by", "covered_at", "status", "grace_started_at"}
        assert result["id"] == ORG

    async def test_unarchive_without_slot_raises_no_slot_error(self, monkeypatch):
        """Service raises standing.NoSlotError directly (like claim_coverage) —
        the router maps it to 402, same as create/claim."""
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

        def _raise_no_slot(*a, **kw):
            raise standing.NoSlotError("no slot")

        monkeypatch.setattr(standing, "require_free_slot", _raise_no_slot)
        row = _org_row(archived_at="2026-08-01T00:00:00+00:00")
        db = _db_seq({"organizations": [MagicMock(data=row, count=1)]})

        with pytest.raises(standing.NoSlotError):
            await service.unarchive_org(db, U1, ORG)

    async def test_unarchive_storage_guard_false_raises_402_with_free_up_space_copy(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        monkeypatch.setattr(standing, "require_free_slot", lambda *a, **kw: None)
        monkeypatch.setattr(storage_guard, "reactivation_allowed", lambda *a, **kw: False)
        row = _org_row(archived_at="2026-08-01T00:00:00+00:00")
        db = _db_seq({"organizations": [MagicMock(data=row, count=1)]})

        with pytest.raises(HTTPException) as exc_info:
            await service.unarchive_org(db, U1, ORG)
        assert exc_info.value.status_code == 402
        assert "free up space" in exc_info.value.detail

    async def test_unarchive_enterprise_org_raises_409(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        row = _org_row(kind="enterprise", archived_at="2026-08-01T00:00:00+00:00")
        db = _db_seq({"organizations": [MagicMock(data=row, count=1)]})

        with pytest.raises(HTTPException) as exc_info:
            await service.unarchive_org(db, U1, ORG)
        assert exc_info.value.status_code == 409

    async def test_unarchive_dissolved_org_raises_409(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        row = _org_row(archived_at="2026-08-01T00:00:00+00:00", dissolved_at="2026-08-02T00:00:00+00:00")
        db = _db_seq({"organizations": [MagicMock(data=row, count=1)]})

        with pytest.raises(HTTPException) as exc_info:
            await service.unarchive_org(db, U1, ORG)
        assert exc_info.value.status_code == 409
        assert "dissolved" in exc_info.value.detail.lower()

    async def test_unarchive_not_archived_raises_400(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        row = _org_row(archived_at=None)
        db = _db_seq({"organizations": [MagicMock(data=row, count=1)]})

        with pytest.raises(HTTPException) as exc_info:
            await service.unarchive_org(db, U1, ORG)
        assert exc_info.value.status_code == 400

    async def test_unarchive_requires_admin_403(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
        db = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await service.unarchive_org(db, U2, ORG)
        assert exc_info.value.status_code == 403


def test_count_covered_orgs_query_excludes_archived_and_dissolved():
    """Filter-pin: standing.count_covered_orgs is the mechanism by which an
    archived org's slot frees up (spec §3) — unarchive_org's require_free_slot
    call depends on archived rows NOT counting toward the slot. Captures the
    exact .eq()/.is_() filters, mirroring test_org_standing.py's
    TestPoolState.test_query_scopes_to_self_serve_covered_by_owner_excludes_dissolved."""
    captured = {"eq": [], "is_": []}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            original_eq, original_is_ = b.eq, b.is_

            def _eq(field, value):
                captured["eq"].append((field, value))
                return original_eq(field, value)

            def _is(field, value):
                captured["is_"].append((field, value))
                return original_is_(field, value)

            b.eq, b.is_ = _eq, _is
            b.execute.return_value = MagicMock(data=[], count=0)
        return b

    db = MagicMock()
    db.table.side_effect = _side
    standing.count_covered_orgs(db, U1)

    assert ("kind", "self_serve") in captured["eq"]
    assert ("covered_by", U1) in captured["eq"]
    assert ("archived_at", "null") in captured["is_"]
    assert ("dissolved_at", "null") in captured["is_"]


# ---------------------------------------------------------------------------
# Recurring top-up follows the PURCHASING ADMIN (spec 2026-08-15 §4.3, Task 11)
# ---------------------------------------------------------------------------


def _offboard_db(topup_admin, topup_sub="sub_topup_1", admins=(U1,)):
    """DB for an offboard of an ADMIN member (U2), with the organizations reads
    the top-up hook makes: 0) _offboard's own Task 17 _require_live_org check,
    1) topup_admin_id check, 2) cancel_topup's subscription id, 3) the update's
    echo + _org_name's read (clamped)."""
    revoked = "2026-07-20T12:00:00+00:00"
    seqs = {
        "org_members": [
            MagicMock(data=_member_row(role="admin"), count=1),
            MagicMock(data=[_member_row(role="admin", status="removed", revoked_at=revoked)], count=1),
            MagicMock(data=_member_row(role="admin", status="removed", revoked_at=revoked), count=1),
            MagicMock(data=[{"user_id": a} for a in admins], count=len(admins)),
        ],
        "organizations": [
            MagicMock(data={"archived_at": None, "dissolved_at": None}, count=1),
            MagicMock(data={"topup_admin_id": topup_admin}, count=1),
            MagicMock(data={"topup_stripe_subscription_id": topup_sub}, count=1),
            MagicMock(data={"name": "Acme"}, count=1),
        ],
    }
    counters = {k: 0 for k in seqs}
    captured = {"org_updates": [], "notifications": []}

    def _side(name):
        b = MockQueryBuilder()
        if name in seqs:
            i = min(counters[name], len(seqs[name]) - 1)
            counters[name] += 1
            b.execute.return_value = seqs[name][i]
        if name == "organizations":

            def _update(payload, *a, **k):
                captured["org_updates"].append(payload)
                return b

            b.update = _update
        if name == "notifications":

            def _insert(rows, *a, **k):
                captured["notifications"].append(rows)
                return b

            b.insert = _insert
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db, captured


async def test_offboarding_the_topup_purchaser_cancels_the_subscription(monkeypatch):
    """The top-up is billed to the purchasing admin's OWN card, so removing
    them must stop the charge — and the remaining admins must be told the pool
    stopped refilling."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db, captured = _offboard_db(topup_admin=U2)
    fake_stripe = MagicMock()

    with patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe):
        result = await service.remove_member(db, U1, ORG, MEMBER)

    assert result["status"] == "removed"
    fake_stripe.Subscription.delete.assert_called_once_with("sub_topup_1")
    assert {"topup_stripe_subscription_id": None, "topup_admin_id": None} in captured["org_updates"]
    assert captured["notifications"] and captured["notifications"][0][0]["user_id"] == U1


async def test_offboarding_a_non_purchasing_admin_leaves_the_topup_alone(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db, captured = _offboard_db(topup_admin=U1)  # someone else pays
    fake_stripe = MagicMock()

    with patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe):
        result = await service.remove_member(db, U1, ORG, MEMBER)

    assert result["status"] == "removed"
    fake_stripe.Subscription.delete.assert_not_called()
    assert captured["org_updates"] == []
    assert captured["notifications"] == []


async def test_offboard_survives_a_failing_topup_cancel(monkeypatch):
    """Best-effort, exactly like the access revocation next to it: a Stripe
    outage must not strand the offboard (the org's columns still point at the
    subscription, so /orgs/{id}/cancel-topup can retry)."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db, _ = _offboard_db(topup_admin=U2)
    fake_stripe = MagicMock()
    fake_stripe.Subscription.delete.side_effect = RuntimeError("stripe down")

    with patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe):
        result = await service.remove_member(db, U1, ORG, MEMBER)

    assert result["status"] == "removed"


def _cancel_topup_db(topup_sub):
    captured = {"org_updates": []}

    def _side(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data={"topup_stripe_subscription_id": topup_sub}, count=1)

            def _update(payload, *a, **k):
                captured["org_updates"].append(payload)
                return b

            b.update = _update
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db, captured


async def test_cancel_topup_cancels_at_stripe_and_clears_both_columns():
    """The one implementation behind POST /orgs/{id}/cancel-topup and the
    offboard hook."""
    db, captured = _cancel_topup_db("sub_topup_1")
    fake_stripe = MagicMock()
    with patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe):
        assert service.cancel_topup(db, ORG) is True
    fake_stripe.Subscription.delete.assert_called_once_with("sub_topup_1")
    assert captured["org_updates"] == [{"topup_stripe_subscription_id": None, "topup_admin_id": None}]


async def test_cancel_topup_is_a_no_op_without_a_subscription():
    db, captured = _cancel_topup_db(None)
    fake_stripe = MagicMock()
    with patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe):
        assert service.cancel_topup(db, ORG) is False
    fake_stripe.Subscription.delete.assert_not_called()
    assert captured["org_updates"] == []


async def test_cancel_topup_keeps_the_columns_when_stripe_refuses():
    """A failed cancel must NOT clear the pointer: the columns are what lets
    the admin retry (and what customer.subscription.deleted clears for real)."""
    db, captured = _cancel_topup_db("sub_topup_1")
    fake_stripe = MagicMock()
    fake_stripe.Subscription.delete.side_effect = RuntimeError("stripe down")
    with (
        patch("subscriptions.stripe_client.get_stripe", return_value=fake_stripe),
        pytest.raises(RuntimeError),
    ):
        service.cancel_topup(db, ORG)
    assert captured["org_updates"] == []


# ---------------------------------------------------------------------------
# Task 17: _require_live_org gates every mutating /orgs endpoint (the control-
# plane gap RLS's can_access_artist didn't cover). Per endpoint-CLASS, not per
# endpoint — update_member_role stands in for the ~14 guarded mutations, all
# wired the same way (authz first, then the shared guard). The lifecycle-op
# exemption is already covered by TestUnarchiveOrg.
# test_unarchive_happy_path_asserts_update_payload (unarchive succeeding on an
# archived org IS that proof), so it isn't repeated here.
# ---------------------------------------------------------------------------


async def test_archived_org_blocks_a_member_management_mutation(monkeypatch):
    """update_member_role stands in for the guarded mutation class: the 409
    fires before any org_members write is attempted (no org_members table
    configured on this db at all)."""
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"organizations": [MagicMock(data=_org_row(archived_at="2026-08-01T00:00:00+00:00"), count=1)]})

    with pytest.raises(HTTPException) as exc_info:
        await service.update_member_role(db, U1, ORG, MEMBER, "admin")
    assert exc_info.value.status_code == 409
    assert "archived" in exc_info.value.detail.lower()


async def test_dissolved_org_blocks_a_member_management_mutation(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    db = _db_seq({"organizations": [MagicMock(data=_org_row(dissolved_at="2026-08-01T00:00:00+00:00"), count=1)]})

    with pytest.raises(HTTPException) as exc_info:
        await service.update_member_role(db, U1, ORG, MEMBER, "admin")
    assert exc_info.value.status_code == 409
    assert "dissolved" in exc_info.value.detail.lower()


async def test_get_org_read_still_works_on_an_archived_org(monkeypatch):
    """Reads are deliberately NOT gated (Task 17): an admin (or support,
    reading through the same endpoint) must still be able to see an archived
    org's state — mirrors test_get_org_zero_balance_when_no_wallet's shape,
    with archived_at set on the row."""
    monkeypatch.setattr(service.authz, "is_org_member", lambda *a: True)
    db = _db_seq(
        {
            "organizations": [
                MagicMock(
                    data={
                        "id": ORG,
                        "status": "active",
                        "archived_at": "2026-08-01T00:00:00+00:00",
                        "min_initial_purchase_credits": None,
                    },
                    count=1,
                )
            ],
            "org_members": [
                MagicMock(data={"role": "member"}, count=1),
                MagicMock(data=[], count=0),
                MagicMock(data=[], count=0),
            ],
            "credit_wallets": [MagicMock(data=[], count=0)],
        }
    )

    result = await service.get_org(db, U1, ORG)
    assert result["archived_at"] == "2026-08-01T00:00:00+00:00"
