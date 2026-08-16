"""orgs/standing.py — slots, dials, coverage counting. Also orgs.service's
coverage claim/release endpoints and orgs/storage_guard.py (Task 5)."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from orgs import service, standing, storage_guard
from tests.conftest import MockQueryBuilder
from tests.test_orgs_service import _db_seq

U1 = "00000000-0000-0000-0000-000000000001"
U2 = "00000000-0000-0000-0000-000000000002"
ORG = "20000000-0000-0000-0000-000000000001"


def _db(*, is_admin=False, tier="basic", tier_row=None, covered=0):
    db = MagicMock()

    def side_effect(name):
        b = MockQueryBuilder()
        if name == "profiles":
            b.execute.return_value = MagicMock(data=[{"is_admin": is_admin}], count=1)
        elif name == "subscriptions":
            b.execute.return_value = MagicMock(data=[{"tier": tier}], count=1)
        elif name == "tier_entitlements":
            row = tier_row or {
                "tier": tier,
                "max_teams": 1,
                "max_team_members": 3,
                "team_storage_bytes": 10 * 2**30,
            }
            b.execute.return_value = MagicMock(data=[row], count=1)
        elif name == "organizations":
            b.execute.return_value = MagicMock(data=[], count=covered)
        return b

    db.table.side_effect = side_effect
    return db


class TestTeamDials:
    def test_reads_subscription_tier_row(self):
        d = standing.team_dials_for_user(_db(tier="basic"), U1)
        assert (d.max_teams, d.max_team_members) == (1, 3)

    def test_msanii_admin_resolves_pro_dials(self):
        db = _db(
            is_admin=True,
            tier="free",
            tier_row={"tier": "pro", "max_teams": 3, "max_team_members": 10, "team_storage_bytes": 100 * 2**30},
        )
        assert standing.team_dials_for_user(db, U1).max_teams == 3

    def test_missing_rows_fail_closed(self):
        db = MagicMock()
        db.table.side_effect = lambda name: MockQueryBuilder()
        assert standing.team_dials_for_user(db, U1).max_teams == 0


class TestCreateOrgSlotGate:
    async def test_free_user_cannot_create(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        db = _db(
            tier="free",
            tier_row={"tier": "free", "max_teams": 0, "max_team_members": 0, "team_storage_bytes": 0},
        )
        with pytest.raises(standing.NoSlotError):
            await service.create_org(db, U1, "My Team")

    async def test_basic_user_creates_active_self_serve(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        captured = {}
        org_calls = {"n": 0}

        def _side(name):
            b = MockQueryBuilder()
            if name == "profiles":
                b.execute.return_value = MagicMock(data=[{"is_admin": False}], count=1)
            elif name == "subscriptions":
                b.execute.return_value = MagicMock(data=[{"tier": "basic"}], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(
                    data=[{"tier": "basic", "max_teams": 1, "max_team_members": 3, "team_storage_bytes": 10 * 2**30}],
                    count=1,
                )
            elif name == "organizations":
                org_calls["n"] += 1
                if org_calls["n"] == 1:
                    # count_covered_orgs: no existing covered orgs yet -> slot free
                    b.execute.return_value = MagicMock(data=[], count=0)
                else:
                    # the actual insert
                    original_insert = b.insert

                    def _capture_insert(payload, *a, **kw):
                        captured["payload"] = payload
                        return original_insert(payload, *a, **kw)

                    b.insert = _capture_insert
                    b.execute.return_value = MagicMock(
                        data=[{"id": "org1", "name": "My Team", "kind": "self_serve", "status": "active"}], count=1
                    )
            return b

        db = MagicMock()
        db.table.side_effect = _side
        result = await service.create_org(db, U1, "My Team")

        payload = captured["payload"]
        assert payload["kind"] == "self_serve" and payload["status"] == "active"
        assert payload["covered_by"] == U1 and payload["covered_at"]
        assert result["id"] == "org1"
        assert result["my_role"] == "admin"

    async def test_second_basic_team_hits_the_slot_wall(self, monkeypatch):
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.setenv("CREDITS_ENABLED", "true")
        db = _db(
            tier="basic",
            tier_row={"tier": "basic", "max_teams": 1, "max_team_members": 3, "team_storage_bytes": 10 * 2**30},
            covered=1,
        )
        with pytest.raises(standing.NoSlotError):
            await service.create_org(db, U1, "Second Team")

    async def test_flags_off_is_legacy_create(self, monkeypatch):
        monkeypatch.delenv("LICENSING_ENABLED", raising=False)
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)
        captured = {}

        def _side(name):
            b = MockQueryBuilder()
            if name == "organizations":
                original_insert = b.insert

                def _capture_insert(payload, *a, **kw):
                    captured["payload"] = payload
                    return original_insert(payload, *a, **kw)

                b.insert = _capture_insert
                b.execute.return_value = MagicMock(
                    data=[{"id": "org1", "name": "Legacy Team", "status": "pending"}], count=1
                )
            return b

        db = MagicMock()
        db.table.side_effect = _side
        result = await service.create_org(db, U1, "Legacy Team")

        payload = captured["payload"]
        assert payload == {"name": "Legacy Team", "created_by": U1}
        assert "kind" not in payload
        assert "covered_by" not in payload
        assert result["status"] == "pending"

    async def test_half_flag_window_refuses_with_503(self, monkeypatch):
        """LICENSING_ENABLED on + CREDITS_ENABLED off must NOT fall through
        to the legacy branch — that would silently mint a permanent
        kind='enterprise' row nobody governs. Refuse outright instead."""
        monkeypatch.setenv("LICENSING_ENABLED", "true")
        monkeypatch.delenv("CREDITS_ENABLED", raising=False)

        db = MagicMock()
        db.table.side_effect = lambda name: MockQueryBuilder()  # must not be reached
        with pytest.raises(HTTPException) as exc_info:
            await service.create_org(db, U1, "My Team")
        assert exc_info.value.status_code == 503
        db.table.assert_not_called()


# ---------------------------------------------------------------------------
# Coverage claim / release (spec 2026-08-15 §2 rev 2, Task 5)
# ---------------------------------------------------------------------------


def _org_row(**overrides):
    row = {
        "kind": "self_serve",
        "status": "active",
        "covered_by": None,
        "covered_at": None,
        "archived_at": None,
        "dissolved_at": None,
    }
    row.update(overrides)
    return row


def _capture_update_db(first_read: dict, updated_row: dict):
    """organizations table: the FIRST execute() serves _first_org's read; the
    SECOND serves the update, with its payload captured for assertion."""
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


class TestClaimCoverage:
    async def test_claim_happy_path_asserts_update_payload(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        monkeypatch.setattr(standing, "require_free_slot", lambda *a, **kw: None)
        db, captured = _capture_update_db(_org_row(), {"id": ORG, "covered_by": U1})

        result = await service.claim_coverage(db, U1, ORG)

        payload = captured["payload"]
        assert payload["covered_by"] == U1
        assert payload["covered_at"] is not None
        assert payload["grace_started_at"] is None
        assert "status" not in payload  # not lapsed -> no status flip
        assert result["id"] == ORG

    async def test_claim_without_slot_raises_no_slot_error(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

        def _raise_no_slot(*a, **kw):
            raise standing.NoSlotError("no slot")

        monkeypatch.setattr(standing, "require_free_slot", _raise_no_slot)
        db = _db_seq({"organizations": [MagicMock(data=_org_row(), count=1)]})

        with pytest.raises(standing.NoSlotError):
            await service.claim_coverage(db, U1, ORG)

    async def test_claim_by_non_admin_raises_403(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
        db = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await service.claim_coverage(db, U1, ORG)
        assert exc_info.value.status_code == 403

    async def test_claim_enterprise_org_raises_409(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        db = _db_seq({"organizations": [MagicMock(data=_org_row(kind="enterprise"), count=1)]})

        with pytest.raises(HTTPException) as exc_info:
            await service.claim_coverage(db, U1, ORG)
        assert exc_info.value.status_code == 409

    async def test_claim_archived_org_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        row = _org_row(archived_at="2026-08-01T00:00:00+00:00")
        db = _db_seq({"organizations": [MagicMock(data=row, count=1)]})

        with pytest.raises(ValueError):
            await service.claim_coverage(db, U1, ORG)

    async def test_claim_dissolved_org_raises_value_error(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        row = _org_row(dissolved_at="2026-08-01T00:00:00+00:00")
        db = _db_seq({"organizations": [MagicMock(data=row, count=1)]})

        with pytest.raises(ValueError):
            await service.claim_coverage(db, U1, ORG)

    async def test_claim_already_covering_is_idempotent_noop(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)

        def _fail_if_called(*a, **kw):
            raise AssertionError("require_free_slot must not be called on the idempotent no-op path")

        monkeypatch.setattr(standing, "require_free_slot", _fail_if_called)
        row = _org_row(covered_by=U1, covered_at="2026-08-01T00:00:00+00:00")
        db = _db_seq({"organizations": [MagicMock(data=row, count=1)]})

        result = await service.claim_coverage(db, U1, ORG)
        assert result == row

    async def test_release_then_reclaim_by_same_admin_is_full_claim(self, monkeypatch):
        """covered_by == user_id but covered_at is NULL (released) must NOT
        hit the idempotent early-return — it falls through to a full claim
        (review r2 hole 3: release-then-reclaim by the same admin)."""
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        monkeypatch.setattr(standing, "require_free_slot", lambda *a, **kw: None)
        row = _org_row(covered_by=U1, covered_at=None)
        db, captured = _capture_update_db(row, {"id": ORG, "covered_by": U1})

        result = await service.claim_coverage(db, U1, ORG)

        payload = captured["payload"]
        assert payload["covered_by"] == U1
        assert payload["covered_at"] is not None
        assert payload["grace_started_at"] is None
        assert result == {"id": ORG, "covered_by": U1}

    async def test_claim_lapsed_blocked_when_reactivation_not_allowed(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        monkeypatch.setattr(standing, "require_free_slot", lambda *a, **kw: None)
        monkeypatch.setattr(storage_guard, "reactivation_allowed", lambda *a, **kw: False)
        db = _db_seq({"organizations": [MagicMock(data=_org_row(status="lapsed"), count=1)]})

        with pytest.raises(HTTPException) as exc_info:
            await service.claim_coverage(db, U1, ORG)
        assert exc_info.value.status_code == 402

    async def test_claim_lapsed_allowed_flips_status_active(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        monkeypatch.setattr(standing, "require_free_slot", lambda *a, **kw: None)
        monkeypatch.setattr(storage_guard, "reactivation_allowed", lambda *a, **kw: True)
        db, captured = _capture_update_db(_org_row(status="lapsed"), {"id": ORG, "status": "active"})

        result = await service.claim_coverage(db, U1, ORG)

        assert captured["payload"]["status"] == "active"
        assert result["status"] == "active"


class TestReleaseCoverage:
    async def test_release_by_non_coverer_raises_403(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        db = _db_seq({"organizations": [MagicMock(data=_org_row(covered_by=U2), count=1)]})

        with pytest.raises(HTTPException) as exc_info:
            await service.release_coverage(db, U1, ORG)
        assert exc_info.value.status_code == 403

    async def test_release_payload_does_not_touch_covered_by(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        row = _org_row(covered_by=U1, covered_at="2026-08-01T00:00:00+00:00")
        db, captured = _capture_update_db(row, {"id": ORG, "covered_by": U1, "covered_at": None})

        result = await service.release_coverage(db, U1, ORG)

        assert captured["payload"] == {"covered_at": None}
        assert "covered_by" not in captured["payload"]
        assert result["covered_by"] == U1

    async def test_release_enterprise_org_raises_409(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        db = _db_seq({"organizations": [MagicMock(data=_org_row(kind="enterprise", covered_by=U1), count=1)]})

        with pytest.raises(HTTPException) as exc_info:
            await service.release_coverage(db, U1, ORG)
        assert exc_info.value.status_code == 409

    async def test_release_conditional_update_rejects_lost_race(self, monkeypatch):
        """The read at the top says covered_by=U1, but a rival admin's claim
        landed before this write — the update is conditioned on
        covered_by=user_id, so the race matches zero rows: no covered_at
        change lands, and the caller gets 403 rather than a silent no-op
        success that would have nulled the rival's fresh covered_at."""
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        row = _org_row(covered_by=U1, covered_at="2026-08-01T00:00:00+00:00")
        captured = {"eq": [], "update_payload": None}
        calls = {"n": 0}

        def _side(name):
            b = MockQueryBuilder()
            if name == "organizations":
                calls["n"] += 1
                if calls["n"] == 1:
                    b.execute.return_value = MagicMock(data=row, count=1)
                else:
                    original_update, original_eq = b.update, b.eq

                    def _capture_update(payload, *a, **kw):
                        captured["update_payload"] = payload
                        return original_update(payload, *a, **kw)

                    def _capture_eq(field, value):
                        captured["eq"].append((field, value))
                        return original_eq(field, value)

                    b.update, b.eq = _capture_update, _capture_eq
                    # The race: covered_by changed underneath us between the
                    # read and this write -> the conditional match is empty.
                    b.execute.return_value = MagicMock(data=[], count=0)
            return b

        db = MagicMock()
        db.table.side_effect = _side

        with pytest.raises(HTTPException) as exc_info:
            await service.release_coverage(db, U1, ORG)

        assert exc_info.value.status_code == 403
        assert captured["update_payload"] == {"covered_at": None}
        assert ("covered_by", U1) in captured["eq"]


# ---------------------------------------------------------------------------
# orgs/storage_guard.py — pool_state + reactivation_allowed (Task 5's REAL
# bodies; Task 12 adds upload_allowed on top).
# ---------------------------------------------------------------------------


def _storage_db(*, org_rows, tier="pro", tier_row=None, is_admin=False):
    db = MagicMock()

    def side_effect(name):
        b = MockQueryBuilder()
        if name == "organizations":
            b.execute.return_value = MagicMock(data=org_rows, count=len(org_rows))
        elif name == "profiles":
            b.execute.return_value = MagicMock(data=[{"is_admin": is_admin}], count=1)
        elif name == "subscriptions":
            b.execute.return_value = MagicMock(data=[{"tier": tier}], count=1)
        elif name == "tier_entitlements":
            row = tier_row or {
                "tier": tier,
                "max_teams": 3,
                "max_team_members": 10,
                "team_storage_bytes": 100 * 2**30,
            }
            b.execute.return_value = MagicMock(data=[row], count=1)
        return b

    db.table.side_effect = side_effect
    return db


class TestPoolState:
    def test_sums_storage_bytes_across_returned_orgs(self):
        """archived/lapsed orgs are covered by this sum — pool_state's query
        carries no status filter, only kind/covered_by/dissolved_at (asserted
        below), so whatever the DB returns (including archived/lapsed rows)
        is summed."""
        db = _storage_db(org_rows=[{"storage_bytes": 5 * 2**30}, {"storage_bytes": 2 * 2**30}, {"storage_bytes": None}])
        used, pool, _ = storage_guard.pool_state(db, U1)
        assert used == 7 * 2**30
        assert pool == 100 * 2**30

    def test_query_scopes_to_self_serve_covered_by_owner_excludes_dissolved(self):
        captured = {"eq": [], "is_": []}

        def side(name):
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
            elif name == "profiles":
                b.execute.return_value = MagicMock(data=[{"is_admin": False}], count=1)
            elif name == "subscriptions":
                b.execute.return_value = MagicMock(data=[{"tier": "basic"}], count=1)
            elif name == "tier_entitlements":
                b.execute.return_value = MagicMock(
                    data=[{"tier": "basic", "max_teams": 1, "max_team_members": 3, "team_storage_bytes": 10 * 2**30}],
                    count=1,
                )
            return b

        db = MagicMock()
        db.table.side_effect = side
        storage_guard.pool_state(db, U1)

        assert ("kind", "self_serve") in captured["eq"]
        assert ("covered_by", U1) in captured["eq"]
        assert not any(f == "status" for f, _ in captured["eq"])  # archived/lapsed included
        assert ("dissolved_at", "null") in captured["is_"]

    def test_pro_like_threshold_over_10gib(self):
        db = _storage_db(
            org_rows=[],
            tier="pro",
            tier_row={"tier": "pro", "max_teams": 3, "max_team_members": 10, "team_storage_bytes": 10 * 2**30 + 1},
        )
        _, _, is_pro_like = storage_guard.pool_state(db, U1)
        assert is_pro_like is True

    def test_not_pro_like_at_exactly_10gib(self):
        db = _storage_db(
            org_rows=[],
            tier="basic",
            tier_row={"tier": "basic", "max_teams": 1, "max_team_members": 3, "team_storage_bytes": 10 * 2**30},
        )
        _, _, is_pro_like = storage_guard.pool_state(db, U1)
        assert is_pro_like is False


class TestReactivationAllowed:
    def test_true_when_used_within_pool(self):
        db = _storage_db(
            org_rows=[{"storage_bytes": 5 * 2**30}],
            tier="basic",
            tier_row={"tier": "basic", "max_teams": 1, "max_team_members": 3, "team_storage_bytes": 10 * 2**30},
        )
        assert storage_guard.reactivation_allowed(db, U1) is True

    def test_false_when_over_pool_and_not_pro_like(self):
        db = _storage_db(
            org_rows=[{"storage_bytes": 11 * 2**30}],
            tier="basic",
            tier_row={"tier": "basic", "max_teams": 1, "max_team_members": 3, "team_storage_bytes": 10 * 2**30},
        )
        assert storage_guard.reactivation_allowed(db, U1) is False

    def test_true_when_pro_like_even_over_pool(self):
        db = _storage_db(
            org_rows=[{"storage_bytes": 200 * 2**30}],
            tier="pro",
            tier_row={"tier": "pro", "max_teams": 3, "max_team_members": 10, "team_storage_bytes": 100 * 2**30},
        )
        assert storage_guard.reactivation_allowed(db, U1) is True


# ---------------------------------------------------------------------------
# evaluate_standing (Task 6, spec §3) — grace -> lapsed -> recovery. A small
# store-backed fake (like test_registry_derive.py's _db) rather than the
# per-table MockQueryBuilder above: evaluate_standing chains eq/is_/in_ over
# "organizations" and then issues follow-up reads/writes (team_dials_for_user,
# _holds_active_admin_seat, the org update, _notify_standing's admin lookup +
# notification insert) whose call ORDER depends on the data, so a call-
# sequence mock would be brittle across the seven scenarios below.
# ---------------------------------------------------------------------------


class _StoreBuilder:
    """Chainable query builder over one in-memory table (store[table_name], a
    list of row dicts). Supports exactly the eq/in_/is_ filters and
    select/insert/update ops evaluate_standing's whole call graph issues.
    update/insert mutate the SAME store, so later reads in the same
    evaluate_standing() call observe earlier writes — mirrors real Postgres
    read-your-writes within one request."""

    def __init__(self, store, table_name):
        self._store = store
        self._table = table_name
        self._filters = []
        self._op = "select"
        self._payload = None
        self._limit_n = None

    def select(self, *_a, **_k):
        self._op = "select"
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def update(self, payload):
        self._op = "update"
        self._payload = payload
        return self

    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def in_(self, col, vals):
        self._filters.append(("in", col, list(vals)))
        return self

    def is_(self, col, val):
        self._filters.append(("isnull" if val == "null" else "eq", col, None if val == "null" else val))
        return self

    def limit(self, n):
        self._limit_n = n
        return self

    def _match(self, row):
        for kind, col, val in self._filters:
            rv = row.get(col)
            if kind == "eq" and rv != val:
                return False
            if kind == "in" and rv not in val:
                return False
            if kind == "isnull" and rv is not None:
                return False
        return True

    def execute(self):
        rows = self._store.setdefault(self._table, [])
        if self._op == "insert":
            new_rows = self._payload if isinstance(self._payload, list) else [self._payload]
            for r in new_rows:
                rows.append(deepcopy(r))
            return MagicMock(data=[deepcopy(r) for r in new_rows])
        if self._op == "update":
            matched = [r for r in rows if self._match(r)]
            for r in matched:
                r.update(self._payload)
            return MagicMock(data=[deepcopy(r) for r in matched])
        matched = [deepcopy(r) for r in rows if self._match(r)]
        if self._limit_n is not None:
            matched = matched[: self._limit_n]
        return MagicMock(data=matched, count=len(matched))


class _StandingDB:
    def __init__(self, **tables):
        self.store = {name: list(rows) for name, rows in tables.items()}

    def table(self, name):
        return _StoreBuilder(self.store, name)


def _iso_days_ago(days):
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _dials_rows(user_id, *, tier="basic", max_teams=1):
    """(profiles, subscriptions, tier_entitlements) rows for a non-admin user
    on `tier` with `max_teams` slots — the trio team_dials_for_user reads."""
    return (
        [{"id": user_id, "is_admin": False}],
        [{"user_id": user_id, "tier": tier}],
        [{"tier": tier, "max_teams": max_teams, "max_team_members": 3, "team_storage_bytes": 10 * 2**30}],
    )


def _admin_row(org_id, user_id, email="admin@example.com"):
    return {"org_id": org_id, "user_id": user_id, "role": "admin", "status": "active", "email": email}


def _standing_org(**overrides):
    row = {
        "id": "org-1",
        "name": "Test Org",
        "kind": "self_serve",
        "status": "active",
        "covered_by": None,
        "covered_at": None,
        "grace_started_at": None,
        "archived_at": None,
        "dissolved_at": None,
        "storage_bytes": 0,
    }
    row.update(overrides)
    return row


class TestEvaluateStanding:
    @pytest.fixture(autouse=True)
    def sent_emails(self, monkeypatch):
        """Patches orgs.emails._send (house norm — see
        test_orgs_service.py::test_invite_email_links_to_the_tokened_claim_page)
        for EVERY test in this class, so nothing here can hit live Resend even
        though `_send` already no-ops without RESEND_API_KEY/RESEND_FROM_EMAIL.
        Returns the list of captured `_send(**kwargs)` calls (subject/html_body/
        recipients) for tests that want to assert on the notification email."""
        from orgs import emails

        calls: list = []
        monkeypatch.setattr(emails, "_send", lambda **kw: calls.append(kw))
        return calls

    def test_covered_org_untouched(self):
        """(a) A fully-covered org (ranked within max_teams, coverer holds an
        active admin seat) is left exactly as-is — no update, no notify."""
        org = _standing_org(id="org-a", covered_by=U1, covered_at=_iso_days_ago(1))
        expected = deepcopy(org)
        profiles, subs, tiers = _dials_rows(U1, tier="pro", max_teams=1)
        db = _StandingDB(
            organizations=[org],
            org_members=[_admin_row("org-a", U1)],
            profiles=profiles,
            subscriptions=subs,
            tier_entitlements=tiers,
            notifications=[],
        )

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 1, "lapsed": 0, "grace_started": 0}
        assert db.store["organizations"][0] == expected
        assert db.store["notifications"] == []

    def test_oldest_of_four_orgs_uncovered_gets_grace_stamped(self):
        """(b) A pro owner with 4 orgs but only 2 slots: the two ranked lowest
        by covered_at (including the oldest) fall out of coverage -> grace."""
        orgs = [_standing_org(id=f"org-{i}", covered_by=U1, covered_at=_iso_days_ago(i)) for i in range(4)]
        profiles, subs, tiers = _dials_rows(U1, tier="pro", max_teams=2)
        members = [_admin_row(f"org-{i}", U1) for i in range(4)]
        db = _StandingDB(
            organizations=orgs,
            org_members=members,
            profiles=profiles,
            subscriptions=subs,
            tier_entitlements=tiers,
            notifications=[],
        )

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 4, "lapsed": 0, "grace_started": 2}
        by_id = {o["id"]: o for o in db.store["organizations"]}
        assert by_id["org-0"]["grace_started_at"] is None  # newest -> ranked in
        assert by_id["org-1"]["grace_started_at"] is None  # 2nd newest -> ranked in
        assert by_id["org-2"]["grace_started_at"] is not None  # falls out -> grace
        assert by_id["org-3"]["grace_started_at"] is not None  # oldest -> grace

    def test_grace_older_than_window_lapses_and_notifies_admins(self, sent_emails):
        """(c) Grace older than grace_days() flips the org to lapsed, inserts
        a notification row for every active admin, and emails them (kind +
        recipients asserted against the patched orgs.emails._send call)."""
        org = _standing_org(id="org-b", covered_by=None, grace_started_at=_iso_days_ago(20))
        db = _StandingDB(
            organizations=[org],
            org_members=[_admin_row("org-b", U2)],
            profiles=[],
            subscriptions=[],
            tier_entitlements=[],
            notifications=[],
        )

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 1, "lapsed": 1, "grace_started": 0}
        assert db.store["organizations"][0]["status"] == "lapsed"
        notifs = db.store["notifications"]
        assert len(notifs) == 1
        assert notifs[0]["user_id"] == U2
        assert notifs[0]["type"] == "status_change"
        assert notifs[0]["entity_type"] == "org"
        assert notifs[0]["entity_id"] == "org-b"
        assert notifs[0]["metadata"]["reason"] == "standing_lapsed"
        assert len(sent_emails) == 1
        assert sent_emails[0]["recipients"] == ["admin@example.com"]
        assert "paused" in sent_emails[0]["subject"]

    def test_lapsed_org_whose_owner_reupgraded_reactivates(self, monkeypatch, sent_emails):
        """(d) Owner re-upgraded (now has a free slot again): a lapsed org
        that is covered again un-lapses and clears grace, guarded by
        storage_guard.reactivation_allowed (mocked True here). The write must
        land BEFORE the notify — asserted indirectly since reactivation_allowed
        is mocked, but the store state is checked before touching notifications
        so a swapped order would still show up as a stale org row."""
        monkeypatch.setattr(storage_guard, "reactivation_allowed", lambda *a, **kw: True)
        org = _standing_org(
            id="org-c",
            status="lapsed",
            covered_by=U1,
            covered_at=_iso_days_ago(30),
            grace_started_at=_iso_days_ago(30),
        )
        profiles, subs, tiers = _dials_rows(U1, tier="pro", max_teams=1)
        db = _StandingDB(
            organizations=[org],
            org_members=[_admin_row("org-c", U1)],
            profiles=profiles,
            subscriptions=subs,
            tier_entitlements=tiers,
            notifications=[],
        )

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 1, "lapsed": 0, "grace_started": 0}
        stored = db.store["organizations"][0]
        assert stored["status"] == "active"
        assert stored["grace_started_at"] is None
        assert len(db.store["notifications"]) == 1
        assert db.store["notifications"][0]["metadata"]["reason"] == "standing_reactivated"
        assert len(sent_emails) == 1
        assert sent_emails[0]["recipients"] == ["admin@example.com"]
        assert "active again" in sent_emails[0]["subject"]

    def test_suspended_org_untouched(self):
        """(e) A suspended org is never even selected by the sweep's query
        (status IN ('active','lapsed') only) — the guard lives in the query,
        not in per-row logic."""
        org = _standing_org(id="org-e", status="suspended", covered_by=U1, covered_at=_iso_days_ago(1))
        expected = deepcopy(org)
        db = _StandingDB(
            organizations=[org],
            org_members=[],
            profiles=[],
            subscriptions=[],
            tier_entitlements=[],
            notifications=[],
        )

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 0, "lapsed": 0, "grace_started": 0}
        assert db.store["organizations"][0] == expected
        assert db.store["notifications"] == []

    def test_released_org_gets_grace(self, sent_emails):
        """(f) Released (covered_by still set, covered_at NULL) is always
        uncovered regardless of ranking — grace, not lapse yet."""
        org = _standing_org(id="org-f", covered_by=U1, covered_at=None)
        profiles, subs, tiers = _dials_rows(U1, tier="pro", max_teams=3)
        db = _StandingDB(
            organizations=[org],
            org_members=[_admin_row("org-f", U1)],
            profiles=profiles,
            subscriptions=subs,
            tier_entitlements=tiers,
            notifications=[],
        )

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 1, "lapsed": 0, "grace_started": 1}
        stored = db.store["organizations"][0]
        assert stored["grace_started_at"] is not None
        assert stored["status"] == "active"
        assert len(sent_emails) == 1
        assert "losing access soon" in sent_emails[0]["subject"]

    def test_enterprise_org_never_selected(self):
        """(g) An enterprise org (kind != 'self_serve') is excluded by the
        query's kind filter, whatever its status/coverage looks like."""
        org = _standing_org(id="org-g", kind="enterprise", covered_by=None, status="active")
        expected = deepcopy(org)
        db = _StandingDB(organizations=[org])

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 0, "lapsed": 0, "grace_started": 0}
        assert db.store["organizations"][0] == expected

    def test_empty_scan_returns_zeros(self):
        assert standing.evaluate_standing(_StandingDB(organizations=[])) == {
            "evaluated": 0,
            "lapsed": 0,
            "grace_started": 0,
        }

    def test_lapsed_org_over_pool_stays_lapsed(self, monkeypatch):
        """The storage guard's negative branch: covered but over-pool ->
        stays lapsed, no notify, no write at all."""
        monkeypatch.setattr(storage_guard, "reactivation_allowed", lambda *a, **kw: False)
        grace_at = _iso_days_ago(30)
        org = _standing_org(
            id="org-d2", status="lapsed", covered_by=U1, covered_at=_iso_days_ago(30), grace_started_at=grace_at
        )
        profiles, subs, tiers = _dials_rows(U1, tier="pro", max_teams=1)
        db = _StandingDB(
            organizations=[org],
            org_members=[_admin_row("org-d2", U1)],
            profiles=profiles,
            subscriptions=subs,
            tier_entitlements=tiers,
            notifications=[],
        )

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 1, "lapsed": 0, "grace_started": 0}
        stored = db.store["organizations"][0]
        assert stored["status"] == "lapsed"
        assert stored["grace_started_at"] == grace_at
        assert db.store["notifications"] == []

    def test_bad_row_is_isolated_and_logged(self, monkeypatch, caplog):
        """A raising per-org step (a corrupt grace_started_at) must not stop
        the sweep from processing the rest of the scan."""
        import logging

        bad = _standing_org(id="org-bad", covered_by=None, grace_started_at="not-a-timestamp")
        good = _standing_org(id="org-good", covered_by=None, grace_started_at=_iso_days_ago(20))
        db = _StandingDB(
            organizations=[bad, good],
            org_members=[],
            profiles=[],
            subscriptions=[],
            tier_entitlements=[],
            notifications=[],
        )

        with caplog.at_level(logging.ERROR):
            result = standing.evaluate_standing(db)

        assert result == {"evaluated": 2, "lapsed": 1, "grace_started": 0}
        by_id = {o["id"]: o for o in db.store["organizations"]}
        assert by_id["org-good"]["status"] == "lapsed"
        assert by_id["org-bad"]["status"] == "active"  # untouched by the failed row
        assert any("standing evaluation failed" in r.getMessage() for r in caplog.records)

    # -----------------------------------------------------------------------
    # _holds_active_admin_seat returning False: ranked in (within max_teams,
    # covered_at recent) is not enough on its own — the identity check must
    # also hold, or the org enters grace exactly as if it were never ranked.
    # -----------------------------------------------------------------------

    def test_ranked_in_but_demoted_to_member_enters_grace(self):
        """Coverer demoted admin -> member: keeps the ranking slot but not
        the seat that earns coverage."""
        org = _standing_org(id="org-h1", covered_by=U1, covered_at=_iso_days_ago(1))
        profiles, subs, tiers = _dials_rows(U1, tier="pro", max_teams=1)
        db = _StandingDB(
            organizations=[org],
            org_members=[{"org_id": "org-h1", "user_id": U1, "role": "member", "status": "active", "email": "a@x.com"}],
            profiles=profiles,
            subscriptions=subs,
            tier_entitlements=tiers,
            notifications=[],
        )

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 1, "lapsed": 0, "grace_started": 1}
        assert db.store["organizations"][0]["grace_started_at"] is not None

    def test_ranked_in_but_suspended_admin_enters_grace(self):
        """Coverer's admin seat suspended: same guard, same outcome as a
        demotion — an inactive seat confers no coverage."""
        org = _standing_org(id="org-h2", covered_by=U1, covered_at=_iso_days_ago(1))
        profiles, subs, tiers = _dials_rows(U1, tier="pro", max_teams=1)
        db = _StandingDB(
            organizations=[org],
            org_members=[
                {"org_id": "org-h2", "user_id": U1, "role": "admin", "status": "suspended", "email": "a@x.com"}
            ],
            profiles=profiles,
            subscriptions=subs,
            tier_entitlements=tiers,
            notifications=[],
        )

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 1, "lapsed": 0, "grace_started": 1}
        assert db.store["organizations"][0]["grace_started_at"] is not None

    def test_ranked_in_but_no_org_members_row_enters_grace(self):
        """covered_by points at a user with NO org_members row at all in this
        org (e.g. a fully-removed seat, or a stale covered_by) — same guard."""
        org = _standing_org(id="org-h3", covered_by=U1, covered_at=_iso_days_ago(1))
        profiles, subs, tiers = _dials_rows(U1, tier="pro", max_teams=1)
        db = _StandingDB(
            organizations=[org],
            org_members=[],
            profiles=profiles,
            subscriptions=subs,
            tier_entitlements=tiers,
            notifications=[],
        )

        result = standing.evaluate_standing(db)

        assert result == {"evaluated": 1, "lapsed": 0, "grace_started": 1}
        assert db.store["organizations"][0]["grace_started_at"] is not None

    # -----------------------------------------------------------------------
    # Idempotency: a second consecutive run over the same store must not
    # double-fire on a transition the first run already made.
    # -----------------------------------------------------------------------

    def test_second_consecutive_run_is_a_no_op(self, sent_emails):
        """Run 1 grace-stamps an uncovered org and notifies once. Run 2, on
        the SAME store immediately after, must not re-stamp, re-notify, or
        re-email — the fresh grace_started_at is nowhere near grace_days()."""
        org = _standing_org(id="org-i", covered_by=None)
        db = _StandingDB(
            organizations=[org],
            org_members=[_admin_row("org-i", U2)],
            profiles=[],
            subscriptions=[],
            tier_entitlements=[],
            notifications=[],
        )

        first = standing.evaluate_standing(db)
        state_after_first = deepcopy(db.store["organizations"][0])
        notifs_after_first = len(db.store["notifications"])
        emails_after_first = len(sent_emails)

        second = standing.evaluate_standing(db)

        assert first == {"evaluated": 1, "lapsed": 0, "grace_started": 1}
        assert second == {"evaluated": 1, "lapsed": 0, "grace_started": 0}
        assert db.store["organizations"][0] == state_after_first
        assert len(db.store["notifications"]) == notifs_after_first
        assert len(sent_emails) == emails_after_first


# ---------------------------------------------------------------------------
# GET /orgs/{org_id}/ledger (Task 15, spec §6) — admin-only pool ledger feed
# for the org billing console. service.get_org_ledger.
# ---------------------------------------------------------------------------


class TestGetOrgLedger:
    async def test_admin_reads_newest_fifty_ledger_rows(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        rows = [
            {
                "kind": "transfer_in",
                "action": "org_transfer",
                "delta": 500,
                "metadata": {"org_id": ORG, "admin_user_id": U1},
                "created_at": "2026-08-10T00:00:00+00:00",
            },
            {
                "kind": "debit",
                "action": "oneclick_run",
                "delta": -12,
                "metadata": {"org_member_id": "m1"},
                "created_at": "2026-08-09T00:00:00+00:00",
            },
        ]
        captured = {}

        def _side(name):
            b = MockQueryBuilder()
            if name == "credit_wallets":
                b.execute.return_value = MagicMock(data=[{"id": "w1"}], count=1)
            elif name == "credit_ledger":
                original_order, original_limit = b.order, b.limit

                def _capture_order(*a, **kw):
                    captured["order"] = (a, kw)
                    return original_order(*a, **kw)

                def _capture_limit(*a, **kw):
                    captured["limit"] = a
                    return original_limit(*a, **kw)

                b.order, b.limit = _capture_order, _capture_limit
                b.execute.return_value = MagicMock(data=rows, count=2)
            return b

        db = MagicMock()
        db.table.side_effect = _side

        result = await service.get_org_ledger(db, U1, ORG)

        assert result == rows
        assert captured["order"] == (("created_at",), {"desc": True})
        assert captured["limit"] == (50,)

    async def test_non_admin_raises_403(self, monkeypatch):
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
        db = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            await service.get_org_ledger(db, U1, ORG)
        assert exc_info.value.status_code == 403
        db.table.assert_not_called()

    async def test_no_wallet_yet_returns_empty_list(self, monkeypatch):
        """A brand-new org with no ledger activity resolves a wallet via
        create-on-miss and returns [] rather than 404ing. `_db_seq` (not a
        hand-rolled side_effect) is needed here because read_or_create_org_wallet
        calls `db.table("credit_wallets")` TWICE (the miss, then the insert) —
        each call gets a FRESH MockQueryBuilder, so the two executes must be
        queued per-table-name, which is exactly what _db_seq does."""
        monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
        db = _db_seq(
            {
                "credit_wallets": [
                    MagicMock(data=[], count=0),
                    MagicMock(data=[{"id": "w1"}], count=1),
                ],
                "credit_ledger": [MagicMock(data=[], count=0)],
            }
        )

        result = await service.get_org_ledger(db, U1, ORG)
        assert result == []
