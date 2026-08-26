"""Dissolve — preview + execute (self-serve licensing, Task 9, spec §3).

The execute is the only op in this module that moves money, artists, seats and
a Stripe subscription in one call, so most of what is asserted here is ORDER: a
store-backed recorder captures every write, RPC and external call in sequence,
and the tests pin the money-first ordering rather than just the individual
payloads. Mirrors tests/test_orgs_service.py's `_db_seq` idiom, extended to
record writes.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from orgs import service
from tests.conftest import MockQueryBuilder

U1 = "00000000-0000-0000-0000-000000000001"  # the dissolving admin
U2 = "00000000-0000-0000-0000-000000000002"  # an active member
U3 = "00000000-0000-0000-0000-000000000003"  # offboarded creator (no seat)
ORG = "20000000-0000-0000-0000-000000000001"
WALLET = "50000000-0000-0000-0000-000000000002"
A1 = "10000000-0000-0000-0000-0000000000a1"
A2 = "10000000-0000-0000-0000-0000000000a2"
A3 = "10000000-0000-0000-0000-0000000000a3"
NAME = "Acme Records"


class _Builder(MockQueryBuilder):
    """Records writes/deletes; serves table-scoped reads in call order."""

    def __init__(self, table, reads, counters, calls, fail):
        super().__init__()
        self.table_name = table
        self._reads = reads
        self._counters = counters
        self._calls = calls
        self._fail = fail
        self._op = "select"
        self._payload = None
        self._filters = {}
        self.execute = MagicMock(side_effect=self._execute)
        self.delete = MagicMock(side_effect=self._delete)

    def update(self, payload, *a, **kw):
        self._op, self._payload = "update", payload
        return self

    def eq(self, field, value):
        self._filters[field] = value
        return self

    def neq(self, field, value):
        self._filters[f"neq:{field}"] = value
        return self

    def _delete(self, *a, **kw):
        self._calls.append(("delete", self.table_name))
        return self

    def _execute(self):
        if self._op == "update":
            self._calls.append(("update", self.table_name, self._payload, dict(self._filters)))
            if self._fail == f"update:{self.table_name}":
                raise RuntimeError("db write rejected")
            return MagicMock(data=[{"id": self._filters.get("id", ORG), **self._payload}], count=1)
        # Reads are recorded too — the filters ARE the behaviour on a read like
        # "which seats are still active", so they have to be assertable.
        self._calls.append(("select", self.table_name, dict(self._filters)))
        seq = self._reads.get(self.table_name, [None])
        i = min(self._counters.get(self.table_name, 0), len(seq) - 1)
        self._counters[self.table_name] = i + 1
        return MagicMock(data=seq[i], count=1)


def _fake_db(reads, calls, fail=None):
    """reads: table -> list of `data` values, one per select (last repeats).
    calls: shared recorder of ("update"|"delete"|"rpc", ...) tuples, in order."""
    counters: dict[str, int] = {}
    db = MagicMock()
    db.table.side_effect = lambda name: _Builder(name, reads, counters, calls, fail)

    def _rpc(name, params=None):
        m = MagicMock()

        def _ex():
            calls.append(("rpc", name, params))
            if fail == f"rpc:{name}":
                raise RuntimeError("rpc failed")
            return MagicMock(data={"duplicate": False, "removed": (params or {}).get("p_amount", 0)})

        m.execute = MagicMock(side_effect=_ex)
        return m

    db.rpc.side_effect = _rpc
    return db


def _writes(calls):
    """Everything that isn't a read: mutations, RPCs and external calls, in
    order. The recorder captures selects as well, so ordering and
    "nothing was written" assertions filter them back out."""
    return [c for c in calls if c[0] != "select"]


def _org_row(**overrides):
    """_first_org's select list: kind, status, covered_by, covered_at,
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


MEMBERS = [
    {"id": "m1", "user_id": U1, "email": "admin@example.com"},
    {"id": "m2", "user_id": U2, "email": "member@example.com"},
]
# A1's creator holds a seat; A2's creator was offboarded; A3 has no creator.
ARTISTS = [
    {"id": A1, "name": "Nova", "user_id": U2},
    {"id": A2, "name": "Kite", "user_id": U3},
    {"id": A3, "name": "Orphan", "user_id": None},
]


def _reads(*, artists=None, members=None, org=None, name=NAME, topup="sub_topup_1", reserve=400, bundle=90):
    return {
        "organizations": [
            org if org is not None else _org_row(),
            {"name": name, "topup_stripe_subscription_id": topup},
        ],
        "org_members": [MEMBERS if members is None else members],
        "artists": [ARTISTS if artists is None else artists],
        "credit_wallets": [[{"id": WALLET, "reserve_balance": reserve, "bundle_balance": bundle}]],
    }


@pytest.fixture(autouse=True)
def _admin(monkeypatch):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)


@pytest.fixture
def calls(monkeypatch):
    """The shared call recorder — Stripe and the org-grant teardown record into
    the same list as the DB writes, so one sequence covers all six steps."""
    recorded: list = []
    stripe = MagicMock()
    stripe.Subscription.delete.side_effect = lambda sub_id: recorded.append(("stripe_cancel", sub_id))
    monkeypatch.setattr("subscriptions.stripe_client.get_stripe", lambda: stripe)
    monkeypatch.setattr(
        service, "_teardown_archived_org_grants", lambda db, org_id: recorded.append(("teardown", org_id))
    )
    return recorded


# ---------------------------------------------------------------------------
# dissolve_preview
# ---------------------------------------------------------------------------


async def test_preview_resolves_recipients_and_fallbacks(calls):
    out = await service.dissolve_preview(_fake_db(_reads(), calls), U1, ORG)

    assert out["recipients"] == [
        {"artistId": A1, "artistName": "Nova", "userId": U2, "email": "member@example.com", "fallback": False},
        # creator U3 no longer holds a seat -> the dissolving admin, flagged
        {"artistId": A2, "artistName": "Kite", "userId": U1, "email": "admin@example.com", "fallback": True},
        # NULL creator -> same fallback
        {"artistId": A3, "artistName": "Orphan", "userId": U1, "email": "admin@example.com", "fallback": True},
    ]
    assert out["memberCount"] == 2
    assert _writes(calls) == []  # a preview writes nothing
    # Both member reads (the roster and the recipient rule) count ACTIVE seats
    # only — that filter is what puts U3's artist on the fallback path.
    assert [c[2] for c in calls if c[0] == "select" and c[1] == "org_members"] == [
        {"org_id": ORG, "status": "active"},
        {"org_id": ORG, "status": "active"},
    ]


async def test_preview_splits_reserve_from_inert_bundle(calls):
    """Clawback is reserve-only; the dispersal bundle is stated, not forfeited."""
    out = await service.dissolve_preview(_fake_db(_reads(reserve=400, bundle=90), calls), U1, ORG)
    assert out["forfeitReserve"] == 400
    assert out["inertBundle"] == 90


async def test_preview_requires_admin(monkeypatch, calls):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc:
        await service.dissolve_preview(_fake_db(_reads(), calls), U2, ORG)
    assert exc.value.status_code == 403


async def test_preview_enterprise_409(calls):
    db = _fake_db(_reads(org=_org_row(kind="enterprise")), calls)
    with pytest.raises(HTTPException) as exc:
        await service.dissolve_preview(db, U1, ORG)
    assert exc.value.status_code == 409
    assert exc.value.detail == "This organization is managed by Msanii"


# ---------------------------------------------------------------------------
# dissolve_org — guards (nothing may be written before the name matches)
# ---------------------------------------------------------------------------


async def test_dissolve_name_mismatch_400_writes_nothing(calls):
    db = _fake_db(_reads(name=NAME), calls)
    with pytest.raises(HTTPException) as exc:
        await service.dissolve_org(db, U1, ORG, "acme records")
    assert exc.value.status_code == 400
    assert _writes(calls) == []


async def test_dissolve_confirm_matches_a_stored_name_with_stray_whitespace(calls):
    """A name that picked up a trailing space must still be typable — comparing
    a trimmed input against an untrimmed stored name would make that org
    impossible to dissolve, forever."""
    db = _fake_db(_reads(name="Acme Records "), calls)
    result = await service.dissolve_org(db, U1, ORG, NAME)
    assert result["dissolved_at"] is not None


async def test_dissolve_enterprise_409_writes_nothing(calls):
    db = _fake_db(_reads(org=_org_row(kind="enterprise")), calls)
    with pytest.raises(HTTPException) as exc:
        await service.dissolve_org(db, U1, ORG, NAME)
    assert exc.value.status_code == 409
    assert _writes(calls) == []


async def test_dissolve_requires_admin(monkeypatch, calls):
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: False)
    with pytest.raises(HTTPException) as exc:
        await service.dissolve_org(_fake_db(_reads(), calls), U2, ORG, NAME)
    assert exc.value.status_code == 403
    assert _writes(calls) == []


async def test_dissolve_already_dissolved_is_idempotent_noop(calls):
    db = _fake_db(_reads(org=_org_row(dissolved_at="2026-08-10T00:00:00+00:00")), calls)
    assert await service.dissolve_org(db, U1, ORG, NAME) == {"already": True}
    assert _writes(calls) == []


# ---------------------------------------------------------------------------
# dissolve_org — the six-step execute
# ---------------------------------------------------------------------------


async def test_dissolve_runs_six_steps_in_order(calls):
    result = await service.dissolve_org(_fake_db(_reads(), calls), U1, ORG, NAME)

    kinds = [(c[0], c[1]) if c[0] in ("update", "rpc") else (c[0],) for c in _writes(calls)]
    assert kinds == [
        ("rpc", "debit_credits"),  # 1. money first
        ("stripe_cancel",),  # 2. stop the meter...
        ("update", "organizations"),  # ...and clear the two top-up columns (cancel_topup)
        ("update", "artists"),  # 3. artists back to people...
        ("update", "artists"),
        ("update", "artists"),
        ("rpc", "recalc_user_storage"),  # 4. ...then re-derive the bytes
        ("rpc", "recalc_user_storage"),
        ("rpc", "recalc_team_storage"),
        ("update", "org_members"),  # 5. seats, invites, grants
        ("update", "pending_org_invites"),
        ("teardown",),
        ("update", "organizations"),  # 6. terminal state
    ]
    assert result["dissolved_at"] is not None


async def test_dissolve_reads_only_this_org_and_only_active_seats(calls):
    """The filters ARE the recipient rule: `status='active'` is what makes an
    offboarded creator fall back to the admin, and `team_id=org_id` is what
    keeps another team's artists out of the revert loop."""
    await service.dissolve_org(_fake_db(_reads(), calls), U1, ORG, NAME)
    reads = [(c[1], c[2]) for c in calls if c[0] == "select"]
    assert ("org_members", {"org_id": ORG, "status": "active"}) in reads
    assert ("artists", {"team_id": ORG}) in reads
    # ...and nothing read org_members without the active filter.
    assert [f for t, f in reads if t == "org_members"] == [{"org_id": ORG, "status": "active"}]


async def test_dissolve_artist_updates_detach_and_reassign(calls):
    await service.dissolve_org(_fake_db(_reads(), calls), U1, ORG, NAME)
    artist_writes = [(c[3]["id"], c[2]) for c in calls if c[0] == "update" and c[1] == "artists"]
    assert artist_writes == [
        (A1, {"team_id": None, "user_id": U2}),  # creator kept their seat
        (A2, {"team_id": None, "user_id": U1}),  # offboarded creator -> admin
        (A3, {"team_id": None, "user_id": U1}),  # NULL creator -> admin
    ]


async def test_dissolve_recalcs_each_distinct_recipient_once(calls):
    await service.dissolve_org(_fake_db(_reads(), calls), U1, ORG, NAME)
    recalcs = [c[2] for c in calls if c[0] == "rpc" and c[1] == "recalc_user_storage"]
    assert recalcs == [{"p_user_id": U2}, {"p_user_id": U1}]  # U1 receives twice, recalculated once
    assert ("rpc", "recalc_team_storage", {"p_org_id": ORG}) in calls


async def test_dissolve_clawback_args_are_exact(calls):
    await service.dissolve_org(_fake_db(_reads(reserve=400), calls), U1, ORG, NAME)
    params = next(c[2] for c in calls if c[0] == "rpc" and c[1] == "debit_credits")
    assert params["p_wallet_id"] == WALLET
    assert params["p_amount"] == 400  # reserve only — the bundle is left inert
    assert params["p_action"] == "dissolve_forfeit"
    assert params["p_request_id"] == f"dissolve:{ORG}"  # fixed key: a retry can't forfeit twice
    assert params["p_kind"] == "clawback"
    # debit_credits RAISES on a clawback that names a member (a forfeit is not
    # member spend and must not move anyone's cap counter).
    assert "p_member_id" not in params


async def test_dissolve_zero_reserve_skips_clawback(calls):
    await service.dissolve_org(_fake_db(_reads(reserve=0, bundle=90), calls), U1, ORG, NAME)
    assert not [c for c in calls if c[0] == "rpc" and c[1] == "debit_credits"]
    assert [c for c in calls if c[0] == "update" and c[1] == "artists"]  # the rest still ran


async def test_dissolve_without_topup_subscription_skips_stripe(calls):
    await service.dissolve_org(_fake_db(_reads(topup=None), calls), U1, ORG, NAME)
    assert not [c for c in calls if c[0] == "stripe_cancel"]


async def test_dissolve_soft_removes_seats_and_expires_invites(calls):
    await service.dissolve_org(_fake_db(_reads(), calls), U1, ORG, NAME)
    seats = next(c for c in calls if c[0] == "update" and c[1] == "org_members")
    assert seats[2]["status"] == "removed" and seats[2]["revoked_at"]
    # The dissolving admin's own seat is excluded: org_members_admin_guard
    # rejects the row that would leave a live org with no active admin, and one
    # rejected row aborts the whole statement (stranding everyone else's seat).
    # ...and already-removed seats keep the revoked_at of their real offboarding.
    assert seats[3] == {"org_id": ORG, "neq:user_id": U1, "neq:status": "removed"}
    invites = next(c for c in calls if c[0] == "update" and c[1] == "pending_org_invites")
    assert invites[2] == {"status": "expired"}
    assert invites[3] == {"org_id": ORG, "status": "pending"}


async def test_dissolve_stamps_archived_at_and_keeps_an_existing_one(calls):
    await service.dissolve_org(_fake_db(_reads(), calls), U1, ORG, NAME)
    patch = next(c[2] for c in calls if c[0] == "update" and c[1] == "organizations" and "dissolved_at" in c[2])
    assert patch["archived_at"] == patch["dissolved_at"]  # not previously archived

    calls.clear()
    db = _fake_db(_reads(org=_org_row(archived_at="2026-07-01T00:00:00+00:00")), calls)
    await service.dissolve_org(db, U1, ORG, NAME)
    patch = next(c[2] for c in calls if c[0] == "update" and c[1] == "organizations" and "dissolved_at" in c[2])
    assert patch["archived_at"] == "2026-07-01T00:00:00+00:00"
    assert patch["dissolved_at"] != patch["archived_at"]


async def test_dissolve_deletes_nothing(calls):
    """SOFT: the org row, the pool wallet and the ledger are all retained."""
    await service.dissolve_org(_fake_db(_reads(), calls), U1, ORG, NAME)
    assert not [c for c in calls if c[0] == "delete"]


# ---------------------------------------------------------------------------
# Abort semantics — half-dissolved is worse than not dissolved
# ---------------------------------------------------------------------------


async def test_dissolve_aborts_when_an_artist_revert_fails(calls):
    """Artists run BEFORE the seats precisely so this leaves an intact admin
    who can retry: no seat is closed and the org is NOT marked dissolved."""
    db = _fake_db(_reads(), calls, fail="update:artists")
    with pytest.raises(HTTPException) as exc:
        await service.dissolve_org(db, U1, ORG, NAME)
    assert exc.value.status_code == 500
    # Non-technical, and honest about the partial state + the way out.
    assert exc.value.detail == (
        "Some artists were handed back but the team could not be dissolved. "
        "Nothing was lost — run dissolve again to finish."
    )
    # No seat is closed and the terminal patch never lands (the only
    # organizations update allowed before the abort is cancel_topup's
    # column clear in step 2).
    assert not [c for c in calls if c[0] == "update" and c[1] == "org_members"]
    assert not [c for c in calls if c[0] == "update" and c[1] == "organizations" and "dissolved_at" in c[2]]


async def test_dissolve_survives_a_failed_forfeit(calls):
    """Best-effort money step: an unreachable RPC must not strand the team."""
    db = _fake_db(_reads(), calls, fail="rpc:debit_credits")
    result = await service.dissolve_org(db, U1, ORG, NAME)
    assert result["dissolved_at"] is not None


async def test_dissolve_survives_a_failed_seat_teardown(calls):
    db = _fake_db(_reads(), calls, fail="update:org_members")
    result = await service.dissolve_org(db, U1, ORG, NAME)
    assert result["dissolved_at"] is not None
