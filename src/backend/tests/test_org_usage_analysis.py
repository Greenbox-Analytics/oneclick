"""GET /orgs/{id}/usage ranges and the analysis fields (spec 2026-09-06 §6):
byAction on seats and keys, the per-day series, the previous window, plus the
key folders / inactive-key hiding / per-creator API attribution added
2026-09-06."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orgs import service
from tests.conftest import MockQueryBuilder

ORG = "20000000-0000-0000-0000-000000000099"
U_ADMIN = "00000000-0000-0000-0000-0000000000c1"
MEMBER = {
    "id": "m1",
    "user_id": U_ADMIN,
    "role": "admin",
    "status": "active",
    "email": "admin@label.test",
    "monthly_cap": None,
    "cap_used": 0,
}
WALLET = {
    "id": "w1",
    "owner_type": "org",
    "owner_id": ORG,
    "bundle_balance": 0,
    "reserve_balance": 500,
    "period_start": "2026-09-01T00:00:00+00:00",
    "period_end": "2026-10-01T00:00:00+00:00",
}
KEY = "k-1"
KEY_ROW = {
    "id": KEY,
    "label": "Prod",
    "key_prefix": "mk_live_aaaa",
    "status": "active",
    "expires_at": None,
    "revoked_at": None,
    "folder_id": None,
    "created_by": None,
}
LONG_AGO = "2020-01-01T00:00:00+00:00"
YESTERDAY = (datetime.now(UTC) - timedelta(days=1)).isoformat()


def _key(key_id, **over):
    return {**KEY_ROW, "id": key_id, **over}


def _row(delta, action, created, *, member=None, key=None):
    meta = {}
    if member:
        meta["org_member_id"] = member
    if key:
        meta.update({"source": "partner_api", "partner_key_id": key})
    return {"kind": "debit", "delta": delta, "action": action, "metadata": meta, "created_at": created}


def _db(pages, keys=None, folders=None, members=None):
    """One response per credit_ledger call, in order (the current window,
    then the previous one); the other tables are fixed."""
    pages = list(pages)
    keys = [KEY_ROW] if keys is None else keys
    folders = folders or []
    members = [MEMBER] if members is None else members
    queries = []
    key_queries = []

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=members, count=len(members))
        elif name == "partner_api_keys":
            b.execute.return_value = MagicMock(data=keys, count=len(keys))
            b.eq = MagicMock(side_effect=lambda *a: b)
            key_queries.append(b)
        elif name == "partner_key_folders":
            b.execute.return_value = MagicMock(data=folders, count=len(folders))
        elif name == "credit_wallets":
            b.execute.return_value = MagicMock(data=[WALLET], count=1)
        elif name == "credit_ledger":
            rows = pages.pop(0) if pages else []
            b.execute.return_value = MagicMock(data=rows, count=len(rows))
            # Still chainable, but now recording — the window bounds and the
            # kind filter are the query's whole contract.
            b.gte = MagicMock(side_effect=lambda *a: b)
            b.lt = MagicMock(side_effect=lambda *a: b)
            b.eq = MagicMock(side_effect=lambda *a: b)
            queries.append(b)
        elif name == "organizations":
            b.execute.return_value = MagicMock(
                data=[{"default_member_cap": 2000, "monthly_dispersal_credits": 0}], count=1
            )
        return b

    db = MagicMock()
    db.table.side_effect = _side
    db.ledger_queries = queries
    db.key_queries = key_queries
    return db


@pytest.fixture(autouse=True)
def admin(monkeypatch):
    # /orgs/* 404s without the flag (orgs.router.require_licensing), and it is
    # off by default in tests — same autouse idiom as tests/test_orgs_router.py.
    monkeypatch.setenv("LICENSING_ENABLED", "true")
    monkeypatch.setattr(service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(service.wallets, "cumulative_paid_in", lambda db, wallet_id: 0)


def test_usage_window_floors():
    now = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
    assert service.usage_window("all", WALLET["period_start"], now) == (None, None)
    assert service.usage_window("mtd", None, now) == (None, None)
    since, prev = service.usage_window("mtd", WALLET["period_start"], now)
    assert since == "2026-09-01T00:00:00+00:00" and prev == "2026-08-17T12:00:00+00:00"
    since, prev = service.usage_window("7d", WALLET["period_start"], now)
    assert since == "2026-09-08T12:00:00+00:00" and prev == "2026-09-01T12:00:00+00:00"
    assert service.usage_window("14d", None, now)[0] == "2026-09-01T12:00:00+00:00"
    assert service.usage_window("1y", None, now)[0] == "2025-09-15T12:00:00+00:00"
    with pytest.raises(ValueError):
        service.usage_window("30d", None, now)


async def test_by_action_and_series_group_member_and_key_spend_separately():
    rows = [
        _row(-30, "oneclick_run", "2026-09-02T10:00:00+00:00", member="m1"),
        _row(-5, "zoe_message", "2026-09-02T11:00:00+00:00", member="m1"),
        _row(-30, "partner_oneclick_run", "2026-09-03T10:00:00+00:00", key=KEY),
        _row(-20, "partner_split_sheet", "2026-09-03T12:00:00+00:00", key=KEY),
        _row(-30, "partner_oneclick_run", "2026-09-04T09:00:00+00:00", key=KEY),
        {"kind": "dispersal", "delta": 2000, "action": None, "metadata": {}, "created_at": "2026-09-01T00:00:00+00:00"},
    ]
    out = await service.get_org_usage(_db([rows, []]), U_ADMIN, ORG)
    assert out["range"] == "mtd" and out["since"] == WALLET["period_start"]
    (seat,) = out["seats"]
    assert seat["spentThisPeriod"] == 35
    assert seat["byAction"] == [
        {"action": "oneclick_run", "credits": 30, "runs": 1},
        {"action": "zoe_message", "credits": 5, "runs": 1},
    ]
    (key,) = out["byKey"]
    assert key["credits"] == 80 and key["runs"] == 3
    assert key["lastUsedAt"] == "2026-09-04T09:00:00+00:00"
    assert (key["label"], key["keyPrefix"], key["status"]) == ("Prod", "mk_live_aaaa", "active")
    assert key["folderId"] is None and key["folderName"] is None
    assert key["byAction"] == [
        {"action": "partner_oneclick_run", "credits": 60, "runs": 2},
        {"action": "partner_split_sheet", "credits": 20, "runs": 1},
    ]
    assert out["series"] == [
        {
            "day": "2026-09-02",
            "actions": [
                {"action": "oneclick_run", "credits": 30, "runs": 1},
                {"action": "zoe_message", "credits": 5, "runs": 1},
            ],
        },
        {
            "day": "2026-09-03",
            "actions": [
                {"action": "partner_oneclick_run", "credits": 30, "runs": 1},
                {"action": "partner_split_sheet", "credits": 20, "runs": 1},
            ],
        },
        {"day": "2026-09-04", "actions": [{"action": "partner_oneclick_run", "credits": 30, "runs": 1}]},
    ]
    assert out["previous"] == {"credits": 0, "runs": 0}


async def test_previous_window_is_a_second_scan_of_debits_only():
    current = [_row(-30, "oneclick_run", "2026-09-02T10:00:00+00:00", member="m1")]
    previous = [
        _row(-30, "oneclick_run", "2026-08-20T10:00:00+00:00", member="m1"),
        _row(-5, "zoe_message", "2026-08-21T10:00:00+00:00", member="m1"),
        {"kind": "admin_grant", "delta": 500, "metadata": {}},
    ]
    db = _db([current, previous])
    out = await service.get_org_usage(db, U_ADMIN, ORG, range_="7d")
    assert out["range"] == "7d"
    assert out["previous"] == {"credits": 35, "runs": 2}
    assert len(db.ledger_queries) == 2
    cur, prev_q = db.ledger_queries
    assert cur.gte.call_args.args == ("created_at", out["since"])
    assert prev_q.gte.call_args.args[0] == "created_at"
    assert prev_q.lt.call_args.args == ("created_at", out["since"])
    # Non-debit rows are excluded by the query, not just by the Python guard.
    for b in (cur, prev_q):
        assert ("kind", "debit") in [c.args for c in b.eq.call_args_list]


async def test_all_time_has_no_floor_and_no_previous():
    db = _db([[]])
    out = await service.get_org_usage(db, U_ADMIN, ORG, range_="all")
    assert out["since"] is None and out["previous"] is None and out["series"] == []
    assert len(db.ledger_queries) == 1
    assert db.ledger_queries[0].gte.call_count == 0


async def test_rows_without_action_or_timestamp_still_count_as_spend():
    # Older ledger rows (and existing test fixtures) carry neither.
    rows = [{"kind": "debit", "delta": -7, "metadata": {"org_member_id": "m1"}}]
    out = await service.get_org_usage(_db([rows, []]), U_ADMIN, ORG)
    assert out["seats"][0]["spentThisPeriod"] == 7
    assert out["seats"][0]["byAction"] == [] and out["series"] == []


# ---- key folders + inactive-key hiding ---------------------------------------


async def test_by_key_lists_every_visible_key_and_hides_long_inactive_ones():
    """A key with no spend still gets a row (an admin needs to see it exists);
    a key revoked/expired over 30 days ago gets none — but its spend is still
    in the org's totals, its folder's total and the series."""
    keys = [
        _key("k-live", label="Live", folder_id="f1"),
        _key("k-idle", label="Idle"),  # never used
        _key("k-old", label="Old", status="revoked", revoked_at=LONG_AGO),
        _key("k-lapsed", label="Lapsed", expires_at=LONG_AGO),
        # Inactive but recent: still listed, with its real status.
        _key("k-fresh", label="Fresh", status="revoked", revoked_at=YESTERDAY),
        # Revoked before revoked_at existed: no clock, so never hidden.
        _key("k-legacy", label="Legacy", status="revoked"),
    ]
    rows = [
        _row(-30, "partner_oneclick_run", "2026-09-02T10:00:00+00:00", key="k-live"),
        _row(-20, "partner_split_sheet", "2026-09-03T10:00:00+00:00", key="k-old"),
    ]
    out = await service.get_org_usage(
        _db([rows, []], keys=keys, folders=[{"id": "f1", "name": "Ingest"}]), U_ADMIN, ORG
    )
    listed = {k["keyId"]: k for k in out["byKey"]}
    assert set(listed) == {"k-live", "k-idle", "k-fresh", "k-legacy"}
    assert listed["k-idle"] == {
        "keyId": "k-idle",
        "label": "Idle",
        "keyPrefix": "mk_live_aaaa",
        "status": "active",
        "folderId": None,
        "folderName": None,
        "credits": 0,
        "runs": 0,
        "lastUsedAt": None,
        "byAction": [],
        "series": [],
    }
    assert listed["k-live"]["folderName"] == "Ingest"
    assert listed["k-fresh"]["status"] == "revoked" and listed["k-legacy"]["status"] == "revoked"
    assert out["byKey"][0]["keyId"] == "k-live"  # credits desc
    # The hidden key's 20 credits never leave the org's books.
    assert out["series"][1] == {
        "day": "2026-09-03",
        "actions": [{"action": "partner_split_sheet", "credits": 20, "runs": 1}],
    }


async def test_by_folder_rolls_hidden_spend_up_and_keeps_empty_folders():
    keys = [
        _key("k-live", label="Live", folder_id="f1"),
        _key("k-old", label="Old", folder_id="f1", status="revoked", revoked_at=LONG_AGO),
        _key("k-loose", label="Loose"),
    ]
    folders = [{"id": "f1", "name": "Ingest"}, {"id": "f2", "name": "Empty"}]
    rows = [
        _row(-30, "partner_oneclick_run", "2026-09-02T10:00:00+00:00", key="k-live"),
        _row(-20, "partner_split_sheet", "2026-09-03T10:00:00+00:00", key="k-old"),
        _row(-5, "partner_zoe_message", "2026-09-03T11:00:00+00:00", key="k-loose"),
        # A ledger row for a key that no longer exists: unfiled, never a byKey row.
        _row(-7, "partner_zoe_message", "2026-09-04T11:00:00+00:00", key="k-gone"),
    ]
    out = await service.get_org_usage(_db([rows, []], keys=keys, folders=folders), U_ADMIN, ORG)
    assert out["byFolder"] == [
        # 30 (listed) + 20 (hidden key, same folder), one LISTED key counted.
        {
            "folderId": "f1",
            "name": "Ingest",
            "keys": 1,
            "credits": 50,
            "runs": 2,
            "byAction": [
                {"action": "partner_oneclick_run", "credits": 30, "runs": 1},
                {"action": "partner_split_sheet", "credits": 20, "runs": 1},
            ],
            "series": [
                {"day": "2026-09-02", "actions": [{"action": "partner_oneclick_run", "credits": 30, "runs": 1}]},
                {"day": "2026-09-03", "actions": [{"action": "partner_split_sheet", "credits": 20, "runs": 1}]},
            ],
        },
        {
            "folderId": None,
            "name": "No folder",
            "keys": 1,
            "credits": 12,
            "runs": 2,
            "byAction": [{"action": "partner_zoe_message", "credits": 12, "runs": 2}],
            "series": [
                {"day": "2026-09-03", "actions": [{"action": "partner_zoe_message", "credits": 5, "runs": 1}]},
                {"day": "2026-09-04", "actions": [{"action": "partner_zoe_message", "credits": 7, "runs": 1}]},
            ],
        },
        # A folder nobody has filed anything into still shows, at zero.
        {"folderId": "f2", "name": "Empty", "keys": 0, "credits": 0, "runs": 0, "byAction": [], "series": []},
    ]


async def test_no_folder_row_is_absent_when_nothing_is_unfiled():
    keys = [_key("k-live", folder_id="f1")]
    out = await service.get_org_usage(_db([[], []], keys=keys, folders=[{"id": "f1", "name": "Ingest"}]), U_ADMIN, ORG)
    assert [f["folderId"] for f in out["byFolder"]] == ["f1"]


# ---- API spend attributed to the key's creator --------------------------------

MEMBER_B = {**MEMBER, "id": "m2", "user_id": "u-b", "email": "b@label.test", "status": "active"}


async def test_api_spend_lands_on_the_key_creators_seat_but_not_on_the_cap():
    keys = [_key("k-a", created_by=U_ADMIN), _key("k-orphan", created_by="u-never-a-member")]
    rows = [
        _row(-30, "oneclick_run", "2026-09-02T10:00:00+00:00", member="m1"),
        _row(-30, "partner_oneclick_run", "2026-09-03T10:00:00+00:00", key="k-a"),
        _row(-5, "partner_zoe_message", "2026-09-03T11:00:00+00:00", key="k-a"),
        # Nobody's seat: the creator holds no row in this org.
        _row(-40, "partner_oneclick_run", "2026-09-04T10:00:00+00:00", key="k-orphan"),
    ]
    out = await service.get_org_usage(_db([rows, []], keys=keys, members=[MEMBER, MEMBER_B]), U_ADMIN, ORG)
    seats = {s["orgMemberId"]: s for s in out["seats"]}
    assert seats["m1"]["apiCredits"] == 35 and seats["m1"]["apiRuns"] == 2
    # The cap-relevant number stays product-only.
    assert seats["m1"]["spentThisPeriod"] == 30
    assert seats["m1"]["byAction"] == [
        {"action": "oneclick_run", "credits": 30, "runs": 1},
        {"action": "partner_oneclick_run", "credits": 30, "runs": 1},
        {"action": "partner_zoe_message", "credits": 5, "runs": 1},
    ]
    # A member who created no key shows zeros, and the orphan key's 40 credits
    # land on nobody while still being in byKey.
    assert (seats["m2"]["apiCredits"], seats["m2"]["apiRuns"]) == (0, 0)
    assert sum(s["apiCredits"] for s in out["seats"]) == 35
    assert {k["keyId"] for k in out["byKey"]} == {"k-a", "k-orphan"}


async def test_removed_member_with_only_api_spend_is_included():
    gone = {**MEMBER, "id": "m3", "user_id": "u-gone", "email": "gone@label.test", "status": "removed"}
    keys = [_key("k-g", created_by="u-gone")]
    rows = [_row(-30, "partner_oneclick_run", "2026-09-03T10:00:00+00:00", key="k-g")]
    out = await service.get_org_usage(_db([rows, []], keys=keys, members=[MEMBER, gone]), U_ADMIN, ORG)
    seats = {s["orgMemberId"]: s for s in out["seats"]}
    assert seats["m3"]["apiCredits"] == 30 and seats["m3"]["spentThisPeriod"] == 0


async def test_only_created_by_narrows_the_keys_read():
    db = _db([[], []], keys=[_key("k-a", created_by=U_ADMIN)])
    out = await service.org_usage_rollup(db, ORG, only_created_by=U_ADMIN)
    assert [k["keyId"] for k in out["byKey"]] == ["k-a"]
    (q,) = db.key_queries
    assert ("created_by", U_ADMIN) in [c.args for c in q.eq.call_args_list]


async def test_only_created_by_ignores_another_members_key_spend():
    """The keys read is filtered server-side; a partner debit on a key that
    read did not return must not surface as unfiled spend."""
    db = _db(
        [[_row(-30, "partner_oneclick_run", "2026-09-03T10:00:00+00:00", key="k-theirs")], []],
        keys=[_key("k-mine", created_by=U_ADMIN)],
    )
    out = await service.org_usage_rollup(db, ORG, only_created_by=U_ADMIN)
    assert [k["keyId"] for k in out["byKey"]] == ["k-mine"]
    assert out["byKey"][0]["credits"] == 0
    # Only my one unfiled key — their 30 credits are nowhere in my view.
    assert out["byFolder"] == [
        {"folderId": None, "name": "No folder", "keys": 1, "credits": 0, "runs": 0, "byAction": [], "series": []}
    ]


async def test_only_created_by_hides_folders_the_caller_has_nothing_in():
    """byFolder must not seed from the org-wide folder list when narrowing —
    otherwise a plain member sees every colleague's folder name at 0 credits.
    A folder still earns a row when it holds one of the caller's own keys."""
    folders = [{"id": "f1", "name": "Mine"}, {"id": "f2", "name": "Theirs"}]

    # Unscoped: both folders show, including the one with no caller key.
    all_keys = [
        _key("k-mine", created_by=U_ADMIN, folder_id="f1"),
        _key("k-theirs", created_by="other", folder_id="f2"),
    ]
    out = await service.org_usage_rollup(_db([[], []], keys=all_keys, folders=folders), ORG)
    assert {f["folderId"] for f in out["byFolder"]} == {"f1", "f2"}

    # Scoped: the server-side filter already narrowed `keys` to the caller's
    # own — only their folder should appear, never "Theirs".
    my_keys = [_key("k-mine", created_by=U_ADMIN, folder_id="f1")]
    out = await service.org_usage_rollup(_db([[], []], keys=my_keys, folders=folders), ORG, only_created_by=U_ADMIN)
    assert {f["folderId"] for f in out["byFolder"]} == {"f1"}
    assert [f["name"] for f in out["byFolder"]] == ["Mine"]


async def test_only_created_by_skips_admin_only_reads():
    """The scoped path is cheap: one ledger scan (no previous-window second
    scan), no cumulative_paid_in, no per-member email resolution, and the
    payload carries only what /me/api-usage reads."""

    def _boom(*a, **k):
        raise AssertionError("must not be called when only_created_by narrows the rollup")

    with (
        patch.object(service.wallets, "cumulative_paid_in", side_effect=_boom),
        patch.object(service, "_member_email", side_effect=_boom),
    ):
        db = _db(
            [[_row(-30, "partner_oneclick_run", "2026-09-03T10:00:00+00:00", key="k-a")]],
            keys=[_key("k-a", created_by=U_ADMIN)],
        )
        out = await service.org_usage_rollup(db, ORG, only_created_by=U_ADMIN)

    assert len(db.ledger_queries) == 1
    assert set(out) == {"range", "since", "byKey", "byFolder"}
    assert "seats" not in out


def test_route_rejects_an_unknown_range(client):
    with patch("orgs.router.service.get_org_usage", new=AsyncMock(return_value={})) as svc:
        r = client.get(f"/orgs/{ORG}/usage?range=30d")
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_range"
    svc.assert_not_called()


def test_route_passes_the_range_through(client):
    with patch("orgs.router.service.get_org_usage", new=AsyncMock(return_value={"range": "1y"})) as svc:
        r = client.get(f"/orgs/{ORG}/usage?range=1y")
    assert r.status_code == 200 and r.json() == {"range": "1y"}
    assert svc.call_args.kwargs["range_"] == "1y"


# ---- per-entity time series ---------------------------------------------------


async def test_keys_folders_and_seats_each_carry_their_own_series():
    """Same shape as the top-level series, same attribution rules: a seat's
    series holds its PRODUCT debits plus the API spend of the keys it created,
    a folder's follows the key's CURRENT folder, and a listed key nobody has
    used gets an empty one rather than being absent."""
    keys = [_key("k-a", created_by=U_ADMIN, folder_id="f1"), _key("k-idle", label="Idle")]
    rows = [
        _row(-30, "oneclick_run", "2026-09-02T10:00:00+00:00", member="m1"),
        _row(-30, "partner_oneclick_run", "2026-09-02T11:00:00+00:00", key="k-a"),
        _row(-5, "partner_zoe_message", "2026-09-02T12:00:00+00:00", key="k-a"),
        _row(-20, "partner_split_sheet", "2026-09-03T09:00:00+00:00", key="k-a"),
        # No timestamp: spend, but nothing a series can place on a day.
        {"kind": "debit", "delta": -7, "action": "zoe_message", "metadata": {"org_member_id": "m1"}},
    ]
    out = await service.get_org_usage(
        _db([rows, []], keys=keys, folders=[{"id": "f1", "name": "Ingest"}]), U_ADMIN, ORG
    )

    by_key = {k["keyId"]: k for k in out["byKey"]}
    key_series = [
        {
            "day": "2026-09-02",
            "actions": [
                {"action": "partner_oneclick_run", "credits": 30, "runs": 1},
                {"action": "partner_zoe_message", "credits": 5, "runs": 1},
            ],
        },
        {"day": "2026-09-03", "actions": [{"action": "partner_split_sheet", "credits": 20, "runs": 1}]},
    ]
    assert by_key["k-a"]["series"] == key_series
    assert by_key["k-idle"]["series"] == []

    folders = {f["folderId"]: f for f in out["byFolder"]}
    assert folders["f1"]["series"] == key_series
    assert folders[None]["series"] == []  # k-idle is unfiled and unused

    # Product spend and the key's API spend land on the one seat, merged per day.
    (seat,) = out["seats"]
    assert seat["series"] == [
        {
            "day": "2026-09-02",
            "actions": [
                {"action": "oneclick_run", "credits": 30, "runs": 1},
                {"action": "partner_oneclick_run", "credits": 30, "runs": 1},
                {"action": "partner_zoe_message", "credits": 5, "runs": 1},
            ],
        },
        {"day": "2026-09-03", "actions": [{"action": "partner_split_sheet", "credits": 20, "runs": 1}]},
    ]
    # The timestamp-less row still counts as spend, just not on any day.
    assert seat["spentThisPeriod"] == 37


async def test_folder_series_merges_every_key_currently_filed_in_it():
    keys = [_key("k-a", folder_id="f1"), _key("k-b", label="B", folder_id="f1")]
    rows = [
        _row(-30, "partner_oneclick_run", "2026-09-02T10:00:00+00:00", key="k-a"),
        _row(-5, "partner_zoe_message", "2026-09-02T11:00:00+00:00", key="k-b"),
        _row(-20, "partner_split_sheet", "2026-09-04T11:00:00+00:00", key="k-b"),
    ]
    out = await service.get_org_usage(
        _db([rows, []], keys=keys, folders=[{"id": "f1", "name": "Ingest"}]), U_ADMIN, ORG
    )
    (folder,) = out["byFolder"]
    assert folder["series"] == [
        {
            "day": "2026-09-02",
            "actions": [
                {"action": "partner_oneclick_run", "credits": 30, "runs": 1},
                {"action": "partner_zoe_message", "credits": 5, "runs": 1},
            ],
        },
        {"day": "2026-09-04", "actions": [{"action": "partner_split_sheet", "credits": 20, "runs": 1}]},
    ]


async def test_scoped_path_rows_carry_series_too():
    """/me/api-usage reads byKey/byFolder only — both must still be plottable."""
    db = _db(
        [[_row(-30, "partner_oneclick_run", "2026-09-03T10:00:00+00:00", key="k-a")]],
        keys=[_key("k-a", created_by=U_ADMIN)],
    )
    out = await service.org_usage_rollup(db, ORG, only_created_by=U_ADMIN)
    expected = [{"day": "2026-09-03", "actions": [{"action": "partner_oneclick_run", "credits": 30, "runs": 1}]}]
    assert out["byKey"][0]["series"] == expected
    assert out["byFolder"][0]["series"] == expected
