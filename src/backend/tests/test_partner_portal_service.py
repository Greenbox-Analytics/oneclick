"""Phase-2 portal: per-key spend on get_org_usage + Created-by labels."""

from unittest.mock import MagicMock

from orgs import service as orgs_service
from partner_api import service as psvc
from tests.conftest import MockQueryBuilder

ORG_ID = "00000000-0000-0000-0000-0000000000aa"
KEY_A = "00000000-0000-0000-0000-0000000000bb"
KEY_B = "00000000-0000-0000-0000-0000000000cc"
KEY_C = "00000000-0000-0000-0000-0000000000c1"  # keyed row, non-partner source
KEY_D = "00000000-0000-0000-0000-0000000000c2"  # keyed row, non-debit kind
U_ADMIN = "00000000-0000-0000-0000-0000000000d1"
U_GONE = "00000000-0000-0000-0000-0000000000d2"

WALLET = {
    "id": "w1",
    "owner_type": "org",
    "owner_id": ORG_ID,
    "bundle_balance": 0,
    "reserve_balance": 500,
    "period_start": "2026-09-01T00:00:00+00:00",
    "period_end": "2026-10-01T00:00:00+00:00",
}
# email on the row => _member_email reads it straight off, no auth-admin call.
MEMBER = {
    "id": "00000000-0000-0000-0000-0000000000ee",
    "user_id": U_ADMIN,
    "role": "admin",
    "status": "active",
    "email": "admin@label.test",
    "monthly_cap": None,
    "cap_used": 0,
}


def _debit(key_id, delta, created, source="partner_api", member=None):
    meta = {"source": source, "partner_key_id": key_id} if key_id else {}
    if member:
        meta["org_member_id"] = member
    return {"kind": "debit", "delta": delta, "metadata": meta, "created_at": created}


# ---- get_org_usage.byKey ----------------------------------------------------


def _usage_db(ledger):
    """Table-mocked db for orgs.service.get_org_usage, mirroring
    test_orgs_router.TestGetOrgUsageService._db: one fixed response per
    table. MockQueryBuilder.range() returns itself, so fetch_all's one page
    is this one execute()."""

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=[MEMBER], count=1)
        elif name == "credit_wallets":
            b.execute.return_value = MagicMock(data=[WALLET], count=1)
        elif name == "credit_ledger":
            b.execute.return_value = MagicMock(data=ledger, count=len(ledger))
        elif name == "organizations":
            b.execute.return_value = MagicMock(
                data=[{"default_member_cap": 2000, "monthly_dispersal_credits": 0}], count=1
            )
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db


async def test_get_org_usage_by_key_groups_partner_rows_only(monkeypatch):
    monkeypatch.setattr(orgs_service.authz, "is_org_admin", lambda *a: True)
    monkeypatch.setattr(orgs_service.wallets, "cumulative_paid_in", lambda db, wallet_id: 0)
    rows = [
        # KEY_A newest-first: a "last row wins" lastUsedAt would read 09-02.
        _debit(KEY_A, -37, "2026-09-05T10:00:00+00:00"),
        _debit(KEY_A, -30, "2026-09-02T10:00:00+00:00"),
        _debit(KEY_B, -30, "2026-09-03T10:00:00+00:00"),
        _debit(None, -5, "2026-09-04T10:00:00+00:00", source=None, member=MEMBER["id"]),  # member spend
        # Keyed but not partner traffic — only the source filter excludes it.
        _debit(KEY_C, -99, "2026-09-06T10:00:00+00:00", source="product"),
        # Keyed partner metadata on a credit row — only the kind filter excludes
        # it, and abs() would ADD its 2000 to the total if it slipped through.
        {
            "kind": "dispersal",
            "delta": 2000,
            "metadata": {"source": "partner_api", "partner_key_id": KEY_D},
            "created_at": "2026-09-01T00:00:00+00:00",
        },
    ]
    out = await orgs_service.get_org_usage(_usage_db(rows), U_ADMIN, ORG_ID)
    by_id = {k["keyId"]: k for k in out["byKey"]}
    assert by_id[KEY_A] == {"keyId": KEY_A, "credits": 67, "runs": 2, "lastUsedAt": "2026-09-05T10:00:00+00:00"}
    assert by_id[KEY_B] == {"keyId": KEY_B, "credits": 30, "runs": 1, "lastUsedAt": "2026-09-03T10:00:00+00:00"}
    # Member spend, product-source rows and non-debit rows all stay out.
    assert set(by_id) == {KEY_A, KEY_B}
    assert out["byKey"][0]["keyId"] == KEY_A  # credits desc
    # ...and partner spend never lands in a seat: the one member spent 5.
    assert [s["spentThisPeriod"] for s in out["seats"]] == [5]


# ---- created_by_labels ------------------------------------------------------


def test_created_by_labels_reads_members_then_falls_back(monkeypatch):
    members = MockQueryBuilder()
    members.execute = MagicMock(return_value=MagicMock(data=[{"user_id": U_ADMIN, "email": "admin@label.test"}]))
    sb = MagicMock()
    sb.table.side_effect = lambda name: {"org_members": members}[name]
    monkeypatch.setattr(
        "orgs.service._resolve_user_email", lambda db, uid: "gone@label.test" if uid == U_GONE else None
    )
    keys = [{"created_by": U_ADMIN}, {"created_by": U_GONE}, {"created_by": None}, {"created_by": "unknown"}]
    labels = psvc.created_by_labels(sb, ORG_ID, keys)
    assert labels == {U_ADMIN: "admin@label.test", U_GONE: "gone@label.test"}


def test_created_by_labels_never_raises():
    sb = MagicMock()
    sb.table.side_effect = RuntimeError("db down")
    assert psvc.created_by_labels(sb, ORG_ID, [{"created_by": U_ADMIN}]) == {}
