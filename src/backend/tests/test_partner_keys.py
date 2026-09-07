from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from partner_api import service as psvc
from tests.conftest import MockQueryBuilder

ORG_ID = "00000000-0000-0000-0000-0000000000aa"
KEY_ID = "00000000-0000-0000-0000-0000000000bb"
FOLDER_ID = "00000000-0000-0000-0000-0000000000f1"

ACTIVE_ORG = {"id": ORG_ID, "status": "active", "archived_at": None, "partner_api_enabled": True}
ACTIVE_KEY = {"id": KEY_ID, "org_id": ORG_ID, "status": "active", "expires_at": None}
FOLDER = {"id": FOLDER_ID, "org_id": ORG_ID, "name": "Ingest", "created_at": "2026-09-01T00:00:00+00:00"}

NOW = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)


def _ago(days):
    return (NOW - timedelta(days=days)).isoformat()


def _sb(key_rows, org_row=ACTIVE_ORG, folder_rows=()):
    """Mock supabase dispatching per-table builders (conftest pattern)."""
    keys = MockQueryBuilder()
    keys.execute = MagicMock(return_value=MagicMock(data=key_rows))
    orgs = MockQueryBuilder()
    orgs.execute = MagicMock(return_value=MagicMock(data=[org_row] if org_row else []))
    folders = MockQueryBuilder()
    folders.execute = MagicMock(return_value=MagicMock(data=list(folder_rows)))
    sb = MagicMock()
    sb.table.side_effect = lambda name: {
        "partner_api_keys": keys,
        "organizations": orgs,
        "partner_key_folders": folders,
    }[name]
    sb.folders = folders
    return sb, keys


def test_mint_returns_secret_once_and_never_stores_it():
    sb, keys = _sb([])
    keys.execute = MagicMock(return_value=MagicMock(data=[{"id": KEY_ID, "org_id": ORG_ID}]))
    out = psvc.mint_key(sb, ORG_ID, label="Production backend")
    assert out["secret"].startswith("mk_live_")
    inserted = keys.insert.call_args[0][0]
    assert "secret" not in inserted
    assert inserted["key_hash"] == psvc._hash_key(out["secret"])
    assert inserted["key_prefix"] == out["secret"][:12]
    # One key type: nothing about a hierarchy is written.
    assert "parent_key_id" not in inserted and "user_ref" not in inserted


def test_resolve_rejects_wrong_prefix_and_unknown_key():
    sb, _ = _sb([])
    assert psvc.resolve_key(sb, None) is None
    assert psvc.resolve_key(sb, "sk-something-else") is None
    assert psvc.resolve_key(sb, "mk_live_unknown") is None


def test_resolve_active_key_returns_context():
    sb, _ = _sb([ACTIVE_KEY])
    ctx = psvc.resolve_key(sb, "mk_live_" + "x" * 43)
    assert ctx == psvc.PartnerContext(org_id=ORG_ID, key_id=KEY_ID)


def test_resolve_rejects_expired_key():
    sb, _ = _sb([{**ACTIVE_KEY, "expires_at": "2020-01-01T00:00:00+00:00"}])
    assert psvc.resolve_key(sb, "mk_live_" + "x" * 43) is None


def test_resolve_rejects_disabled_or_archived_org():
    for org in (
        {**ACTIVE_ORG, "partner_api_enabled": False},
        {**ACTIVE_ORG, "archived_at": "2026-01-01T00:00:00+00:00"},
        {**ACTIVE_ORG, "status": "suspended"},
    ):
        sb, _ = _sb([ACTIVE_KEY], org_row=org)
        assert psvc.resolve_key(sb, "mk_live_" + "x" * 43) is None


def test_resolve_filters_on_active_status():
    # MockQueryBuilder.eq is a no-op, so the SQL-side filter has to be asserted
    # on the calls — a revoked key would otherwise resolve fine in Python.
    sb, keys = _sb([ACTIVE_KEY])
    keys.eq = MagicMock(return_value=keys)
    assert psvc.resolve_key(sb, "mk_live_" + "x" * 43) is not None
    assert ("status", "active") in [c.args for c in keys.eq.call_args_list]


def test_revoke_reports_whether_anything_matched_and_stamps_the_clock():
    sb, keys = _sb([])
    keys.eq = MagicMock(return_value=keys)
    keys.update = MagicMock(return_value=keys)
    keys.execute = MagicMock(return_value=MagicMock(data=[{"id": KEY_ID}]))
    assert psvc.revoke_key(sb, ORG_ID, KEY_ID) is True
    filters = [c.args for c in keys.eq.call_args_list]
    assert ("org_id", ORG_ID) in filters and ("id", KEY_ID) in filters
    # revoked_at is what the 30-day hiding counts from — a status with no
    # timestamp can never be hidden.
    patch = keys.update.call_args[0][0]
    assert patch["status"] == "revoked"
    assert datetime.fromisoformat(patch["revoked_at"]).tzinfo is not None

    sb2, keys2 = _sb([])  # execute -> data=[] : nothing matched
    assert psvc.revoke_key(sb2, ORG_ID, "not-ours") is False
    assert keys2.execute.call_count == 1  # one UPDATE, nothing to cascade


def test_list_keys_is_scoped_to_the_org_and_never_selects_the_hash():
    sb, keys = _sb([])
    keys.eq = MagicMock(return_value=keys)
    keys.select = MagicMock(return_value=keys)
    psvc.list_keys(sb, ORG_ID)
    assert [c.args[0] for c in keys.eq.call_args_list] == ["org_id"]
    selected = keys.select.call_args[0][0]
    assert "key_hash" not in selected
    # The console renders folder + inactivity from these two.
    assert "revoked_at" in selected and "folder_id" in selected


# ---- inactive keys are hidden, never deleted --------------------------------


@pytest.mark.parametrize(
    "row,hidden",
    [
        ({"status": "revoked", "revoked_at": _ago(29)}, False),
        ({"status": "revoked", "revoked_at": _ago(31)}, True),
        # Pre-migration row: revoked, but no clock to measure from.
        ({"status": "revoked", "revoked_at": None}, False),
        ({"status": "active", "expires_at": _ago(31)}, True),
        ({"status": "active", "expires_at": _ago(29)}, False),
        ({"status": "active", "expires_at": None}, False),
    ],
)
def test_is_hidden_boundary(row, hidden):
    assert psvc.is_hidden(row, NOW) is hidden


def test_list_keys_drops_hidden_rows():
    rows = [
        {"id": "live", "status": "active", "expires_at": None, "revoked_at": None},
        {"id": "old", "status": "revoked", "revoked_at": _ago(400)},
        {"id": "lapsed", "status": "active", "expires_at": _ago(90)},
    ]
    sb, _ = _sb(rows)
    assert [k["id"] for k in psvc.list_keys(sb, ORG_ID)] == ["live"]


def test_key_status_prefers_revoked_over_expired():
    assert psvc.key_status({"status": "active", "expires_at": None}, NOW) == "active"
    assert psvc.key_status({"status": "active", "expires_at": _ago(1)}, NOW) == "expired"
    assert psvc.key_status({"status": "revoked", "expires_at": _ago(1)}, NOW) == "revoked"


# ---- folders ----------------------------------------------------------------


def test_list_folders_is_scoped_and_ordered():
    sb, _ = _sb([], folder_rows=[FOLDER])
    sb.folders.eq = MagicMock(return_value=sb.folders)
    sb.folders.order = MagicMock(return_value=sb.folders)
    assert psvc.list_folders(sb, ORG_ID) == [FOLDER]
    assert [c.args for c in sb.folders.eq.call_args_list] == [("org_id", ORG_ID)]
    assert sb.folders.order.call_args[0] == ("name",)


def test_create_folder_is_idempotent_case_insensitively():
    sb, _ = _sb([], folder_rows=[FOLDER])
    out = psvc.create_folder(sb, ORG_ID, "  ingest ")
    assert out["id"] == FOLDER_ID and out["created"] is False
    sb.folders.insert.assert_not_called()


def test_create_folder_inserts_a_new_name():
    sb, _ = _sb([], folder_rows=[])
    sb.folders.execute = MagicMock(side_effect=[MagicMock(data=[]), MagicMock(data=[FOLDER])])
    out = psvc.create_folder(sb, ORG_ID, " Ingest ")
    assert out["id"] == FOLDER_ID and out["created"] is True
    assert sb.folders.insert.call_args[0][0] == {"org_id": ORG_ID, "name": "Ingest"}


def test_create_folder_rejects_a_blank_or_oversized_name():
    sb, _ = _sb([])
    for bad in ("   ", "x" * 81):
        with pytest.raises(ValueError):
            psvc.create_folder(sb, ORG_ID, bad)


def test_set_key_folder_refuses_a_foreign_folder_without_writing():
    sb, keys = _sb([{"id": KEY_ID}], folder_rows=[])  # folder lookup -> no row
    assert psvc.set_key_folder(sb, ORG_ID, KEY_ID, "not-ours") is False
    keys.execute.assert_not_called()


def test_set_key_folder_reports_an_unknown_key_as_false():
    sb, keys = _sb([], folder_rows=[FOLDER])  # key update matches nothing
    keys.eq = MagicMock(return_value=keys)
    assert psvc.set_key_folder(sb, ORG_ID, "not-ours", FOLDER_ID) is False
    assert ("org_id", ORG_ID) in [c.args for c in keys.eq.call_args_list]


def test_set_key_folder_unfiles_without_touching_the_folders_table():
    sb, keys = _sb([{"id": KEY_ID}])
    keys.update = MagicMock(return_value=keys)
    assert psvc.set_key_folder(sb, ORG_ID, KEY_ID, None) is True
    assert keys.update.call_args[0][0] == {"folder_id": None}
    sb.folders.execute.assert_not_called()


def test_mint_stores_the_folder_and_rejects_an_unknown_one():
    sb, keys = _sb([], folder_rows=[FOLDER])
    keys.execute = MagicMock(return_value=MagicMock(data=[{"id": KEY_ID}]))
    psvc.mint_key(sb, ORG_ID, label="Prod", folder_id=FOLDER_ID)
    assert keys.insert.call_args[0][0]["folder_id"] == FOLDER_ID

    sb2, keys2 = _sb([], folder_rows=[])
    with pytest.raises(ValueError):
        psvc.mint_key(sb2, ORG_ID, label="Prod", folder_id="not-ours")
    keys2.insert.assert_not_called()


def test_surface_flag_requires_all_three(monkeypatch):
    monkeypatch.setenv("PARTNER_API_ENABLED", "true")
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")
    assert psvc.partner_surface_enabled() is True
    monkeypatch.delenv("PARTNER_API_ENABLED")
    assert psvc.partner_surface_enabled() is False
