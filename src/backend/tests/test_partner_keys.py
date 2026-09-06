from unittest.mock import MagicMock

from partner_api import service as psvc
from tests.conftest import MockQueryBuilder

ORG_ID = "00000000-0000-0000-0000-0000000000aa"
KEY_ID = "00000000-0000-0000-0000-0000000000bb"

ACTIVE_ORG = {"id": ORG_ID, "status": "active", "archived_at": None, "partner_api_enabled": True}
ACTIVE_KEY = {"id": KEY_ID, "org_id": ORG_ID, "status": "active", "expires_at": None}


def _sb(key_rows, org_row=ACTIVE_ORG):
    """Mock supabase dispatching per-table builders (conftest pattern)."""
    keys = MockQueryBuilder()
    keys.execute = MagicMock(return_value=MagicMock(data=key_rows))
    orgs = MockQueryBuilder()
    orgs.execute = MagicMock(return_value=MagicMock(data=[org_row] if org_row else []))
    sb = MagicMock()
    sb.table.side_effect = lambda name: {"partner_api_keys": keys, "organizations": orgs}[name]
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


def test_revoke_reports_whether_anything_matched():
    sb, keys = _sb([])
    keys.eq = MagicMock(return_value=keys)
    keys.execute = MagicMock(return_value=MagicMock(data=[{"id": KEY_ID}]))
    assert psvc.revoke_key(sb, ORG_ID, KEY_ID) is True
    filters = [c.args for c in keys.eq.call_args_list]
    assert ("org_id", ORG_ID) in filters and ("id", KEY_ID) in filters

    sb2, keys2 = _sb([])  # execute -> data=[] : nothing matched
    assert psvc.revoke_key(sb2, ORG_ID, "not-ours") is False
    assert keys2.execute.call_count == 1  # one UPDATE, nothing to cascade


def test_list_keys_is_scoped_to_the_org_and_never_selects_the_hash():
    sb, keys = _sb([])
    keys.eq = MagicMock(return_value=keys)
    keys.select = MagicMock(return_value=keys)
    psvc.list_keys(sb, ORG_ID)
    assert [c.args[0] for c in keys.eq.call_args_list] == ["org_id"]
    assert "key_hash" not in keys.select.call_args[0][0]


def test_surface_flag_requires_all_three(monkeypatch):
    monkeypatch.setenv("PARTNER_API_ENABLED", "true")
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")
    assert psvc.partner_surface_enabled() is True
    monkeypatch.delenv("PARTNER_API_ENABLED")
    assert psvc.partner_surface_enabled() is False
