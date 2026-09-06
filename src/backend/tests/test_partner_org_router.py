"""Phase-2 portal: org-admin key endpoints on the PRODUCT backend."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

ORG_ID = "00000000-0000-0000-0000-0000000000aa"
KEY_ID = "00000000-0000-0000-0000-0000000000bb"
CHILD_ID = "00000000-0000-0000-0000-0000000000cd"

# _first_org no longer selects partner_api_enabled — the gate reads that bit
# with its own organizations query (see _seed_capability).
LIVE_ORG = {
    "id": ORG_ID,
    "status": "active",
    "archived_at": None,
    "dissolved_at": None,
}


def _seed_capability(mock_supabase, enabled):
    """Answer _gate's own `organizations.partner_api_enabled` read; every other
    table keeps the conftest default. enabled=None seeds NO row."""
    from tests.conftest import MockQueryBuilder, _default_table_side_effect

    rows = [] if enabled is None else [{"partner_api_enabled": enabled}]

    def _side(name):
        if name != "organizations":
            return _default_table_side_effect(name)
        b = MockQueryBuilder()
        b.execute.return_value = MagicMock(data=rows, count=len(rows))
        return b

    mock_supabase.table.side_effect = _side


@pytest.fixture
def flags(monkeypatch, client):
    # Depends on `client` so this runs AFTER main.py's import-time load_dotenv(),
    # which would otherwise re-seed a var this fixture just deleted.
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")
    monkeypatch.delenv("PARTNER_API_ENABLED", raising=False)


@pytest.fixture
def admin_of_live_org(monkeypatch, mock_supabase):
    monkeypatch.setattr("partner_api.org_router.authz.require_admin", lambda sb, uid, org: None)
    monkeypatch.setattr("partner_api.org_router.orgs_service._first_org", lambda sb, org: dict(LIVE_ORG))
    _seed_capability(mock_supabase, True)


def test_flags_off_404(client, monkeypatch):
    # conftest clears both credit flags -> the whole router is a 404. Re-clear
    # explicitly: main.py's load_dotenv() can re-seed them from a dev's .env.
    monkeypatch.delenv("CREDITS_ENABLED", raising=False)
    monkeypatch.delenv("LICENSING_ENABLED", raising=False)
    assert client.get(f"/orgs/{ORG_ID}/partner-keys").status_code == 404


def test_partner_host_lockdown_404s_this_router(client, flags, monkeypatch, admin_of_live_org):
    monkeypatch.setenv("PARTNER_API_ENABLED", "true")
    assert client.get(f"/orgs/{ORG_ID}/partner-keys").status_code == 404


def test_non_admin_403_before_anything_about_the_org_leaks(client, flags, monkeypatch):
    def deny(sb, uid, org):
        raise HTTPException(status_code=403, detail="Admin access required")

    monkeypatch.setattr("partner_api.org_router.authz.require_admin", deny)
    first_org, capability = MagicMock(), MagicMock()
    monkeypatch.setattr("partner_api.org_router.orgs_service._first_org", first_org)
    monkeypatch.setattr("partner_api.org_router._partner_api_enabled", capability)
    assert client.get(f"/orgs/{ORG_ID}/partner-keys").status_code == 403
    first_org.assert_not_called()
    capability.assert_not_called()


def test_missing_org_404s_before_the_capability_read(client, flags, monkeypatch):
    monkeypatch.setattr("partner_api.org_router.authz.require_admin", lambda sb, uid, org: None)
    monkeypatch.setattr("partner_api.org_router.orgs_service._first_org", lambda sb, org: None)
    capability = MagicMock()
    monkeypatch.setattr("partner_api.org_router._partner_api_enabled", capability)
    assert client.get(f"/orgs/{ORG_ID}/partner-keys").status_code == 404
    capability.assert_not_called()


def test_disabled_org_403_access_disabled(client, flags, monkeypatch, mock_supabase):
    monkeypatch.setattr("partner_api.org_router.authz.require_admin", lambda sb, uid, org: None)
    monkeypatch.setattr("partner_api.org_router.orgs_service._first_org", lambda sb, org: dict(LIVE_ORG))
    _seed_capability(mock_supabase, False)
    r = client.get(f"/orgs/{ORG_ID}/partner-keys")
    assert r.status_code == 403
    assert r.json()["detail"] == {"code": "access_disabled"}


def test_no_capability_row_denies(client, flags, monkeypatch, mock_supabase):
    """No row back (unreadable org) => denied, never opened or 500."""
    monkeypatch.setattr("partner_api.org_router.authz.require_admin", lambda sb, uid, org: None)
    monkeypatch.setattr("partner_api.org_router.orgs_service._first_org", lambda sb, org: dict(LIVE_ORG))
    _seed_capability(mock_supabase, None)
    assert client.get(f"/orgs/{ORG_ID}/partner-keys").status_code == 403


def test_archived_org_409_on_writes_200_on_reads(client, flags, monkeypatch, mock_supabase):
    monkeypatch.setattr("partner_api.org_router.authz.require_admin", lambda sb, uid, org: None)
    monkeypatch.setattr(
        "partner_api.org_router.orgs_service._first_org",
        lambda sb, org: {**LIVE_ORG, "archived_at": "2026-08-01T00:00:00+00:00"},
    )
    _seed_capability(mock_supabase, True)
    monkeypatch.setattr("partner_api.org_router.psvc.list_keys", lambda sb, org: [])
    monkeypatch.setattr("partner_api.org_router.psvc.created_by_labels", lambda sb, org, keys: {})
    assert client.get(f"/orgs/{ORG_ID}/partner-keys").status_code == 200
    assert client.post(f"/orgs/{ORG_ID}/partner-keys", json={"label": "x"}).status_code == 409
    assert client.delete(f"/orgs/{ORG_ID}/partner-keys/{KEY_ID}").status_code == 409


def test_list_labels_created_by_and_never_leaks_secrets(client, flags, admin_of_live_org, monkeypatch):
    keys = [
        {
            "id": KEY_ID,
            "label": "Prod",
            "key_prefix": "mk_live_abcd",
            "status": "active",
            "expires_at": None,
            "created_at": "2026-09-01T00:00:00+00:00",
            "created_by": "u-admin",
            "last_used_at": None,
        },
        {
            "id": CHILD_ID,
            "label": "Staging",
            "key_prefix": "mk_live_efgh",
            "status": "active",
            "expires_at": None,
            "created_at": "2026-09-02T00:00:00+00:00",
            "created_by": None,
            "last_used_at": None,
        },
    ]
    monkeypatch.setattr(
        "partner_api.org_router.psvc.list_keys",
        lambda sb, org: [dict(k) for k in keys],
    )
    monkeypatch.setattr(
        "partner_api.org_router.psvc.created_by_labels",
        lambda sb, org, ks: {"u-admin": "admin@label.test"},
    )
    r = client.get(f"/orgs/{ORG_ID}/partner-keys")
    assert r.status_code == 200
    out = r.json()["keys"]
    assert out[0]["created_by_label"] == "admin@label.test"
    assert out[1]["created_by_label"] is None  # creator unresolvable -> null, never a guess
    assert all("secret" not in k and "key_hash" not in k for k in out)


def test_create_key(client, flags, admin_of_live_org, monkeypatch):
    minted = []

    def fake_mint(sb, org_id, **kw):
        minted.append({"org_id": org_id, **kw})
        return {"id": "new", "secret": "mk_live_new", **kw}

    monkeypatch.setattr("partner_api.org_router.psvc.mint_key", fake_mint)
    events = []
    monkeypatch.setattr("partner_api.org_router.analytics_capture", lambda d, e, p=None: events.append((e, p)))

    r = client.post(f"/orgs/{ORG_ID}/partner-keys", json={"label": "Production backend"})
    assert r.status_code == 200 and r.json()["secret"] == "mk_live_new"
    assert "user_ref" not in minted[0] and "parent_key_id" not in minted[0]  # one key type
    assert minted[0]["created_by"]  # the caller's id from get_current_user_id (client fixture)

    r = client.post(f"/orgs/{ORG_ID}/partner-keys", json={"label": "Staging", "expires_at": "2099-01-01"})
    assert r.status_code == 200
    assert minted[1]["expires_at"].endswith("+00:00")  # naive date normalized to UTC by ExpiringKeyCreate

    assert [e for e, _ in events] == ["partner_key_created", "partner_key_created"]
    assert events[1][1] == {"org_id": ORG_ID, "via": "portal"}


def test_create_rejects_past_expiry_and_blank_label(client, flags, admin_of_live_org):
    assert (
        client.post(f"/orgs/{ORG_ID}/partner-keys", json={"label": "x", "expires_at": "2020-01-01"}).status_code == 422
    )
    assert client.post(f"/orgs/{ORG_ID}/partner-keys", json={"label": ""}).status_code == 422


def test_revoke_404s_on_foreign_key_or_lost_race(client, flags, admin_of_live_org, monkeypatch):
    # revoke_key scopes on org_id and returns False for a foreign id OR a race
    # where another admin revoked it first — the router turns that into 404
    # without reading the key list first.
    revoked = []
    monkeypatch.setattr(
        "partner_api.org_router.psvc.revoke_key",
        lambda sb, org, kid: revoked.append(kid) or kid == KEY_ID,
    )
    events = []
    monkeypatch.setattr("partner_api.org_router.analytics_capture", lambda d, e, p=None: events.append((e, p)))

    assert client.delete(f"/orgs/{ORG_ID}/partner-keys/not-ours").status_code == 404
    assert events == []

    r = client.delete(f"/orgs/{ORG_ID}/partner-keys/{KEY_ID}")
    assert r.status_code == 200 and revoked == ["not-ours", KEY_ID]
    assert events == [("partner_key_revoked", {"org_id": ORG_ID, "via": "portal"})]
