from unittest.mock import MagicMock

import pytest

from partner_api.service import PartnerContext
from tests.conftest import MockQueryBuilder

ORG_ID = "00000000-0000-0000-0000-0000000000aa"
CTX = PartnerContext(org_id=ORG_ID, key_id="00000000-0000-0000-0000-0000000000bb")


@pytest.fixture
def flags(monkeypatch):
    monkeypatch.setenv("PARTNER_API_ENABLED", "true")
    monkeypatch.setenv("CREDITS_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")


def test_flag_matrix_404s(client, monkeypatch):
    # conftest clears CREDITS/LICENSING, but main.py's import-time load_dotenv()
    # can re-seed them from a dev's .env AFTER that fixture ran (the client
    # fixture is what first imports main). Clear all three here so the matrix
    # tests the gate, not the developer's environment.
    for flag in ("PARTNER_API_ENABLED", "CREDITS_ENABLED", "LICENSING_ENABLED"):
        monkeypatch.delenv(flag, raising=False)
    h = {"Authorization": "Bearer mk_live_x"}
    assert client.get("/zoe/v1/models", headers=h).status_code == 404
    assert client.post("/zoe/v1/chat/completions", headers=h, json={"messages": []}).status_code == 404
    assert client.post("/registry/v1/splits", headers=h).status_code == 404
    assert client.post("/splitsheet/v1/documents", headers=h, json={}).status_code == 404
    # each flag alone is not enough
    for present in ("PARTNER_API_ENABLED", "CREDITS_ENABLED", "LICENSING_ENABLED"):
        monkeypatch.setenv(present, "true")
        assert client.get("/zoe/v1/models", headers=h).status_code == 404
        monkeypatch.delenv(present)


def test_every_route_requires_a_valid_key(client, flags, monkeypatch):
    # GET /zoe/v1/models is the free key check; a bad key is a 401 there and
    # on every billed route alike.
    monkeypatch.setattr("partner_api.router.psvc.resolve_key", lambda sb, b: None)
    h = {"Authorization": "Bearer mk_live_bad"}
    assert client.get("/zoe/v1/models", headers=h).status_code == 401
    assert client.post("/oneclick/v1/royalties", headers=h).status_code == 401
    assert client.post("/registry/v1/splits", headers=h).status_code == 401
    assert client.post("/splitsheet/v1/documents", headers=h, json={}).status_code == 401


def test_no_account_level_routes(client, flags, monkeypatch):
    # /me and /test were removed (2026-09-06): nothing on the surface reads
    # the balance or answers for free except the OpenAI model probe.
    monkeypatch.setattr("partner_api.router.psvc.resolve_key", lambda sb, b: CTX)
    h = {"Authorization": "Bearer mk_live_ok"}
    assert client.get("/me", headers=h).json()["detail"] == "Not found"
    assert client.post("/test", headers=h).json()["detail"] == "Not found"


def test_machine_surface_has_no_key_management(client, flags, monkeypatch):
    # One key type, minted by humans in the console or admin UI — nothing on
    # the bearer-key surface mints, lists or revokes.
    monkeypatch.setattr("partner_api.router.psvc.resolve_key", lambda sb, b: CTX)
    h = {"Authorization": "Bearer mk_live_ok"}
    assert client.post("/keys", headers=h, json={"label": "x"}).status_code == 404
    assert client.get("/keys", headers=h).status_code == 404
    assert client.delete("/keys/anything", headers=h).status_code == 404


def test_partner_host_lockdown_404s_everything_but_api_routes_and_health(client, flags, monkeypatch):
    # With PARTNER_API_ENABLED set this process IS the API host: only the
    # partner routers' own routes and /health may answer, whatever auth a
    # request carries. The allowlist is by ROUTE, not prefix — product routes
    # under the shared /oneclick and /zoe prefixes stay dark.
    monkeypatch.setattr("partner_api.router.psvc.resolve_key", lambda sb, b: None)
    assert client.get("/health").status_code == 200
    # Reached the router, key rejected: the four tools' own routes.
    assert client.post("/oneclick/v1/royalties").status_code == 401
    assert client.post("/registry/v1/splits").status_code == 401
    assert client.post("/splitsheet/v1/documents").status_code == 401
    assert client.post("/zoe/v1/chat/completions").status_code == 401
    assert client.get("/zoe/v1/models").status_code == 401
    # Product routes under the shared prefixes stay dark.
    assert client.post("/oneclick/calculate-royalties").json()["detail"] == "Not found"
    assert client.post("/registry/parse-contract-splits").json()["detail"] == "Not found"
    assert client.post("/splitsheet/generate").json()["detail"] == "Not found"
    assert client.post("/zoe/ask-stream").json()["detail"] == "Not found"
    assert client.get("/partner/v1/me").json()["detail"] == "Not found"  # the old prefix is gone
    assert client.get("/me").json()["detail"] == "Not found"  # and the old account routes
    assert client.post("/test").json()["detail"] == "Not found"
    # Exactly "/health", not a prefix: "Not found" is the lockdown's own body,
    # FastAPI's no-route 404 says "Not Found".
    assert client.get("/healthz").json()["detail"] == "Not found"
    assert client.get(f"/orgs/{ORG_ID}").status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.put(f"/admin/orgs/{ORG_ID}/partner-api", json={"enabled": True}).status_code == 404


def test_admin_mints_lists_revokes_keys_on_product_host(client, monkeypatch, mock_supabase):
    # PRODUCT host: PARTNER_API_ENABLED deliberately NOT set (conftest clears
    # the credit flags; this one is never set here). The admin key endpoints
    # must work exactly where an operator will call them — flag-independent.
    import main
    from subscriptions.admin_auth import require_admin

    monkeypatch.delenv("PARTNER_API_ENABLED", raising=False)
    main.app.dependency_overrides[require_admin] = lambda: "admin@example.com"
    try:
        orgs = MockQueryBuilder()
        orgs.execute = MagicMock(return_value=MagicMock(data={"id": ORG_ID}))
        mock_supabase.table.side_effect = lambda name: orgs

        minted = {}

        def fake_mint(sb, org_id, **kw):
            minted.update(kw, org_id=org_id)
            return {"id": "k1", "secret": "mk_live_k1"}

        monkeypatch.setattr("subscriptions.admin_router.psvc.mint_key", fake_mint)
        r = client.post(f"/admin/orgs/{ORG_ID}/partner-keys", json={"label": "Production backend"})
        assert r.status_code == 200
        assert r.json()["secret"] == "mk_live_k1"
        assert minted["created_by"]  # the UUID from get_current_user_id, not the admin email
        assert "user_ref" not in minted

        monkeypatch.setattr("subscriptions.admin_router.psvc.list_keys", lambda sb, org_id: [{"id": "k1"}])
        r = client.get(f"/admin/orgs/{ORG_ID}/partner-keys")
        assert r.status_code == 200
        assert "secret" not in r.json()["keys"][0]

        revoked = {}

        def fake_revoke(sb, org_id, key_id):
            revoked.update(org=org_id, key=key_id)
            return True

        monkeypatch.setattr("subscriptions.admin_router.psvc.revoke_key", fake_revoke)
        r = client.delete(f"/admin/orgs/{ORG_ID}/partner-keys/k1")
        assert r.status_code == 200
        assert revoked == {"org": ORG_ID, "key": "k1"}

        # Foreign / unknown id: revoke_key reports no match -> 404, not "revoked".
        monkeypatch.setattr("subscriptions.admin_router.psvc.revoke_key", lambda sb, org_id, key_id: False)
        assert client.delete(f"/admin/orgs/{ORG_ID}/partner-keys/nope").status_code == 404

        # Past expiry is a 422 at the edge on this path too (ExpiringKeyCreate).
        r = client.post(f"/admin/orgs/{ORG_ID}/partner-keys", json={"label": "x", "expires_at": "2020-01-01"})
        assert r.status_code == 422
    finally:
        main.app.dependency_overrides.pop(require_admin, None)


def test_cors_exposes_the_billing_headers_to_browsers(client):
    # The docs console reads Msanii-Credits cross-origin; without expose_headers
    # a browser hides every custom response header. Behavioural, not introspective.
    r = client.get("/health", headers={"Origin": "http://localhost:8080"})
    exposed = {h.strip() for h in r.headers["access-control-expose-headers"].split(",")}
    assert {"Msanii-Credits", "Msanii-Request-Id", "Msanii-Replayed", "Content-Disposition"} <= exposed
