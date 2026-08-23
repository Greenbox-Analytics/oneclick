"""The `?scope=` contract at the ROUTER layer, across every listing endpoint.

Two properties the service-level tests can't pin:

1. A forged/foreign org scope is a uniform 404 ("Organization not found") on
   EVERY listing endpoint — never a 500. GET /artists matters especially: its
   handler wraps the body in `except Exception -> 500`, so resolve_scope must
   run OUTSIDE that net or the no-existence-oracle contract silently breaks.
2. With the flags off, `?scope=` is INERT — no membership RPC fires and the
   request succeeds exactly as before the feature (the rollback path).

Uses the shared `client` fixture (auth overridden, supabase mocked); the org
membership check is the `is_org_member` RPC on the mocked client.
"""

from unittest.mock import MagicMock

import pytest

from tests.conftest import TEST_USER_ID

FOREIGN_ORG = "20000000-0000-0000-0000-0000000000ff"

# Every GET that resolves the workspace scope. Kept in one list so a future
# endpoint only needs one line — and so the 404-uniformity test can't drift
# per-endpoint.
SCOPED_LIST_ENDPOINTS = [
    "/artists",
    "/projects",
    "/zoe/context-tree",
    "/registry/works",
    "/registry/works/my-collaborations",
    "/registry/artists/with-teamcards",
    "/boards/boards",
    "/boards/archived",
    "/boards/columns",
    "/boards/tasks",
    "/boards/parents",
    "/expenses/summary",
]


@pytest.fixture
def scoping_on(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SCOPING_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")


@pytest.mark.parametrize("endpoint", SCOPED_LIST_ENDPOINTS)
def test_foreign_org_scope_is_a_uniform_404(scoping_on, client, mock_supabase, endpoint):
    """A foreign org and a nonexistent one must be indistinguishable — 404,
    same body, never a 500 — or ?scope= becomes an org-id existence probe."""
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=False)

    resp = client.get(endpoint, params={"scope": FOREIGN_ORG})

    assert resp.status_code == 404, f"{endpoint} returned {resp.status_code}"
    assert resp.json()["detail"] == "Organization not found"


def test_calendar_foreign_org_scope_404s(scoping_on, client, mock_supabase):
    """/boards/calendar needs its own case — it has required params."""
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=False)

    resp = client.get(
        "/boards/calendar",
        params={"start": "2026-08-01", "end": "2026-08-31", "scope": FOREIGN_ORG},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Organization not found"


def test_flags_off_scope_param_is_inert(client, mock_supabase):
    """Rollback: with the flags off a scope param confers nothing and checks
    nothing — no membership RPC, no 404, the pre-scoping response."""
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=False)

    resp = client.get("/artists", params={"scope": FOREIGN_ORG})

    assert resp.status_code == 200
    mock_supabase.rpc.assert_not_called()


def test_personal_scope_needs_no_membership_check(scoping_on, client, mock_supabase):
    resp = client.get("/artists", params={"scope": "personal"})

    assert resp.status_code == 200
    mock_supabase.rpc.assert_not_called()


def test_member_org_scope_is_accepted(scoping_on, client, mock_supabase):
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=True)

    resp = client.get("/artists", params={"scope": FOREIGN_ORG})

    assert resp.status_code == 200
    mock_supabase.rpc.assert_called_with("is_org_member", {"p_user_id": TEST_USER_ID, "p_org_id": FOREIGN_ORG})
