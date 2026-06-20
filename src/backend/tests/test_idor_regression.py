"""Cross-tenant (IDOR) regression guard.

Representative — not an exhaustive enumeration of all endpoints. Asserts that the
previously-vulnerable work-access and project-membership routes deny access when
the caller lacks it. See docs/superpowers/plans/2026-06-17-backend-idor-remediation.md.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from registry.access import WorkAccess
from tests.conftest import MockQueryBuilder, _default_table_side_effect

# Work-access-gated routes: deny by patching get_work_access to an empty WorkAccess
# (can_view / can_edit / can_manage all False).
DENIED_WORK_ROUTES = [
    ("get", "/registry/works/w/files"),
    ("get", "/registry/works/w/audio"),
    ("get", "/registry/works/w/full"),
    ("get", "/registry/stakes?work_id=w"),
]


@pytest.mark.parametrize("method,path", DENIED_WORK_ROUTES)
def test_work_routes_denied(client, method, path):
    with (
        patch("registry.work_links_service.get_work_access", AsyncMock(return_value=WorkAccess())),
        patch("registry.service.get_work_access", AsyncMock(return_value=WorkAccess())),
    ):
        resp = getattr(client, method)(path)
    assert resp.status_code in (403, 404)


# Membership-gated routes: deny when get_user_role returns None and the caller owns no artists.
DENIED_MEMBERSHIP_ROUTES = [
    ("get", "/projects/victim/members"),
    ("get", "/projects/victim/pending-invites"),
    ("get", "/registry/works/by-project/victim"),
]


@pytest.mark.parametrize("method,path", DENIED_MEMBERSHIP_ROUTES)
def test_membership_routes_denied(client, mock_supabase, method, path):
    def _router(name):
        b = MockQueryBuilder()
        if name == "project_members":
            b.execute.return_value = MagicMock(data=None)  # get_user_role -> None
        elif name == "artists":
            b.execute.return_value = MagicMock(data=[], count=0)  # owns nothing
        return b

    mock_supabase.table.side_effect = _router
    resp = getattr(client, method)(path)
    assert resp.status_code in (403, 404)


def test_parse_contract_splits_denied_for_unowned_contract(client, mock_supabase):
    """Closes the coverage gap noted in the Task 5 review."""

    def _router(name):
        if name == "project_files":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[{"project_id": "victim-proj"}], count=1)
            return b
        if name == "artists":
            b = MockQueryBuilder()
            b.execute.return_value = MagicMock(data=[], count=0)  # caller owns no artists
            return b
        # Subscription/entitlement tables fall through to default (Pro) so the
        # feature gate passes and the ownership check is what produces the 403.
        return _default_table_side_effect(name)

    mock_supabase.table.side_effect = _router
    # Endpoint takes multipart Form fields, not JSON.
    resp = client.post("/registry/parse-contract-splits", data={"contract_file_id": "victim-file"})
    assert resp.status_code == 403
