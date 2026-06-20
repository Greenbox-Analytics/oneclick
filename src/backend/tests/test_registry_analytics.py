"""Tests for registry step event instrumentation:
- registry_collaborator_invited (router)
- registry_work_registered (service, both call sites)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import TEST_USER_ID, MockQueryBuilder, _default_table_side_effect

_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})


def _sub_wrap(fn):
    """Route subscription tables through the Pro-tier default so gating passes."""

    def _wrapped(name):
        if name in _SUBSCRIPTION_TABLES:
            return _default_table_side_effect(name)
        return fn(name)

    return _wrapped


WORK_ID = "aaaaaaaa-0000-0000-0000-000000000001"
COLLAB_ID = "bbbbbbbb-0000-0000-0000-000000000002"


def _collab_row(**overrides):
    base = {
        "id": COLLAB_ID,
        "work_id": WORK_ID,
        "invited_by": TEST_USER_ID,
        "email": "collab@example.com",
        "name": "Alice Collab",
        "role": "writer",
        "status": "invited",
        "invite_token": "tok-abc",
        "expires_at": None,
        "collaborator_user_id": None,
        "stake_id": None,
        "responded_at": None,
        "invited_at": "2026-04-10T00:00:00+00:00",
    }
    base.update(overrides)
    return base


# ============================================================
# 1. registry_collaborator_invited (router)
# ============================================================


def test_invite_collaborator_fires_event(client, mock_supabase):
    """POST /registry/collaborators/invite fires registry_collaborator_invited."""
    captured = []

    work_title_builder = MockQueryBuilder()
    work_title_builder.execute.return_value = MagicMock(data={"title": "Track A"}, count=1)

    mock_supabase.rpc.return_value = MockQueryBuilder()
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=None)

    insert_builder = MockQueryBuilder()
    insert_builder.execute.return_value = MagicMock(data=[_collab_row(role="writer")], count=1)

    profile_builder = MockQueryBuilder()
    profile_builder.execute.return_value = MagicMock(data={"full_name": "Test User"}, count=1)

    call_count = {"n": 0}

    def table_side_effect(name):
        call_count["n"] += 1
        n = call_count["n"]
        if n == 1:
            return work_title_builder  # router: works_registry title lookup
        elif n == 2:
            return insert_builder  # service: registry_collaborators insert
        else:
            return profile_builder  # router: profiles inviter name

    mock_supabase.table.side_effect = _sub_wrap(table_side_effect)

    from registry.access import WorkAccess

    with (
        patch("registry.emails.send_invitation_email"),
        patch("registry.router.get_work_access", AsyncMock(return_value=WorkAccess(work_role="owner"))),
        patch(
            "registry.router.analytics_capture",
            side_effect=lambda uid, event, props=None: captured.append((uid, event, dict(props or {}))),
        ),
    ):
        resp = client.post(
            "/registry/collaborators/invite",
            json={
                "work_id": WORK_ID,
                "email": "collab@example.com",
                "name": "Alice Collab",
                "role": "writer",
            },
        )

    assert resp.status_code == 200, resp.text
    invited = [c for c in captured if c[1] == "registry_collaborator_invited"]
    assert len(invited) == 1, f"expected registry_collaborator_invited, got {captured}"
    assert invited[0][0] == TEST_USER_ID
    assert invited[0][2]["tool"] == "registry"
    assert invited[0][2]["role"] == "writer"


# ============================================================
# 2. registry_work_registered (service — check_and_update_work_status)
# ============================================================


def test_check_and_update_fires_registered_when_status_flips(monkeypatch):
    """check_and_update_work_status fires registry_work_registered when it flips
    a work to status='registered'."""
    from registry import service

    captured = []
    monkeypatch.setattr(
        service,
        "analytics_capture",
        lambda uid, event, props=None: captured.append((uid, event, dict(props or {}))),
    )

    # The function makes two table() calls in the success branch:
    #   1) select registry_collaborators (all confirmed)
    #   2) update works_registry status=registered
    #   3) (our new code) select works_registry user_id
    # Use a fresh MockQueryBuilder per table call so we can configure
    # different execute return values for the read calls.
    collabs_builder = MockQueryBuilder()
    collabs_builder.execute.return_value = MagicMock(data=[{"status": "confirmed"}, {"status": "confirmed"}], count=2)

    update_builder = MockQueryBuilder()
    update_builder.execute.return_value = MagicMock(data=[{"id": WORK_ID}], count=1)

    owner_builder = MockQueryBuilder()
    owner_builder.execute.return_value = MagicMock(data={"user_id": "owner-u1"}, count=1)

    call_log = []

    def table_side_effect(name):
        call_log.append(name)
        # First works_registry call is update; second is the owner lookup.
        if name == "registry_collaborators":
            return collabs_builder
        if name == "works_registry":
            # First call = update path, second = owner select
            count = sum(1 for n in call_log if n == "works_registry")
            return update_builder if count == 1 else owner_builder
        return MockQueryBuilder()

    db = MagicMock()
    db.table.side_effect = table_side_effect

    asyncio.run(service.check_and_update_work_status(db, WORK_ID))

    registered = [c for c in captured if c[1] == "registry_work_registered"]
    assert len(registered) == 1, f"expected registry_work_registered, got {captured}"
    assert registered[0][0] == "owner-u1"
    assert registered[0][2]["tool"] == "registry"
    assert registered[0][2]["work_id"] == WORK_ID


def test_check_and_update_no_event_when_not_all_confirmed(monkeypatch):
    """No event fires when at least one collaborator is still pending."""
    from registry import service

    captured = []
    monkeypatch.setattr(
        service,
        "analytics_capture",
        lambda uid, event, props=None: captured.append((uid, event, dict(props or {}))),
    )

    collabs_builder = MockQueryBuilder()
    collabs_builder.execute.return_value = MagicMock(data=[{"status": "confirmed"}, {"status": "invited"}], count=2)

    db = MagicMock()
    db.table.return_value = collabs_builder

    asyncio.run(service.check_and_update_work_status(db, WORK_ID))

    assert not [c for c in captured if c[1] == "registry_work_registered"]


# ============================================================
# 3. registry_work_registered (service — _check_auto_register)
# ============================================================


def test_check_auto_register_fires_registered_when_status_flips(monkeypatch):
    """_check_auto_register fires registry_work_registered when all confirmed."""
    from registry import service

    captured = []
    monkeypatch.setattr(
        service,
        "analytics_capture",
        lambda uid, event, props=None: captured.append((uid, event, dict(props or {}))),
    )

    collabs_builder = MockQueryBuilder()
    collabs_builder.execute.return_value = MagicMock(data=[{"status": "confirmed"}], count=1)

    update_builder = MockQueryBuilder()
    update_builder.execute.return_value = MagicMock(data=[{"id": WORK_ID}], count=1)

    owner_builder = MockQueryBuilder()
    owner_builder.execute.return_value = MagicMock(data={"user_id": "owner-u2"}, count=1)

    call_log = []

    def table_side_effect(name):
        call_log.append(name)
        if name == "registry_collaborators":
            return collabs_builder
        if name == "works_registry":
            count = sum(1 for n in call_log if n == "works_registry")
            return update_builder if count == 1 else owner_builder
        return MockQueryBuilder()

    db = MagicMock()
    db.table.side_effect = table_side_effect

    asyncio.run(service._check_auto_register(db, WORK_ID))

    registered = [c for c in captured if c[1] == "registry_work_registered"]
    assert len(registered) == 1, f"expected registry_work_registered, got {captured}"
    assert registered[0][0] == "owner-u2"
    assert registered[0][2]["tool"] == "registry"
    assert registered[0][2]["work_id"] == WORK_ID


# ============================================================
# 4. Module-scope import wiring
# ============================================================


def test_service_imports_analytics_capture_at_module_scope():
    """Sanity: analytics_capture must be a module-level symbol so monkeypatch works."""
    from registry import service

    assert hasattr(service, "analytics_capture")
    assert hasattr(service, "logger")
