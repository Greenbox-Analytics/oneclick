"""Shared test fixtures for the Msanii backend test suite.

Provides:
- TEST_USER_ID constant for all tests
- MockQueryBuilder: chainable mock mimicking supabase-py query builder
- MockStorageBucket: mock for Supabase storage operations
- mock_supabase fixture: fully mocked Supabase client
- client fixture: FastAPI TestClient with auth override and mocked Supabase
"""

import os

# Pin PostHog off before any module that imports analytics gets a chance to
# initialize it. Belt-and-suspenders against a developer's local .env leaking
# POSTHOG_ENABLED=true into pytest and polluting the shared PostHog project.
os.environ["POSTHOG_ENABLED"] = "false"

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

TEST_USER_ID = "00000000-0000-0000-0000-000000000001"


class MockQueryBuilder:
    """Chainable mock that mimics the supabase-py query builder interface.

    Most filter / modifier methods return ``self`` so calls can be chained
    just like the real client.  ``execute()`` is a :class:`MagicMock` so
    tests can configure ``return_value`` freely.

    ``insert`` is a MagicMock with ``return_value=self`` so that both:
    - ``builder.insert({...}).execute()`` → ``builder.execute()`` (existing pattern)
    - ``builder.insert.return_value.execute.side_effect = ...`` (new test pattern)
    work identically, since ``insert.return_value`` is ``self``.

    ``delete`` is a plain MagicMock whose ``return_value`` is an independent
    auto-MagicMock chain (NOT ``self``). This allows webhook tests to configure
    ``builder.delete.return_value.eq.return_value.execute = mock`` independently
    from ``builder.execute``. Tests that need to verify the eq argument should
    use ``builder.delete.return_value.eq.assert_called_with(...)`` rather than
    the old ``b.eq = _capture`` pattern.
    """

    def __init__(self):
        self.execute = MagicMock(return_value=MagicMock(data=[], count=0))
        # insert.return_value = self so that insert({...}).execute() → self.execute()
        # and insert.return_value.execute IS self.execute (configurable by tests)
        self.insert = MagicMock(return_value=self)
        # delete is an independent MagicMock: delete().eq().execute() goes through
        # an auto-MagicMock chain separate from self.execute
        self.delete = MagicMock()

    # --- chainable methods ---------------------------------------------------

    def select(self, *args, **kwargs):
        return self

    def update(self, *args, **kwargs):
        return self

    def upsert(self, *args, **kwargs):
        return self

    def eq(self, *args, **kwargs):
        return self

    def neq(self, *args, **kwargs):
        return self

    def in_(self, *args, **kwargs):
        return self

    def gt(self, *args, **kwargs):
        return self

    def gte(self, *args, **kwargs):
        return self

    def lt(self, *args, **kwargs):
        return self

    def lte(self, *args, **kwargs):
        return self

    def like(self, *args, **kwargs):
        return self

    def ilike(self, *args, **kwargs):
        return self

    def is_(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def range(self, *args, **kwargs):
        return self

    def single(self, *args, **kwargs):
        return self

    def maybe_single(self, *args, **kwargs):
        return self

    def or_(self, *args, **kwargs):
        return self


class MockStorageBucket:
    """Mock for Supabase storage bucket operations."""

    def __init__(self):
        self.upload = MagicMock(return_value={"Key": "mock-key"})
        self.download = MagicMock(return_value=b"mock-file-content")
        self.remove = MagicMock(return_value=None)
        self.create_signed_url = MagicMock(return_value={"signedURL": "https://example.com/signed"})
        self.get_public_url = MagicMock(return_value="https://example.com/public")
        self.list = MagicMock(return_value=[])


_PRO_TIER_ROW = {
    "tier": "pro",
    "max_artists": -1,
    "max_projects": -1,
    "max_boards": -1,
    "max_tasks": -1,
    "max_storage_bytes": -1,
    "max_split_sheets_per_month": -1,
    "max_oneclick_runs_per_month": -1,
    "zoe_enabled": True,
    "oneclick_enabled": True,
    "registry_enabled": True,
    "integrations_allowed": ["google_drive", "slack", "notion"],
    "updated_at": "2026-05-09T00:00:00+00:00",
}

_PRO_SUB_ROW = {
    "id": "s-default",
    "user_id": TEST_USER_ID,
    "tier": "pro",
    "status": "active",
    "stripe_customer_id": None,
    "stripe_subscription_id": None,
    "stripe_price_id": None,
    "current_period_start": None,
    "current_period_end": None,
    "cancel_at_period_end": False,
    "canceled_at": None,
    "created_at": "2026-05-01T00:00:00+00:00",
    "updated_at": "2026-05-01T00:00:00+00:00",
}

_DEFAULT_USAGE_ROW = {
    "user_id": TEST_USER_ID,
    "total_storage_bytes": 0,
    "split_sheets_this_period": 0,
    "zoe_queries_this_period": 0,
    "oneclick_runs_this_period": 0,
    "period_start": "2026-05-09T00:00:00+00:00",
    "period_end": "2099-05-09T00:00:00+00:00",
    "updated_at": "2026-05-09T00:00:00+00:00",
}

# Includes 'profiles' because EntitlementsService.get_for_user now calls
# is_db_admin(supabase, user_id) which reads from the profiles table — tests
# that route subscription-related lookups through this set should also include
# profiles so the admin short-circuit lookup doesn't fall through to whatever
# domain builder the test installed.
_SUBSCRIPTION_TABLES = frozenset({"subscriptions", "tier_entitlements", "tier_overrides", "usage_counters", "profiles"})


def _default_table_side_effect(name):
    """Default table mock: returns Pro data for subscription tables, empty for all others.

    Tests that need specific data for subscription tables should override
    mock_supabase.table.side_effect after receiving the fixture.
    """
    b = MockQueryBuilder()
    if name == "subscriptions":
        b.execute.return_value = MagicMock(data=[_PRO_SUB_ROW], count=1)
    elif name == "tier_entitlements":
        b.execute.return_value = MagicMock(data=[_PRO_TIER_ROW], count=1)
    elif name == "tier_overrides":
        b.execute.return_value = MagicMock(data=[], count=0)
    elif name == "usage_counters":
        b.execute.return_value = MagicMock(data=[_DEFAULT_USAGE_ROW], count=1)
    elif name == "profiles":
        # Default to non-admin so EntitlementsService.is_db_admin short-circuit
        # is bypassed and tests see normal Pro-tier entitlements (via the
        # subscriptions/tier_entitlements rows above) rather than the admin
        # implicit-Pro shape. Tests that specifically need admin behaviour
        # should override profiles to {"is_admin": True}.
        b.execute.return_value = MagicMock(data=[], count=0)
    return b


@pytest.fixture(autouse=True)
def _disable_paywall_bypass_by_default(monkeypatch):
    """Ensure BYPASS_PAYWALLS is unset for every test by default.

    Background: main.py calls load_dotenv() at import time, which seeds
    os.environ from the developer's local .env. Any dev who has
    BYPASS_PAYWALLS=true locally (to demo full-Pro UX) would otherwise see
    ~16 gating/entitlement tests fail with assertions like `assert -1 == 3`
    (expected Free cap of 3, got Pro cap of -1).

    This fixture clears the var per-test so tests deterministically exercise
    the gated paths regardless of the dev's local config. Tests that
    intentionally exercise the bypass path (see
    test_entitlements_service.py::TestBypassPaywallsBehavior) monkeypatch
    BYPASS_PAYWALLS back to "true" themselves — that still works because
    monkeypatch.setenv overrides the delete this fixture performs.
    """
    monkeypatch.delenv("BYPASS_PAYWALLS", raising=False)


@pytest.fixture()
def mock_supabase():
    """Return a MagicMock Supabase client with `.table()` and `.storage.from_()` wired up.

    Subscription/entitlements tables return Pro-tier data by default so subscription
    gates pass in tests that aren't specifically testing gate behaviour.
    Tests that override ``mock_supabase.table.side_effect`` are responsible for
    providing subscription data themselves (or using the monkeypatch approach to
    patch ``enforcement._service`` directly).
    """
    mock = MagicMock()
    mock.table.side_effect = _default_table_side_effect
    mock.storage.from_.return_value = MockStorageBucket()
    return mock


TEST_USER_EMAIL = "test@example.com"


def grant_owner_access():
    """Context manager that patches registry.service.get_work_access to return an
    owner-level WorkAccess (all_visible).

    Read endpoints (stakes/licenses/agreements/works/collaborators) now gate reads
    through get_work_access, which issues several table lookups in a specific order.
    Endpoint-contract tests that use a single shared MockQueryBuilder for every table
    can't satisfy that resolver, so they patch it to an authorized result here.
    Access resolution itself is covered by tests/test_registry_access.py and
    tests/test_registry_read_filtering.py.
    """
    from unittest.mock import AsyncMock, patch

    from registry import service
    from registry.access import WorkAccess

    wa = WorkAccess(work_role="owner", can_see_full_ownership=True)
    wa._all_visible = True
    return patch.object(service, "get_work_access", AsyncMock(return_value=wa))


@pytest.fixture()
def client(mock_supabase):
    """FastAPI TestClient with Supabase mocked and auth overridden.

    * ``main.get_supabase_client`` always returns *mock_supabase*.
    * ``main.supabase`` is replaced by *mock_supabase*.
    * ``get_current_user_id`` dependency yields ``TEST_USER_ID``.
    * ``get_current_user_email`` dependency yields ``TEST_USER_EMAIL``.
    * ``subscriptions.deps._entitlements_service`` singleton is reset so the
      EntitlementsService is re-created with the current mock on each test.
    """
    import main
    import subscriptions.deps as _sub_deps
    from auth import get_current_user_email, get_current_user_id

    original_get_supabase = main.get_supabase_client
    original_supabase = main.supabase
    original_ent_service = _sub_deps._entitlements_service

    main.get_supabase_client = lambda: mock_supabase
    main.supabase = mock_supabase
    # Reset the singleton so it is re-built with the current mock_supabase
    _sub_deps._entitlements_service = None

    async def _override_user_id():
        return TEST_USER_ID

    async def _override_user_email():
        return TEST_USER_EMAIL

    main.app.dependency_overrides[get_current_user_id] = _override_user_id
    main.app.dependency_overrides[get_current_user_email] = _override_user_email

    with TestClient(main.app) as tc:
        yield tc

    # Restore originals
    main.get_supabase_client = original_get_supabase
    main.supabase = original_supabase
    _sub_deps._entitlements_service = original_ent_service
    main.app.dependency_overrides.clear()
