"""Shared test fixtures for the Msanii backend test suite.

Provides:
- TEST_USER_ID constant for all tests
- MockQueryBuilder: chainable mock mimicking supabase-py query builder
- MockStorageBucket: mock for Supabase storage operations
- mock_supabase fixture: fully mocked Supabase client
- client fixture: FastAPI TestClient with auth override and mocked Supabase
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

TEST_USER_ID = "00000000-0000-0000-0000-000000000001"


class MockQueryBuilder:
    """Chainable mock that mimics the supabase-py query builder interface.

    Every filter / modifier method returns ``self`` so calls can be chained
    just like the real client.  ``execute()`` is a :class:`MagicMock` so
    tests can configure ``return_value`` freely.
    """

    def __init__(self):
        self.execute = MagicMock(return_value=MagicMock(data=[], count=0))

    # --- chainable methods ---------------------------------------------------

    def select(self, *args, **kwargs):
        return self

    def insert(self, *args, **kwargs):
        return self

    def update(self, *args, **kwargs):
        return self

    def delete(self, *args, **kwargs):
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
        self.list = MagicMock(return_value=[])


@pytest.fixture()
def mock_supabase():
    """Return a MagicMock Supabase client with `.table()` and `.storage.from_()` wired up."""
    mock = MagicMock()
    mock.table.side_effect = lambda name: MockQueryBuilder()
    mock.storage.from_.return_value = MockStorageBucket()
    return mock


@pytest.fixture()
def client(mock_supabase):
    """FastAPI TestClient with Supabase mocked and auth overridden.

    * ``main.get_supabase_client`` always returns *mock_supabase*.
    * ``main.supabase`` is replaced by *mock_supabase*.
    * ``get_current_user_id`` dependency yields ``TEST_USER_ID``.
    """
    import main
    from auth import get_current_user_id

    original_get_supabase = main.get_supabase_client
    original_supabase = main.supabase

    main.get_supabase_client = lambda: mock_supabase
    main.supabase = mock_supabase

    async def _override_user_id():
        return TEST_USER_ID

    main.app.dependency_overrides[get_current_user_id] = _override_user_id

    with TestClient(main.app) as tc:
        yield tc

    # Restore originals
    main.get_supabase_client = original_get_supabase
    main.supabase = original_supabase
    main.app.dependency_overrides.clear()
