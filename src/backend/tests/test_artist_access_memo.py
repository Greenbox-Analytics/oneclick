"""Per-request memoization in artist_access.

The point of these tests is not the speed-up — it's the boundary. This module
is the authorization for every artist-scoped read (the service-role client
bypasses RLS), so the memo must live for exactly one request and not one
instant longer. A cache that outlives the request would keep an offboarded
member's seat alive.
"""

from unittest.mock import MagicMock

import artist_access


class _Chain:
    """Chainable supabase-py stub. Records how many times execute() ran."""

    def __init__(self, registry: dict, name: str, data: list):
        self._registry = registry
        self._name = name
        self._data = data

    def _noop(self, *a, **kw):
        return self

    select = eq = in_ = is_ = neq = or_ = order = limit = _noop

    def execute(self):
        self._registry[self._name] = self._registry.get(self._name, 0) + 1
        return MagicMock(data=self._data)


def _make_db(counts: dict):
    """db whose org_members/organizations/artists reads are all counted."""
    rows = {
        "org_members": [{"org_id": "org-1"}],
        "organizations": [{"id": "org-1"}],
        "artists": [{"id": "artist-1"}],
    }
    db = MagicMock()
    db.table.side_effect = lambda name: _Chain(counts, name, rows.get(name, []))
    return db


class TestLiveOrgIdsMemo:
    def test_repeated_calls_in_one_scope_hit_supabase_once(self):
        counts: dict = {}
        db = _make_db(counts)
        token = artist_access.begin_request_scope()
        try:
            for _ in range(5):
                assert artist_access.live_org_ids(db, "user-1") == ["org-1"]
        finally:
            artist_access.end_request_scope(token)

        # 2 round trips total, not 2 per call — this is the trace the fix targets.
        assert counts["org_members"] == 1
        assert counts["organizations"] == 1

    def test_no_scope_means_no_caching(self):
        """Scripts, the sweep and tests run outside a request. They must not
        accumulate authorization state in a long-lived process."""
        counts: dict = {}
        db = _make_db(counts)
        artist_access.live_org_ids(db, "user-1")
        artist_access.live_org_ids(db, "user-1")
        assert counts["org_members"] == 2

    def test_a_second_request_re_reads(self):
        """The safety property: offboarding a member takes effect on the very
        next request, exactly as before the memo existed."""
        counts: dict = {}
        db = _make_db(counts)
        for _ in range(2):
            token = artist_access.begin_request_scope()
            try:
                artist_access.live_org_ids(db, "user-1")
            finally:
                artist_access.end_request_scope(token)
        assert counts["org_members"] == 2

    def test_different_users_do_not_share_an_entry(self):
        """Two users inside one scope must never collapse into one answer."""
        counts: dict = {}
        db = _make_db(counts)
        token = artist_access.begin_request_scope()
        try:
            artist_access.live_org_ids(db, "user-1")
            artist_access.live_org_ids(db, "user-2")
        finally:
            artist_access.end_request_scope(token)
        assert counts["org_members"] == 2

    def test_scope_is_torn_down_even_when_the_request_raises(self):
        counts: dict = {}
        db = _make_db(counts)
        token = artist_access.begin_request_scope()
        try:
            artist_access.live_org_ids(db, "user-1")
            raise RuntimeError("boom")
        except RuntimeError:
            pass
        finally:
            artist_access.end_request_scope(token)

        # Back outside a scope: uncached again, so nothing leaked.
        artist_access.live_org_ids(db, "user-1")
        assert counts["org_members"] == 2


class TestAccessibleArtistIdsMemo:
    def test_repeated_calls_in_one_scope_hit_supabase_once(self):
        counts: dict = {}
        db = _make_db(counts)
        token = artist_access.begin_request_scope()
        try:
            for _ in range(4):
                assert artist_access.accessible_artist_ids(db, "user-1") == ["artist-1"]
        finally:
            artist_access.end_request_scope(token)

        assert counts["artists"] == 1
        # And its own dependency collapsed too.
        assert counts["org_members"] == 1
