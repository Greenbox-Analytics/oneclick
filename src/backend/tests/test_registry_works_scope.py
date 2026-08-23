"""Registry works, listed by ARTIST OWNERSHIP and narrowed to the workspace.

`get_works` used to filter on `works_registry.user_id` — the CREATOR column —
which was wrong in both directions, the same way it was for the 21 RLS policies
20260803000002 re-scoped: colleagues could not see each other's works on their
own org's artist, and an offboarded creator kept seeing works on an artist they
no longer reach.

The switch is gated on the scope being ACTIVE, because it is the visible,
one-way half of this feature.
"""

import asyncio
from unittest.mock import MagicMock

import artist_access
from artist_access import Scope
from registry import service
from tests.conftest import TEST_USER_ID, MockQueryBuilder

ORG_A = "20000000-0000-0000-0000-00000000000a"
MY_ARTIST = "aaaaaaaa-0000-0000-0000-000000000001"
ORG_ARTIST = "aaaaaaaa-0000-0000-0000-000000000002"
OTHER_ARTIST = "aaaaaaaa-0000-0000-0000-000000000009"


class _NegationAware(MockQueryBuilder):
    """MockQueryBuilder plus a record of which filters were NEGATED.

    The base mock's `not_` returns `self` (matching supabase-py's
    `.not_.in_(...)` chaining), so a negated filter would otherwise be
    indistinguishable from a positive one — and "exclude these artists" vs
    "only these artists" is the difference between hiding a duplicate and
    showing nothing but duplicates."""

    def __init__(self):
        super().__init__()
        self._negate_next = False
        self.negations: list = []

    @property
    def not_(self):
        self._negate_next = True
        return self


class _Recorder:
    def __init__(self, *, seats=None, live_orgs=None, artists=None, works=None, collabs=None):
        self.seats = seats if seats is not None else []
        self.live_orgs = live_orgs if live_orgs is not None else []
        self.artists = artists if artists is not None else []
        self.works = works if works is not None else []
        self.collabs = collabs if collabs is not None else []
        self.work_filters = []
        self.negations = []

    def __call__(self, name):
        b = _NegationAware() if name == "works_registry" else MockQueryBuilder()
        data = {
            "org_members": self.seats,
            "organizations": self.live_orgs,
            "artists": self.artists,
            "works_registry": self.works,
            "registry_collaborators": self.collabs,
        }[name]
        b.execute.return_value = MagicMock(data=data, count=len(data))
        if name == "works_registry":
            for method in ("eq", "in_", "neq", "order"):
                original = getattr(b, method)

                def _capture(*args, _m=method, _o=original, _b=b, **kwargs):
                    self.work_filters.append((_m, args))
                    if _b._negate_next:
                        self.negations.append((_m, args))
                        _b._negate_next = False
                    return _o(*args, **kwargs)

                setattr(b, method, _capture)
        return b


def _db(rec):
    db = MagicMock()
    db.table.side_effect = rec
    return db


def _run(rec, **kwargs):
    return asyncio.run(service.get_works(_db(rec), TEST_USER_ID, **kwargs))


# ---------------------------------------------------------------------------
# The rollback path
# ---------------------------------------------------------------------------


def test_unscoped_keeps_the_original_creator_filter():
    """Flag off => byte-identical to the pre-feature query. The creator→owner
    switch is one-way in users' eyes, so it must wait for the flag."""
    rec = _Recorder(works=[{"id": "w1"}])
    _run(rec, scope=Scope.unscoped())

    assert ("eq", ("user_id", TEST_USER_ID)) in rec.work_filters
    assert not [f for f in rec.work_filters if f[0] == "in_" and f[1][0] == "artist_id"]


def test_unscoped_still_honours_artist_id():
    rec = _Recorder(works=[{"id": "w1"}])
    _run(rec, artist_id=MY_ARTIST, scope=Scope.unscoped())

    assert ("eq", ("artist_id", MY_ARTIST)) in rec.work_filters


# ---------------------------------------------------------------------------
# Scoped listing
# ---------------------------------------------------------------------------


def test_personal_scope_lists_by_artist_ownership():
    rec = _Recorder(artists=[{"id": MY_ARTIST}], works=[{"id": "w1"}])
    _run(rec, scope=Scope.personal())

    assert ("in_", ("artist_id", [MY_ARTIST])) in rec.work_filters
    # The creator column is no longer the filter.
    assert not [f for f in rec.work_filters if f[0] == "eq" and f[1][0] == "user_id"]


def test_org_scope_lists_every_members_work_on_the_orgs_artists():
    """The org-sharing half: colleague A's work on the org's artist is B's to
    see, because the ORG owns the artist."""
    rec = _Recorder(
        seats=[{"org_id": ORG_A}],
        live_orgs=[{"id": ORG_A}],
        artists=[{"id": ORG_ARTIST}],
        works=[{"id": "w1", "user_id": "someone-else"}],
    )
    _run(rec, scope=Scope.org(ORG_A))

    assert ("in_", ("artist_id", [ORG_ARTIST])) in rec.work_filters


def test_empty_workspace_short_circuits_without_querying_works():
    """No reachable artists => no works. Must not fall through to an unfiltered
    works query."""
    rec = _Recorder(artists=[], works=[{"id": "leak"}])
    assert _run(rec, scope=Scope.personal()) == []
    assert rec.work_filters == []


# ---------------------------------------------------------------------------
# ?artist_id= must narrow, never escape
# ---------------------------------------------------------------------------


def test_artist_id_narrows_within_the_workspace():
    rec = _Recorder(artists=[{"id": MY_ARTIST}, {"id": ORG_ARTIST}], works=[{"id": "w1"}])
    _run(rec, artist_id=MY_ARTIST, scope=Scope.personal())

    assert ("in_", ("artist_id", [MY_ARTIST])) in rec.work_filters


def test_artist_id_outside_the_workspace_returns_nothing():
    """The load-bearing one: `?artist_id=<other workspace's artist>` would be a
    one-parameter bypass of the entire scope if it were taken on trust."""
    rec = _Recorder(artists=[{"id": MY_ARTIST}], works=[{"id": "leak"}])

    assert _run(rec, artist_id=OTHER_ARTIST, scope=Scope.personal()) == []
    assert rec.work_filters == []


# ---------------------------------------------------------------------------
# "Shared with me" is personal-only
# ---------------------------------------------------------------------------


def test_collaborations_are_empty_in_an_org_workspace():
    """A collaboration on the org's own artist already shows by ownership;
    listing it here too would show one row twice on one page."""
    rec = _Recorder(collabs=[{"work_id": "w1"}], works=[{"id": "w1"}])
    out = asyncio.run(service.get_works_as_collaborator(_db(rec), TEST_USER_ID, scope=Scope.org(ORG_A)))

    assert out == []
    assert rec.work_filters == []


def test_collaborations_exclude_org_owned_artists_in_personal_scope():
    """Works on artists an org owns are listed by ownership in that org's
    workspace, so they must not ALSO appear under "Shared with me".

    The exclusion is applied to the QUERY, never to the returned page:
    PostgREST computes count="exact" on the query, so post-filtering after
    paginate_query would make `total` lie and hand back short pages."""
    rec = _Recorder(
        seats=[{"org_id": ORG_A}],
        live_orgs=[{"id": ORG_A}],
        artists=[{"id": ORG_ARTIST}],
        collabs=[{"work_id": "w1"}],
        works=[{"id": "w1"}],
    )
    asyncio.run(service.get_works_as_collaborator(_db(rec), TEST_USER_ID, scope=Scope.personal()))

    # MockQueryBuilder.not_ returns self, so a negated filter records under the
    # same name — this pins that the artist_id exclusion was built with the
    # org's artists, before pagination.
    assert ("in_", ("artist_id", [ORG_ARTIST])) in rec.work_filters
    assert rec.negations == [("in_", ("artist_id", [ORG_ARTIST]))], "exclusion must be NEGATED"


def test_collaborations_unscoped_is_unchanged():
    rec = _Recorder(collabs=[{"work_id": "w1"}], works=[{"id": "w1"}])
    asyncio.run(service.get_works_as_collaborator(_db(rec), TEST_USER_ID, scope=Scope.unscoped()))

    assert ("in_", ("id", ["w1"])) in rec.work_filters
    assert ("neq", ("user_id", TEST_USER_ID)) in rec.work_filters


def test_scope_is_optional_everywhere_it_is_threaded():
    """Callers that predate the scope must keep working (and stay unscoped)."""
    rec = _Recorder(works=[{"id": "w1"}])
    _run(rec)
    assert ("eq", ("user_id", TEST_USER_ID)) in rec.work_filters
    assert artist_access.Scope.unscoped().active is False


# ---------------------------------------------------------------------------
# get_artists_with_teamcards — the Artists page roster
# ---------------------------------------------------------------------------


class _ArtistRecorder(_Recorder):
    """Same fixture data, but records the filters built on `artists` — the
    teamcards roster narrows the artist query itself, not a works query."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.artist_filters: list = []

    def __call__(self, name):
        b = super().__call__(name)
        if name == "artists":
            for method in ("eq", "in_", "is_", "or_"):
                original = getattr(b, method)

                def _capture(*args, _o=original, _m=method, **kw):
                    self.artist_filters.append((_m, args))
                    return _o(*args, **kw)

                setattr(b, method, _capture)
        return b


def test_teamcards_roster_narrows_to_the_org_and_keeps_the_union():
    """scoped_artists composition: the union predicate FIRST (or_), then the
    narrowing ANDed on — a bare eq(team_id) alone would widen to dead orgs."""
    rec = _ArtistRecorder(
        seats=[{"org_id": ORG_A}],
        live_orgs=[{"id": ORG_A}],
        artists=[{"id": ORG_ARTIST, "name": "Nova", "verified": False}],
    )
    asyncio.run(
        service.get_artists_with_teamcards(
            MagicMock(table=MagicMock(side_effect=rec)), TEST_USER_ID, scope=Scope.org(ORG_A)
        )
    )

    assert [f for f in rec.artist_filters if f[0] == "or_"], "union predicate missing"
    assert ("eq", ("team_id", ORG_A)) in rec.artist_filters


def test_teamcards_roster_unscoped_is_the_plain_union():
    rec = _ArtistRecorder(
        seats=[{"org_id": ORG_A}],
        live_orgs=[{"id": ORG_A}],
        artists=[{"id": ORG_ARTIST, "name": "Nova", "verified": False}],
    )
    asyncio.run(
        service.get_artists_with_teamcards(MagicMock(table=MagicMock(side_effect=rec)), TEST_USER_ID, scope=None)
    )

    assert [f for f in rec.artist_filters if f[0] == "or_"]
    assert not [f for f in rec.artist_filters if f[0] == "eq" and f[1][0] == "team_id"]
