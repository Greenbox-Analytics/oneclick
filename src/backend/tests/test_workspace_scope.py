"""Workspace scoping: which workspace a LISTING shows.

`artist_access` holds two function families and the whole design rests on not
merging them (see its module docstring):

    visible_artists / can_access_artist  -> AUTHORIZATION, always the union
    scoped_artists  / scoped_artist_ids  -> PRESENTATION, the union narrowed

These tests pin three properties:

1. `scoped_artists` narrows by CONSTRUCTION — `visible_artists` runs first and
   unconditionally, so a forged or stale `Scope` can only ever return less.
2. `resolve_scope` falls closed, and its two failure modes stay different: an
   explicit foreign org 404s (no existence oracle), a stale STORED context
   quietly degrades to personal rather than 404-ing the caller's whole app.
3. Both flags off => byte-identical to pre-scoping behaviour.

Mocks can't evaluate a PostgREST filter, so — as in test_artist_access.py —
these pin the SHAPE of the query: which filters get built, with which args.
"""

from unittest.mock import MagicMock

import pytest

import artist_access
from artist_access import Scope
from tests.conftest import TEST_USER_ID, MockQueryBuilder

ARTIST = "aaaaaaaa-0000-0000-0000-000000000001"
ORG_A = "20000000-0000-0000-0000-00000000000a"
ORG_B = "20000000-0000-0000-0000-00000000000b"


class _Recorder:
    """Routes table() by name and records the filters built on `artists`."""

    def __init__(self, seats=None, live_orgs=None, artists=None, profile=None):
        self.seats = seats if seats is not None else []
        self.live_orgs = live_orgs if live_orgs is not None else []
        self.artists = artists if artists is not None else []
        self.profile = profile if profile is not None else []
        self.artist_filters = []

    def __call__(self, name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=self.seats, count=len(self.seats))
        elif name == "organizations":
            b.execute.return_value = MagicMock(data=self.live_orgs, count=len(self.live_orgs))
        elif name == "profiles":
            b.execute.return_value = MagicMock(data=self.profile, count=len(self.profile))
        elif name == "artists":
            b.execute.return_value = MagicMock(data=self.artists, count=len(self.artists))
            for method in ("eq", "neq", "is_", "or_", "in_"):
                original = getattr(b, method)

                def _capture(*args, _m=method, _o=original, **kwargs):
                    self.artist_filters.append((_m, args))
                    return _o(*args, **kwargs)

                setattr(b, method, _capture)
        return b


def _db(recorder):
    db = MagicMock()
    db.table.side_effect = recorder
    return db


@pytest.fixture
def scoping_on(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SCOPING_ENABLED", "true")
    monkeypatch.setenv("LICENSING_ENABLED", "true")


# ---------------------------------------------------------------------------
# scoped_artists — narrowing by construction
# ---------------------------------------------------------------------------


def test_unscoped_is_exactly_visible_artists():
    """The rollback path: Scope.unscoped() must add nothing at all."""
    rec = _Recorder(seats=[{"org_id": ORG_A}], live_orgs=[{"id": ORG_A}], artists=[{"id": ARTIST}])
    artist_access.scoped_artist_ids(_db(rec), TEST_USER_ID, Scope.unscoped())

    assert [f for f in rec.artist_filters if f[0] == "or_"], "union filter missing"
    # No narrowing beyond the union.
    assert not [f for f in rec.artist_filters if f[0] == "eq" and f[1][0] == "team_id"]


def test_personal_scope_narrows_to_personal_artists():
    rec = _Recorder(seats=[{"org_id": ORG_A}], live_orgs=[{"id": ORG_A}], artists=[{"id": ARTIST}])
    artist_access.scoped_artist_ids(_db(rec), TEST_USER_ID, Scope.personal())

    # Union FIRST (the member holds a seat, so visible_artists emits or_) ...
    assert [f for f in rec.artist_filters if f[0] == "or_"]
    # ... then the narrowing, ANDed onto it.
    assert ("is_", ("team_id", "null")) in rec.artist_filters
    assert ("eq", ("user_id", TEST_USER_ID)) in rec.artist_filters


def test_org_scope_narrows_to_that_org_and_keeps_the_union():
    """The load-bearing assertion: the union filter is still applied. Without
    it, `.eq("team_id", X)` alone would WIDEN access to any org's artists,
    making every construction site of a Scope an authorization decision."""
    rec = _Recorder(
        seats=[{"org_id": ORG_A}, {"org_id": ORG_B}],
        live_orgs=[{"id": ORG_A}, {"id": ORG_B}],
        artists=[{"id": ARTIST}],
    )
    artist_access.scoped_artist_ids(_db(rec), TEST_USER_ID, Scope.org(ORG_A))

    ors = [f for f in rec.artist_filters if f[0] == "or_"]
    assert len(ors) == 1, "union predicate was skipped — scope could widen access"
    assert ORG_A in ors[0][1][0] and ORG_B in ors[0][1][0]
    assert ("eq", ("team_id", ORG_A)) in rec.artist_filters


def test_org_scope_for_an_unreachable_org_falls_closed():
    """An archived/lapsed org contributes no id to the union, so visible_artists
    emits the personal-only filter and the narrowing ANDs down to
    `team_id IS NULL AND team_id = <org>` — the empty set, with no sentinel."""
    rec = _Recorder(seats=[{"org_id": ORG_A}], live_orgs=[], artists=[])
    artist_access.scoped_artist_ids(_db(rec), TEST_USER_ID, Scope.org(ORG_A))

    assert ("is_", ("team_id", "null")) in rec.artist_filters
    assert ("eq", ("team_id", ORG_A)) in rec.artist_filters
    assert not [f for f in rec.artist_filters if f[0] == "or_"]


def test_visible_artists_is_never_narrowed_by_scoping():
    """The authorization family must stay the union. If this ever fails,
    transfer, invite-accept and deep links break with it."""
    rec = _Recorder(seats=[{"org_id": ORG_A}], live_orgs=[{"id": ORG_A}], artists=[{"id": ARTIST}])
    artist_access.accessible_artist_ids(_db(rec), TEST_USER_ID)

    assert [f for f in rec.artist_filters if f[0] == "or_"]
    assert not [f for f in rec.artist_filters if f[0] == "eq" and f[1][0] == "team_id"]


# ---------------------------------------------------------------------------
# resolve_scope
# ---------------------------------------------------------------------------


def test_flags_off_resolves_unscoped():
    """Neither flag set: scoping is inert and every list is today's list."""
    rec = _Recorder()
    assert artist_access.resolve_scope(_db(rec), TEST_USER_ID, ORG_A) == Scope.unscoped()


def test_licensing_off_resolves_unscoped_even_with_scoping_on(monkeypatch):
    """The 'frozen but live' state: orgs may already exist and RLS still grants
    their artists, but no switcher renders. Narrowing to personal there would
    hide every transferred artist behind a control the user cannot see."""
    monkeypatch.setenv("WORKSPACE_SCOPING_ENABLED", "true")
    monkeypatch.delenv("LICENSING_ENABLED", raising=False)

    rec = _Recorder()
    assert artist_access.resolve_scope(_db(rec), TEST_USER_ID, ORG_A) == Scope.unscoped()


def test_explicit_personal_token(scoping_on):
    rec = _Recorder()
    assert artist_access.resolve_scope(_db(rec), TEST_USER_ID, "personal") == Scope.personal()


def test_explicit_org_requires_an_active_seat(scoping_on):
    db = _db(_Recorder())
    db.rpc.return_value.execute.return_value = MagicMock(data=True)

    assert artist_access.resolve_scope(db, TEST_USER_ID, ORG_A) == Scope.org(ORG_A)
    db.rpc.assert_called_with("is_org_member", {"p_user_id": TEST_USER_ID, "p_org_id": ORG_A})


def test_explicit_foreign_org_404s_with_no_existence_oracle(scoping_on):
    """A foreign org and a nonexistent one must be indistinguishable — otherwise
    ?scope= becomes a probe for which org ids exist."""
    from fastapi import HTTPException

    db = _db(_Recorder())
    db.rpc.return_value.execute.return_value = MagicMock(data=False)

    with pytest.raises(HTTPException) as exc:
        artist_access.resolve_scope(db, TEST_USER_ID, ORG_B)
    assert exc.value.status_code == 404
    assert exc.value.detail == "Organization not found"


def test_absent_param_falls_back_to_the_stored_context(scoping_on):
    rec = _Recorder(
        profile=[{"billing_context_org_id": ORG_A}],
        seats=[{"org_id": ORG_A}],
        live_orgs=[{"id": ORG_A}],
    )
    assert artist_access.resolve_scope(_db(rec), TEST_USER_ID, None) == Scope.org(ORG_A)


def test_stale_stored_context_degrades_to_personal_not_404(scoping_on):
    """Offboarding a member must not 404 their whole app on the next request."""
    rec = _Recorder(profile=[{"billing_context_org_id": ORG_A}], seats=[], live_orgs=[])
    assert artist_access.resolve_scope(_db(rec), TEST_USER_ID, None) == Scope.personal()


def test_no_stored_context_is_personal(scoping_on):
    rec = _Recorder(profile=[{"billing_context_org_id": None}])
    assert artist_access.resolve_scope(_db(rec), TEST_USER_ID, None) == Scope.personal()


def test_stored_context_uses_the_access_predicate_not_the_billing_one(scoping_on):
    """A PENDING org is personal for BILLING (_resolve_context returns None for
    it) but is still the caller's workspace: live_org_ids only excludes archived
    and 'lapsed'. Driving the view filter off the billing answer would make a
    pending org's entire roster vanish."""
    rec = _Recorder(
        profile=[{"billing_context_org_id": ORG_A}],
        seats=[{"org_id": ORG_A}],
        live_orgs=[{"id": ORG_A, "status": "pending"}],
    )
    assert artist_access.resolve_scope(_db(rec), TEST_USER_ID, None) == Scope.org(ORG_A)
