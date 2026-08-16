"""Artist visibility: the backend's mirror of the SQL can_access_artist().

The backend runs on the service-role client, which BYPASSES RLS — so these
helpers ARE the authorization for every artist-scoped endpoint in main.py, and
they must answer exactly what can_access_artist() answers for RLS
(supabase/migrations/20260803000001_team_owned_artists.sql):

    personal: team_id IS NULL AND user_id = me
    team:     team_id = an org where I hold an ACTIVE seat, org not archived

`artists.user_id` is the CREATOR and keeps pointing at them after a transfer,
so the pre-fix `.eq("user_id", me)` was wrong in BOTH directions: it hid team
artists from the colleagues who should see them, and kept showing them to an
offboarded creator who should not.

Mocks can't evaluate a PostgREST filter, so these tests pin the SHAPE of the
query — which filters get built, with which arguments — the same approach
test_orgs_service.test_list_my_orgs_excludes_removed_memberships takes.
"""

from unittest.mock import MagicMock

import artist_access
import main
from tests.conftest import TEST_USER_ID, MockQueryBuilder

ARTIST = "aaaaaaaa-0000-0000-0000-000000000001"
ORG_A = "20000000-0000-0000-0000-00000000000a"
ORG_B = "20000000-0000-0000-0000-00000000000b"
PROJECT = "cccccccc-0000-0000-0000-000000000001"


class _Recorder:
    """Routes table() by name, records the filter calls made on `artists`."""

    def __init__(self, seats=None, live_orgs=None, artists=None, projects=None):
        self.seats = seats if seats is not None else []
        self.live_orgs = live_orgs if live_orgs is not None else []
        self.artists = artists if artists is not None else []
        self.projects = projects if projects is not None else []
        self.artist_filters = []
        self.org_filters = []

    def __call__(self, name):
        b = MockQueryBuilder()
        if name == "org_members":
            b.execute.return_value = MagicMock(data=self.seats, count=len(self.seats))
        elif name == "organizations":
            b.execute.return_value = MagicMock(data=self.live_orgs, count=len(self.live_orgs))
            self._record(b, self.org_filters)
        elif name == "artists":
            b.execute.return_value = MagicMock(data=self.artists, count=len(self.artists))
            self._record(b, self.artist_filters)
        elif name == "projects":
            b.execute.return_value = MagicMock(data=self.projects, count=len(self.projects))
        return b

    @staticmethod
    def _record(builder, sink):
        for method in ("eq", "neq", "is_", "or_", "in_"):
            original = getattr(builder, method)

            def _capture(*args, _m=method, _o=original, **kwargs):
                sink.append((_m, args))
                return _o(*args, **kwargs)

            setattr(builder, method, _capture)


def _install(monkeypatch, recorder):
    db = MagicMock()
    db.table.side_effect = recorder
    monkeypatch.setattr(main, "get_supabase_client", lambda: db)
    return db


# ---------------------------------------------------------------------------
# The visibility rule itself
# ---------------------------------------------------------------------------


def test_no_org_membership_restricts_to_personal_artists(monkeypatch):
    """With no seats, visibility is personal-only — and `team_id IS NULL` is
    load-bearing: without it a creator keeps reaching an artist they already
    handed to a team."""
    rec = _Recorder(seats=[], artists=[{"id": ARTIST}])
    _install(monkeypatch, rec)

    assert main.verify_user_owns_artist(TEST_USER_ID, ARTIST) is True
    assert ("is_", ("team_id", "null")) in rec.artist_filters
    assert ("eq", ("user_id", TEST_USER_ID)) in rec.artist_filters
    assert not [f for f in rec.artist_filters if f[0] == "or_"]


def test_active_seat_widens_visibility_to_that_orgs_artists(monkeypatch):
    """An active seat in a live org adds `team_id IN (...)` alongside the
    personal branch — the colleague-can-see-team-artists half of the fix."""
    rec = _Recorder(
        seats=[{"org_id": ORG_A}, {"org_id": ORG_B}],
        live_orgs=[{"id": ORG_A}, {"id": ORG_B}],
        artists=[{"id": ARTIST}],
    )
    _install(monkeypatch, rec)

    assert main.verify_user_owns_artist(TEST_USER_ID, ARTIST) is True
    ors = [f for f in rec.artist_filters if f[0] == "or_"]
    assert len(ors) == 1
    clause = ors[0][1][0]
    assert f"and(team_id.is.null,user_id.eq.{TEST_USER_ID})" in clause
    assert ORG_A in clause and ORG_B in clause


def test_archived_org_confers_no_visibility(monkeypatch):
    """can_access_artist denies on archived_at, so an archived org must not
    contribute its id — the seat row alone is not enough."""
    rec = _Recorder(seats=[{"org_id": ORG_A}], live_orgs=[], artists=[])
    _install(monkeypatch, rec)

    assert main.verify_user_owns_artist(TEST_USER_ID, ARTIST) is False
    assert ("is_", ("archived_at", "null")) in rec.org_filters
    assert not [f for f in rec.artist_filters if f[0] == "or_"]


def test_lapsed_org_confers_no_visibility(monkeypatch):
    """20260816000002_self_serve_orgs.sql: a self-serve org whose grace period
    ran out goes inert for its whole roster, admins included — same one-rule
    semantics as archived_at, mirrored here since the backend runs
    service-role and the SQL clause alone is decorative for API paths.
    Modeled the same way archived is: the lapsed org is absent from the
    (mocked) DB response, and the query must still ask for the exclusion."""
    rec = _Recorder(seats=[{"org_id": ORG_A}], live_orgs=[], artists=[])
    _install(monkeypatch, rec)

    assert main.verify_user_owns_artist(TEST_USER_ID, ARTIST) is False
    assert ("neq", ("status", "lapsed")) in rec.org_filters


def test_pending_org_stays_visible(monkeypatch):
    """Pre-existing behaviour pinned: only 'lapsed' is excluded.
    'pending'/'suspended' are deliberately still unchecked (documented in
    20260803000001) — a pending org is still being set up, and its members
    should be able to build the roster they are about to pay for."""
    rec = _Recorder(
        seats=[{"org_id": ORG_A}],
        live_orgs=[{"id": ORG_A}],
        artists=[{"id": ARTIST}],
    )
    _install(monkeypatch, rec)

    assert main.verify_user_owns_artist(TEST_USER_ID, ARTIST) is True
    assert ("neq", ("status", "lapsed")) in rec.org_filters


def test_only_active_seats_are_considered(monkeypatch):
    """A suspended/removed seat confers nothing — the org_members read must
    filter status='active' rather than counting any row."""
    captured = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "org_members":
            original = b.eq

            def _capture(field, value):
                captured[field] = value
                return original(field, value)

            b.eq = _capture
        return b

    db = MagicMock()
    db.table.side_effect = _side
    monkeypatch.setattr(main, "get_supabase_client", lambda: db)

    main.verify_user_owns_artist(TEST_USER_ID, ARTIST)
    assert captured.get("status") == "active"
    assert captured.get("user_id") == TEST_USER_ID


# ---------------------------------------------------------------------------
# The helpers that inherit the rule
# ---------------------------------------------------------------------------


def test_get_user_artist_ids_uses_the_same_rule(monkeypatch):
    rec = _Recorder(
        seats=[{"org_id": ORG_A}],
        live_orgs=[{"id": ORG_A}],
        artists=[{"id": ARTIST}, {"id": "aaaaaaaa-0000-0000-0000-000000000002"}],
    )
    _install(monkeypatch, rec)

    ids = main.get_user_artist_ids(TEST_USER_ID)
    assert ids == [ARTIST, "aaaaaaaa-0000-0000-0000-000000000002"]
    assert [f for f in rec.artist_filters if f[0] == "or_"]


def test_verify_user_owns_project_defers_to_artist_access(monkeypatch):
    """A project's reachability is its ARTIST's reachability — resolve the
    artist_id and ask once, rather than fetching every accessible artist id."""
    rec = _Recorder(
        seats=[],
        artists=[{"id": ARTIST}],
        projects=[{"artist_id": ARTIST}],
    )
    _install(monkeypatch, rec)

    assert main.verify_user_owns_project(TEST_USER_ID, PROJECT) is True
    assert ("is_", ("team_id", "null")) in rec.artist_filters


def test_verify_user_owns_project_false_for_unknown_project(monkeypatch):
    rec = _Recorder(seats=[], artists=[{"id": ARTIST}], projects=[])
    _install(monkeypatch, rec)

    assert main.verify_user_owns_project(TEST_USER_ID, PROJECT) is False
    # Never reached the artists table — no project, nothing to authorize.
    assert rec.artist_filters == []


def test_verify_user_owns_project_false_when_artist_not_accessible(monkeypatch):
    rec = _Recorder(seats=[], artists=[], projects=[{"artist_id": ARTIST}])
    _install(monkeypatch, rec)

    assert main.verify_user_owns_project(TEST_USER_ID, PROJECT) is False


# ---------------------------------------------------------------------------
# The same rule, reached from the other modules that gate on artists.
# main.py was swept first; registry/ and boards/ held their own copies of the
# creator-keyed query and are the reason artist_access.py is a shared module
# rather than a main.py helper.
# ---------------------------------------------------------------------------


def test_registry_artist_roster_includes_team_artists(monkeypatch):
    """GET /registry/artists/with-teamcards renders the Artist Profiles page —
    a creator-keyed roster there hides the org's artists from its own members."""
    from registry import service as registry_service

    rec = _Recorder(
        seats=[{"org_id": ORG_A}],
        live_orgs=[{"id": ORG_A}],
        artists=[{"id": ARTIST, "team_id": ORG_A, "linked_user_id": None}],
    )
    db = MagicMock()
    db.table.side_effect = rec

    import asyncio

    out = asyncio.run(registry_service.get_artists_with_teamcards(db, TEST_USER_ID))
    assert [a["id"] for a in out] == [ARTIST]
    assert [f for f in rec.artist_filters if f[0] == "or_"], "team branch never applied"


def test_create_work_is_allowed_on_a_team_artist(monkeypatch):
    """A member creating a work on their org's artist must not be refused."""
    from registry import service as registry_service

    rec = _Recorder(seats=[{"org_id": ORG_A}], live_orgs=[{"id": ORG_A}], artists=[{"id": ARTIST}])
    db = MagicMock()
    db.table.side_effect = rec

    import asyncio

    asyncio.run(registry_service.create_work(db, TEST_USER_ID, {"artist_id": ARTIST, "title": "T"}))
    assert [f for f in rec.artist_filters if f[0] == "or_"]


def test_create_work_refused_when_artist_unreachable():
    """The gate still denies — widening visibility must not become no gate."""
    from registry import service as registry_service

    rec = _Recorder(seats=[], artists=[])
    db = MagicMock()
    db.table.side_effect = rec

    import asyncio

    assert asyncio.run(registry_service.create_work(db, TEST_USER_ID, {"artist_id": ARTIST})) is None


def test_boards_artist_filter_uses_the_shared_rule():
    from boards import service as boards_service

    rec = _Recorder(seats=[{"org_id": ORG_A}], live_orgs=[{"id": ORG_A}], artists=[{"id": ARTIST}])
    db = MagicMock()
    db.table.side_effect = rec

    assert boards_service._owned_artist_ids(db, TEST_USER_ID, [ARTIST]) == [ARTIST]
    assert [f for f in rec.artist_filters if f[0] == "or_"]


def test_accessible_artist_ids_is_empty_without_reachable_artists():
    rec = _Recorder(seats=[], artists=[])
    db = MagicMock()
    db.table.side_effect = rec

    assert artist_access.accessible_artist_ids(db, TEST_USER_ID) == []


def test_can_access_artist_short_circuits_on_missing_id():
    """A null artist_id must be a denial, not an unfiltered query."""
    rec = _Recorder(seats=[], artists=[{"id": ARTIST}])
    db = MagicMock()
    db.table.side_effect = rec

    assert artist_access.can_access_artist(db, TEST_USER_ID, None) is False
    assert rec.artist_filters == []
