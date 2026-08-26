"""Boards filed by workspace (`boards.org_id`).

Boards carry TWO org-valued columns (post 20260818000001_boards_to_orgs), and
every assertion below depends on keeping them apart:

    boards.team_id -> organizations(id)  SHARING: an org-owned board, visible
                                         to the org's live seats. NULL = personal.
    boards.org_id  -> organizations(id)  FILING: which workspace a PERSONAL
                                         board is filed under.

`org_id` is a filing label, not a sharing grant — a personal board filed under
an org stays visible to its owner alone; sharing is team_id's job.
"""

from unittest.mock import MagicMock

import pytest

from artist_access import Scope
from boards import service
from tests.conftest import TEST_USER_ID, MockQueryBuilder

ORG_A = "20000000-0000-0000-0000-00000000000a"
ARTIST = "aaaaaaaa-0000-0000-0000-000000000001"
BOARD = "bbbbbbbb-0000-0000-0000-000000000001"


class _Recorder:
    def __init__(self, *, boards=None, artists=None):
        self.boards = boards if boards is not None else []
        self.artists = artists if artists is not None else []
        self.board_filters = []
        self.inserted = None

    def __call__(self, name):
        b = MockQueryBuilder()
        if name == "boards":
            b.execute.return_value = MagicMock(data=self.boards, count=len(self.boards))

            def _insert(payload):
                self.inserted = payload
                # The insert's own execute() must return the created row, or
                # ensure_personal_board takes its lost-the-race re-read branch.
                b.execute.return_value = MagicMock(data=[{"id": BOARD, **payload}], count=1)
                return b

            b.insert.side_effect = _insert
            for method in ("eq", "is_", "in_"):
                original = getattr(b, method)

                def _capture(*args, _m=method, _o=original, **kwargs):
                    self.board_filters.append((_m, args))
                    return _o(*args, **kwargs)

                setattr(b, method, _capture)
        elif name == "artists":
            b.execute.return_value = MagicMock(data=self.artists, count=len(self.artists))
        return b


def _db(rec):
    db = MagicMock()
    db.table.side_effect = rec
    return db


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def test_personal_scope_lists_only_unfiled_boards():
    rec = _Recorder(boards=[{"id": BOARD}])
    service._personal_board_ids(_db(rec), TEST_USER_ID, Scope.personal())

    assert ("is_", ("team_id", "null")) in rec.board_filters  # not a legacy team board
    assert ("is_", ("org_id", "null")) in rec.board_filters  # filed under Personal


def test_org_scope_lists_only_that_orgs_boards():
    rec = _Recorder(boards=[{"id": BOARD}])
    service._personal_board_ids(_db(rec), TEST_USER_ID, Scope.org(ORG_A))

    assert ("eq", ("org_id", ORG_A)) in rec.board_filters


def test_unscoped_does_not_filter_by_org():
    """Rollback path: with the flag off, every board the caller owns is listed,
    exactly as before org_id existed."""
    rec = _Recorder(boards=[{"id": BOARD}])
    service._personal_board_ids(_db(rec), TEST_USER_ID, Scope.unscoped())

    assert not [f for f in rec.board_filters if f[1] and f[1][0] == "org_id"]


def test_no_scope_argument_behaves_as_unscoped():
    rec = _Recorder(boards=[{"id": BOARD}])
    service._personal_board_ids(_db(rec), TEST_USER_ID)

    assert not [f for f in rec.board_filters if f[1] and f[1][0] == "org_id"]


# ---------------------------------------------------------------------------
# ensure_personal_board — where org_id comes from
# ---------------------------------------------------------------------------


def test_artist_board_derives_org_from_the_artist_not_the_scope():
    """THE bug this guards against. ensure_personal_board is keyed
    (owner_id, artist_id), so if org_id came from the ambient scope the SAME
    artist would get one board when opened from Personal and another from the
    org — splitting one artist's tasks across two half-empty boards."""
    rec = _Recorder(boards=[], artists=[{"name": "Nova", "team_id": ORG_A}])
    # Opened while working as PERSONAL, on an artist the ORG owns.
    service.ensure_personal_board(_db(rec), TEST_USER_ID, ARTIST, Scope.personal())

    assert rec.inserted["org_id"] == ORG_A, "org_id must follow the artist, not the scope"
    assert rec.inserted["artist_id"] == ARTIST


def test_artist_board_lookup_does_not_filter_on_org():
    """Same reason: the FIND half must also ignore the scope, or the lookup
    misses the existing board and creates a duplicate."""
    rec = _Recorder(boards=[{"id": BOARD}], artists=[])
    service.ensure_personal_board(_db(rec), TEST_USER_ID, ARTIST, Scope.org(ORG_A))

    assert ("eq", ("artist_id", ARTIST)) in rec.board_filters
    assert not [f for f in rec.board_filters if f[1] and f[1][0] == "org_id"]


def test_artistless_board_is_one_per_workspace():
    """The unscoped "Personal" board has no artist to derive from, so it takes
    the scope — each workspace gets its own."""
    rec = _Recorder(boards=[], artists=[])
    service.ensure_personal_board(_db(rec), TEST_USER_ID, None, Scope.org(ORG_A))

    assert ("eq", ("org_id", ORG_A)) in rec.board_filters  # looked for the org's board
    assert rec.inserted["org_id"] == ORG_A
    assert rec.inserted["artist_id"] is None


def test_artistless_board_in_personal_scope_is_null_filed():
    rec = _Recorder(boards=[], artists=[])
    service.ensure_personal_board(_db(rec), TEST_USER_ID, None, Scope.personal())

    assert ("is_", ("org_id", "null")) in rec.board_filters
    assert rec.inserted["org_id"] is None


# ---------------------------------------------------------------------------
# create_board
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_board_files_under_the_active_workspace():
    rec = _Recorder(boards=[{"id": BOARD}], artists=[])
    await service.create_board(_db(rec), TEST_USER_ID, "Q3 plan", scope=Scope.org(ORG_A))

    assert rec.inserted["org_id"] == ORG_A


@pytest.mark.asyncio
async def test_create_board_with_an_artist_follows_the_artist():
    rec = _Recorder(boards=[{"id": BOARD}], artists=[{"team_id": ORG_A}])
    # Created while working as Personal, but on an org-owned artist.
    await service.create_board(_db(rec), TEST_USER_ID, "Nova", artist_id=ARTIST, scope=Scope.personal())

    assert rec.inserted["org_id"] == ORG_A


@pytest.mark.asyncio
async def test_create_board_unscoped_files_nowhere():
    rec = _Recorder(boards=[{"id": BOARD}], artists=[])
    await service.create_board(_db(rec), TEST_USER_ID, "Q3 plan", scope=Scope.unscoped())

    assert rec.inserted["org_id"] is None
