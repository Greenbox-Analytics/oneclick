"""The artist-ownership layer in `get_work_access`.

docs/licensing.md describes THREE access layers that OR together — artist
ownership, project membership, work-only collaboration — but this resolver only
implemented the last two. That mattered once org works became listable: a
colleague could see the work row and then open an empty detail page, because
every sub-read (stakes, licences, agreements, files, audio) goes through
`WorkAccess`.

The layer is not a new rule. `works_team_artist` (20260803000002) is a `FOR ALL`
policy calling `can_access_artist`, so RLS has always granted org members full
read/write on the work row — the service-role path just never asked, which is
why direct PostgREST reads and API reads disagreed.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from registry import access

ORG = "20000000-0000-0000-0000-00000000000a"
ARTIST = "aaaaaaaa-0000-0000-0000-000000000001"
CREATOR = "11111111-0000-0000-0000-000000000001"
COLLEAGUE = "11111111-0000-0000-0000-000000000002"
OUTSIDER = "11111111-0000-0000-0000-000000000003"


def _db(*, seats, live_orgs, artists):
    """Mock db: a work on a team-owned artist, plus the org/seat rows
    `artist_access.live_org_ids` reads."""
    db = MagicMock()

    def table(name):
        q = MagicMock()
        rows = {
            "works_registry": [{"id": "w1", "user_id": CREATOR, "project_id": "p1", "artist_id": ARTIST}],
            "project_members": [],
            "registry_collaborators": [],
            "ownership_stakes": [],
            "registry_access_grants": [],
            "org_members": seats,
            "organizations": live_orgs,
            "artists": artists,
        }[name]
        q.select.return_value = q
        for method in ("eq", "in_", "neq", "is_", "or_", "order", "limit"):
            getattr(q, method).return_value = q
        q.execute.return_value = MagicMock(data=rows)
        single = MagicMock()
        single.execute.return_value = MagicMock(data=(rows[0] if rows else None))
        q.single.return_value = single
        q.maybe_single.return_value = single
        return q

    db.table.side_effect = table
    return db


@pytest.fixture
def licensing_on(monkeypatch):
    monkeypatch.setenv("LICENSING_ENABLED", "true")


def test_org_colleague_reaches_a_work_on_the_orgs_artist(licensing_on):
    """The whole point: a teammate who is neither the creator, nor a project
    member, nor an invited collaborator still sees the work — because the org
    owns the artist it hangs off."""
    db = _db(seats=[{"org_id": ORG}], live_orgs=[{"id": ORG}], artists=[{"id": ARTIST}])
    wa = asyncio.run(access.get_work_access(db, COLLEAGUE, "w1"))

    assert wa.can_view
    # `elevated` — matching the FOR ALL policy. Anything less would let the
    # colleague list the work and then refuse its splits.
    assert wa.elevated and wa.can_edit and wa.can_manage and wa.can_delete
    assert wa.all_visible() and wa.can_see_full_ownership


def test_non_member_still_gets_nothing(licensing_on):
    """Widening the resolver must not become removing the gate."""
    db = _db(seats=[], live_orgs=[], artists=[])
    wa = asyncio.run(access.get_work_access(db, OUTSIDER, "w1"))

    assert wa.work_role == "none"
    assert not wa.can_view


def test_archived_or_lapsed_org_confers_nothing(licensing_on):
    """live_org_ids denies an archived/'lapsed' org, so the seat alone is not
    enough — modelled the way test_artist_access.py does it, with the dead org
    absent from the organizations read."""
    db = _db(seats=[{"org_id": ORG}], live_orgs=[], artists=[])
    wa = asyncio.run(access.get_work_access(db, COLLEAGUE, "w1"))

    assert not wa.can_view


def test_creator_is_still_the_owner_not_merely_elevated(licensing_on):
    """The owner branch must win: `work_role == "owner"` drives copy and the
    collaborator-management affordances, so a creator must not be demoted to
    'admin' just because the artist layer would also have granted them."""
    db = _db(seats=[{"org_id": ORG}], live_orgs=[{"id": ORG}], artists=[{"id": ARTIST}])
    wa = asyncio.run(access.get_work_access(db, CREATOR, "w1"))

    assert wa.work_role == "owner"


def test_layer_is_inert_when_licensing_is_off(monkeypatch):
    """True rollback: with licensing off, works are creator+project scoped
    exactly as they were before orgs existed."""
    monkeypatch.delenv("LICENSING_ENABLED", raising=False)
    db = _db(seats=[{"org_id": ORG}], live_orgs=[{"id": ORG}], artists=[{"id": ARTIST}])
    wa = asyncio.run(access.get_work_access(db, COLLEAGUE, "w1"))

    assert not wa.can_view
