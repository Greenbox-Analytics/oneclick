"""The personal -> team artist transfer (orgs.artists, Team-Owned Artists Task 5).

Mock-based, in the same posture as tests/test_org_projects.py: authz is patched
at the shared `orgs.authz` module so both the service under test and its callers
see the same answer.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from orgs import artists as org_artists
from orgs import authz
from tests.conftest import MockQueryBuilder

U1 = "00000000-0000-0000-0000-000000000001"
ORG_ID = "20000000-0000-0000-0000-000000000001"
ARTIST_ID = "30000000-0000-0000-0000-000000000001"
GB = 2**30


def _db(*, owner_id=U1, team_id=None, exists=True, archived_at=None, kind=None, covered_by=None):
    """Shared by TestTransferArtist and TestTransferStorageGuard below.
    `kind`/`covered_by` default to None -> `.get("kind") == "self_serve"` is
    False, so callers that never pass them exercise the SAME
    enterprise/no-op path as before Task 12 touched this fixture."""
    captured: dict = {}

    def _side(name):
        b = MockQueryBuilder()
        if name == "artists":
            rows = [{"id": ARTIST_ID, "user_id": owner_id, "team_id": team_id}] if exists else []
            b.execute.return_value = MagicMock(data=rows, count=len(rows))

            def _update(payload):
                captured["payload"] = payload
                return b

            b.update = _update
        elif name == "organizations":
            b.execute.return_value = MagicMock(
                data=[{"id": ORG_ID, "archived_at": archived_at, "kind": kind, "covered_by": covered_by}],
                count=1,
            )
        return b

    db = MagicMock()
    db.table.side_effect = _side
    return db, captured


class TestTransferArtist:
    """personal -> team, one way. The reverse has a genuinely nasty case — an
    artist pulled out of a team whose credits paid for its files — so v1 does
    not offer it; support moves it back by hand if it is ever needed."""

    _db = staticmethod(_db)

    async def test_non_member_of_destination_404s(self, monkeypatch):
        """Membership is checked FIRST, so a stranger probing artist ids learns
        nothing beyond the 404 a bad org_id already gives them."""
        monkeypatch.setattr(authz, "is_org_member", lambda *a: False)
        db, _ = self._db()

        with pytest.raises(HTTPException) as exc_info:
            await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert exc_info.value.status_code == 404

    async def test_missing_artist_404s(self, monkeypatch):
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, _ = self._db(exists=False)

        with pytest.raises(HTTPException) as exc_info:
            await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert exc_info.value.status_code == 404

    async def test_non_creator_gets_404_not_403(self, monkeypatch):
        """No existence oracle: someone else's personal artist answers with the
        SAME 404 as a nonexistent one — a 403 would confirm the artist exists."""
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, _ = self._db(owner_id="someone-else")

        with pytest.raises(HTTPException) as exc_info:
            await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Artist not found"

    async def test_non_creator_team_owned_gets_404_not_409(self, monkeypatch):
        """No existence oracle: a 409 on someone ELSE's team-owned artist would
        tell any org member 'this artist exists and is team-owned'. Only the
        creator gets the 409."""
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, _ = self._db(owner_id="someone-else", team_id="0rg00000-0000-0000-0000-0000000000ff")

        with pytest.raises(HTTPException) as exc_info:
            await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Artist not found"

    async def test_creator_already_team_owned_is_409(self, monkeypatch):
        """The helpful 'already transferred' signal is reserved for the
        artist's creator."""
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, _ = self._db(team_id="0rg00000-0000-0000-0000-0000000000ff")

        with pytest.raises(org_artists.ArtistAlreadyTeamOwnedError):
            await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

    async def test_archived_destination_org_is_409(self, monkeypatch):
        """FIX 7: can_access_artist denies on archived_at, so transferring into
        an archived org would permanently lock the creator out of their own
        artist — refused up front with a clear message."""
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, captured = self._db(archived_at="2026-01-01T00:00:00+00:00")

        with pytest.raises(HTTPException) as exc_info:
            await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert exc_info.value.status_code == 409
        assert "archived" in exc_info.value.detail.lower()
        assert "payload" not in captured  # nothing written

    async def test_transfer_stamps_and_recomputes_both_totals(self, monkeypatch):
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, captured = self._db()

        await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert captured["payload"]["team_id"] == ORG_ID
        assert captured["payload"]["transferred_by"] == U1
        assert captured["payload"]["transferred_at"]
        rpcs = [c.args[0] for c in db.rpc.call_args_list if c.args]
        assert "recalc_user_storage" in rpcs, "the ex-owner's total must drop"
        assert "recalc_team_storage" in rpcs, "the team's total must rise"

    async def test_a_failed_recalc_does_not_fail_the_transfer(self, monkeypatch):
        """The write is the transfer; the totals are a cache the triggers keep
        live from here on. A stale number must not strand an artist half-moved."""
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, captured = self._db()
        db.rpc.side_effect = RuntimeError("postgres said no")

        await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert captured["payload"]["team_id"] == ORG_ID


class TestTransferStorageGuard:
    """Task 12, spec §5 byte inlet #2: a transfer into a self_serve org must
    not let a team route around the upload gate by moving an artist's files in
    wholesale instead of uploading them one at a time."""

    _db = staticmethod(_db)

    async def test_self_serve_transfer_over_pool_is_409_with_both_numbers(self, monkeypatch):
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, captured = self._db(kind="self_serve", covered_by=U1)
        monkeypatch.setattr(org_artists.storage_guard, "artist_subtree_bytes", lambda *a: 11 * GB)
        monkeypatch.setattr(org_artists.storage_guard, "pool_state", lambda *a: (0, 10 * GB, False))

        with pytest.raises(HTTPException) as exc_info:
            await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert exc_info.value.status_code == 409
        assert "11.0 GB" in exc_info.value.detail
        assert "10.0 GB" in exc_info.value.detail
        assert "payload" not in captured  # nothing written

    async def test_self_serve_transfer_within_pool_proceeds(self, monkeypatch):
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, captured = self._db(kind="self_serve", covered_by=U1)
        monkeypatch.setattr(org_artists.storage_guard, "artist_subtree_bytes", lambda *a: 4 * GB)
        monkeypatch.setattr(org_artists.storage_guard, "pool_state", lambda *a: (5 * GB, 10 * GB, False))

        await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert captured["payload"]["team_id"] == ORG_ID

    async def test_self_serve_transfer_landing_exactly_on_the_pool_boundary_proceeds(self, monkeypatch):
        """used + subtree == pool is a fit, not an overage — same inclusive
        `<=` boundary as upload_allowed itself (the guard delegates to it)."""
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, captured = self._db(kind="self_serve", covered_by=U1)
        monkeypatch.setattr(org_artists.storage_guard, "artist_subtree_bytes", lambda *a: 4 * GB)
        monkeypatch.setattr(org_artists.storage_guard, "pool_state", lambda *a: (6 * GB, 10 * GB, False))  # 6 + 4 == 10

        await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert captured["payload"]["team_id"] == ORG_ID

    async def test_self_serve_transfer_pro_like_owner_bypasses_the_pool(self, monkeypatch):
        """A pro-like covering owner is allowed regardless of the pool numbers
        — PAYG absorbs it (mirrors upload_allowed / reactivation_allowed)."""
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, captured = self._db(kind="self_serve", covered_by=U1)
        monkeypatch.setattr(org_artists.storage_guard, "artist_subtree_bytes", lambda *a: 200 * GB)
        monkeypatch.setattr(org_artists.storage_guard, "pool_state", lambda *a: (0, 100 * GB, True))

        await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert captured["payload"]["team_id"] == ORG_ID

    async def test_enterprise_transfer_never_touches_the_pool(self, monkeypatch):
        """Enterprise targets are untouched: the guard doesn't even ask —
        no artist_subtree_bytes call, no pool_state call, no tier_entitlements
        read (pool_state is the only thing in this module that reads it)."""
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, captured = self._db(kind="enterprise", covered_by=None)

        def _boom(*a):
            raise AssertionError("enterprise transfer must not consult the pool")

        monkeypatch.setattr(org_artists.storage_guard, "artist_subtree_bytes", _boom)
        monkeypatch.setattr(org_artists.storage_guard, "pool_state", _boom)

        await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert captured["payload"]["team_id"] == ORG_ID
