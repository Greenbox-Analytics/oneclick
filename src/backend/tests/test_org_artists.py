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


class TestTransferArtist:
    """personal -> team, one way. The reverse has a genuinely nasty case — an
    artist pulled out of a team whose credits paid for its files — so v1 does
    not offer it; support moves it back by hand if it is ever needed."""

    def _db(self, *, owner_id=U1, team_id=None, exists=True):
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
            return b

        db = MagicMock()
        db.table.side_effect = _side
        return db, captured

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

    async def test_non_owner_cannot_transfer(self, monkeypatch):
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, _ = self._db(owner_id="someone-else")

        with pytest.raises(HTTPException) as exc_info:
            await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

        assert exc_info.value.status_code == 403

    async def test_already_team_owned_is_409(self, monkeypatch):
        monkeypatch.setattr(authz, "is_org_member", lambda *a: True)
        db, _ = self._db(team_id="0rg00000-0000-0000-0000-0000000000ff")

        with pytest.raises(org_artists.ArtistAlreadyTeamOwnedError):
            await org_artists.transfer_artist_to_team(db, U1, ORG_ID, ARTIST_ID)

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
