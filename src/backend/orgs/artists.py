"""The personal -> team artist transfer.

Mirrors orgs/projects.py: authz first, then no-existence-oracle 404s, then the
work. Artists are the ownership root — ten tables cascade off one — so anything
that changes `artists.team_id` lives here rather than being scattered. The
`artists_lock_team_id` trigger (20260803000001) enforces that: any client
holding a user JWT is refused, and this module runs on the service-role client
where auth.uid() is NULL.
"""

from datetime import UTC, datetime

from fastapi import HTTPException
from supabase import Client

from orgs import authz


class ArtistAlreadyTeamOwnedError(Exception):
    """The artist already belongs to a team. Transfer is one-way in v1: moving
    an artist OUT of a team whose credits paid for its files is a support
    decision with a refund question attached, not a self-serve button."""


async def transfer_artist_to_team(db: Client, user_id: str, org_id: str, artist_id: str) -> dict:
    """Hand a personal artist to a team, with its whole subtree.

    Nothing is copied: ten tables hang off `artists` by foreign key, so changing
    one column moves the projects, works, files, audio, credentials and boards
    with it. What DOES move explicitly is the two storage totals — the triggers
    keep them live from here on, but the bytes already on disk were counted
    against the wrong side and have to be re-derived once.

    Order: authz (destination seat), then the destination's archived check,
    then creator-ship, then the state check, then the write. Anyone who is NOT
    the artist's creator gets the SAME 404 whether the artist doesn't exist,
    is someone else's personal artist, or is already team-owned — a 403/409
    split there would let any org member probe arbitrary artist ids for
    existence and ownership. The 409 "already belongs to a team" is reserved
    for the creator themselves (the helpful "already transferred" signal).

    An ARCHIVED destination org is refused with a 409: `can_access_artist`
    denies on `archived_at`, so transferring in would permanently lock the
    creator out of their own artist.
    """
    authz.require_member(db, user_id, org_id)

    org_rows = db.table("organizations").select("archived_at").eq("id", org_id).execute().data
    org_row = (org_rows or [None])[0]
    if org_row and org_row.get("archived_at"):
        raise HTTPException(
            status_code=409,
            detail="This organization has been archived — artists can't be moved into it.",
        )

    row = db.table("artists").select("id, user_id, team_id").eq("id", artist_id).execute().data
    artist = (row or [None])[0]
    if not artist or artist.get("user_id") != user_id:
        # Non-creators (and unknown ids) always get the same 404 — no oracle.
        raise HTTPException(status_code=404, detail="Artist not found")
    if artist.get("team_id"):
        raise ArtistAlreadyTeamOwnedError("This artist already belongs to a team")

    updated = (
        db.table("artists")
        .update(
            {
                "team_id": org_id,
                "transferred_at": datetime.now(UTC).isoformat(),
                "transferred_by": user_id,
            }
        )
        .eq("id", artist_id)
        .execute()
    )

    # Re-derive both totals. Best-effort and in this order: the ex-owner's total
    # is the one that would otherwise overstate a real person's usage and
    # wrongly block their next upload.
    for rpc, params in (
        ("recalc_user_storage", {"p_user_id": user_id}),
        ("recalc_team_storage", {"p_org_id": org_id}),
    ):
        try:
            db.rpc(rpc, params).execute()
        except Exception as exc:  # noqa: BLE001 - a stale total must not fail the transfer
            print(f"transfer_artist_to_team: {rpc} failed artist={artist_id}: {exc}")

    return (updated.data or [{}])[0]
