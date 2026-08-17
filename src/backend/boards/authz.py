"""Access-control helpers for boards.

A team board belongs to an ORGANIZATION (`boards.team_id → organizations.id`,
the same edge as `artists.team_id`); a personal board has team_id NULL. The
backend uses the service-role client, which BYPASSES RLS, so these checks are
the real gate — the SQL `can_access_board(board_id, user_id)` in
20260818000001_boards_to_orgs.sql is the RLS mirror of `_can_access` below, and
its liveness half, SQL `is_live_org_member(user_id, org_id)`, mirrors
`artist_access.live_org_ids` here. Keep all of them in step: three definitions
of "live seat" that disagree is exactly the drift this design exists to avoid.

Predicate (spec 2026-08-16 §1):
  personal → owner
  team     → ACTIVE seat in a LIVE org (artist_access.live_org_ids: not archived,
             not lapsed) AND (open board OR owner OR org admin OR listed in board_members)
"""

from fastapi import HTTPException
from supabase import Client

import artist_access
from orgs import authz as org_authz


def get_board(db: Client, board_id: str) -> dict | None:
    """Board row (id, team_id, owner_id, archived, restricted) or None.

    `select("*")`, NOT an explicit column list: the deploy order is new backend
    first, migration second (20260818000001's header), so for the length of
    that window `boards.restricted` does not exist yet and naming it would
    make PostgREST raise 42703 on EVERY board read — personal boards included.
    `*` degrades to "no such key" instead, which `restricted = False` below
    turns into pre-migration semantics (no narrowing exists yet).

    The `setdefault` is load-bearing for `_can_access`: it lets that function
    read `board["restricted"]` STRICTLY, so a dict that came from some other,
    narrower select (e.g. `delete_board`'s `select("id, name, team_id,
    owner_id")`) raises KeyError -> 500 -> fails CLOSED, rather than reading a
    missing key as "not restricted" and silently opening a private board.
    """
    res = db.table("boards").select("*").eq("id", board_id).limit(1).execute()
    rows = res.data or []
    if not rows:
        return None
    board = rows[0]
    board.setdefault("restricted", False)
    return board


def is_live_org_member(db: Client, user_id: str, org_id: str) -> bool:
    """Active seat in a non-archived, non-lapsed org — one definition, shared with artists.

    Deliberately NOT `db.rpc("is_live_org_member", ...)`, even though the
    migration creates exactly that SQL function and `orgs/authz.py` argues for
    the RPC form elsewhere: the new backend goes live BEFORE the migration
    runs, and an RPC to a function that does not exist yet is a 500 on every
    board endpoint for the whole deploy window. `live_org_ids` reads only
    `org_members` + `organizations`, which both already exist. Two round-trips
    is the price of a deployable window; revisit only if boards get hot.
    """
    return org_id in artist_access.live_org_ids(db, user_id)


def _listed(db: Client, board_id: str, user_id: str) -> bool:
    res = db.table("board_members").select("id").eq("board_id", board_id).eq("user_id", user_id).limit(1).execute()
    return bool(res.data)


def _can_access(db: Client, user_id: str, board: dict) -> bool:
    if board["team_id"] is None:
        return board["owner_id"] == user_id
    if not is_live_org_member(db, user_id, board["team_id"]):
        return False
    if not board["restricted"]:  # strict on purpose — see get_board's docstring
        return True
    if board["owner_id"] == user_id:
        return True
    if org_authz.is_org_admin(db, user_id, board["team_id"]):
        return True
    return _listed(db, board["id"], user_id)


def can_access_board(db: Client, user_id: str, board_id: str) -> bool:
    board = get_board(db, board_id)
    return bool(board) and _can_access(db, user_id, board)


def can_edit_board(db: Client, user_id: str, board_id: str) -> bool:
    """No viewer role: anyone who can see a board can edit its columns/tasks."""
    return can_access_board(db, user_id, board_id)


def can_assign_user(db: Client, target_user_id: str, board_id: str) -> bool:
    """A task can only be assigned to someone who can open the board.

    NOTE the second parameter is the TARGET, not the caller — same `(db, str,
    str)` shape as `can_access_board`, so a caller that passes the actor here
    would fail OPEN. The actor is gated separately and first: see
    `boards/service.py:add_assignee`, which calls `require_board_edit(actor)`
    before this.
    """
    board = get_board(db, board_id)
    return bool(board) and _can_access(db, target_user_id, board)


def can_manage_board(db: Client, user_id: str, board: dict) -> bool:
    """Settings gate (name is NOT gated here — see rename): visibility + member list.
    Personal → owner; team → live seat AND (owner OR org admin)."""
    if board["team_id"] is None:
        return board["owner_id"] == user_id
    if not is_live_org_member(db, user_id, board["team_id"]):
        return False
    return board["owner_id"] == user_id or org_authz.is_org_admin(db, user_id, board["team_id"])


def is_board_admin(db: Client, user_id: str, board: dict) -> bool:
    """Archive / delete / restore / archived-list gate. Personal → owner; team → live org admin."""
    if board["team_id"] is None:
        return board["owner_id"] == user_id
    return is_live_org_member(db, user_id, board["team_id"]) and org_authz.is_org_admin(db, user_id, board["team_id"])


def require_board_access(db: Client, user_id: str, board_id: str) -> None:
    """404 (never 403) so a foreign board id can't be told from a missing one."""
    if not can_access_board(db, user_id, board_id):
        raise HTTPException(status_code=404, detail="Board not found")


def require_board_edit(db: Client, user_id: str, board_id: str) -> None:
    if not can_edit_board(db, user_id, board_id):
        raise HTTPException(status_code=404, detail="Board not found")


def require_org_admin(db: Client, user_id: str, org_id: str) -> None:
    """403: the caller may know the org exists (they're a member) but isn't an admin."""
    if not (is_live_org_member(db, user_id, org_id) and org_authz.is_org_admin(db, user_id, org_id)):
        raise HTTPException(status_code=403, detail="Admin access required")
