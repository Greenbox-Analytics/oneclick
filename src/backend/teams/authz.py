"""Access-control helpers for team boards.

The backend uses the Supabase service-role client, which BYPASSES RLS — so these
per-request checks are the real authorization gate (RLS is defense-in-depth for any
direct frontend reads). Spec §5. Each helper takes the service-role `db` client and
returns a boolean, plus thin `require_*` wrappers that raise.

Argument order is (db, user_id, team_id) to match the SQL helpers
is_team_member(p_user_id, p_team_id) / is_team_admin(p_user_id, p_team_id).
"""

from fastapi import HTTPException
from supabase import Client


def is_team_member(db: Client, user_id: str, team_id: str) -> bool:
    """True if user_id is any member (admin or member) of team_id."""
    res = db.table("team_members").select("id").eq("team_id", team_id).eq("user_id", user_id).limit(1).execute()
    return bool(res.data)


def is_team_admin(db: Client, user_id: str, team_id: str) -> bool:
    """True if user_id is an admin of team_id."""
    res = (
        db.table("team_members")
        .select("id")
        .eq("team_id", team_id)
        .eq("user_id", user_id)
        .eq("role", "admin")
        .limit(1)
        .execute()
    )
    return bool(res.data)


def get_board(db: Client, board_id: str) -> dict | None:
    """Fetch the board row (id, team_id, owner_id, archived) or None."""
    res = db.table("boards").select("id, team_id, owner_id, archived").eq("id", board_id).limit(1).execute()
    rows = res.data or []
    return rows[0] if rows else None


def can_access_board(db: Client, user_id: str, board_id: str) -> bool:
    """True if user can view a board: personal owner, or a member of its team.

    Membership/ownership only. Archived-board *list* filtering (spec §5.8) is applied
    by list endpoints, not here — an admin still needs access to an archived board to
    un-archive it.
    """
    board = get_board(db, board_id)
    if not board:
        return False
    if board["team_id"] is None:
        return board["owner_id"] == user_id
    return is_team_member(db, user_id, board["team_id"])


def can_edit_board(db: Client, user_id: str, board_id: str) -> bool:
    """True if user can edit columns/tasks on a board.

    In v1 there is no viewer role, so any team member (admin or member) can edit, and a
    personal owner can edit their own — identical to can_access_board.
    """
    return can_access_board(db, user_id, board_id)


def can_assign_user(db: Client, target_user_id: str, board_id: str) -> bool:
    """True if target_user_id may be assigned to a task on board_id (spec §5.6).

    Personal board (team_id IS NULL): only the owner (self-assignment).
    Team board: the target must be a member of the team.
    """
    board = get_board(db, board_id)
    if not board:
        return False
    if board["team_id"] is None:
        return target_user_id == board["owner_id"]
    return is_team_member(db, target_user_id, board["team_id"])


def require_board_access(db: Client, user_id: str, board_id: str) -> None:
    """Raise 404 (not 403, to avoid leaking existence) if the user can't access the board."""
    if not can_access_board(db, user_id, board_id):
        raise HTTPException(status_code=404, detail="Board not found")


def require_board_edit(db: Client, user_id: str, board_id: str) -> None:
    """Raise 404 if the user can't edit the board."""
    if not can_edit_board(db, user_id, board_id):
        raise HTTPException(status_code=404, detail="Board not found")


def require_team_admin(db: Client, user_id: str, team_id: str) -> None:
    """Raise 403 if the user is not an admin of the team."""
    if not is_team_admin(db, user_id, team_id):
        raise HTTPException(status_code=403, detail="Admin access required")
