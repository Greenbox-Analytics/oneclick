"""Teams & membership business logic. Mirrors projects/service.py.

The backend uses the service-role client (RLS-bypassing), so authz goes through
teams/authz.py per call. create_team is a single insert; the creator's admin membership is
added atomically by the auto_create_team_admin trigger (keyed on created_by, since auth.uid()
is NULL under the service role) — so a team can never exist without an admin.
"""

from datetime import UTC, datetime, timedelta

from supabase import Client

from confirm import ConfirmationError, normalize_name  # shared trim+NFC + confirmation error
from teams import authz


class DuplicateInviteError(Exception):
    """Already a member, or a duplicate invite for (team, email)."""


class LastAdminError(Exception):
    """The DB last-admin guard rejected a removal/demotion."""


class InviteInvalidError(Exception):
    """Invite is expired, declined, or otherwise no longer actionable."""


class NotArchivedError(Exception):
    """Team must be archived before it can be permanently deleted."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _seven_days_iso() -> str:
    return (datetime.now(UTC) + timedelta(days=7)).isoformat()


def _is_last_admin_error(exc: Exception) -> bool:
    return "only admin" in str(exc).lower()


def _find_user_id_by_email(db: Client, email: str) -> str | None:
    """Look up a user id by email via the get_user_id_by_email(lookup_email) SECURITY DEFINER
    RPC (pre-existing, from the registry migration 20260329000000).

    This project's PostgREST does not expose the `auth` schema, so db.schema("auth") fails
    (PGRST106); the RPC reads auth.users on the backend's behalf. Returns None if unknown.
    """
    try:
        result = db.rpc("get_user_id_by_email", {"lookup_email": email}).execute()
    except Exception as exc:
        print(f"get_user_id_by_email failed for {email!r}: {exc}")
        return None
    data = result.data
    if isinstance(data, list):  # defensive: some PostgREST/client versions wrap scalars
        data = data[0] if data else None
    return data or None


async def create_team(db: Client, user_id: str, name: str, description: str | None) -> dict:
    """Create a team. A single insert — the auto_create_team_admin trigger (Phase 2 migration)
    atomically adds the creator as admin, keyed on created_by (works under the service role,
    unlike an auth.uid() trigger). One write means no orphan-team-with-no-admin window."""
    res = db.table("teams").insert({"name": name, "description": description, "created_by": user_id}).execute()
    team = res.data[0] if res.data else None
    if not team:
        raise RuntimeError("Failed to create team")
    # The creator is always the admin (auto_create_team_admin), so annotate my_role like
    # list_my_teams — the client doesn't need a follow-up GET to learn its role.
    return {**team, "my_role": "admin"}


async def list_my_teams(db: Client, user_id: str) -> list[dict]:
    """Active (non-archived) teams the user belongs to, each annotated with `my_role`."""
    memberships = db.table("team_members").select("team_id, role").eq("user_id", user_id).execute()
    rows = memberships.data or []
    if not rows:
        return []
    role_by_team = {m["team_id"]: m["role"] for m in rows}
    teams = (
        db.table("teams")
        .select("*")
        .in_("id", list(role_by_team.keys()))
        .is_("archived_at", "null")
        .order("created_at")
        .execute()
    )
    out = []
    for t in teams.data or []:
        t["my_role"] = role_by_team.get(t["id"])
        out.append(t)
    return out


async def get_team(db: Client, user_id: str, team_id: str) -> dict:
    """Fetch a team the caller belongs to."""
    if not authz.is_team_member(db, user_id, team_id):
        raise PermissionError("Not a team member")
    res = db.table("teams").select("*").eq("id", team_id).maybe_single().execute()
    if not res.data:
        raise ValueError("Team not found")
    return res.data


async def update_team(db: Client, user_id: str, team_id: str, fields: dict) -> dict:
    """Update team name/description. Admin only."""
    if not authz.is_team_admin(db, user_id, team_id):
        raise PermissionError("Admin access required")
    clean = {k: v for k, v in fields.items() if v is not None}
    if not clean:
        return await get_team(db, user_id, team_id)
    res = db.table("teams").update(clean).eq("id", team_id).execute()
    return res.data[0] if res.data else None


async def archive_team(db: Client, user_id: str, team_id: str) -> dict:
    """Soft-delete (archive) a team. Admin only. Returns impact counts for the UI confirm."""
    if not authz.is_team_admin(db, user_id, team_id):
        raise PermissionError("Admin access required")
    boards = db.table("boards").select("id", count="exact").eq("team_id", team_id).execute()
    board_ids = [b["id"] for b in (boards.data or [])]
    task_count = 0
    if board_ids:
        tasks = db.table("board_tasks").select("id", count="exact").in_("board_id", board_ids).execute()
        task_count = tasks.count or 0
    members = db.table("team_members").select("id", count="exact").eq("team_id", team_id).execute()
    db.table("teams").update({"archived_at": _now_iso()}).eq("id", team_id).execute()
    return {
        "archived": team_id,
        "boards": boards.count or 0,
        "tasks": task_count,
        "members": members.count or 0,
    }


async def delete_team(db: Client, user_id: str, team_id: str, confirm_name: str) -> dict:
    """Permanently delete an ALREADY-ARCHIVED team (cascade removes boards/columns/tasks/
    assignees/members/invites). Admin-only + normalized typed-name confirmation. Requires
    migration 20260703000000 for the ≥2-member cascade (last-admin trigger 27000 fix)."""
    if not authz.is_team_admin(db, user_id, team_id):
        raise PermissionError("Admin access required")
    team = db.table("teams").select("id, name, archived_at").eq("id", team_id).limit(1).execute()
    if not team.data:
        raise ValueError("Team not found")
    row = team.data[0]
    if not row.get("archived_at"):
        raise NotArchivedError("Archive the team before deleting it")
    if normalize_name(confirm_name) != normalize_name(row.get("name")):
        raise ConfirmationError("Confirmation does not match")

    boards = db.table("boards").select("id", count="exact").eq("team_id", team_id).execute()
    board_ids = [b["id"] for b in (boards.data or [])]
    task_count = 0
    if board_ids:
        tasks = db.table("board_tasks").select("id", count="exact").in_("board_id", board_ids).execute()
        task_count = tasks.count or 0
    members = db.table("team_members").select("id", count="exact").eq("team_id", team_id).execute()

    # team_member_removal_cleanup handles member notifications during the cascade; this mop-up
    # covers invitee (non-member) team notifications the trigger can't reach. (spec §2.1)
    db.table("notifications").delete().eq("entity_type", "team").eq("entity_id", team_id).execute()
    db.table("teams").delete().eq("id", team_id).execute()  # FK cascade
    return {"deleted": team_id, "boards": boards.count or 0, "tasks": task_count, "members": members.count or 0}


async def restore_team(db: Client, user_id: str, team_id: str) -> dict:
    """Un-archive a team. Admin-only."""
    if not authz.is_team_admin(db, user_id, team_id):
        raise PermissionError("Admin access required")
    db.table("teams").update({"archived_at": None}).eq("id", team_id).execute()
    return {"restored": team_id}


async def list_archived_teams(db: Client, user_id: str) -> list[dict]:
    """Archived teams where the caller is an ADMIN, each with boards/tasks/members counts."""
    memberships = (
        db.table("team_members").select("team_id").eq("user_id", user_id).eq("role", "admin").execute().data or []
    )
    team_ids = [m["team_id"] for m in memberships]
    if not team_ids:
        return []
    teams = (
        db.table("teams")
        .select("*")
        .in_("id", team_ids)
        .not_.is_("archived_at", "null")
        .order("archived_at", desc=True)
        .execute()
    ).data or []
    for t in teams:
        b = db.table("boards").select("id", count="exact").eq("team_id", t["id"]).execute()
        board_ids = [x["id"] for x in (b.data or [])]
        tc = 0
        if board_ids:
            tc = db.table("board_tasks").select("id", count="exact").in_("board_id", board_ids).execute().count or 0
        m = db.table("team_members").select("id", count="exact").eq("team_id", t["id"]).execute()
        t["boards"], t["tasks"], t["members"] = b.count or 0, tc, m.count or 0
    return teams


async def list_members(db: Client, user_id: str, team_id: str) -> list[dict]:
    """List team members with profile name/avatar. Members only."""
    if not authz.is_team_member(db, user_id, team_id):
        raise PermissionError("Not a team member")
    members = db.table("team_members").select("*").eq("team_id", team_id).order("created_at").execute()
    rows = members.data or []
    user_ids = [m["user_id"] for m in rows]
    profiles = {}
    if user_ids:
        res = db.table("profiles").select("id, full_name, avatar_url").in_("id", user_ids).execute()
        profiles = {p["id"]: p for p in (res.data or [])}
    for m in rows:
        prof = profiles.get(m["user_id"], {})
        m["full_name"] = prof.get("full_name")
        m["avatar_url"] = prof.get("avatar_url")
    return rows


async def update_member_role(db: Client, user_id: str, team_id: str, member_id: str, role: str) -> dict:
    """Change a member's role. Admin only. DB last-admin guard may reject (→ LastAdminError)."""
    if not authz.is_team_admin(db, user_id, team_id):
        raise PermissionError("Admin access required")
    if role not in ("admin", "member"):
        raise ValueError("Invalid role")
    try:
        res = db.table("team_members").update({"role": role}).eq("id", member_id).eq("team_id", team_id).execute()
    except Exception as exc:
        if _is_last_admin_error(exc):
            raise LastAdminError("You are the only admin — promote another member first") from exc
        raise
    return res.data[0] if res.data else None


async def remove_member(db: Client, user_id: str, team_id: str, member_id: str) -> dict:
    """Remove a member. Admins remove anyone; a member may remove themselves (leave).
    DB triggers enforce the last-admin invariant + clean up the removed user's footprint."""
    target = db.table("team_members").select("*").eq("id", member_id).eq("team_id", team_id).maybe_single().execute()
    if not target.data:
        raise ValueError("Member not found")
    is_self = target.data["user_id"] == user_id
    if not is_self and not authz.is_team_admin(db, user_id, team_id):
        raise PermissionError("Admin access required to remove other members")
    try:
        db.table("team_members").delete().eq("id", member_id).eq("team_id", team_id).execute()
    except Exception as exc:
        if _is_last_admin_error(exc):
            raise LastAdminError("You are the only admin — promote another member first") from exc
        raise
    return {"deleted": member_id}


# ============================================================================
# Invite flow (§6)
# ============================================================================


async def invite_member(db: Client, user_id: str, team_id: str, email: str, role: str) -> dict:
    """Idempotent invite (spec §6). Admin only.

    - already a member            -> DuplicateInviteError
    - existing invite row         -> UPDATE it (role/expiry/status=pending), resend
    - otherwise                   -> INSERT a new pending invite
    Returns {"type": "invited", "invite": <row>, "notify_user_id": <existing id or None>}.
    The expression UNIQUE (team_id, LOWER(email)) can't be an upsert target, so we
    SELECT-then-UPDATE/INSERT explicitly.
    """
    if not authz.is_team_admin(db, user_id, team_id):
        raise PermissionError("Admin access required")
    if role not in ("admin", "member"):
        raise ValueError("Invalid role")
    email_l = email.lower()

    existing_user_id = _find_user_id_by_email(db, email)
    if existing_user_id and authz.is_team_member(db, existing_user_id, team_id):
        raise DuplicateInviteError("User is already a member of this team")

    existing = (
        db.table("pending_team_invites")
        .select("*")
        .eq("team_id", team_id)
        .eq("email", email_l)
        .maybe_single()
        .execute()
    )
    if existing and existing.data:
        updated = (
            db.table("pending_team_invites")
            .update(
                {
                    "role": role,
                    "status": "pending",
                    "expires_at": _seven_days_iso(),
                    "invited_by": user_id,
                }
            )
            .eq("id", existing.data["id"])
            .execute()
        )
        invite = updated.data[0] if updated.data else None
    else:
        try:
            created = (
                db.table("pending_team_invites")
                .insert({"team_id": team_id, "email": email_l, "role": role, "invited_by": user_id})
                .execute()
            )
        except Exception as exc:
            # Race / pre-existing row on the (team_id, LOWER(email)) unique index → clean 409
            # instead of a raw 500 (mirrors projects.add_member's 23505 handling).
            if "23505" in str(exc) or "duplicate key" in str(exc).lower():
                raise DuplicateInviteError("An invite for this email already exists on this team") from exc
            raise
        invite = created.data[0] if created.data else None

    return {"type": "invited", "invite": invite, "notify_user_id": existing_user_id}


async def get_pending_invites(db: Client, user_id: str, team_id: str) -> list[dict]:
    """List a team's pending invites. Admin only."""
    if not authz.is_team_admin(db, user_id, team_id):
        raise PermissionError("Admin access required")
    res = db.table("pending_team_invites").select("*").eq("team_id", team_id).order("created_at", desc=True).execute()
    return res.data or []


async def cancel_invite(db: Client, user_id: str, team_id: str, invite_id: str) -> dict:
    """Delete a pending invite. Admin only."""
    if not authz.is_team_admin(db, user_id, team_id):
        raise PermissionError("Admin access required")
    db.table("pending_team_invites").delete().eq("id", invite_id).eq("team_id", team_id).execute()
    return {"deleted": invite_id}


async def get_invite_by_token(db: Client, token: str) -> dict | None:
    res = db.table("pending_team_invites").select("*").eq("token", token).maybe_single().execute()
    return res.data


async def accept_invite(db: Client, user_id: str, user_email: str, token: str) -> dict:
    """Accept an invite by token. The caller's email must match the invite."""
    invite = await get_invite_by_token(db, token)
    if not invite:
        raise ValueError("Invite not found")
    if invite["email"].lower() != user_email.lower():
        raise PermissionError("This invite was sent to a different email")
    if invite["status"] == "accepted":
        return {"type": "already_accepted", "team_id": invite["team_id"]}
    if invite["status"] != "pending":
        raise InviteInvalidError("This invite is no longer valid")
    if datetime.fromisoformat(invite["expires_at"]) < datetime.now(UTC):
        raise InviteInvalidError("This invite has expired")

    # ignore_duplicates → ON CONFLICT DO NOTHING: if already a member, leave their row/role
    # untouched (no silent demotion) and don't fire the last-admin UPDATE guard.
    db.table("team_members").upsert(
        {
            "team_id": invite["team_id"],
            "user_id": user_id,
            "role": invite["role"],
            "invited_by": invite["invited_by"],
        },
        on_conflict="team_id,user_id",
        ignore_duplicates=True,
    ).execute()
    db.table("pending_team_invites").update({"status": "accepted"}).eq("id", invite["id"]).execute()
    return {"type": "accepted", "team_id": invite["team_id"]}


async def decline_invite(db: Client, user_id: str, user_email: str, token: str) -> dict:
    """Decline an invite by token. The caller's email must match the invite."""
    invite = await get_invite_by_token(db, token)
    if not invite:
        raise ValueError("Invite not found")
    if invite["email"].lower() != user_email.lower():
        raise PermissionError("This invite was sent to a different email")
    db.table("pending_team_invites").update({"status": "declined"}).eq("id", invite["id"]).execute()
    return {"type": "declined", "team_id": invite["team_id"]}


def create_team_invite_notification(
    db: Client, target_user_id: str, team_id: str, team_name: str, inviter_name: str, token: str
) -> None:
    """In-app notification for an invited existing user (Accept/Decline). Writes to the
    unified `notifications` table (entity_type='team')."""
    db.table("notifications").insert(
        {
            "user_id": target_user_id,
            "type": "team_invite",
            "title": f"Invited to {team_name}",
            "message": f'{inviter_name} invited you to join the team "{team_name}".',
            "entity_type": "team",
            "entity_id": team_id,
            "metadata": {"team_id": team_id, "token": token},
        }
    ).execute()
