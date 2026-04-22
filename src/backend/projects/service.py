from supabase import Client


class DuplicateInviteError(Exception):
    """Raised when an invite already exists for (project_id, email)."""


async def get_members(db: Client, user_id: str, project_id: str):
    """List all members of a project."""
    result = db.table("project_members").select("*").eq("project_id", project_id).order("created_at").execute()
    return result.data or []


async def get_user_role(db: Client, user_id: str, project_id: str):
    """Get the caller's role on a project, or None if not a member."""
    result = (
        db.table("project_members")
        .select("role")
        .eq("project_id", project_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    return result.data["role"] if result.data else None


def _find_user_id_by_email(db: Client, email: str) -> str | None:
    """Look up a user id by email via auth.users (service-role client)."""
    try:
        result = db.schema("auth").from_("users").select("id").ilike("email", email).limit(1).execute()
    except Exception as exc:
        print(f"auth.users lookup failed for {email!r}: {exc}")
        return None
    rows = result.data or []
    return rows[0]["id"] if rows else None


async def add_member(db: Client, user_id: str, project_id: str, email: str, role: str):
    """Add a member by email. Auto-adds if account exists, creates pending invite if not."""
    caller_role = await get_user_role(db, user_id, project_id)
    if caller_role not in ("owner", "admin"):
        raise PermissionError("Only admins can add members")
    if role not in ("admin", "editor", "viewer"):
        raise ValueError("Invalid role")

    existing_user_id = _find_user_id_by_email(db, email)

    if existing_user_id:
        already = (
            db.table("project_members")
            .select("id")
            .eq("project_id", project_id)
            .eq("user_id", existing_user_id)
            .maybe_single()
            .execute()
        )
        if already.data:
            raise DuplicateInviteError("User is already a member of this project")

        result = (
            db.table("project_members")
            .insert(
                {
                    "project_id": project_id,
                    "user_id": existing_user_id,
                    "role": role,
                    "invited_by": user_id,
                }
            )
            .execute()
        )
        return {"type": "added", "member": result.data[0] if result.data else None}

    try:
        result = (
            db.table("pending_project_invites")
            .insert(
                {
                    "project_id": project_id,
                    "email": email.lower(),
                    "role": role,
                    "invited_by": user_id,
                }
            )
            .execute()
        )
    except Exception as exc:
        # Postgres unique_violation is SQLSTATE 23505. supabase-py surfaces this as
        # a generic Exception whose message contains the code; inspect the string.
        if "23505" in str(exc) or "duplicate key" in str(exc).lower():
            raise DuplicateInviteError("An invite for this email already exists on this project") from exc
        raise

    return {"type": "pending", "invite": result.data[0] if result.data else None}


async def record_invite_email_result(
    db: Client,
    invite_id: str,
    error_message: str | None,
):
    """Persist the outcome of the last Resend attempt on a pending invite."""
    try:
        db.table("pending_project_invites").update(
            {
                "last_email_error": error_message,
                "last_email_attempt_at": "now()",
            }
        ).eq("id", invite_id).execute()
    except Exception as exc:
        print(f"Failed to record invite email result for {invite_id}: {exc}")


async def update_member_role(db: Client, user_id: str, project_id: str, member_id: str, role: str):
    """Update a member's role. Caller must be admin+."""
    caller_role = await get_user_role(db, user_id, project_id)
    if caller_role not in ("owner", "admin"):
        raise PermissionError("Only admins can change roles")
    if role not in ("admin", "editor", "viewer"):
        raise ValueError("Invalid role — cannot set to owner")

    result = (
        db.table("project_members").update({"role": role}).eq("id", member_id).eq("project_id", project_id).execute()
    )
    return result.data[0] if result.data else None


async def remove_member(db: Client, user_id: str, project_id: str, member_id: str):
    """Remove a member. Admin+ can remove others; non-owners can remove themselves."""
    target = db.table("project_members").select("*").eq("id", member_id).eq("project_id", project_id).single().execute()
    if not target.data:
        raise ValueError("Member not found")

    is_self = target.data["user_id"] == user_id
    caller_role = await get_user_role(db, user_id, project_id)

    if target.data["role"] == "owner":
        raise PermissionError("Cannot remove the project owner")
    if not is_self and caller_role not in ("owner", "admin"):
        raise PermissionError("Only admins can remove other members")

    db.table("project_members").delete().eq("id", member_id).execute()
    return {"deleted": member_id}


async def get_pending_invites(db: Client, user_id: str, project_id: str):
    """List pending invites for non-existing users."""
    result = (
        db.table("pending_project_invites")
        .select("*")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


async def get_pending_invite(db: Client, project_id: str, invite_id: str):
    """Fetch a single pending invite, scoped to a project."""
    result = (
        db.table("pending_project_invites")
        .select("*")
        .eq("id", invite_id)
        .eq("project_id", project_id)
        .maybe_single()
        .execute()
    )
    return result.data


async def delete_pending_invite(db: Client, user_id: str, project_id: str, invite_id: str):
    """Cancel a pending invite."""
    caller_role = await get_user_role(db, user_id, project_id)
    if caller_role not in ("owner", "admin"):
        raise PermissionError("Only admins can cancel invites")
    db.table("pending_project_invites").delete().eq("id", invite_id).execute()
    return {"deleted": invite_id}
