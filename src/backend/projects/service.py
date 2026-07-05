from supabase import Client


class DuplicateInviteError(Exception):
    """Raised when an invite already exists for (project_id, email)."""


async def get_members(db: Client, user_id: str, project_id: str):
    """List all members of a project."""
    if await get_user_role(db, user_id, project_id) is None:
        raise PermissionError("denied")
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
    """Look up a user id by email via the get_user_id_by_email(lookup_email) SECURITY DEFINER
    RPC (pre-existing, registry migration 20260329000000).

    This project's PostgREST does not expose the `auth` schema (PGRST106), so the previous
    db.schema("auth") approach silently returned None — every invitee was treated as new.
    """
    try:
        result = db.rpc("get_user_id_by_email", {"lookup_email": email}).execute()
    except Exception as exc:
        print(f"get_user_id_by_email failed for {email!r}: {exc}")
        return None
    data = result.data
    if isinstance(data, list):
        data = data[0] if data else None
    return data or None


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
    if await get_user_role(db, user_id, project_id) is None:
        raise PermissionError("denied")
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
    db.table("pending_project_invites").delete().eq("id", invite_id).eq("project_id", project_id).execute()
    return {"deleted": invite_id}


# ============================================================================
# Project expenses (net-vs-gross royalty support)
# ============================================================================

_WRITE_ROLES = ("owner", "admin", "editor")


def _attach_expense_works(db: Client, expenses: list[dict]) -> list[dict]:
    """Populate a `work_ids` list on each expense from the join table."""
    if not expenses:
        return expenses
    expense_ids = [e["id"] for e in expenses]
    links = db.table("project_expense_works").select("expense_id, work_id").in_("expense_id", expense_ids).execute()
    by_expense: dict[str, list[str]] = {}
    for row in links.data or []:
        by_expense.setdefault(row["expense_id"], []).append(row["work_id"])
    for e in expenses:
        e["work_ids"] = by_expense.get(e["id"], [])
    return expenses


def _replace_expense_works(db: Client, expense_id: str, work_ids: list[str]) -> None:
    """Replace the set of work links for an expense."""
    db.table("project_expense_works").delete().eq("expense_id", expense_id).execute()
    rows = [{"expense_id": expense_id, "work_id": wid} for wid in dict.fromkeys(work_ids)]
    if rows:
        db.table("project_expense_works").insert(rows).execute()


async def get_expenses(db: Client, user_id: str, project_id: str):
    """List all expenses for a project, each with its linked work_ids."""
    if await get_user_role(db, user_id, project_id) is None:
        raise PermissionError("Access denied")
    result = (
        db.table("project_expenses")
        .select("*")
        .eq("project_id", project_id)
        .order("incurred_on", desc=True)
        .order("created_at", desc=True)
        .execute()
    )
    return _attach_expense_works(db, result.data or [])


async def create_expense(db: Client, user_id: str, project_id: str, data: dict):
    """Create a project expense. Editors+ only."""
    if await get_user_role(db, user_id, project_id) not in _WRITE_ROLES:
        raise PermissionError("Only editors and above can add expenses")
    work_ids = data.pop("work_ids", []) or []
    result = (
        db.table("project_expenses")
        .insert(
            {
                "project_id": project_id,
                "created_by": user_id,
                "description": data["description"],
                "amount": data["amount"],
                "category": data.get("category"),
                "incurred_on": data.get("incurred_on"),
            }
        )
        .execute()
    )
    expense = result.data[0] if result.data else None
    if expense:
        _replace_expense_works(db, expense["id"], work_ids)
        expense["work_ids"] = list(dict.fromkeys(work_ids))
    return expense


async def update_expense(db: Client, user_id: str, project_id: str, expense_id: str, data: dict):
    """Update a project expense. Editors+ only. `work_ids=None` leaves links unchanged."""
    if await get_user_role(db, user_id, project_id) not in _WRITE_ROLES:
        raise PermissionError("Only editors and above can edit expenses")
    work_ids = data.pop("work_ids", None)
    fields = {k: v for k, v in data.items() if v is not None}
    if fields:
        db.table("project_expenses").update(fields).eq("id", expense_id).eq("project_id", project_id).execute()
    if work_ids is not None:
        _replace_expense_works(db, expense_id, work_ids)
    result = (
        db.table("project_expenses")
        .select("*")
        .eq("id", expense_id)
        .eq("project_id", project_id)
        .maybe_single()
        .execute()
    )
    expense = result.data
    if expense:
        _attach_expense_works(db, [expense])
    return expense


async def delete_expense(db: Client, user_id: str, project_id: str, expense_id: str):
    """Delete a project expense. Editors+ only. Join rows cascade."""
    if await get_user_role(db, user_id, project_id) not in _WRITE_ROLES:
        raise PermissionError("Only editors and above can delete expenses")
    db.table("project_expenses").delete().eq("id", expense_id).eq("project_id", project_id).execute()
    return {"deleted": expense_id}


async def get_expenses_summary(db: Client, user_id: str):
    """Cross-project expense rollup for the standalone Expense Tracker tool.

    Returns every expense across the projects the caller is a member of (any
    role — read parity with the per-project Expenses tab), enriched with
    project and artist names and an ``is_tagged`` flag (linked to >=1 track).
    """
    memberships = db.table("project_members").select("project_id").eq("user_id", user_id).execute()
    project_ids = list({m["project_id"] for m in (memberships.data or [])})
    if not project_ids:
        return []

    projects = db.table("projects").select("id, name, artist_id").in_("id", project_ids).execute()
    proj_by_id = {p["id"]: p for p in (projects.data or [])}

    artist_ids = list({p["artist_id"] for p in proj_by_id.values() if p.get("artist_id")})
    artist_by_id = {}
    if artist_ids:
        artists = db.table("artists").select("id, name").in_("id", artist_ids).execute()
        artist_by_id = {a["id"]: a["name"] for a in (artists.data or [])}

    expenses = db.table("project_expenses").select("*").in_("project_id", project_ids).execute()
    rows = expenses.data or []
    if not rows:
        return []

    expense_ids = [e["id"] for e in rows]
    links = db.table("project_expense_works").select("expense_id").in_("expense_id", expense_ids).execute()
    tagged_ids = {row["expense_id"] for row in (links.data or [])}

    summary = []
    for e in rows:
        proj = proj_by_id.get(e["project_id"], {})
        artist_id = proj.get("artist_id")
        summary.append(
            {
                "id": e["id"],
                "project_id": e["project_id"],
                "project_name": proj.get("name"),
                "artist_id": artist_id,
                "artist_name": artist_by_id.get(artist_id),
                "description": e.get("description"),
                "amount": float(e.get("amount") or 0),
                "category": e.get("category"),
                "incurred_on": e.get("incurred_on"),
                "is_tagged": e["id"] in tagged_ids,
            }
        )
    return summary
