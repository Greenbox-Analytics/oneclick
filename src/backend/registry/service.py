"""Service layer for the Rights & Ownership Registry with collaboration, TeamCards, notes, and verification."""

import hashlib
from supabase import Client


# ============================================================
# Works
# ============================================================

async def get_works(db: Client, user_id: str, artist_id: str = None):
    query = db.table("works_registry").select("*").eq("user_id", user_id)
    if artist_id:
        query = query.eq("artist_id", artist_id)
    result = query.order("created_at", desc=True).execute()
    return result.data


async def get_works_as_collaborator(db: Client, user_id: str):
    """Get ALL works where user is a collaborator (not the creator) — any status."""
    collab_rows = (
        db.table("registry_collaborators")
        .select("work_id")
        .eq("collaborator_user_id", user_id)
        .neq("status", "revoked")
        .execute()
    )
    work_ids = [r["work_id"] for r in (collab_rows.data or [])]
    if not work_ids:
        return []
    result = (
        db.table("works_registry")
        .select("*")
        .in_("id", work_ids)
        .neq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data


async def get_works_by_project(db: Client, user_id: str, project_id: str):
    result = (
        db.table("works_registry")
        .select("*")
        .eq("project_id", project_id)
        .order("created_at")
        .execute()
    )
    return result.data


async def get_work(db: Client, user_id: str, work_id: str):
    result = (
        db.table("works_registry")
        .select("*")
        .eq("id", work_id)
        .single()
        .execute()
    )
    return result.data


async def create_work(db: Client, user_id: str, data: dict):
    # Verify the artist belongs to this user before allowing work creation
    artist = db.table("artists").select("id").eq("id", data.get("artist_id")).eq("user_id", user_id).single().execute()
    if not artist.data:
        return None  # artist_id doesn't belong to this user
    data["user_id"] = user_id
    result = db.table("works_registry").insert(data).execute()
    return result.data[0] if result.data else None


async def update_work(db: Client, user_id: str, work_id: str, data: dict):
    # Block edits on registered works — requires re-approval if modified
    work = db.table("works_registry").select("status").eq("id", work_id).eq("user_id", user_id).single().execute()
    if not work.data:
        return None
    if work.data["status"] == "registered":
        # Reset to draft if owner edits a registered work — forces re-approval
        data["status"] = "draft"
    result = (
        db.table("works_registry")
        .update(data)
        .eq("id", work_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def delete_work(db: Client, user_id: str, work_id: str):
    # Notify collaborators before deletion so they don't silently lose access
    collabs = (
        db.table("registry_collaborators")
        .select("collaborator_user_id, work_id")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .execute()
    )
    work = db.table("works_registry").select("title").eq("id", work_id).single().execute()
    work_title = (work.data or {}).get("title", "Untitled")
    for c in (collabs.data or []):
        if c.get("collaborator_user_id"):
            await create_notification(
                db,
                user_id=c["collaborator_user_id"],
                work_id=None,  # work is being deleted
                notification_type="status_change",
                title="Work deleted",
                message=f'The work "{work_title}" has been deleted by its owner.',
                metadata={"work_title": work_title},
            )
    return db.table("works_registry").delete().eq("id", work_id).eq("user_id", user_id).execute().data


# ============================================================
# Ownership Stakes
# ============================================================

async def get_stakes(db: Client, user_id: str, work_id: str):
    result = (
        db.table("ownership_stakes")
        .select("*")
        .eq("work_id", work_id)
        .order("stake_type")
        .order("percentage", desc=True)
        .execute()
    )
    return result.data


async def validate_stake_percentage(
    db: Client, user_id: str, work_id: str, stake_type: str,
    new_percentage: float, exclude_stake_id: str = None,
):
    existing = (
        db.table("ownership_stakes")
        .select("id, percentage")
        .eq("work_id", work_id)
        .eq("user_id", user_id)
        .eq("stake_type", stake_type)
        .execute()
    )
    total = sum(
        row["percentage"] for row in (existing.data or [])
        if row["id"] != exclude_stake_id
    )
    return (total + new_percentage) <= 100.0


async def create_stake(db: Client, user_id: str, data: dict):
    data["user_id"] = user_id
    result = db.table("ownership_stakes").insert(data).execute()
    return result.data[0] if result.data else None


async def update_stake(db: Client, user_id: str, stake_id: str, data: dict):
    result = (
        db.table("ownership_stakes")
        .update(data)
        .eq("id", stake_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def delete_stake(db: Client, user_id: str, stake_id: str):
    return db.table("ownership_stakes").delete().eq("id", stake_id).eq("user_id", user_id).execute().data


# ============================================================
# Licensing Rights
# ============================================================

async def get_licenses(db: Client, user_id: str, work_id: str):
    result = (
        db.table("licensing_rights")
        .select("*")
        .eq("work_id", work_id)
        .order("start_date", desc=True)
        .execute()
    )
    return result.data


async def create_license(db: Client, user_id: str, data: dict):
    data["user_id"] = user_id
    result = db.table("licensing_rights").insert(data).execute()
    return result.data[0] if result.data else None


async def update_license(db: Client, user_id: str, license_id: str, data: dict):
    result = (
        db.table("licensing_rights").update(data)
        .eq("id", license_id).eq("user_id", user_id).execute()
    )
    return result.data[0] if result.data else None


async def delete_license(db: Client, user_id: str, license_id: str):
    return db.table("licensing_rights").delete().eq("id", license_id).eq("user_id", user_id).execute().data


# ============================================================
# Agreements
# ============================================================

async def get_agreements(db: Client, user_id: str, work_id: str):
    result = (
        db.table("registry_agreements")
        .select("*")
        .eq("work_id", work_id)
        .order("effective_date", desc=True)
        .execute()
    )
    return result.data


async def create_agreement(db: Client, user_id: str, data: dict):
    data["user_id"] = user_id
    result = db.table("registry_agreements").insert(data).execute()
    return result.data[0] if result.data else None


# ============================================================
# Collaboration
# ============================================================

async def get_collaborators(db: Client, work_id: str):
    result = (
        db.table("registry_collaborators")
        .select("*")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .order("invited_at", desc=True)
        .execute()
    )
    return result.data


async def check_user_exists(db: Client, email: str):
    """Check if a user with this email exists on the platform. Returns user_id or None.
    Uses a database function that queries auth.users directly (O(1) via index)
    instead of iterating paginated list_users() results."""
    result = db.rpc("get_user_id_by_email", {"lookup_email": email}).execute()
    if result.data:
        return result.data
    return None


async def invite_collaborator(db: Client, invited_by: str, data: dict, work_title: str = ""):
    """Create a collaborator invitation. Auto-links if user exists on platform."""
    data["invited_by"] = invited_by

    existing_user_id = await check_user_exists(db, data["email"])
    if existing_user_id:
        data["collaborator_user_id"] = existing_user_id

    result = db.table("registry_collaborators").insert(data).execute()
    collab = result.data[0] if result.data else None

    if collab and existing_user_id:
        inviter_profile = db.table("profiles").select("full_name").eq("id", invited_by).single().execute()
        inviter_name = (inviter_profile.data or {}).get("full_name") or "Someone"
        await create_notification(
            db,
            user_id=existing_user_id,
            work_id=data.get("work_id"),
            notification_type="invitation",
            title="New collaboration request",
            message=f'{inviter_name} listed you as {data.get("role", "collaborator")} on "{work_title}"',
            metadata={"inviter_name": inviter_name, "work_title": work_title, "role": data.get("role")},
        )

        # Auto-verify: link collaborator's user_id to matching artist entries
        await auto_verify_artist(db, invited_by, data["email"], existing_user_id)

    return collab


async def is_invite_expired(collab: dict) -> bool:
    """Check if an invitation has passed its 48h expiry."""
    from datetime import datetime, timezone
    expires_at = collab.get("expires_at")
    if not expires_at:
        return False
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    return datetime.now(timezone.utc) > expires_at


async def claim_invitation(db: Client, invite_token: str, user_id: str):
    """Link a logged-in user to their invitation by token. Returns (collab, error)."""
    result = (
        db.table("registry_collaborators")
        .select("*")
        .eq("invite_token", invite_token)
        .single()
        .execute()
    )
    if not result.data:
        return None, "not_found"

    collab = result.data
    if collab.get("collaborator_user_id") and collab["collaborator_user_id"] != user_id:
        return None, "already_claimed"

    if await is_invite_expired(collab):
        return None, "expired"

    updated = (
        db.table("registry_collaborators")
        .update({"collaborator_user_id": user_id})
        .eq("id", collab["id"])
        .execute()
    )
    claimed = updated.data[0] if updated.data else collab

    # Auto-verify artist link on claim
    await auto_verify_artist(db, collab["invited_by"], collab["email"], user_id)

    # Notify the work creator that someone claimed their invitation
    work_row = db.table("works_registry").select("user_id, title").eq("id", collab["work_id"]).single().execute()
    if work_row.data:
        await create_notification(
            db,
            user_id=work_row.data["user_id"],
            work_id=collab["work_id"],
            notification_type="invitation",
            title="Invitation claimed",
            message=f'{collab["name"]} accepted your invitation for "{work_row.data["title"]}"',
            metadata={"collaborator_name": collab["name"]},
        )

    return claimed, None


async def confirm_stake(db: Client, collaborator_id: str, user_id: str):
    """Collaborator confirms their stake. Can change from disputed->confirmed
    as long as the work is still in pending_approval status."""
    from datetime import datetime, timezone
    # Verify the work is still pending (allow changing decisions)
    collab_check = (
        db.table("registry_collaborators")
        .select("work_id, status")
        .eq("id", collaborator_id)
        .eq("collaborator_user_id", user_id)
        .single()
        .execute()
    )
    if not collab_check.data:
        return None
    work = (
        db.table("works_registry")
        .select("status")
        .eq("id", collab_check.data["work_id"])
        .single()
        .execute()
    )
    if work.data and work.data["status"] == "registered":
        return None  # Can't change after fully registered

    result = (
        db.table("registry_collaborators")
        .update({
            "status": "confirmed",
            "responded_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", collaborator_id)
        .eq("collaborator_user_id", user_id)
        .execute()
    )
    if result.data:
        collab = result.data[0]
        await _check_auto_register(db, collab["work_id"])
        # Notify the work creator that a collaborator responded
        work_row = db.table("works_registry").select("user_id, title").eq("id", collab["work_id"]).single().execute()
        if work_row.data:
            await create_notification(
                db,
                user_id=work_row.data["user_id"],
                work_id=collab["work_id"],
                notification_type="confirmation",
                title="Stake confirmed",
                message=f'{collab["name"]} confirmed their stake on "{work_row.data["title"]}"',
                metadata={"collaborator_name": collab["name"]},
            )
    return result.data[0] if result.data else None


async def dispute_stake(db: Client, collaborator_id: str, user_id: str, reason: str):
    """Collaborator disputes their stake. Can change from confirmed->disputed
    as long as the work is still in pending_approval status."""
    from datetime import datetime, timezone
    # Verify the work is still pending (allow changing decisions)
    collab_check = (
        db.table("registry_collaborators")
        .select("work_id, status")
        .eq("id", collaborator_id)
        .eq("collaborator_user_id", user_id)
        .single()
        .execute()
    )
    if not collab_check.data:
        return None
    work = (
        db.table("works_registry")
        .select("status")
        .eq("id", collab_check.data["work_id"])
        .single()
        .execute()
    )
    if work.data and work.data["status"] == "registered":
        return None  # Can't change after fully registered

    result = (
        db.table("registry_collaborators")
        .update({
            "status": "disputed",
            "dispute_reason": reason,
            "responded_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", collaborator_id)
        .eq("collaborator_user_id", user_id)
        .execute()
    )
    if result.data:
        collab = result.data[0]
        await check_and_update_work_status(db, collab["work_id"])
        # Notify the work creator about the dispute
        work_row = db.table("works_registry").select("user_id, title").eq("id", collab["work_id"]).single().execute()
        if work_row.data:
            await create_notification(
                db,
                user_id=work_row.data["user_id"],
                work_id=collab["work_id"],
                notification_type="dispute",
                title="Stake disputed",
                message=f'{collab["name"]} disputed their stake on "{work_row.data["title"]}": {reason}',
                metadata={"collaborator_name": collab["name"], "reason": reason},
            )
    return result.data[0] if result.data else None


async def check_and_update_work_status(db: Client, work_id: str):
    collabs = (
        db.table("registry_collaborators")
        .select("status")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .neq("status", "declined")
        .execute()
    )
    if not collabs.data:
        return
    if all(c["status"] == "confirmed" for c in collabs.data):
        db.table("works_registry").update({"status": "registered"}).eq("id", work_id).execute()


async def resend_invitation(db: Client, user_id: str, collaborator_id: str):
    """Resend an expired or pending invitation — generates new token, resets expiry."""
    import uuid
    from datetime import datetime, timezone, timedelta
    # Verify the caller is the inviter
    collab = (
        db.table("registry_collaborators")
        .select("*")
        .eq("id", collaborator_id)
        .eq("invited_by", user_id)
        .single()
        .execute()
    )
    if not collab.data:
        return None, "not_found"
    if collab.data["status"] not in ("invited", "revoked"):
        return None, f"Cannot resend: collaborator status is {collab.data['status']}"

    new_token = str(uuid.uuid4())
    updated = (
        db.table("registry_collaborators")
        .update({
            "invite_token": new_token,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat(),
            "status": "invited",
        })
        .eq("id", collaborator_id)
        .execute()
    )
    return updated.data[0] if updated.data else None, None


async def revoke_collaborator(db: Client, user_id: str, collaborator_id: str):
    """Revoke a collaborator. Deletes associated stakes. Reverts registered works to draft."""
    collab = (
        db.table("registry_collaborators")
        .select("*")
        .eq("id", collaborator_id)
        .eq("invited_by", user_id)
        .single()
        .execute()
    )
    if not collab.data:
        return None

    # Revoke
    db.table("registry_collaborators").update({"status": "revoked"}).eq("id", collaborator_id).execute()

    # Delete associated stake
    if collab.data.get("stake_id"):
        db.table("ownership_stakes").delete().eq("id", collab.data["stake_id"]).execute()

    # Revert registered work to draft
    work = db.table("works_registry").select("status").eq("id", collab.data["work_id"]).single().execute()
    if work.data and work.data["status"] == "registered":
        db.table("works_registry").update({"status": "draft"}).eq("id", collab.data["work_id"]).execute()

    return collab.data


async def submit_for_approval(db: Client, user_id: str, work_id: str):
    work = (
        db.table("works_registry")
        .select("status")
        .eq("id", work_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not work.data:
        return None, "Work not found"
    if work.data["status"] not in ("draft",):
        return None, f"Cannot submit: work is already {work.data['status']}"

    collabs = (
        db.table("registry_collaborators")
        .select("id")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .execute()
    )
    if not collabs.data:
        return None, "No collaborators invited — add at least one before submitting"

    # Only reset collaborators that aren't already confirmed
    db.table("registry_collaborators").update(
        {"status": "invited", "responded_at": None}
    ).eq("work_id", work_id).in_("status", ["declined"]).execute()

    result = (
        db.table("works_registry")
        .update({"status": "pending_approval"})
        .eq("id", work_id)
        .eq("user_id", user_id)
        .execute()
    )
    updated_work = result.data[0] if result.data else None

    # Re-notify all collaborators (especially those reset from disputed->invited)
    if updated_work:
        work_title = updated_work.get("title", "Untitled")
        all_collabs = (
            db.table("registry_collaborators")
            .select("collaborator_user_id, email, name, invite_token")
            .eq("work_id", work_id)
            .eq("status", "invited")
            .execute()
        )
        inviter_profile = db.table("profiles").select("full_name").eq("id", user_id).single().execute()
        inviter_name = (inviter_profile.data or {}).get("full_name") or "The project owner"
        for c in (all_collabs.data or []):
            if c.get("collaborator_user_id"):
                await create_notification(
                    db,
                    user_id=c["collaborator_user_id"],
                    work_id=work_id,
                    notification_type="invitation",
                    title="Work resubmitted for approval",
                    message=f'{inviter_name} resubmitted "{work_title}" for your review',
                    metadata={"inviter_name": inviter_name, "work_title": work_title},
                )
        # Attach collaborator list so the router can re-send emails
        updated_work["_renotify_collabs"] = all_collabs.data or []

    return updated_work, None


async def invite_with_stakes(db: Client, user_id: str, data):
    """Create collaborator + ownership stakes atomically."""
    import secrets
    from datetime import datetime, timedelta, timezone

    work = db.table("works_registry").select("*").eq("id", data.work_id).single().execute()
    if not work.data or work.data["user_id"] != user_id:
        raise PermissionError("Not the work owner")

    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat()
    collab = db.table("registry_collaborators").insert({
        "work_id": data.work_id,
        "invited_by": user_id,
        "email": data.email,
        "name": data.name,
        "role": data.role,
        "status": "invited",
        "invite_token": token,
        "expires_at": expires,
    }).execute()
    collab_row = collab.data[0] if collab.data else None

    created_stakes = []
    for stake in data.stakes:
        s = db.table("ownership_stakes").insert({
            "work_id": data.work_id,
            "user_id": user_id,
            "stake_type": stake.stake_type,
            "holder_name": data.name,
            "holder_role": data.role,
            "percentage": stake.percentage,
            "holder_email": data.email,
        }).execute()
        if s.data:
            created_stakes.append(s.data[0])

    if created_stakes and collab_row:
        db.table("registry_collaborators").update({
            "stake_id": created_stakes[0]["id"]
        }).eq("id", collab_row["id"]).execute()

    # If work is registered, revert to draft (ownership changed)
    if work.data["status"] == "registered":
        db.table("works_registry").update({"status": "draft"}).eq("id", data.work_id).execute()

    return {"collaborator": collab_row, "stakes": created_stakes, "invite_token": token}


async def decline_invitation(db: Client, user_id: str, collaborator_id: str):
    """Decline an invitation. Validates email match."""
    from datetime import datetime, timezone
    collab = db.table("registry_collaborators").select("*").eq("id", collaborator_id).single().execute()
    if not collab.data:
        raise ValueError("Collaborator not found")

    profile = db.table("profiles").select("email").eq("id", user_id).maybe_single().execute()
    user_email = profile.data["email"] if profile.data else None
    if not user_email or user_email.lower() != collab.data["email"].lower():
        raise PermissionError("Email does not match invitation")

    db.table("registry_collaborators").update({
        "status": "declined",
        "collaborator_user_id": user_id,
        "responded_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", collaborator_id).execute()

    # Notify work owner
    work = db.table("works_registry").select("user_id, title").eq("id", collab.data["work_id"]).single().execute()
    if work.data:
        await create_notification(
            db,
            user_id=work.data["user_id"],
            work_id=collab.data["work_id"],
            notification_type="status_change",
            title="Invitation declined",
            message=f'{collab.data["name"]} declined the invitation for "{work.data["title"]}"',
            metadata={"collaborator_name": collab.data["name"]},
        )

    return {"declined": collaborator_id}


async def accept_from_dashboard(db: Client, user_id: str, collaborator_id: str):
    """Claim + confirm atomically from the registry dashboard."""
    from datetime import datetime, timezone
    collab = db.table("registry_collaborators").select("*").eq("id", collaborator_id).single().execute()
    if not collab.data:
        raise ValueError("Collaborator not found")

    profile = db.table("profiles").select("email").eq("id", user_id).maybe_single().execute()
    user_email = profile.data["email"] if profile.data else None
    if not user_email or user_email.lower() != collab.data["email"].lower():
        raise PermissionError("Email does not match invitation")

    db.table("registry_collaborators").update({
        "collaborator_user_id": user_id,
        "status": "confirmed",
        "responded_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", collaborator_id).execute()

    await _check_auto_register(db, collab.data["work_id"])

    # Notify work owner
    work = db.table("works_registry").select("user_id, title").eq("id", collab.data["work_id"]).single().execute()
    if work.data:
        await create_notification(
            db,
            user_id=work.data["user_id"],
            work_id=collab.data["work_id"],
            notification_type="confirmation",
            title="Invitation accepted",
            message=f'{collab.data["name"]} accepted the invitation for "{work.data["title"]}"',
            metadata={"collaborator_name": collab.data["name"]},
        )

    return {"accepted": collaborator_id}


async def get_my_invites(db: Client, user_id: str):
    """Get invites for current user by email match (for Action Required tab)."""
    profile = db.table("profiles").select("email").eq("id", user_id).maybe_single().execute()
    if not profile.data or not profile.data.get("email"):
        return []

    email = profile.data["email"].lower()
    result = (
        db.table("registry_collaborators")
        .select("*, works_registry(id, title, project_id, status)")
        .ilike("email", email)
        .eq("status", "invited")
        .order("invited_at", desc=True)
        .execute()
    )
    return result.data or []


async def _check_auto_register(db: Client, work_id: str):
    """After a confirm, check if ALL collaborators are confirmed -> auto-register."""
    collabs = (
        db.table("registry_collaborators")
        .select("status")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .neq("status", "declined")
        .execute()
    )
    if not collabs.data:
        return
    all_confirmed = all(c["status"] == "confirmed" for c in collabs.data)
    if all_confirmed:
        db.table("works_registry").update({"status": "registered"}).eq("id", work_id).execute()


# ============================================================
# Full Work Data
# ============================================================

async def get_work_full(db: Client, user_id: str, work_id: str):
    work = await get_work(db, user_id, work_id)
    if not work:
        return None
    stakes = await get_stakes(db, user_id, work_id)
    licenses = await get_licenses(db, user_id, work_id)
    agreements = await get_agreements(db, user_id, work_id)
    collaborators = await get_collaborators(db, work_id)
    return {
        **work,
        "stakes": stakes or [],
        "licenses": licenses or [],
        "agreements": agreements or [],
        "collaborators": collaborators or [],
    }


# ============================================================
# Verification — Option C merge logic
# ============================================================

async def auto_verify_artist(db: Client, manager_user_id: str, email: str, collaborator_user_id: str):
    """When a collaborator is linked, find any artist entries the manager created with
    that email and set linked_user_id + verified. This powers Option C merge."""
    from datetime import datetime, timezone

    # Primary match: artist email matches invite email
    artists = (
        db.table("artists")
        .select("id, linked_user_id")
        .eq("user_id", manager_user_id)
        .eq("email", email)
        .execute()
    )

    # Fallback: try the collaborator's actual auth email
    if not artists.data:
        tc = db.table("team_cards").select("email").eq("user_id", collaborator_user_id).single().execute()
        auth_email = (tc.data or {}).get("email", "")
        if auth_email and auth_email.lower() != email.lower():
            artists = (
                db.table("artists")
                .select("id, linked_user_id")
                .eq("user_id", manager_user_id)
                .eq("email", auth_email)
                .execute()
            )
    for artist in (artists.data or []):
        if not artist.get("linked_user_id"):
            db.table("artists").update({
                "linked_user_id": collaborator_user_id,
                "verified": True,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", artist["id"]).execute()

            # Notify the collaborator about verification
            await create_notification(
                db,
                user_id=collaborator_user_id,
                work_id=None,
                notification_type="verification",
                title="Artist profile verified",
                message="Your identity has been verified on an artist profile. Your TeamCard info is now visible to collaborators.",
                metadata={},
            )


async def get_artists_with_teamcards(db: Client, user_id: str):
    """Batch fetch: all artists for a user with TeamCard overlays in 2 queries (not N+1)."""
    artists_result = db.table("artists").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    artists = artists_result.data or []
    if not artists:
        return []

    # Batch fetch all relevant TeamCards in one query
    linked_ids = [a["linked_user_id"] for a in artists if a.get("linked_user_id") and a.get("verified")]
    tc_map = {}
    if linked_ids:
        tc_result = db.table("team_cards").select("*").in_("user_id", linked_ids).execute()
        for tc in (tc_result.data or []):
            visible = tc.get("visible_fields") or []
            filtered = {"user_id": tc["user_id"]}
            for field in visible:
                if field in tc:
                    filtered[field] = tc[field]
            tc_map[tc["user_id"]] = filtered

    results = []
    for a in artists:
        merged = {**a, "teamcard": None}
        if a.get("linked_user_id") and a.get("verified") and a["linked_user_id"] in tc_map:
            merged["teamcard"] = tc_map[a["linked_user_id"]]
        results.append(merged)
    return results


async def get_artist_with_teamcard(db: Client, artist_id: str):
    """Get an artist with Option C merge: if verified, overlay TeamCard fields on shared identity fields."""
    artist = db.table("artists").select("*").eq("id", artist_id).single().execute()
    if not artist.data:
        return None

    a = artist.data
    result = {**a, "teamcard": None}

    if a.get("linked_user_id") and a.get("verified"):
        tc = (
            db.table("team_cards")
            .select("*")
            .eq("user_id", a["linked_user_id"])
            .single()
            .execute()
        )
        if tc.data:
            card = tc.data
            visible = card.get("visible_fields") or []
            # Build filtered teamcard with only visible fields
            filtered_card = {"user_id": card["user_id"]}
            for field in visible:
                if field in card:
                    filtered_card[field] = card[field]
            result["teamcard"] = filtered_card
    return result


# ============================================================
# TeamCard
# ============================================================

async def get_team_card(db: Client, user_id: str):
    result = db.table("team_cards").select("*").eq("user_id", user_id).single().execute()
    return result.data


async def update_team_card(db: Client, user_id: str, data: dict):
    # email cannot be changed
    data.pop("email", None)
    # display_name, first_name, last_name cannot be empty
    for field in ("display_name", "first_name", "last_name"):
        if field in data and not data[field]:
            data.pop(field)
    result = (
        db.table("team_cards")
        .update(data)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def get_collaborator_team_card(db: Client, collaborator_user_id: str):
    """Get a collaborator's TeamCard filtered to only visible fields."""
    tc = db.table("team_cards").select("*").eq("user_id", collaborator_user_id).single().execute()
    if not tc.data:
        return None
    card = tc.data
    visible = card.get("visible_fields") or []
    filtered = {"user_id": card["user_id"], "email": card["email"]}
    for field in visible:
        if field in card:
            filtered[field] = card[field]
    return filtered


# ============================================================
# Notes
# ============================================================

async def get_notes(db: Client, user_id: str, artist_id: str = None, project_id: str = None, folder_id: str = None):
    """List notes. For artist-scoped notes, filters by user_id (private notes).
    For project-scoped notes, does NOT filter by user_id — relies on RLS to allow
    collaborator reads."""
    query = db.table("notes").select("*")
    if artist_id:
        # Artist notes are always private — filter by owner
        query = query.eq("user_id", user_id).eq("artist_id", artist_id)
    elif project_id:
        # Project notes: don't filter by user_id — RLS handles collaborator access
        query = query.eq("project_id", project_id)
    else:
        # No scope — return only user's own notes
        query = query.eq("user_id", user_id)
    if folder_id:
        query = query.eq("folder_id", folder_id)
    result = query.order("pinned", desc=True).order("updated_at", desc=True).execute()
    return result.data


async def get_note(db: Client, user_id: str, note_id: str):
    """Get a single note. RLS handles access control."""
    result = db.table("notes").select("*").eq("id", note_id).single().execute()
    return result.data


async def create_note(db: Client, user_id: str, data: dict):
    data["user_id"] = user_id
    result = db.table("notes").insert(data).execute()
    return result.data[0] if result.data else None


async def update_note(db: Client, user_id: str, note_id: str, data: dict):
    result = (
        db.table("notes")
        .update(data)
        .eq("id", note_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def delete_note(db: Client, user_id: str, note_id: str):
    return db.table("notes").delete().eq("id", note_id).eq("user_id", user_id).execute().data


async def get_folders(db: Client, user_id: str, artist_id: str = None, project_id: str = None):
    """List folders. Same access pattern as get_notes."""
    query = db.table("note_folders").select("*")
    if artist_id:
        query = query.eq("user_id", user_id).eq("artist_id", artist_id)
    elif project_id:
        query = query.eq("project_id", project_id)
    else:
        query = query.eq("user_id", user_id)
    result = query.order("sort_order").order("name").execute()
    return result.data


async def create_folder(db: Client, user_id: str, data: dict):
    data["user_id"] = user_id
    result = db.table("note_folders").insert(data).execute()
    return result.data[0] if result.data else None


async def update_folder(db: Client, user_id: str, folder_id: str, data: dict):
    result = (
        db.table("note_folders")
        .update(data)
        .eq("id", folder_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def delete_folder(db: Client, user_id: str, folder_id: str):
    return db.table("note_folders").delete().eq("id", folder_id).eq("user_id", user_id).execute().data


# ============================================================
# Project About
# ============================================================

async def get_project_about(db: Client, project_id: str):
    result = db.table("projects").select("about_content").eq("id", project_id).single().execute()
    return (result.data or {}).get("about_content", [])


async def update_project_about(db: Client, user_id: str, project_id: str, about_content: list):
    """Update project about content. Verifies the caller owns the project via artist ownership."""
    # Verify ownership: project -> artist -> user_id
    project = (
        db.table("projects")
        .select("id, artist_id")
        .eq("id", project_id)
        .single()
        .execute()
    )
    if not project.data:
        return None
    artist = (
        db.table("artists")
        .select("user_id")
        .eq("id", project.data["artist_id"])
        .single()
        .execute()
    )
    if not artist.data or artist.data["user_id"] != user_id:
        return None  # Not the owner

    result = (
        db.table("projects")
        .update({"about_content": about_content})
        .eq("id", project_id)
        .execute()
    )
    return result.data[0] if result.data else None


# ============================================================
# Notifications
# ============================================================

async def create_notification(
    db: Client, user_id: str, work_id: str,
    notification_type: str, title: str, message: str,
    metadata: dict = None,
):
    db.table("registry_notifications").insert({
        "user_id": user_id,
        "work_id": work_id,
        "type": notification_type,
        "title": title,
        "message": message,
        "metadata": metadata or {},
    }).execute()


async def get_notifications(db: Client, user_id: str, unread_only: bool = False):
    query = db.table("registry_notifications").select("*").eq("user_id", user_id)
    if unread_only:
        query = query.eq("read", False)
    result = query.order("created_at", desc=True).limit(50).execute()
    return result.data


async def mark_notification_read(db: Client, user_id: str, notification_id: str):
    db.table("registry_notifications").update({"read": True}).eq("id", notification_id).eq("user_id", user_id).execute()


async def mark_all_notifications_read(db: Client, user_id: str):
    db.table("registry_notifications").update({"read": True}).eq("user_id", user_id).eq("read", False).execute()


# ============================================================
# Utility
# ============================================================

def compute_document_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()
