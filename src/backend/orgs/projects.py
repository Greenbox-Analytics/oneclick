"""Org-admin-driven project membership management (grant/adjust/revoke seat
access on a team-owned project through ordinary `project_members` writes) plus
the provenance-scoped revocation helper the offboard/archive paths share.

WHICH PROJECTS AN ORG MAY MANAGE comes from artist ownership: a project belongs
to an org when its ARTIST does (`projects.artist_id -> artists.team_id`). The
`org_project_links` table that used to answer this one project at a time was
retired in 20260804000001 — one edge, one answer, for both access and billing.

Ground truth for ACCESS stays `project_members` (spec §6's architecture note):
owning the artist does not itself write anyone a membership row. It records
which ONE org may subsequently manage this project's memberships, and billing/
caps derive from the same edge. This module never writes a `project_members`
row itself except via `revoke_org_granted_memberships`'s DELETEs.

No-existence-oracle discipline (mirrors `orgs.authz.require_member` and the
spec §5 billing-context posture): this module never lets a caller distinguish
"resource doesn't exist" from "resource exists but you're not authorized" by
the 404 body. Two pairs collapse to one body each:
  - unknown member_id OR wrong org OR not an ACTIVE seat -> "Member not found"
  - unknown project OR its artist belongs to no org OR to a DIFFERENT org (an
    org admin must never learn that some project belongs elsewhere)
    -> "Project not found"

Wrap-vs-direct-insert decision: `projects.service.add_member`,
`update_member_role`, and `remove_member` all gate authorization on the
CALLER holding a `project_members` row with role owner/admin
(`get_user_role(caller_id, project_id)`). An org admin acting through this
module is authorized by ORG admin status (`orgs.authz.require_admin`), NOT
by project membership, and very often holds no `project_members` row on the
project at all — that is the entire point (admins manage access without being
individually invited to every project). Calling those functions with the real
org-admin id would spuriously deny legitimate actions; calling them with a
substituted caller id (e.g. the project owner's, who always passes the check)
would happen to work today only because none of the three functions persist an
actor/audit column, but that is a fragile impersonation hack that a future
audit-trail addition to `projects/service.py` could silently corrupt. So: the
INSERT/UPDATE writes below go directly to `project_members` rather than through
`add_member`/`update_member_role`. The only invariant those functions add
BEYOND the caller-authorization check — role must be one of admin/editor/viewer,
never 'owner' — is already enforced more strongly here by the endpoint's
Literal-typed role payload (`orgs.models.ProjectMemberRoleUpdate`) plus this
module's own explicit owner-target check (`ProjectMemberIsOwnerError`), so
nothing is reimplemented, only referenced. The DELETE path reuses
`revoke_org_granted_memberships` (this module's OWN helper), which already is
the correct provenance-scoped delete.
"""

from fastapi import HTTPException
from supabase import Client

from orgs import authz
from orgs.service import _resolve_user_email


class ProjectMemberIsOwnerError(Exception):
    """Task 3: the target member_id resolves to the project's OWNER row.
    Owners are never org-granted (rule 3 — belt and suspenders with
    `one_owner_per_project`), and their access can never be adjusted or
    revoked by an org admin, no matter how the project got linked. Mapped by
    the router to 409 with this exact copy — deliberately distinct from the
    generic organic no-op, so the caller understands WHY this particular
    member can't be managed rather than being told "they have organic
    access" (true but misleading for the owner specifically)."""


def revoke_org_granted_memberships(
    sb: Client, org_id: str, *, user_id: str | None = None, project_id: str | None = None
) -> int:
    """Rule 3: subtractive, provenance-scoped revocation — the ONE
    implementation every offboard/unlink/archive path (Task 2 unlink here;
    Task 4's `_offboard`/`archive_org`/account-deletion teardown) shares, so
    "delete only the rows this org granted" never drifts across call sites.

    DELETEs `project_members` rows WHERE `org_id` = this org, optionally
    narrowed further by `user_id` (a single seat's offboarding) and/or
    `project_id` (a single project's unlink) — both narrowing filters are
    AND'd when both are given. Organic rows (`org_id IS NULL`) and rows
    stamped by a DIFFERENT org are never touched: the `org_id` filter is the
    entire safety mechanism, and it is unconditional (never optional) in
    every call. The owner row is never a target either way — owners are
    never org-granted (rule 3 — belt and suspenders with `one_owner_per_project`).

    Returns the number of rows deleted (the caller-facing "N teammates lost
    access" count — Task 2's `{"revoked": n}` response, Task 8's unlink
    confirmation copy).
    """
    query = sb.table("project_members").delete().eq("org_id", org_id)
    if user_id is not None:
        query = query.eq("user_id", user_id)
    if project_id is not None:
        query = query.eq("project_id", project_id)
    result = query.execute()
    return len(result.data or [])


async def list_org_projects(db: Client, user_id: str, org_id: str) -> list[dict]:
    """GET /orgs/{org_id}/projects — org ADMIN console view (Task 2 AC 3).

    One `project_members` round trip answers two questions at once (no
    per-project N+1): which row is the OWNER's (role='owner', for the
    ownerEmail lookup) and how many rows THIS org granted (`org_id` = this
    org — owner rows are never org-granted, so they never inflate the
    count). `orgGrantedMemberCount` counts seats THIS org's admins are
    managing on the project, not every collaborator who can see it (organic
    collaborators are invisible to this rollup by design — they aren't the
    org's business).

    ownerEmail resolution: prefer `org_members.email` when the owner
    themselves holds an ACTIVE seat in this org (denormalized column, same
    one `get_org_usage` reads — zero extra lookup cost for the common
    "owner is also a seat-holder" case); otherwise fall back to
    `_resolve_user_email` (one auth-admin lookup per project whose owner
    isn't a seat of this org).
    """
    authz.require_admin(db, user_id, org_id)

    artists_res = db.table("artists").select("id, transferred_at").eq("team_id", org_id).execute()
    artists = artists_res.data or []
    if not artists:
        return []
    transferred_at_by_artist = {a["id"]: a.get("transferred_at") for a in artists}

    projects_res = (
        db.table("projects")
        .select("id, name, artist_id")
        .in_("artist_id", list(transferred_at_by_artist))
        .order("created_at")
        .execute()
    )
    projects = projects_res.data or []
    if not projects:
        return []
    project_by_id = {p["id"]: p for p in projects}
    project_ids = list(project_by_id)

    members_res = (
        db.table("project_members").select("project_id, user_id, role, org_id").in_("project_id", project_ids).execute()
    )
    members = members_res.data or []
    owner_user_id_by_project = {m["project_id"]: m["user_id"] for m in members if m.get("role") == "owner"}
    granted_count_by_project: dict[str, int] = {}
    for m in members:
        if m.get("org_id") == org_id:
            pid = m["project_id"]
            granted_count_by_project[pid] = granted_count_by_project.get(pid, 0) + 1

    owner_user_ids = list({uid for uid in owner_user_id_by_project.values() if uid})
    seat_email_by_user_id: dict[str, str] = {}
    if owner_user_ids:
        seats_res = (
            db.table("org_members")
            .select("user_id, email, status")
            .eq("org_id", org_id)
            .in_("user_id", owner_user_ids)
            .execute()
        )
        for row in seats_res.data or []:
            if row.get("status") == "active" and row.get("email"):
                seat_email_by_user_id[row["user_id"]] = row["email"]

    out = []
    for project_id in project_ids:
        project = project_by_id.get(project_id, {})
        owner_user_id = owner_user_id_by_project.get(project_id)
        owner_email = seat_email_by_user_id.get(owner_user_id) if owner_user_id else None
        if not owner_email and owner_user_id:
            owner_email = _resolve_user_email(db, owner_user_id)
        out.append(
            {
                "projectId": project_id,
                "name": project.get("name"),
                "ownerEmail": owner_email,
                # When the org came to own this project: the moment its ARTIST
                # was transferred. Field name kept so the console contract does
                # not change under a rename that carries no new information.
                "linkedAt": transferred_at_by_artist.get(project.get("artist_id")),
                "orgGrantedMemberCount": granted_count_by_project.get(project_id, 0),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Admin membership management on linked projects (Task 3, rules 2-3). See the
# module docstring's "Wrap-vs-direct-insert decision" section for why these
# write directly to `project_members` instead of calling
# `projects.service.add_member`/`update_member_role`.
# ---------------------------------------------------------------------------

_ORGANIC_NOOP_DETAIL = "This member already has access on their own — the organization doesn't manage it."


def _require_active_org_seat_member(db: Client, org_id: str, member_id: str) -> str:
    """404s identically whether member_id doesn't exist at all, belongs to a
    DIFFERENT org, or isn't an ACTIVE seat (suspended/removed) — same
    no-existence-oracle posture as `_require_project_in_org` below.
    Returns the seat's user_id."""
    res = (
        db.table("org_members")
        .select("user_id")
        .eq("id", member_id)
        .eq("org_id", org_id)
        .eq("status", "active")
        .maybe_single()
        .execute()
    )
    row = res.data if res else None
    if not row:
        raise HTTPException(status_code=404, detail="Member not found")
    return row["user_id"]


def _require_project_in_org(db: Client, org_id: str, project_id: str) -> None:
    """404s identically whether the project doesn't exist, its artist belongs to
    no org, or its artist belongs to a DIFFERENT org — the same
    no-existence-oracle stance the link check had, on the edge that replaced it:
    an org admin must never learn that some project belongs to another org.

    `artists!inner` makes the embed a join rather than a nullable expansion, so
    a project with no artist row drops out here instead of arriving as None."""
    res = db.table("projects").select("id, artists!inner(team_id)").eq("id", project_id).maybe_single().execute()
    row = res.data if res else None
    artist = (row or {}).get("artists") or {}
    if isinstance(artist, list):  # PostgREST returns a list for some embeds
        artist = artist[0] if artist else {}
    if not row or artist.get("team_id") != org_id:
        raise HTTPException(status_code=404, detail="Project not found")


def _get_project_member_row(db: Client, project_id: str, user_id: str) -> dict | None:
    """The one read every branch of both Task 3 endpoints needs: does a
    `project_members` row already exist for (project, target user), and if
    so what's its role/org_id? Neither `projects.service.get_user_role` (role
    only, no org_id) nor any other existing helper exposes org_id, so this is
    a fresh SELECT rather than a reuse of anything."""
    res = (
        db.table("project_members")
        .select("id, role, org_id")
        .eq("project_id", project_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    return res.data if res else None


async def set_org_project_member_role(
    db: Client, user_id: str, org_id: str, project_id: str, member_id: str, role: str
) -> dict:
    """PUT /orgs/{org_id}/projects/{project_id}/members/{member_id} (Task 3
    AC 1, org ADMIN only).

    Order: org-admin authz first, then the two no-leak 404 gates (target
    seat must be ACTIVE in this org; project must be linked to this org),
    then the existing-row decision tree, evaluated in THIS order:

      1. row.role == 'owner'                -> 409 ProjectMemberIsOwnerError
         (MUST be checked before #4: an owner row always has org_id IS NULL
         — owners are never org-granted — so if the organic check ran first
         an owner target would silently match the generic organic no-op
         instead of this specific, more informative 409)
      2. no existing row                    -> INSERT, org_id = this org
         (rule 2: the ONLY place provenance is stamped)
      3. row.org_id == this org             -> UPDATE role, provenance kept
      4. row.org_id is NULL or another org  -> NO-OP, {"status": "organic"}
         (rule 2: never overwrite organic or another org's grant)
    """
    authz.require_admin(db, user_id, org_id)

    target_user_id = _require_active_org_seat_member(db, org_id, member_id)
    _require_project_in_org(db, org_id, project_id)

    row = _get_project_member_row(db, project_id, target_user_id)

    if row and row.get("role") == "owner":
        raise ProjectMemberIsOwnerError("The project owner's access can't be managed by the organization.")

    if not row:
        # No projects.service function fits: add_member is email/invite
        # oriented, gates on the CALLER already being a project member
        # (see module docstring), and has no org_id parameter at all — so
        # this is a direct INSERT, the ONLY place `org_id` is stamped.
        inserted = (
            db.table("project_members")
            .insert({"project_id": project_id, "user_id": target_user_id, "role": role, "org_id": org_id})
            .execute()
        )
        member = inserted.data[0] if inserted.data else None
        return {"status": "granted", "member": member}

    if row.get("org_id") == org_id:
        # Role change on a row THIS org already granted — provenance is
        # untouched (only `role` is written). See module docstring for why
        # this is a direct UPDATE rather than a call to
        # `projects.service.update_member_role`.
        updated = (
            db.table("project_members")
            .update({"role": role})
            .eq("id", row["id"])
            .eq("project_id", project_id)
            .execute()
        )
        member = updated.data[0] if updated.data else None
        return {"status": "granted", "member": member}

    # Organic row (org_id IS NULL) or granted by a DIFFERENT org: rule 2 —
    # never overwrite. No-op, not an error.
    return {"status": "organic", "detail": _ORGANIC_NOOP_DETAIL}


async def remove_org_project_member(db: Client, user_id: str, org_id: str, project_id: str, member_id: str) -> dict:
    """DELETE /orgs/{org_id}/projects/{project_id}/members/{member_id} (Task
    3 AC 2, org ADMIN only).

    Reuses `revoke_org_granted_memberships` with BOTH narrowing filters
    (project_id AND user_id, on top of the unconditional org_id filter) —
    the plan's explicit instruction, and the same helper Task 2's
    `unlink_project`/Task 4's offboard-and-archive paths share, so "delete
    only the rows this org granted" never drifts across call sites.

    The owner and organic branches mirror `set_org_project_member_role`'s
    decision tree exactly (owner checked first, for the same reason): a row
    that exists but isn't THIS org's grant (organic, or another org's) is
    reported with the same `{"status": "organic"}` shape rather than
    deleted. A target with NO `project_members` row at all has nothing to
    revoke either way — `revoke_org_granted_memberships` is called and
    reports `revoked: 0`, which is accurate (there was no org-granted access
    to remove) rather than mislabeled as "organic" (that copy asserts the
    member DOES have independent access, which would be false here).
    """
    authz.require_admin(db, user_id, org_id)

    target_user_id = _require_active_org_seat_member(db, org_id, member_id)
    _require_project_in_org(db, org_id, project_id)

    row = _get_project_member_row(db, project_id, target_user_id)

    if row and row.get("role") == "owner":
        raise ProjectMemberIsOwnerError("The project owner's access can't be managed by the organization.")

    if row and row.get("org_id") != org_id:
        return {"status": "organic", "detail": _ORGANIC_NOOP_DETAIL}

    revoked = revoke_org_granted_memberships(db, org_id, user_id=target_user_id, project_id=project_id)
    return {"status": "revoked", "revoked": revoked}
