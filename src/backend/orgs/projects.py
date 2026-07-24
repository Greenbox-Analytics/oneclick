"""Owner-consented project<->org linking (Licensing Phase C, spec §6, Task 2)
+ the provenance-scoped revocation helper Task 3/4 also share + Task 3's
org-admin-driven project membership management (grant/adjust/revoke seat
access on a linked project through ordinary `project_members` writes).

Ground truth stays `project_members` (spec §6's architecture note): linking a
project to an org does NOT itself grant anyone access. It only (a) records
which ONE org (rule 8 — `org_project_links.UNIQUE(project_id)`) may
subsequently manage this project's memberships (Task 3, via ordinary
`project_members` writes), and (b) later phases (Task 5-7) derive billing/
caps from it. This module never writes a `project_members` row itself except
via `revoke_org_granted_memberships`'s DELETEs.

Rule 1 (linking = consent): only the project OWNER, who must ALSO hold an
ACTIVE seat in an ACTIVE org, may create or remove the link. Org admins can
view links (`list_org_projects`) and manage seat access on linked projects
(Task 3), but can never link/unlink a project themselves — consent is the
owner's alone to give or withdraw.

No-existence-oracle discipline (mirrors `orgs.authz.require_member` and the
spec §5 billing-context posture): this module never lets a caller distinguish
"resource doesn't exist" from "resource exists but you're not authorized" by
the 404 body. Two independent pairs collapse to one body each:
  - unknown project OR caller isn't its owner -> "Project not found"
  - unknown org OR no ACTIVE seat OR org isn't ACTIVE (pending/suspended/
    archived) -> "Organization not found"
A PENDING org 404s here even for its own active seat holders: an org that
confers nothing yet cannot accrue access grants (Task 2 AC — distinct from
`_resolve_context`'s more permissive ambient-billing-context posture, which
deliberately DOES allow a pending org as a preference so it "just starts
working" once activated; linking is a stronger, deliberate action and has no
equivalent reason to tolerate a not-yet-real org).

Task 3's two membership endpoints extend the SAME no-leak posture with a
third pair, collapsed the same way:
  - unknown member_id OR wrong org OR not an ACTIVE seat -> "Member not found"
  - unknown project OR not linked to THIS org (including linked to a
    DIFFERENT org — an org admin must never learn that some project is
    linked elsewhere) -> "Project not found"

Wrap-vs-direct-insert decision (Task 3): `projects.service.add_member`,
`update_member_role`, and `remove_member` all gate authorization on the
CALLER holding a `project_members` row with role owner/admin
(`get_user_role(caller_id, project_id)`). An org admin acting through this
module is authorized by ORG admin status (`orgs.authz.require_admin`), NOT
by project membership, and very often holds no `project_members` row on the
linked project at all — that is the entire point of Task 3 (admins manage
access without being individually invited to every linked project). Calling
those functions with the real org-admin id would spuriously deny legitimate
actions; calling them with a substituted caller id (e.g. the project owner's,
who always passes the check) would happen to work today only because none of
the three functions persist an actor/audit column, but that is a fragile
impersonation hack that a future audit-trail addition to `projects/service.py`
could silently corrupt. `projects/service.py` is also out of scope for this
task (touch list). So: the INSERT/UPDATE writes below go directly to
`project_members` rather than through `add_member`/`update_member_role`. The
only invariant those functions add BEYOND the caller-authorization check —
role must be one of admin/editor/viewer, never 'owner' — is already enforced
more strongly here by the endpoint's Literal-typed role payload
(`orgs.models.ProjectMemberRoleUpdate`) plus this module's own explicit
owner-target check (`ProjectMemberIsOwnerError`), so nothing is reimplemented,
only referenced. The DELETE path is different: it reuses
`revoke_org_granted_memberships` (this module's OWN Task 2/4 helper, not
`projects/service.py`) exactly as the plan directs, since that helper already
is the correct provenance-scoped delete and predates this task.
"""

from fastapi import HTTPException
from postgrest.exceptions import APIError
from supabase import Client

from orgs import authz
from orgs.service import _resolve_user_email
from projects.service import get_user_role


class ProjectAlreadyLinkedError(Exception):
    """`org_project_links` already has a row for this project — to this org
    or a different one (rule 8: UNIQUE(project_id), one org per project).
    Mapped by the router to 409 with the rule-8 copy."""


class ProjectMemberIsOwnerError(Exception):
    """Task 3: the target member_id resolves to the project's OWNER row.
    Owners are never org-granted (rule 3 — belt and suspenders with
    `one_owner_per_project`), and their access can never be adjusted or
    revoked by an org admin, no matter how the project got linked. Mapped by
    the router to 409 with this exact copy — deliberately distinct from the
    generic organic no-op, so the caller understands WHY this particular
    member can't be managed rather than being told "they have organic
    access" (true but misleading for the owner specifically)."""


async def _require_project_owner(db: Client, user_id: str, project_id: str) -> None:
    """404s identically for "no such project" and "caller isn't the owner" —
    reuses `projects.service.get_user_role` (the project-role predicate;
    returns None for both an unknown project and a caller with no
    `project_members` row at all) rather than reimplementing ownership."""
    role = await get_user_role(db, user_id, project_id)
    if role != "owner":
        raise HTTPException(status_code=404, detail="Project not found")


def _require_active_org_seat(db: Client, user_id: str, org_id: str) -> None:
    """404s identically for "no such org", "no ACTIVE seat", and "org isn't
    ACTIVE" (pending/suspended/archived). `authz.is_org_member` already only
    counts ACTIVE org_members rows and returns False for a nonexistent org
    (no matching row either way) — this extends it with the org-row ACTIVE
    check that linking specifically requires (an org-membership check alone
    would let a PENDING org's own seat holder link, which Task 2's AC
    forbids: a pending org confers nothing and must not be able to accrue
    access grants)."""
    if not authz.is_org_member(db, user_id, org_id):
        raise HTTPException(status_code=404, detail="Organization not found")

    org_res = db.table("organizations").select("id, status, archived_at").eq("id", org_id).maybe_single().execute()
    org = org_res.data if org_res else None
    if not org or org.get("archived_at") is not None or org.get("status") != "active":
        raise HTTPException(status_code=404, detail="Organization not found")


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


async def link_project(db: Client, user_id: str, org_id: str, project_id: str) -> dict:
    """POST /orgs/{org_id}/projects/{project_id}/link (Task 2, rule 1).

    Both gates must pass:
      (a) caller is the project OWNER (`_require_project_owner`).
      (b) caller holds an ACTIVE seat in an ACTIVE org (`_require_active_org_seat`).
    Ownership is checked first only because it is the more common failure
    path in practice — neither check leaks into the other's 404 body (see
    module docstring), so the order carries no security weight.

    409s (rule 8) if the project already has ANY `org_project_links` row,
    whether it points at this org or a different one — a project links to at
    most ONE org, ever; re-linking to a different org requires an explicit
    unlink first (Task 2's DELETE endpoint), never an implicit overwrite.

    Rule 10 is NOT enforced here — it's a disclosure, not a gate. Linking
    itself grants no one access and moves no money; it only (1) lets this
    org's admins subsequently manage `project_members` rows on this project
    (Task 3) and (2) once resource-derived billing ships (Task 5-7), makes
    the LINKED org the billing/caps authority for anyone who both works on
    this project AND already holds an active seat in that org — INCLUDING a
    seat-holder with organic (non-org-granted) access, who never asked the
    admin for anything. That is deliberate (billing population ⊇
    access-granted population) and is disclosed to the owner at the consent
    moment (Task 8's link copy), not softened or gated here.
    """
    await _require_project_owner(db, user_id, project_id)
    _require_active_org_seat(db, user_id, org_id)

    existing = db.table("org_project_links").select("id, org_id").eq("project_id", project_id).maybe_single().execute()
    if existing and existing.data:
        raise ProjectAlreadyLinkedError("This project is already linked to an organization — unlink it first.")

    # The probe above is not atomic with the INSERT — a concurrent duplicate
    # (double-submit) can pass both probes and hit UNIQUE(project_id) at the
    # DB. That's still rule 8 doing its job, so map it to the same 409 copy.
    try:
        res = (
            db.table("org_project_links")
            .insert({"org_id": org_id, "project_id": project_id, "linked_by": user_id})
            .execute()
        )
    except APIError as e:
        if getattr(e, "code", None) == "23505":
            raise ProjectAlreadyLinkedError(
                "This project is already linked to an organization — unlink it first."
            ) from e
        raise
    return res.data[0] if res.data else {"org_id": org_id, "project_id": project_id}


async def unlink_project(db: Client, user_id: str, org_id: str, project_id: str) -> dict:
    """DELETE /orgs/{org_id}/projects/{project_id}/link — project OWNER only
    (rule 1: org admins can never unlink someone else's project, no matter
    what's happening with the org).

    Order: owner-check first (404 "Project not found" for non-owners,
    matching `link_project`'s posture and never leaking this project's link
    state to a non-owner prober). Once ownership is established, the link
    itself must exist AND point at THIS org — a project not currently linked
    to `org_id` 404s (distinct, informative copy is fine here: the caller
    already owns the project, so this isn't an anonymous-prober oracle
    concern, just accurate feedback on their own resource's state).

    Unlink REVOKES (rule 3) before it deletes the link row: every
    `project_members` row THIS org granted on THIS project is deleted via
    `revoke_org_granted_memberships` first, so a crash between the two steps
    never leaves a project both access-revoked-but-still-linked in a way that
    contradicts rule 3 — the access removal and the link removal are ordered
    so the WORSE failure mode (link row survives revocation) is the one that
    can happen, never the reverse (link gone but org-granted rows survive).

    Returns `{"revoked": n}` for Task 8's unlink confirmation copy.
    """
    await _require_project_owner(db, user_id, project_id)

    link_res = db.table("org_project_links").select("id, org_id").eq("project_id", project_id).maybe_single().execute()
    link = link_res.data if link_res else None
    if not link or link.get("org_id") != org_id:
        raise HTTPException(status_code=404, detail="This project is not linked to this organization.")

    revoked = revoke_org_granted_memberships(db, org_id, project_id=project_id)

    db.table("org_project_links").delete().eq("id", link["id"]).execute()

    return {"revoked": revoked}


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

    links_res = (
        db.table("org_project_links")
        .select("id, project_id, created_at")
        .eq("org_id", org_id)
        .order("created_at")
        .execute()
    )
    links = links_res.data or []
    if not links:
        return []

    project_ids = [link["project_id"] for link in links]

    projects_res = db.table("projects").select("id, name").in_("id", project_ids).execute()
    project_by_id = {p["id"]: p for p in (projects_res.data or [])}

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
    for link in links:
        project_id = link["project_id"]
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
                "linkedAt": link.get("created_at"),
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
    no-existence-oracle posture as `_require_project_owner`/
    `_require_active_org_seat` above. Returns the seat's user_id."""
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


def _require_project_linked_to_org(db: Client, org_id: str, project_id: str) -> None:
    """404s identically whether the project doesn't exist, isn't linked to
    any org, or is linked to a DIFFERENT org — filtering on BOTH project_id
    AND org_id in one query means "linked elsewhere" and "not linked at all"
    are indistinguishable to the caller, which is the point: an org admin
    must never learn that some project is linked to another org."""
    res = (
        db.table("org_project_links")
        .select("id")
        .eq("project_id", project_id)
        .eq("org_id", org_id)
        .maybe_single()
        .execute()
    )
    row = res.data if res else None
    if not row:
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
    _require_project_linked_to_org(db, org_id, project_id)

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
    _require_project_linked_to_org(db, org_id, project_id)

    row = _get_project_member_row(db, project_id, target_user_id)

    if row and row.get("role") == "owner":
        raise ProjectMemberIsOwnerError("The project owner's access can't be managed by the organization.")

    if row and row.get("org_id") != org_id:
        return {"status": "organic", "detail": _ORGANIC_NOOP_DETAIL}

    revoked = revoke_org_granted_memberships(db, org_id, user_id=target_user_id, project_id=project_id)
    return {"status": "revoked", "revoked": revoked}
