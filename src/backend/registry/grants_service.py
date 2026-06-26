"""Per-collaborator visibility grants: add/remove/list + access-level. Service-role only
(RLS bypassed); authorization is enforced by the endpoints via get_work_access."""

from supabase import Client

_VALID_TYPES = {"project_file", "audio_file", "license", "agreement", "ownership_breakdown"}


async def _resource_belongs_to_work(db: Client, work_id, rtype, rid) -> bool:
    if rtype == "ownership_breakdown":
        return rid is None
    if rid is None:
        return False
    if rtype == "project_file":
        r = db.table("work_files").select("id").eq("work_id", work_id).eq("file_id", rid).execute()
        return bool(r.data)
    if rtype == "audio_file":
        r = db.table("work_audio_links").select("id").eq("work_id", work_id).eq("audio_file_id", rid).execute()
        return bool(r.data)
    if rtype == "license":
        r = db.table("licensing_rights").select("id").eq("id", rid).eq("work_id", work_id).execute()
        return bool(r.data)
    if rtype == "agreement":
        r = db.table("registry_agreements").select("id").eq("id", rid).eq("work_id", work_id).execute()
        return bool(r.data)
    return False


async def add_grant(db: Client, collaborator_id, work_id, rtype, rid, granted_by) -> bool:
    if rtype not in _VALID_TYPES:
        raise ValueError(f"bad resource_type {rtype}")
    if not await _resource_belongs_to_work(db, work_id, rtype, rid):
        raise ValueError("resource does not belong to this work")
    q = (
        db.table("registry_access_grants")
        .select("id")
        .eq("collaborator_id", collaborator_id)
        .eq("resource_type", rtype)
    )
    q = q.is_("resource_id", "null") if rid is None else q.eq("resource_id", rid)
    if q.execute().data:
        return False  # idempotent: already granted
    db.table("registry_access_grants").insert(
        {
            "work_id": work_id,
            "collaborator_id": collaborator_id,
            "resource_type": rtype,
            "resource_id": rid,
            "granted_by": granted_by,
        }
    ).execute()
    return True


async def remove_grant(db: Client, collaborator_id, rtype, rid):
    q = db.table("registry_access_grants").delete().eq("collaborator_id", collaborator_id).eq("resource_type", rtype)
    q = q.is_("resource_id", "null") if rid is None else q.eq("resource_id", rid)
    return q.execute().data


async def get_grant_matrix(db: Client, work_id):
    collabs = (
        db.table("registry_collaborators")
        .select("id, name, email, role, access_level, status")
        .eq("work_id", work_id)
        .neq("status", "revoked")
        .execute()
    ).data or []
    grants = (
        db.table("registry_access_grants")
        .select("collaborator_id, resource_type, resource_id")
        .eq("work_id", work_id)
        .execute()
    ).data or []
    by_collab: dict[str, list] = {}
    for g in grants:
        by_collab.setdefault(g["collaborator_id"], []).append(g)
    return {"collaborators": collabs, "grants_by_collaborator": by_collab}


async def set_access_level(db: Client, collaborator_id, access_level):
    if access_level not in ("viewer", "admin"):
        raise ValueError("bad access_level")
    db.table("registry_collaborators").update({"access_level": access_level}).eq("id", collaborator_id).execute()


async def set_work_role(db: Client, collaborator_id, role):
    if not role or not role.strip():
        raise ValueError("role required")
    db.table("registry_collaborators").update({"role": role.strip()}).eq("id", collaborator_id).execute()
