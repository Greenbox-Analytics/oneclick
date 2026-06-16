"""Central work-access resolver. The backend uses the service-role Supabase client
(RLS bypassed), so THIS is the authoritative authorization layer for registry reads
and writes. RLS mirrors it (a later migration) for direct frontend reads."""

from dataclasses import dataclass, field

from supabase import Client

_ELEVATED_PROJECT_ROLES = {"owner", "admin"}

@dataclass
class WorkAccess:
    work_role: str = "none"  # owner | admin | viewer | none
    project_role: str = "none"  # owner | admin | editor | viewer | none
    my_collaborator_id: str | None = None
    is_project_member: bool = False
    can_see_full_ownership: bool = False
    visible_stake_ids: set[str] = field(default_factory=set)
    visible_file_ids: set[str] = field(default_factory=set)
    visible_audio_ids: set[str] = field(default_factory=set)
    visible_license_ids: set[str] = field(default_factory=set)
    visible_agreement_ids: set[str] = field(default_factory=set)
    _all_visible: bool = False  # internal: elevated/project-member see everything

    @property
    def elevated(self) -> bool:
        return self.work_role in ("owner", "admin") or self.project_role in _ELEVATED_PROJECT_ROLES

    @property
    def can_view(self) -> bool:
        return self.work_role != "none" or self.project_role != "none"

    @property
    def can_edit(self) -> bool:
        return self.elevated

    @property
    def can_manage(self) -> bool:
        return self.elevated

    @property
    def can_delete(self) -> bool:
        return self.elevated

    def all_visible(self) -> bool:
        return self._all_visible


async def get_work_access(db: Client, user_id: str, work_id: str) -> WorkAccess:
    # maybe_single (not single): a missing/duplicate work returns empty data instead of
    # raising, so the gate fails CLOSED (empty WorkAccess => can_view False) rather than 500.
    work = db.table("works_registry").select("user_id, project_id").eq("id", work_id).maybe_single().execute()
    if not work or not work.data:
        return WorkAccess()
    owner_id = work.data["user_id"]
    project_id = work.data.get("project_id")

    wa = WorkAccess()

    # project role
    if project_id:
        pm = db.table("project_members").select("role").eq("project_id", project_id).eq("user_id", user_id).execute()
        if pm.data:
            wa.project_role = pm.data[0]["role"]
            wa.is_project_member = True

    # work role
    if user_id == owner_id:
        wa.work_role = "owner"
    else:
        collab = (
            db.table("registry_collaborators")
            .select("id, access_level, status")
            .eq("work_id", work_id)
            .eq("collaborator_user_id", user_id)
            .eq("status", "confirmed")
            .execute()
        )
        if collab.data:
            row = collab.data[0]
            wa.my_collaborator_id = row["id"]
            wa.work_role = "admin" if row.get("access_level") == "admin" else "viewer"

    if not wa.can_view:
        return wa

    # Elevated or any project member => sees everything
    if wa.elevated or wa.is_project_member:
        wa._all_visible = True
        wa.can_see_full_ownership = True
        return wa

    # work-only viewer: closed-by-default, gated by grants + own/owner stakes
    grants = (
        db.table("registry_access_grants")
        .select("resource_type, resource_id")
        .eq("collaborator_id", wa.my_collaborator_id)
        .execute()
    ).data or []
    by_type: dict[str, set[str]] = {}
    for g in grants:
        if g["resource_type"] == "ownership_breakdown":
            wa.can_see_full_ownership = True
        elif g.get("resource_id"):
            by_type.setdefault(g["resource_type"], set()).add(g["resource_id"])

    wa.visible_file_ids = by_type.get("project_file", set())
    wa.visible_audio_ids = by_type.get("audio_file", set())
    wa.visible_license_ids = by_type.get("license", set())
    wa.visible_agreement_ids = by_type.get("agreement", set())

    stakes = (
        db.table("ownership_stakes").select("id, is_owner_stake, collaborator_id").eq("work_id", work_id).execute()
    ).data or []
    if wa.can_see_full_ownership:
        wa.visible_stake_ids = {s["id"] for s in stakes}
    else:
        wa.visible_stake_ids = {
            s["id"] for s in stakes if s.get("is_owner_stake") or s.get("collaborator_id") == wa.my_collaborator_id
        }
    return wa
