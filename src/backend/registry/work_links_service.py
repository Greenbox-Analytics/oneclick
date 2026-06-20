from supabase import Client

from registry.access import get_work_access


async def get_work_files(db: Client, user_id: str, work_id: str):
    wa = await get_work_access(db, user_id, work_id)
    if not wa.can_view:
        raise PermissionError("denied")
    result = db.table("work_files").select("*, project_files(*)").eq("work_id", work_id).execute()
    return result.data or []


async def link_file_to_work(db: Client, user_id: str, work_id: str, file_id: str):
    wa = await get_work_access(db, user_id, work_id)
    if not wa.can_edit:
        raise PermissionError("denied")
    result = db.table("work_files").insert({"work_id": work_id, "file_id": file_id}).execute()
    return result.data[0] if result.data else None


async def unlink_file_from_work(db: Client, user_id: str, work_id: str, link_id: str):
    wa = await get_work_access(db, user_id, work_id)
    if not wa.can_edit:
        raise PermissionError("denied")
    db.table("work_files").delete().eq("id", link_id).eq("work_id", work_id).execute()
    return {"deleted": link_id}


async def get_work_audio(db: Client, user_id: str, work_id: str):
    wa = await get_work_access(db, user_id, work_id)
    if not wa.can_view:
        raise PermissionError("denied")
    result = db.table("work_audio_links").select("*, audio_files(*)").eq("work_id", work_id).execute()
    return result.data or []


async def link_audio_to_work(db: Client, user_id: str, work_id: str, audio_file_id: str):
    wa = await get_work_access(db, user_id, work_id)
    if not wa.can_edit:
        raise PermissionError("denied")
    result = db.table("work_audio_links").insert({"work_id": work_id, "audio_file_id": audio_file_id}).execute()
    return result.data[0] if result.data else None


async def unlink_audio_from_work(db: Client, user_id: str, work_id: str, link_id: str):
    wa = await get_work_access(db, user_id, work_id)
    if not wa.can_edit:
        raise PermissionError("denied")
    db.table("work_audio_links").delete().eq("id", link_id).eq("work_id", work_id).execute()
    return {"deleted": link_id}
