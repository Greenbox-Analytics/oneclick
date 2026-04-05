from supabase import Client


async def get_work_files(db: Client, work_id: str):
    result = (
        db.table("work_files")
        .select("*, project_files(*)")
        .eq("work_id", work_id)
        .execute()
    )
    return result.data or []


async def link_file_to_work(db: Client, work_id: str, file_id: str):
    result = (
        db.table("work_files")
        .insert({"work_id": work_id, "file_id": file_id})
        .execute()
    )
    return result.data[0] if result.data else None


async def unlink_file_from_work(db: Client, link_id: str):
    db.table("work_files").delete().eq("id", link_id).execute()
    return {"deleted": link_id}


async def get_work_audio(db: Client, work_id: str):
    result = (
        db.table("work_audio_links")
        .select("*, audio_files(*)")
        .eq("work_id", work_id)
        .execute()
    )
    return result.data or []


async def link_audio_to_work(db: Client, work_id: str, audio_file_id: str):
    result = (
        db.table("work_audio_links")
        .insert({"work_id": work_id, "audio_file_id": audio_file_id})
        .execute()
    )
    return result.data[0] if result.data else None


async def unlink_audio_from_work(db: Client, link_id: str):
    db.table("work_audio_links").delete().eq("id", link_id).execute()
    return {"deleted": link_id}
