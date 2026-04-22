"""Business logic for workspace settings."""

from datetime import UTC, datetime

from supabase import Client


async def get_settings(supabase: Client, user_id: str) -> dict:
    """Get workspace settings for a user, creating defaults if none exist."""
    result = supabase.table("workspace_settings").select("*").eq("user_id", user_id).execute()
    if result.data:
        return result.data[0]

    # Upsert default settings
    default = {"user_id": user_id}
    insert_result = supabase.table("workspace_settings").insert(default).execute()
    return insert_result.data[0] if insert_result.data else default


async def update_settings(supabase: Client, user_id: str, data: dict) -> dict:
    """Update workspace settings for a user."""
    clean = {k: v for k, v in data.items() if v is not None}
    if not clean:
        return await get_settings(supabase, user_id)

    clean["updated_at"] = datetime.now(UTC).isoformat()
    result = supabase.table("workspace_settings").update(clean).eq("user_id", user_id).execute()
    if result.data:
        return result.data[0]

    # If no row existed, upsert
    clean["user_id"] = user_id
    insert_result = supabase.table("workspace_settings").insert(clean).execute()
    return insert_result.data[0] if insert_result.data else clean
