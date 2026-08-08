"""Google Drive business logic - file browsing, import, export, and sync."""

import httpx
from supabase import Client

from integrations.storage_import import store_imported_file

DRIVE_API = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_API = "https://www.googleapis.com/upload/drive/v3"


async def list_drive_files(token: str, folder_id: str = "root") -> list:
    """List files and folders in a Google Drive folder."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{DRIVE_API}/files",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "q": f"'{folder_id}' in parents and trashed = false",
                "fields": "files(id, name, mimeType, modifiedTime, size, iconLink, webViewLink)",
                "orderBy": "folder,name",
                "pageSize": 100,
            },
        )
        response.raise_for_status()
        return response.json().get("files", [])


async def search_drive_files(token: str, query: str) -> list:
    """Search files across all of Google Drive by name."""
    escaped = query.replace("'", "\\'")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{DRIVE_API}/files",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "q": f"name contains '{escaped}' and trashed = false",
                "fields": "files(id, name, mimeType, modifiedTime, size, iconLink, webViewLink)",
                "orderBy": "modifiedTime desc",
                "pageSize": 50,
            },
        )
        response.raise_for_status()
        return response.json().get("files", [])


async def download_drive_file(token: str, file_id: str) -> bytes:
    """Download a file from Google Drive."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{DRIVE_API}/files/{file_id}",
            headers={"Authorization": f"Bearer {token}"},
            params={"alt": "media"},
        )
        response.raise_for_status()
        return response.content


async def import_drive_file(token: str, supabase: Client, user_id: str, data: dict) -> dict:
    """Import a file from Drive into a Supabase project."""
    # Check if this Drive file is already imported into this project.
    # No .eq("provider", ...) here on purpose: that column only exists after
    # migration 20260807000000, and this dedup query must keep working on a
    # backend deployed before that migration runs. Dropbox ids are always
    # "id:"-prefixed and Drive ids never are, so a cross-provider collision
    # can't happen — the filter would buy nothing but a 42703 failure mode.
    existing = (
        supabase.table("drive_sync_mappings")
        .select("id")
        .eq("project_id", data["project_id"])
        .eq("drive_file_id", data["drive_file_id"])
        .execute()
    )
    if existing.data:
        raise ValueError("This file has already been imported into this project.")

    # Get file metadata
    async with httpx.AsyncClient() as client:
        meta_response = await client.get(
            f"{DRIVE_API}/files/{data['drive_file_id']}",
            headers={"Authorization": f"Bearer {token}"},
            params={"fields": "id,name,mimeType,size"},
        )
        meta_response.raise_for_status()
        metadata = meta_response.json()

    # Download file content
    content = await download_drive_file(token, data["drive_file_id"])

    file_name = metadata["name"]
    mime = metadata.get("mimeType") or "application/octet-stream"
    file_size = int(metadata["size"]) if metadata.get("size") else None

    # Gate -> Storage write -> project_files insert -> orphan cleanup on
    # failure, shared with the Dropbox import path. owner_user_id is omitted:
    # the router only verified project-member role, not that user_id is the
    # project's storage-counter owner, so the pre-check is skipped and the DB
    # trigger (-> StorageCapExceededError) is the gate.
    file_row = store_imported_file(
        supabase,
        user_id,
        data["project_id"],
        file_name,
        content,
        mime=mime,
        folder_category=data.get("file_type", "contract"),
        file_size=file_size,
    )

    # Create sync mapping
    if file_row:
        supabase.table("drive_sync_mappings").insert(
            {
                "user_id": user_id,
                "project_file_id": file_row["id"],
                "project_id": data["project_id"],
                "drive_file_id": data["drive_file_id"],
                "sync_direction": "from_drive",
            }
        ).execute()

    return {"file": file_row, "source": "google_drive"}


async def export_to_drive(token: str, supabase: Client, user_id: str, data: dict) -> dict:
    """Export a project file to Google Drive."""
    from projects.service import get_user_role

    # Get file from Supabase (without user_id filter — project_files has no user_id column).
    # select("*") preserves all columns (e.g. mime_type) used downstream.
    pf = supabase.table("project_files").select("*").eq("id", data["project_file_id"]).maybe_single().execute()
    if not pf or not pf.data:
        raise PermissionError("not found")
    if await get_user_role(supabase, user_id, pf.data["project_id"]) is None:
        raise PermissionError("denied")
    file_row = pf.data

    content = supabase.storage.from_("project-files").download(file_row["file_path"])

    # Upload to Drive
    folder_id = data.get("drive_folder_id", "root")
    metadata = {
        "name": file_row["file_name"],
        "parents": [folder_id],
    }

    async with httpx.AsyncClient() as client:
        # Multipart upload
        import json

        response = await client.post(
            f"{DRIVE_UPLOAD_API}/files?uploadType=multipart",
            headers={"Authorization": f"Bearer {token}"},
            files={
                "metadata": ("metadata", json.dumps(metadata), "application/json"),
                "file": (file_row["file_name"], content, file_row.get("mime_type", "application/octet-stream")),
            },
        )
        response.raise_for_status()
        drive_file = response.json()

    # Create sync mapping
    supabase.table("drive_sync_mappings").insert(
        {
            "user_id": user_id,
            "project_file_id": data["project_file_id"],
            "project_id": file_row["project_id"],
            "drive_file_id": drive_file["id"],
            "sync_direction": "to_drive",
        }
    ).execute()

    return {"drive_file": drive_file, "source": "export"}


async def export_pdf_to_drive(token: str, pdf_content: bytes, filename: str, folder_id: str | None = None) -> dict:
    """Upload a PDF to Google Drive."""
    import json

    metadata = {"name": filename, "mimeType": "application/pdf"}
    if folder_id:
        metadata["parents"] = [folder_id]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{DRIVE_UPLOAD_API}/files?uploadType=multipart",
            headers={"Authorization": f"Bearer {token}"},
            files={
                "metadata": ("metadata", json.dumps(metadata), "application/json"),
                "file": (filename, pdf_content, "application/pdf"),
            },
        )
        response.raise_for_status()
        return response.json()
