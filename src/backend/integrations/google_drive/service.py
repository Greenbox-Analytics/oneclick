"""Google Drive business logic - file browsing, import, export, and sync."""

import httpx
from supabase import Client

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

    # Store in Supabase storage
    file_name = metadata["name"]
    storage_path = f"{user_id}/{data['project_id']}/{file_name}"
    supabase.storage.from_("project-files").upload(storage_path, content)

    # Create project_files record
    file_record = {
        "user_id": user_id,
        "project_id": data["project_id"],
        "file_name": file_name,
        "file_type": data.get("file_type", "contract"),
        "file_path": storage_path,
        "file_size": metadata.get("size"),
        "mime_type": metadata.get("mimeType"),
    }
    result = supabase.table("project_files").insert(file_record).execute()

    # Create sync mapping
    if result.data:
        supabase.table("drive_sync_mappings").insert(
            {
                "user_id": user_id,
                "project_file_id": result.data[0]["id"],
                "project_id": data["project_id"],
                "drive_file_id": data["drive_file_id"],
                "sync_direction": "from_drive",
            }
        ).execute()

    return {"file": result.data[0] if result.data else {}, "source": "google_drive"}


async def export_to_drive(token: str, supabase: Client, user_id: str, data: dict) -> dict:
    """Export a project file to Google Drive."""
    # Get file from Supabase
    file_record = (
        supabase.table("project_files")
        .select("*")
        .eq("id", data["project_file_id"])
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not file_record.data:
        raise ValueError("File not found")

    file_data = file_record.data
    content = supabase.storage.from_("project-files").download(file_data["file_path"])

    # Upload to Drive
    folder_id = data.get("drive_folder_id", "root")
    metadata = {
        "name": file_data["file_name"],
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
                "file": (file_data["file_name"], content, file_data.get("mime_type", "application/octet-stream")),
            },
        )
        response.raise_for_status()
        drive_file = response.json()

    # Create sync mapping
    supabase.table("drive_sync_mappings").insert(
        {
            "user_id": user_id,
            "project_file_id": data["project_file_id"],
            "project_id": file_data["project_id"],
            "drive_file_id": drive_file["id"],
            "sync_direction": "to_drive",
        }
    ).execute()

    return {"drive_file": drive_file, "source": "export"}
