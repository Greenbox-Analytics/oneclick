"""Dropbox business logic - file browsing, import, export, share links.

Dropbox API v2 is POST-JSON RPC on api.dropboxapi.com; content endpoints live
on content.dropboxapi.com with args JSON-encoded in the Dropbox-API-Arg header.
All output is normalized to the same file shape the Google Drive service
returns, so the frontend browse/import UI is shared between providers.
"""

import json
import mimetypes

import httpx
from supabase import Client

from integrations.storage_import import store_imported_file

RPC_API = "https://api.dropboxapi.com/2"
CONTENT_API = "https://content.dropboxapi.com/2"

# Sentinel the shared import dialog already recognizes as "folder".
FOLDER_MIME = "application/vnd.google-apps.folder"

# files/upload single-call limit; larger files need upload sessions (not built).
# Dropbox-specific: the shared storage_import helper imposes no size ceiling
# of its own, since Drive has no equivalent limit.
MAX_UPLOAD_BYTES = 150 * 1024 * 1024

# httpx defaults to a 5s timeout on every phase (connect/read/write), which is
# fine for the small JSON RPC calls below but guarantees a ReadTimeout on any
# real upload/download body. Content-transfer calls get their own generous
# timeout; connect stays short, read/write cover a slow 150 MB transfer.
TRANSFER_TIMEOUT = httpx.Timeout(30.0, read=300.0, write=300.0)


class FileTooLargeError(Exception):
    """A file exceeds MAX_UPLOAD_BYTES (Dropbox's own upload-size ceiling —
    a per-provider decision; the shared storage_import helper doesn't raise
    this since Drive has no equivalent limit). Subclasses Exception (not
    ValueError) so a router's except-ValueError branches can't accidentally
    catch it — it needs its own 413 mapping, not the 409/other meaning
    ValueError carries at each call site."""


def _normalize_entry(entry: dict) -> dict:
    """Map a Dropbox files/list_folder entry to the Drive file shape."""
    if entry.get(".tag") == "folder":
        # Dropbox folders carry no server_modified or size.
        return {
            "id": entry["id"],
            "name": entry["name"],
            "mimeType": FOLDER_MIME,
            "modifiedTime": None,
            "size": None,
        }
    mime = mimetypes.guess_type(entry["name"])[0] or "application/octet-stream"
    return {
        "id": entry["id"],
        "name": entry["name"],
        "mimeType": mime,
        "modifiedTime": entry.get("server_modified"),
        "size": str(entry["size"]) if entry.get("size") is not None else None,
    }


def _rpc_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def list_dropbox_files(token: str, folder_id: str = "root") -> list:
    """List a Dropbox folder. Accepts "root" (frontend convention) or a folder id."""
    # ponytail: first page only (limit 100), matches Drive's existing cap; add cursor-follow if folders outgrow it
    path = "" if folder_id in ("root", "") else folder_id
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RPC_API}/files/list_folder",
            headers=_rpc_headers(token),
            json={"path": path, "limit": 100},
        )
        response.raise_for_status()
        entries = response.json().get("entries", [])
    # Folders first, matching Drive's orderBy=folder,name.
    folders = sorted((e for e in entries if e.get(".tag") == "folder"), key=lambda e: e["name"].lower())
    files = sorted((e for e in entries if e.get(".tag") == "file"), key=lambda e: e["name"].lower())
    return [_normalize_entry(e) for e in folders + files]


async def search_dropbox_files(token: str, query: str) -> list:
    """Search Dropbox by filename."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RPC_API}/files/search_v2",
            headers=_rpc_headers(token),
            json={"query": query, "options": {"max_results": 50, "filename_only": True}},
        )
        response.raise_for_status()
        matches = response.json().get("matches", [])
    entries = [m.get("metadata", {}).get("metadata", {}) for m in matches]
    return [_normalize_entry(e) for e in entries if e.get(".tag") in ("file", "folder")]


async def get_dropbox_metadata(token: str, file_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RPC_API}/files/get_metadata",
            headers=_rpc_headers(token),
            json={"path": file_id},
        )
        response.raise_for_status()
        return response.json()


async def download_dropbox_file(token: str, file_id: str) -> bytes:
    async with httpx.AsyncClient(timeout=TRANSFER_TIMEOUT) as client:
        response = await client.post(
            f"{CONTENT_API}/files/download",
            headers={
                "Authorization": f"Bearer {token}",
                "Dropbox-API-Arg": json.dumps({"path": file_id}),
            },
        )
        response.raise_for_status()
        return response.content


async def import_dropbox_file(token: str, supabase: Client, user_id: str, data: dict) -> dict:
    """Import a file from Dropbox into a Supabase project."""
    existing = (
        supabase.table("drive_sync_mappings")
        .select("id")
        .eq("project_id", data["project_id"])
        .eq("drive_file_id", data["dropbox_file_id"])
        .eq("provider", "dropbox")
        .execute()
    )
    if existing.data:
        raise ValueError("This file has already been imported into this project.")

    metadata = await get_dropbox_metadata(token, data["dropbox_file_id"])
    file_size = int(metadata["size"]) if metadata.get("size") else None
    if file_size is not None and file_size > MAX_UPLOAD_BYTES:
        raise FileTooLargeError("This file is too large to import (150 MB max).")

    content = await download_dropbox_file(token, data["dropbox_file_id"])

    file_name = metadata["name"]
    mime = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

    # Gate -> Storage write -> project_files insert -> orphan cleanup on
    # failure, shared with the Google Drive import path. owner_user_id is
    # omitted: the router only verified project-member role, not that
    # user_id is the project's storage-counter owner, so the pre-check is
    # skipped and the DB trigger (-> StorageCapExceededError) is the gate.
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

    if file_row:
        supabase.table("drive_sync_mappings").insert(
            {
                "user_id": user_id,
                "project_file_id": file_row["id"],
                "project_id": data["project_id"],
                "drive_file_id": data["dropbox_file_id"],
                "provider": "dropbox",
                "sync_direction": "from_drive",
            }
        ).execute()

    return {"file": file_row, "source": "dropbox"}


async def create_share_link(token: str, path: str) -> str:
    """Create a shared link for a Dropbox path/id, reusing an existing link on 409."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{RPC_API}/sharing/create_shared_link_with_settings",
            headers=_rpc_headers(token),
            json={"path": path},
        )
        if response.status_code == 409 and response.json().get("error", {}).get(".tag") == "shared_link_already_exists":
            fallback = await client.post(
                f"{RPC_API}/sharing/list_shared_links",
                headers=_rpc_headers(token),
                json={"path": path, "direct_only": True},
            )
            fallback.raise_for_status()
            links = fallback.json().get("links", [])
            if links:
                return links[0]["url"]
        response.raise_for_status()
        return response.json()["url"]


async def _resolve_existing_export(token: str, supabase: Client, row: dict) -> dict:
    """Return the caller's existing Dropbox export, minting the share link if
    the row's upload succeeded but the link was never created (either the
    original recovery path, or a racer that lost the unique-index insert)."""
    if row.get("share_url"):
        return {"dropbox_file": None, "share_url": row["share_url"], "already_saved": True}
    share_url = await create_share_link(token, row["drive_file_id"])
    supabase.table("drive_sync_mappings").update({"share_url": share_url}).eq("id", row["id"]).execute()
    return {"dropbox_file": None, "share_url": share_url, "already_saved": True}


async def export_to_dropbox(token: str, supabase: Client, user_id: str, data: dict) -> dict:
    """Save a project file to the caller's Dropbox and return a shared link.

    Idempotent per (user_id, project_file_id): a repeat call returns the stored
    link without re-uploading. The filter MUST include user_id (one member's
    saved copy is never returned to another member) and sync_direction='to_drive'
    (an import-provenance row also carries project_file_id and must not match).
    The idempotency lookup runs before the authz check: a matching row can only
    exist if this same caller already passed that check on a prior call, and
    both short-circuit branches below only ever hand back the caller's own link.

    A partial unique index on (user_id, project_file_id) WHERE provider='dropbox'
    AND sync_direction='to_drive' guards the gap between this pre-check and the
    insert below: two concurrent exports can both pass the pre-check, but only
    one insert wins — the loser's insert is caught and resolved to the winner's
    row instead of erroring.
    """
    from projects.service import get_user_role

    existing = (
        supabase.table("drive_sync_mappings")
        .select("id, share_url, drive_file_id")
        .eq("user_id", user_id)
        .eq("project_file_id", data["project_file_id"])
        .eq("provider", "dropbox")
        .eq("sync_direction", "to_drive")
        .execute()
    )
    if existing.data:
        return await _resolve_existing_export(token, supabase, existing.data[0])

    pf = supabase.table("project_files").select("*").eq("id", data["project_file_id"]).maybe_single().execute()
    if not pf or not pf.data:
        raise PermissionError("not found")
    if await get_user_role(supabase, user_id, pf.data["project_id"]) is None:
        raise PermissionError("denied")
    file_row = pf.data

    # Check the known size FIRST — avoids loading a too-large file into memory
    # just to reject it. Rows with no file_size recorded fall through to the
    # post-download length check below, which is the authority for those.
    file_size = file_row.get("file_size")
    if file_size is not None and file_size > MAX_UPLOAD_BYTES:
        raise FileTooLargeError("This file is too large to save to Dropbox (150 MB max).")

    content = supabase.storage.from_("project-files").download(file_row["file_path"])
    if len(content) > MAX_UPLOAD_BYTES:
        raise FileTooLargeError("This file is too large to save to Dropbox (150 MB max).")

    folder = data.get("dropbox_folder_id") or ""
    # Dropbox accepts "id:<folder-id>/<name>" paths; root is "/<name>".
    upload_path = f"{folder}/{file_row['file_name']}" if folder else f"/{file_row['file_name']}"

    async with httpx.AsyncClient(timeout=TRANSFER_TIMEOUT) as client:
        response = await client.post(
            f"{CONTENT_API}/files/upload",
            headers={
                "Authorization": f"Bearer {token}",
                "Dropbox-API-Arg": json.dumps({"path": upload_path, "mode": "add", "autorename": True}),
                "Content-Type": "application/octet-stream",
            },
            content=content,
        )
        response.raise_for_status()
        dropbox_file = response.json()

    # Record the upload BEFORE requesting the share link: if link creation
    # throws, this row (share_url=None) lets a retry repair the link without
    # re-uploading (which would autorename into a duplicate file).
    try:
        inserted = (
            supabase.table("drive_sync_mappings")
            .insert(
                {
                    "user_id": user_id,
                    "project_file_id": data["project_file_id"],
                    "project_id": file_row["project_id"],
                    "drive_file_id": dropbox_file["id"],
                    "provider": "dropbox",
                    "sync_direction": "to_drive",
                    "share_url": None,
                }
            )
            .execute()
        )
    except Exception as e:
        error_message = str(e)
        if "23505" in error_message or "duplicate key value" in error_message.lower():
            # Lost the race: a concurrent export already holds the row for this
            # (user, project_file). Hand back its link instead of erroring —
            # the file this racer just uploaded becomes an untracked duplicate
            # in Dropbox (autorenamed), which is an acceptable cost of the race.
            after_race = (
                supabase.table("drive_sync_mappings")
                .select("id, share_url, drive_file_id")
                .eq("user_id", user_id)
                .eq("project_file_id", data["project_file_id"])
                .eq("provider", "dropbox")
                .eq("sync_direction", "to_drive")
                .execute()
            )
            if after_race.data:
                return await _resolve_existing_export(token, supabase, after_race.data[0])
        raise

    share_url = await create_share_link(token, dropbox_file["id"])

    if inserted.data:
        supabase.table("drive_sync_mappings").update({"share_url": share_url}).eq(
            "id", inserted.data[0]["id"]
        ).execute()

    return {"dropbox_file": dropbox_file, "share_url": share_url, "already_saved": False}
