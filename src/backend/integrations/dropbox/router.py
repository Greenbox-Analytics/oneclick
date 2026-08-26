"""FastAPI router for Dropbox integration."""

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_current_user_id
from integrations.dropbox.models import DropboxExportRequest, DropboxImportRequest
from integrations.oauth import (
    FRONTEND_URL,
    build_auth_url,
    exchange_code_for_tokens,
    get_valid_token,
    store_connection,
    verify_oauth_state,
)

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


@router.get("/auth")
async def initiate_auth(user_id: str = Depends(get_current_user_id)):
    """Start Dropbox OAuth flow."""
    auth_url = build_auth_url("dropbox", user_id)
    return {"auth_url": auth_url}


@router.get("/callback")
async def oauth_callback(code: str, state: str):
    """Handle Dropbox OAuth callback."""
    try:
        payload = verify_oauth_state(state)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    user_id = payload["user_id"]

    try:
        tokens = await exchange_code_for_tokens("dropbox", code)
    except Exception as e:
        analytics_capture(
            user_id,
            "integration_connect_failed",
            {"tool": "dropbox", "error_code": type(e).__name__},
        )
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {str(e)}")

    try:
        await store_connection(_get_supabase(), user_id, "dropbox", tokens)
    except Exception as e:
        analytics_capture(
            user_id,
            "integration_connect_failed",
            {"tool": "dropbox", "error_code": type(e).__name__},
        )
        raise

    analytics_capture(user_id, "integration_connected", {"tool": "dropbox"})
    return RedirectResponse(url=f"{FRONTEND_URL}/workspace?connected=dropbox")


@router.delete("/disconnect")
async def disconnect(user_id: str = Depends(get_current_user_id)):
    """Disconnect Dropbox integration."""
    _get_supabase().table("integration_connections").delete().eq("user_id", user_id).eq("provider", "dropbox").execute()
    return {"success": True}


@router.get("/browse")
async def browse_files(
    user_id: str = Depends(get_current_user_id),
    folder_id: str = Query(default="root"),
    search: str = Query(default=""),
):
    """List files and folders in Dropbox. Optional full-Dropbox search."""
    token = await get_valid_token(_get_supabase(), user_id, "dropbox")
    if not token:
        raise HTTPException(status_code=401, detail="Dropbox not connected")

    from integrations.dropbox.service import list_dropbox_files, search_dropbox_files

    if search.strip():
        files = await search_dropbox_files(token, search.strip())
    else:
        files = await list_dropbox_files(token, folder_id)
    return {"files": files}


@router.post("/import")
async def import_file(body: DropboxImportRequest, user_id: str = Depends(get_current_user_id)):
    """Import a file from Dropbox into a project."""
    from projects.service import get_user_role

    role = await get_user_role(_get_supabase(), user_id, body.project_id)
    if role not in ("owner", "admin", "editor"):
        raise HTTPException(status_code=403, detail="Access denied")

    token = await get_valid_token(_get_supabase(), user_id, "dropbox")
    if not token:
        raise HTTPException(status_code=401, detail="Dropbox not connected")

    from integrations.dropbox.service import FileTooLargeError, import_dropbox_file
    from integrations.storage_import import StorageCapExceededError

    try:
        result = await import_dropbox_file(token, _get_supabase(), user_id, body.model_dump())
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except StorageCapExceededError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    analytics_capture(
        user_id,
        "integration_used",
        {"tool": "dropbox", "action": "file_imported"},
    )
    return result


@router.post("/export")
async def export_file(body: DropboxExportRequest, user_id: str = Depends(get_current_user_id)):
    """Export a project file to Dropbox."""
    token = await get_valid_token(_get_supabase(), user_id, "dropbox")
    if not token:
        raise HTTPException(status_code=401, detail="Dropbox not connected")

    from integrations.dropbox.service import FileTooLargeError, export_to_dropbox

    try:
        result = await export_to_dropbox(token, _get_supabase(), user_id, body.model_dump())
    except PermissionError:
        raise HTTPException(status_code=403, detail="Access denied")
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))

    analytics_capture(
        user_id,
        "integration_used",
        {"tool": "dropbox", "action": "file_exported"},
    )
    return result


@router.get("/export-status")
async def export_status(
    project_file_id: str = Query(...),
    user_id: str = Depends(get_current_user_id),
):
    """Report whether a project file has already been saved to the caller's Dropbox."""
    result = (
        _get_supabase()
        .table("drive_sync_mappings")
        .select("share_url")
        .eq("user_id", user_id)
        .eq("project_file_id", project_file_id)
        .eq("provider", "dropbox")
        .eq("sync_direction", "to_drive")
        .execute()
    )
    # Two signals: the row's existence means the file is already in the caller's
    # Dropbox; share_url means the link finished. A row with share_url NULL is
    # "uploaded, link pending" — the UI must not offer a folder picker for it.
    if not result.data:
        return {"saved": False, "share_url": None}
    return {"saved": True, "share_url": result.data[0]["share_url"]}
