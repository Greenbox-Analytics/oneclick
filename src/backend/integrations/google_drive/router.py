"""FastAPI router for Google Drive integration."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from integrations.oauth import (
    build_auth_url, verify_oauth_state, exchange_code_for_tokens,
    store_connection, get_valid_token, FRONTEND_URL,
)
from auth import get_current_user_id
from integrations.google_drive.models import DriveImportRequest, DriveExportRequest, DriveSyncSetup

router = APIRouter()


def _get_supabase():
    from main import get_supabase_client
    return get_supabase_client()


@router.get("/auth")
async def initiate_auth(user_id: str = Depends(get_current_user_id)):
    """Start Google Drive OAuth flow."""
    auth_url = build_auth_url("google_drive", user_id)
    return {"auth_url": auth_url}


@router.get("/callback")
async def oauth_callback(code: str, state: str):
    """Handle Google OAuth callback."""
    try:
        payload = verify_oauth_state(state)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    user_id = payload["user_id"]

    try:
        tokens = await exchange_code_for_tokens("google_drive", code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {str(e)}")

    await store_connection(_get_supabase(), user_id, "google_drive", tokens)
    return RedirectResponse(url=f"{FRONTEND_URL}/workspace?connected=google_drive")


@router.delete("/disconnect")
async def disconnect(user_id: str = Depends(get_current_user_id)):
    """Disconnect Google Drive integration."""
    _get_supabase().table("integration_connections").delete().eq(
        "user_id", user_id
    ).eq("provider", "google_drive").execute()
    return {"success": True}


@router.get("/browse")
async def browse_files(
    user_id: str = Depends(get_current_user_id),
    folder_id: str = Query(default="root"),
):
    """List files and folders in Google Drive."""
    token = await get_valid_token(_get_supabase(), user_id, "google_drive")
    if not token:
        raise HTTPException(status_code=401, detail="Google Drive not connected")

    from integrations.google_drive.service import list_drive_files
    files = await list_drive_files(token, folder_id)
    return {"files": files}


@router.post("/import")
async def import_file(body: DriveImportRequest, user_id: str = Depends(get_current_user_id)):
    """Import a file from Google Drive into a project."""
    token = await get_valid_token(_get_supabase(), user_id, "google_drive")
    if not token:
        raise HTTPException(status_code=401, detail="Google Drive not connected")

    from integrations.google_drive.service import import_drive_file
    result = await import_drive_file(token, _get_supabase(), user_id, body.model_dump())
    return result


@router.post("/export")
async def export_file(body: DriveExportRequest, user_id: str = Depends(get_current_user_id)):
    """Export a project file to Google Drive."""
    token = await get_valid_token(_get_supabase(), user_id, "google_drive")
    if not token:
        raise HTTPException(status_code=401, detail="Google Drive not connected")

    from integrations.google_drive.service import export_to_drive
    result = await export_to_drive(token, _get_supabase(), user_id, body.model_dump())
    return result


@router.post("/sync/setup")
async def setup_sync(body: DriveSyncSetup, user_id: str = Depends(get_current_user_id)):
    """Configure bidirectional sync for a project folder."""
    token = await get_valid_token(_get_supabase(), user_id, "google_drive")
    if not token:
        raise HTTPException(status_code=401, detail="Google Drive not connected")

    _get_supabase().table("drive_sync_mappings").insert({
        "user_id": user_id,
        "project_id": body.project_id,
        "drive_folder_id": body.drive_folder_id,
        "sync_direction": body.sync_direction,
    }).execute()

    return {"success": True}


@router.get("/sync/status")
async def sync_status(user_id: str = Depends(get_current_user_id)):
    """Get sync status for all configured projects."""
    result = (
        _get_supabase()
        .table("drive_sync_mappings")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )
    return {"mappings": result.data or []}
