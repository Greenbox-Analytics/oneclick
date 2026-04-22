import base64
import html
import os

import resend
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from supabase import Client

from auth import get_current_user_id
from projects import service

router = APIRouter()

MAX_TOTAL_BYTES = 40 * 1024 * 1024  # Resend enforces a 40 MB attachment cap per email.


class ShareEmailRequest(BaseModel):
    recipient_email: EmailStr
    subject: str = Field(default="Files from Msanii", max_length=200)
    message: str = Field(default="", max_length=4000)
    file_ids: list[str] = Field(default_factory=list)
    audio_file_ids: list[str] = Field(default_factory=list)


def _get_supabase() -> Client:
    from main import get_supabase_client

    return get_supabase_client()


def _fetch_project_file(db: Client, file_id: str, project_id: str) -> dict:
    """Fetch a project_files row scoped to the project; raise 404 if missing."""
    result = (
        db.table("project_files")
        .select("id, file_name, file_path")
        .eq("id", file_id)
        .eq("project_id", project_id)
        .maybe_single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found in project")
    return result.data


def _fetch_audio_file(db: Client, audio_file_id: str, project_id: str) -> dict:
    """Fetch an audio_files row that is linked to the given project."""
    link = (
        db.table("project_audio_links")
        .select("audio_file_id")
        .eq("project_id", project_id)
        .eq("audio_file_id", audio_file_id)
        .maybe_single()
        .execute()
    )
    if not link.data:
        raise HTTPException(status_code=404, detail=f"Audio {audio_file_id} not linked to project")

    result = db.table("audio_files").select("id, file_name, file_path").eq("id", audio_file_id).maybe_single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail=f"Audio {audio_file_id} not found")
    return result.data


def _download(db: Client, bucket: str, path: str) -> bytes:
    try:
        return db.storage.from_(bucket).download(path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download {path}: {exc}") from exc


@router.post("/{project_id}/share-email")
async def share_via_email(
    project_id: str,
    body: ShareEmailRequest,
    user_id: str = Depends(get_current_user_id),
):
    db = _get_supabase()

    caller_role = await service.get_user_role(db, user_id, project_id)
    if caller_role not in ("owner", "admin", "editor"):
        raise HTTPException(status_code=403, detail="Only editors, admins, or the owner can share files")

    if not body.file_ids and not body.audio_file_ids:
        raise HTTPException(status_code=400, detail="Select at least one file or audio to share")

    attachments: list[dict] = []
    total_bytes = 0

    for file_id in body.file_ids:
        row = _fetch_project_file(db, file_id, project_id)
        content = _download(db, "project-files", row["file_path"])
        total_bytes += len(content)
        if total_bytes > MAX_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Total attachment size exceeds {MAX_TOTAL_BYTES // (1024 * 1024)} MB",
            )
        attachments.append({"filename": row["file_name"], "content": base64.b64encode(content).decode("ascii")})

    for audio_id in body.audio_file_ids:
        row = _fetch_audio_file(db, audio_id, project_id)
        content = _download(db, "audio-files", row["file_path"])
        total_bytes += len(content)
        if total_bytes > MAX_TOTAL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Total attachment size exceeds {MAX_TOTAL_BYTES // (1024 * 1024)} MB",
            )
        attachments.append({"filename": row["file_name"], "content": base64.b64encode(content).decode("ascii")})

    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Email service not configured")
    resend.api_key = api_key

    inviter = db.table("profiles").select("full_name").eq("id", user_id).maybe_single().execute()
    inviter_name = (inviter.data or {}).get("full_name") or "A Msanii user"

    safe_message = html.escape(body.message).replace("\n", "<br/>")
    safe_sender = html.escape(inviter_name)
    html_body = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px;">
      <p style="font-size: 15px; color: #333;">
        <strong>{safe_sender}</strong> shared {len(attachments)} file{"s" if len(attachments) != 1 else ""} with you via Msanii.
      </p>
      {f'<p style="font-size: 14px; color: #555;">{safe_message}</p>' if body.message else ""}
      <p style="font-size: 12px; color: #999; margin-top: 24px;">
        Sent from Msanii. Reply directly to reach the sender.
      </p>
    </div>
    """

    from_address = os.getenv("RESEND_FROM_EMAIL", "Msanii <onboarding@resend.dev>")

    try:
        resend.Emails.send(
            {
                "from": from_address,
                "to": [body.recipient_email],
                "subject": body.subject,
                "html": html_body,
                "attachments": attachments,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to send email: {exc}") from exc

    return {"total_bytes": total_bytes, "attachment_count": len(attachments)}
