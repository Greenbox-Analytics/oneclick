import re
import sys
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from analytics import capture as analytics_capture
from auth import get_current_user_id
from splitsheet.docx_generator import generate_split_sheet_docx
from splitsheet.pdf_generator import generate_split_sheet_pdf
from subscriptions.deps import _get_entitlements_service
from subscriptions.enforcement import gated_split_sheet

router = APIRouter()


class ContributorInput(BaseModel):
    name: str
    role: str
    publishing_percentage: float | None = None
    master_percentage: float | None = None
    publisher_or_label: str | None = None
    ipi_number: str | None = None


class SplitSheetRequest(BaseModel):
    work_title: str
    work_type: str = "single"
    split_type: str = "both"
    date: str
    format: str = "pdf"
    contributors: list[ContributorInput]
    save_to_artist: bool = False
    artist_id: str | None = None
    project_id: str | None = None


@router.post("/generate")
async def generate_split_sheet(req: SplitSheetRequest, user_id: str = Depends(get_current_user_id)):
    # Gate: 402 if at per-period split-sheet cap
    gated_split_sheet(user_id)
    started_at = time.perf_counter()

    if not req.contributors:
        raise HTTPException(status_code=400, detail="At least one contributor is required")

    contributors_dicts = [c.model_dump() for c in req.contributors]

    try:
        if req.format == "docx":
            buffer = generate_split_sheet_docx(
                work_title=req.work_title,
                work_type=req.work_type,
                split_type=req.split_type,
                date=req.date,
                contributors=contributors_dicts,
            )
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ext = "docx"
        else:
            buffer = generate_split_sheet_pdf(
                work_title=req.work_title,
                work_type=req.work_type,
                split_type=req.split_type,
                date=req.date,
                contributors=contributors_dicts,
            )
            media_type = "application/pdf"
            ext = "pdf"
    except Exception as e:
        analytics_capture(
            user_id,
            "splitsheet_generation_failed",
            {"tool": "splitsheet", "error_code": type(e).__name__},
        )
        raise HTTPException(status_code=500, detail=f"Failed to generate document: {str(e)}")

    safe_title = re.sub(r"[^a-zA-Z0-9._-]", "_", req.work_title)
    filename = f"Split_Sheet_{safe_title}.{ext}"

    # Save to artist profile if requested
    if req.save_to_artist and req.artist_id and req.project_id:
        try:
            from main import get_supabase_client

            file_bytes = buffer.read()
            buffer.seek(0)

            timestamp = int(time.time())
            storage_path = f"{req.artist_id}/{req.project_id}/split_sheet/{timestamp}_{filename}"

            get_supabase_client().storage.from_("project-files").upload(storage_path, file_bytes)
            file_url = get_supabase_client().storage.from_("project-files").get_public_url(storage_path)

            db_record = {
                "project_id": req.project_id,
                "folder_category": "split_sheet",
                "file_name": filename,
                "file_url": file_url,
                "file_path": storage_path,
                "file_size": len(file_bytes),
                "file_type": media_type,
            }
            get_supabase_client().table("project_files").insert(db_record).execute()
        except Exception as e:
            # Still return the file even if saving fails
            print(f"Warning: Failed to save split sheet to artist profile: {e}")

    # Increment counter on the all-success path only (not in early-return / except branches)
    _get_entitlements_service().increment_usage(user_id, "split_sheets_this_period")

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    analytics_capture(
        user_id,
        "tool_used",
        {
            "tool": "splitsheet",
            "success": True,
            "duration_ms": duration_ms,
        },
    )
    analytics_capture(
        user_id,
        "splitsheet_generated",
        {
            "tool": "splitsheet",
            "format": req.format,
            "collaborator_count": len(req.contributors),
            "duration_ms": duration_ms,
        },
    )

    return StreamingResponse(
        buffer,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
