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
from subscriptions.enforcement import gated_credits, gated_split_sheet
from subscriptions.models import CreditAction

router = APIRouter()


class ContributorInput(BaseModel):
    name: str
    role: str
    # Publishing side — composition. Publishing income splits into a writer's share
    # and a publisher's share. A self-published writer keeps both; a published writer
    # collects the writer's share while their publisher collects the publisher's share.
    publishing_share: float | None = None  # self-published total (used when not is_published)
    writer_share: float | None = None
    publisher_share: float | None = None
    ipi_number: str | None = None  # writer's IPI/CAE — publishing only
    is_published: bool = False
    publisher_name: str | None = None
    publisher_ipi: str | None = None
    # Master side — sound recording. No IPI, no publisher share.
    master_percentage: float | None = None
    label: str | None = None


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
    # TWO gates, and the order is load-bearing (spec 2026-08-17 §6).
    #
    # The cap comes FIRST because it is the wall no amount of money opens:
    # tier_entitlements.max_split_sheets_per_month is a hard monthly ceiling for
    # free users with no purchase path. Offering a capped user a top-up CTA is
    # worse than useless, so the credit gate is never consulted once the cap has
    # denied. Paid tiers carry -1 (unlimited), so there credits are the only
    # governor.
    #
    # In LEGACY mode (CREDITS_ENABLED off) gated_credits falls through to
    # gated_feature(GENERATE_SPLIT_SHEET), which re-reads the same cap. That
    # double-check is deliberate and harmless — neither call increments the
    # counter, and the second is a pure read. Do NOT "simplify" it by deleting
    # the gated_split_sheet call above: that would silently drop the cap the
    # moment credits are re-enabled.
    gated_split_sheet(user_id)
    sheet_grant = gated_credits(user_id, CreditAction.SPLIT_SHEET)
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
        from main import get_supabase_client, verify_user_owns_artist, verify_user_owns_project

        if not verify_user_owns_artist(user_id, req.artist_id) or not verify_user_owns_project(user_id, req.project_id):
            raise HTTPException(status_code=403, detail="Access denied")

        try:
            file_bytes = buffer.read()
            buffer.seek(0)

            timestamp = int(time.time())
            storage_path = f"{req.artist_id}/{req.project_id}/split_sheet/{timestamp}_{filename}"

            get_supabase_client().storage.from_("project-files").upload(
                storage_path, file_bytes, file_options={"content-type": media_type}
            )
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

    # Charge-on-success: counter and debit both live on the all-success path
    # only (not in early-return / except branches). A sheet is charged per
    # FORMAT — SplitSheetRequest.format is one of pdf|docx per request — so
    # generating both is two charges and two cap units. Accepted: they are two
    # deliverables (spec §6).
    _get_entitlements_service().increment_usage(user_id, "split_sheets_this_period")
    _get_entitlements_service().debit_for_action(user_id, sheet_grant)

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
