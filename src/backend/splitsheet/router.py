import io
import re
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from auth import get_current_user_id
from splitsheet.pdf_generator import generate_split_sheet_pdf
from splitsheet.docx_generator import generate_split_sheet_docx

router = APIRouter()


class ContributorInput(BaseModel):
    name: str
    role: str
    publishing_percentage: Optional[float] = None
    master_percentage: Optional[float] = None
    publisher_or_label: Optional[str] = None
    ipi_number: Optional[str] = None


class SplitSheetRequest(BaseModel):
    work_title: str
    work_type: str = "single"
    split_type: str = "both"
    date: str
    format: str = "pdf"
    contributors: list[ContributorInput]
    save_to_artist: bool = False
    artist_id: Optional[str] = None
    project_id: Optional[str] = None


@router.post("/generate")
async def generate_split_sheet(req: SplitSheetRequest, user_id: str = Depends(get_current_user_id)):
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

    return StreamingResponse(
        buffer,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
