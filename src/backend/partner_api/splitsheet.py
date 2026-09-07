"""Split sheets on the API — POST /splitsheet/v1/documents.

JSON in, the finished PDF or Word document out, through the product's own
generators. No AI runs, so a sheet always costs the base, charged per DOCUMENT
(the pdf and the docx of one sheet are two) and debited only after the file is
on the wire. Idempotent under Idempotency-Key. Nothing is stored.

The charge rides in `Msanii-Credits` / `Msanii-Request-Id` headers (plus
`Msanii-Replayed` on a replay) because the body is the document.
"""

import asyncio
import hashlib
import logging
import re

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from analytics import capture as analytics_capture
from partner_api import service as psvc
from partner_api.models import PartnerSplitSheetRequest
from partner_api.router import _get_supabase, get_partner_context, require_partner_api
from partner_api.service import PartnerContext
from subscriptions.ai_pricing import compute_charge

splitsheet_router = APIRouter(prefix="/splitsheet/v1", dependencies=[Depends(require_partner_api)])

MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def render(req: PartnerSplitSheetRequest) -> bytes:
    """The document bytes (worker thread — reportlab is CPU-bound)."""
    from splitsheet.docx_generator import generate_split_sheet_docx
    from splitsheet.pdf_generator import generate_split_sheet_pdf

    generate = generate_split_sheet_docx if req.format == "docx" else generate_split_sheet_pdf
    buffer = generate(
        work_title=req.work_title,
        work_type=req.work_type,
        split_type=req.split_type,
        date=req.date,
        contributors=[c.model_dump() for c in req.contributors],
    )
    return buffer.getvalue() if hasattr(buffer, "getvalue") else buffer.read()


@splitsheet_router.post("/documents")
async def partner_split_sheet(
    req: PartnerSplitSheetRequest,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ctx: PartnerContext = Depends(get_partner_context),
):
    sb = _get_supabase()
    price = psvc.get_price(sb, psvc.SPLIT_SHEET_ACTION)
    pool = psvc.check_pool(sb, ctx.org_id, price)
    if not pool["ok"]:
        raise HTTPException(
            status_code=402, detail={"code": "insufficient_credits", "price": price, "balance": pool["balance"]}
        )
    # The deliverable is a pure function of the validated body, format
    # included — so that is what the idempotency id binds to.
    fingerprint = hashlib.sha256(req.model_dump_json().encode()).hexdigest()
    request_id = psvc.derive_request_id(ctx.key_id, idempotency_key, fingerprint, pool.get("period_end"))
    analytics_capture(
        ctx.org_id,
        "partner_splitsheet_started",
        {"format": req.format, "split_type": req.split_type, "contributor_count": len(req.contributors)},
    )

    try:
        data = await asyncio.to_thread(render, req)
    except Exception:
        logging.exception("partner split sheet failed org=%s key=%s request_id=%s", ctx.org_id, ctx.key_id, request_id)
        analytics_capture(ctx.org_id, "partner_splitsheet_failed", {"error_code": "internal_error"})
        raise HTTPException(status_code=500, detail={"code": "internal_error", "request_id": request_id}) from None

    safe_title = re.sub(r"[^a-zA-Z0-9._-]", "_", req.work_title)
    filename = f"Split_Sheet_{safe_title}.{req.format}"

    # No LLM ran, so the charge is the base and is known before the body — it
    # rides in headers because the body IS the document. A replay was charged
    # on its first run, so the headers say 0 rather than repeat the price.
    charge, charge_meta = compute_charge(psvc.SPLIT_SHEET_ACTION, price, None, None)
    replayed = bool(idempotency_key) and psvc.already_charged(sb, request_id)
    billed = 0 if replayed else charge
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Length": str(len(data)),
        "Msanii-Credits": str(billed),
        "Msanii-Request-Id": request_id,
    }
    if replayed:
        headers["Msanii-Replayed"] = "true"

    async def deliver():
        # ONE frame: yield the file, then bill. Nothing below runs for a gone
        # client.
        yield data
        try:
            psvc.debit_run(
                sb,
                wallet_id=pool["wallet_id"],
                amount=charge,
                request_id=request_id,
                key_id=ctx.key_id,
                metadata=charge_meta,
                action=psvc.SPLIT_SHEET_ACTION,
            )
        except Exception:
            logging.exception("partner split sheet debit failed org=%s request_id=%s", ctx.org_id, request_id)
        analytics_capture(
            ctx.org_id,
            "partner_splitsheet_completed",
            {"format": req.format, "charged": billed, "replayed": replayed},
        )

    return StreamingResponse(deliver(), media_type=MEDIA_TYPES[req.format], headers=headers)
