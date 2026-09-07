"""Splits on the API — POST /registry/v1/splits.

The Registry's contract parse, for partners: send contract PDFs, get the deal
back as data. `contract_terms` is exactly the shape /oneclick/v1/royalties
accepts (parse once, then run every statement against it at the base price);
`splits` is the per-party master / publishing / SoundExchange pivot.

Nothing is stored — the parse cache is keyed by contract text and holds no org
data, so a key never widens into anyone's documents. Multipart in, SSE out,
billed by stream_billed_run.
"""

import asyncio
import hashlib
import shutil
import tempfile
from pathlib import Path as FSPath

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from analytics import capture as analytics_capture
from partner_api import service as psvc
from partner_api.router import (
    MAX_CONTRACT_FILES,
    MAX_CONTRACTS_BYTES,
    _get_supabase,
    _save_upload,
    get_partner_context,
    require_partner_api,
    stream_billed_run,
)
from partner_api.service import PartnerContext

registry_router = APIRouter(prefix="/registry/v1", dependencies=[Depends(require_partner_api)])


@registry_router.post("/splits")
async def partner_splits(
    contracts: list[UploadFile] = File(...),
    main_artist_name: str = Form(""),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ctx: PartnerContext = Depends(get_partner_context),
):
    sb = _get_supabase()

    # ---- pre-stream validation: plain HTTP statuses ----
    if not contracts:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_request", "message": "Send at least one contract PDF as contracts."},
        )
    if len(contracts) > MAX_CONTRACT_FILES:
        raise HTTPException(status_code=413, detail={"code": "too_many_contracts"})
    for c in contracts:
        if not (c.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_request", "message": "Contracts must be PDF files in v1."},
            )

    price = psvc.get_price(sb, psvc.REGISTRY_ACTION)
    pool = psvc.check_pool(sb, ctx.org_id, price)
    if not pool["ok"]:
        raise HTTPException(
            status_code=402, detail={"code": "insufficient_credits", "price": price, "balance": pool["balance"]}
        )

    tmpdir = tempfile.mkdtemp(prefix="partner_parse_")
    tmp = FSPath(tmpdir)
    hasher = hashlib.sha256()
    try:
        contract_paths = []
        total = 0
        for i, c in enumerate(contracts):
            p = tmp / f"contract_{i}.pdf"
            total += _save_upload(c, p, MAX_CONTRACTS_BYTES - total, "file_too_large", hasher)
            contract_paths.append(str(p))
        # The pivot depends on the main artist, so the same PDFs for a
        # different artist are a different deliverable.
        hasher.update(main_artist_name.strip().encode())
        request_id = psvc.derive_request_id(ctx.key_id, idempotency_key, hasher.hexdigest(), pool.get("period_end"))
        analytics_capture(ctx.org_id, "partner_registry_parse_started", {"contract_count": len(contract_paths)})
        # Started HERE: run_partner_parse's finally is the only thing that
        # deletes tmpdir. Must stay LAST in this try.
        task = asyncio.create_task(
            asyncio.to_thread(
                psvc.run_partner_parse,
                sb,
                org_id=ctx.org_id,
                tmpdir=tmpdir,
                contract_paths=contract_paths,
                main_artist_name=main_artist_name.strip(),
            )
        )
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    def on_error(exc):
        # The contract itself: no text, encrypted, not really a PDF.
        if not isinstance(exc, ValueError):
            return None
        return {
            "type": "error",
            "code": "CONTRACT_UNREADABLE",
            "message": "We couldn't read this contract.",
            "suggestion": (
                "Make sure each file is a text PDF (not a scanned image), isn't password-protected, and isn't empty."
            ),
            "details": {"reason": str(exc)},
        }

    return StreamingResponse(
        stream_billed_run(
            sb,
            ctx,
            task=task,
            action=psvc.REGISTRY_ACTION,
            price=price,
            pool=pool,
            request_id=request_id,
            idempotency_key=idempotency_key,
            on_error=on_error,
            completed_props=lambda result: {"party_count": len(result["splits"].get("parties", []))},
        ),
        media_type="text/event-stream",
    )
