"""Splits on the API — POST /registry/v1/splits.

The Registry's contract parse, for partners: send contract PDFs, get the deal
back as data. `contract_terms` is exactly the shape /oneclick/v1/royalties
accepts (parse a contract once, then run every statement against it at the
base price), and `splits` is the per-party master / publishing / SoundExchange
pivot the Registry uses for ownership stakes.

Nothing is stored. The parse cache is keyed by the contract's text — shared
with the product, deliberately cross-tenant — and holds no org data, so a key
never widens into anyone's stored documents.

Same shape as the calculation endpoint: multipart in, SSE out (heartbeats
while the parse runs), priced by credit_prices.partner_registry_parse through
ai_pricing.compute_charge (base / metered tail), debited ONLY after the result
frame is on the wire, idempotent under Idempotency-Key.
"""

import asyncio
import hashlib
import logging
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
    _sse,
    get_partner_context,
    require_partner_api,
)
from partner_api.service import PartnerContext
from subscriptions.ai_pricing import compute_charge

registry_router = APIRouter(prefix="/registry/v1", dependencies=[Depends(require_partner_api)])


@registry_router.post("/splits")
async def partner_splits(
    contracts: list[UploadFile] = File(...),
    main_artist_name: str = Form(""),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ctx: PartnerContext = Depends(get_partner_context),
):
    sb = _get_supabase()

    # ---- pre-stream validation: plain HTTP statuses --------------------------
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
        # The pivot depends on who the main artist is, so the same PDFs for a
        # different artist are a different deliverable.
        hasher.update(main_artist_name.strip().encode())
        request_id = psvc.derive_request_id(ctx.key_id, idempotency_key, hasher.hexdigest(), pool.get("period_end"))
        analytics_capture(ctx.org_id, "partner_registry_parse_started", {"contract_count": len(contract_paths)})
        # Started HERE, not in the generator: run_partner_parse's finally is the
        # only thing that deletes tmpdir. Must stay LAST in this try.
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

    async def generate():
        # Heartbeats: a proxied response dies after ~100 s of silence, and the
        # LLM parse is exactly that long. SSE comments keep bytes flowing.
        while True:
            done, _ = await asyncio.wait({task}, timeout=15)
            if done:
                break
            yield ": ping\n\n"
        try:
            result, measured, usage = task.result()
        except ValueError as e:
            # The contract itself: no text, encrypted, not really a PDF.
            analytics_capture(ctx.org_id, "partner_registry_parse_failed", {"error_code": "CONTRACT_UNREADABLE"})
            yield _sse(
                {
                    "type": "error",
                    "code": "CONTRACT_UNREADABLE",
                    "message": "We couldn't read this contract.",
                    "suggestion": (
                        "Make sure each file is a text PDF (not a scanned image), isn't password-protected, "
                        "and isn't empty."
                    ),
                    "details": {"reason": str(e)},
                }
            )
            return
        except Exception:
            logging.exception("partner parse failed org=%s key=%s request_id=%s", ctx.org_id, ctx.key_id, request_id)
            analytics_capture(ctx.org_id, "partner_registry_parse_failed", {"error_code": "internal_error"})
            yield _sse({"type": "error", "code": "internal_error", "request_id": request_id})
            return
        yield _sse({"type": "result", **result})
        # Bill only AFTER the result is on the wire — a client that dropped
        # closed the generator at the yield above and is never charged. THE
        # charge formula, never a local max(); a debit failure is logged, not
        # surfaced, because the deliverable already went out.
        charge, charge_meta = compute_charge(psvc.REGISTRY_ACTION, price, measured, usage)
        try:
            psvc.debit_run(
                sb,
                wallet_id=pool["wallet_id"],
                amount=charge,
                request_id=request_id,
                key_id=ctx.key_id,
                metadata=charge_meta,
                action=psvc.REGISTRY_ACTION,
            )
        except Exception:
            logging.exception("partner parse debit failed org=%s request_id=%s", ctx.org_id, request_id)
        analytics_capture(
            ctx.org_id,
            "partner_registry_parse_completed",
            {"party_count": len(result["splits"].get("parties", [])), "charged": charge},
        )

    return StreamingResponse(generate(), media_type="text/event-stream")
