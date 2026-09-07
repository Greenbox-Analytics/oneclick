"""Partner API HTTP surface — API-key-authenticated, tool-first paths, one
router per tool, each carrying its own prefix and version:

    POST /oneclick/v1/royalties        oneclick_router (here)
    POST /registry/v1/splits   partner_api.registry.registry_router
    POST /splitsheet/v1/documents      partner_api.splitsheet.splitsheet_router
    …/zoe/v1/*                         partner_api.zoe.zoe_router (OpenAI-compatible)

main.py mounts them bare, and the host lockdown allowlists exactly these
routes — a product route such as /oneclick/calculate-royalties shares the
/oneclick prefix but is NOT one of them, so it stays 404 on the API host.
There is no account-level route: the free key check is GET /zoe/v1/models,
and every other route is a billed deliverable.

Keys are minted/listed/revoked by HUMANS elsewhere: the org's own admins on
the console router (partner_api/org_router.py, on the product backend) and
Msanii admins on subscriptions/admin_router.py. Nothing here mints: this
router 404s unless PARTNER_API_ENABLED is set, and the product service never
sets it.

404s at the router level unless PARTNER_API_ENABLED + CREDITS_ENABLED +
LICENSING_ENABLED are all set (require_licensing idiom: true rollback).
"""

import asyncio
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path as FSPath

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import Json

from analytics import capture as analytics_capture
from oneclick.royalty_calculator import CalculationError
from partner_api import service as psvc
from partner_api.models import PartnerContractTerms, PartnerExpense
from partner_api.service import PartnerContext
from subscriptions.ai_pricing import compute_charge


def require_partner_api() -> None:
    if not psvc.partner_surface_enabled():
        raise HTTPException(status_code=404, detail="Not found")


def _get_supabase():
    from main import get_supabase_client

    return get_supabase_client()


oneclick_router = APIRouter(prefix="/oneclick/v1", dependencies=[Depends(require_partner_api)])


# ---- machine (API key) auth --------------------------------------------------


def get_partner_context(authorization: str | None = Header(None)) -> PartnerContext:
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    ctx = psvc.resolve_key(_get_supabase(), bearer)
    if ctx is None:
        raise HTTPException(status_code=401, detail={"code": "invalid_key"})
    return ctx


# ---- calculate --------------------------------------------------------------

MAX_CONTRACT_FILES = 10
# Cloud Run (HTTP/1) rejects request bodies over 32 MB before the app runs —
# the totals must stay under that or these app-level 413s are unreachable.
MAX_CONTRACTS_BYTES = 20 * 1024 * 1024
MAX_STATEMENT_BYTES = 10 * 1024 * 1024


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _save_upload(upload: UploadFile, dest: FSPath, max_bytes: int, code: str, hasher) -> int:
    """Stream an upload to disk, 413 if it exceeds max_bytes. Feeds hasher so
    the idempotency request id is bound to the payload. Returns bytes written."""
    written = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(status_code=413, detail={"code": code})
            hasher.update(chunk)
            out.write(chunk)
    return written


@oneclick_router.post("/royalties")
async def partner_calculate(
    statement: UploadFile = File(...),
    contracts: list[UploadFile] = File(default=[]),
    # pydantic Json[...] parses the multipart string and 422s malformed input
    # itself (FastAPI's default validation body — spec §5).
    contract_terms: Json[PartnerContractTerms] | None = Form(None),
    expenses: Json[list[PartnerExpense]] | None = Form(None),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ctx: PartnerContext = Depends(get_partner_context),
):
    sb = _get_supabase()

    # ---- pre-stream validation: plain HTTP statuses --------------------------
    if bool(contracts) == (contract_terms is not None):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_request", "message": "Provide exactly one of contracts[] or contract_terms."},
        )
    if len(contracts) > MAX_CONTRACT_FILES:
        raise HTTPException(status_code=413, detail={"code": "too_many_contracts"})
    for c in contracts:
        if not (c.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_request", "message": "Contracts must be PDF files in v1."},
            )

    expense_dicts = [e.model_dump() for e in expenses] if expenses else None

    price = psvc.get_price(sb, psvc.ONECLICK_ACTION)
    pool = psvc.check_pool(sb, ctx.org_id, price)
    if not pool["ok"]:
        raise HTTPException(
            status_code=402, detail={"code": "insufficient_credits", "price": price, "balance": pool["balance"]}
        )

    tmpdir = tempfile.mkdtemp(prefix="partner_calc_")
    tmp = FSPath(tmpdir)
    hasher = hashlib.sha256()
    try:
        statement_ext = FSPath(statement.filename or "statement.csv").suffix or ".csv"
        statement_path = tmp / f"statement{statement_ext}"
        _save_upload(statement, statement_path, MAX_STATEMENT_BYTES, "file_too_large", hasher)
        contract_paths = []
        total = 0
        for i, c in enumerate(contracts):
            p = tmp / f"contract_{i}.pdf"
            total += _save_upload(c, p, MAX_CONTRACTS_BYTES - total, "file_too_large", hasher)
            contract_paths.append(str(p))
        # Hash the PARSED payload, not the raw string: whitespace-different
        # JSON of the same terms is the same deliverable.
        if contract_terms is not None:
            hasher.update(contract_terms.model_dump_json().encode())
        if expense_dicts:
            hasher.update(json.dumps(expense_dicts, sort_keys=True).encode())
        request_id = psvc.derive_request_id(ctx.key_id, idempotency_key, hasher.hexdigest(), pool.get("period_end"))
        analytics_capture(
            ctx.org_id,
            "partner_oneclick_calc_started",
            {
                "contract_count": len(contract_paths),
                "mode": "terms" if contract_terms is not None else "files",
            },
        )
        # Started HERE, not in the generator: a client that disconnects before
        # the stream body runs would never reach it, and run_partner_calc's
        # finally is the only thing that deletes tmpdir. Must stay LAST in this
        # try — past this point the worker owns the directory.
        task = asyncio.create_task(
            asyncio.to_thread(
                psvc.run_partner_calc,
                sb,
                org_id=ctx.org_id,
                tmpdir=tmpdir,
                statement_path=str(statement_path),
                contract_paths=contract_paths,
                contract_terms=contract_terms,
                expenses=expense_dicts,
            )
        )
    except Exception:
        # Pre-stream failure: the worker never started, so the router still owns tmpdir.
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    async def generate():
        # No cleanup here: run_partner_calc owns tmpdir. The worker thread
        # can't be cancelled, so deleting files on client disconnect would
        # yank them out from under it mid-read.
        # Heartbeat: Cloudflare kills a proxied response after ~100s of
        # silence, and the LLM parse is exactly that long. SSE comments
        # keep bytes flowing without polluting the event stream.
        while True:
            done, _ = await asyncio.wait({task}, timeout=15)
            if done:
                break
            yield ": ping\n\n"
        try:
            result, measured, usage = task.result()
        except CalculationError as e:
            analytics_capture(
                ctx.org_id,
                "partner_oneclick_calc_failed",
                {"error_code": e.code},
            )
            # NB: the attribute is user_message (royalty_calculator.py:164) —
            # e.message does not exist and would AttributeError mid-stream.
            yield _sse(
                {
                    "type": "error",
                    "code": e.code,
                    "message": e.user_message,
                    "suggestion": e.suggestion,
                    # Structured context (available_columns, statement_songs…) —
                    # what a partner needs to fix the input without a human here.
                    "details": e.details,
                    "billing": psvc.unbilled(),
                }
            )
            return
        except Exception:
            # request_id too: it is the only id the partner is handed back.
            logging.exception("partner calc failed org=%s key=%s request_id=%s", ctx.org_id, ctx.key_id, request_id)
            analytics_capture(
                ctx.org_id,
                "partner_oneclick_calc_failed",
                {"error_code": "internal_error"},
            )
            yield _sse(
                {"type": "error", "code": "internal_error", "request_id": request_id, "billing": psvc.unbilled()}
            )
            return
        # THE charge formula, computed BEFORE the frame so the body can report
        # what this call costs; the debit still runs after the frame (below).
        # A replay under an Idempotency-Key was charged on its first run — the
        # debit RPC dedupes it — so the body says so instead of repeating the price.
        # A racing duplicate reports the price while only one debit lands —
        # over-reports, never under.
        charge, charge_meta = compute_charge(psvc.ONECLICK_ACTION, price, measured, usage)
        replayed = bool(idempotency_key) and psvc.already_charged(sb, request_id)
        billing = psvc.billing_block(charge, request_id, replayed=replayed)
        yield _sse({"type": "result", **result, "billing": billing})
        # Bill only AFTER the result is on the wire. Nothing below runs if the
        # client is gone: the yield above raises GeneratorExit on close, so a
        # partner who never received an answer is never charged for one. The
        # onus for a dropped connection is ours, not theirs (owner decision
        # 2026-09-04) — the LLM spend on an undelivered run is our loss.
        #
        # Charge-on-success through THE charge formula (ai_pricing.compute_charge:
        # base / metered / base + size tail, 2026-08-27) — the partner path
        # must never grow its own max(). A debit failure must NOT turn a
        # finished calc into a partner-visible error — log loudly, result sent.
        try:
            psvc.debit_run(
                sb,
                wallet_id=pool["wallet_id"],
                amount=charge,
                request_id=request_id,
                key_id=ctx.key_id,
                metadata=charge_meta,
            )
        except Exception:
            logging.exception("partner debit failed org=%s request_id=%s", ctx.org_id, request_id)
        analytics_capture(
            ctx.org_id,
            "partner_oneclick_calc_completed",
            {"total_payments": result["summary"]["payments"], "charged": billing["credits"], "replayed": replayed},
        )

    return StreamingResponse(generate(), media_type="text/event-stream")
