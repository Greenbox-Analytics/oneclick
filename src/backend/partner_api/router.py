"""Partner API HTTP surface — API-key auth, one router per tool, each carrying
its own prefix and version:

    POST /oneclick/v1/royalties     oneclick_router (here)
    POST /registry/v1/splits        partner_api.registry
    POST /splitsheet/v1/documents   partner_api.splitsheet
    …/zoe/v1/*                      partner_api.zoe (OpenAI-compatible)

main.py mounts them bare and the host lockdown allowlists exactly these ROUTES
— a product route like /oneclick/calculate-royalties shares the prefix but is
not one of them, so it stays 404 on the API host. GET /zoe/v1/models is the
only free route; everything else is a billed deliverable.

Nothing here mints keys — humans do that on org_router.py (the org's admins)
or admin_router.py (Msanii admins). 404s unless PARTNER_API_ENABLED +
CREDITS_ENABLED + LICENSING_ENABLED are all set.
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


# Analytics funnel prefix per billed action, so an action implies its events.
EVENT_PREFIX = {
    psvc.ONECLICK_ACTION: "partner_oneclick_calc",
    psvc.REGISTRY_ACTION: "partner_registry_parse",
}


async def stream_billed_run(
    sb,
    ctx: PartnerContext,
    *,
    task,
    action: str,
    price: int,
    pool: dict,
    request_id: str,
    idempotency_key: str | None,
    on_error,
    completed_props,
):
    """THE billed-SSE body: heartbeat, one result-or-error frame, then bill.
    Shared by /oneclick/v1/royalties and /registry/v1/splits so the two can't
    drift on the part that touches money.

    No tmpdir cleanup here — the worker owns it (see run_partner_calc).

    `on_error(exc)` maps a partner-visible failure to its error frame minus
    `billing`; None sends it to the internal_error branch.
    """
    prefix = EVENT_PREFIX[action]
    # Heartbeat: a proxied response dies after ~100s of silence and the LLM
    # parse is that long. SSE comments don't pollute the event stream.
    while True:
        done, _ = await asyncio.wait({task}, timeout=15)
        if done:
            break
        yield ": ping\n\n"
    try:
        result, measured, usage = task.result()
    except Exception as exc:
        frame = on_error(exc)
        if frame is None:
            # request_id too: the only id the partner is handed back.
            logging.exception(
                "partner run failed action=%s org=%s key=%s request_id=%s", action, ctx.org_id, ctx.key_id, request_id
            )
            frame = {"type": "error", "code": "internal_error", "request_id": request_id}
        analytics_capture(ctx.org_id, f"{prefix}_failed", {"error_code": frame["code"]})
        yield _sse({**frame, "billing": psvc.unbilled()})
        return
    # Computed BEFORE the frame so the body can report what the call costs; the
    # debit runs after it. A replay under an Idempotency-Key was charged on its
    # first run (the RPC dedupes), so the body says so instead of the price.
    charge, charge_meta = compute_charge(action, price, measured, usage)
    replayed = bool(idempotency_key) and psvc.already_charged(sb, request_id)
    billing = psvc.billing_block(charge, request_id, replayed=replayed)
    yield _sse({"type": "result", **result, "billing": billing})
    # Bill only AFTER the result is on the wire: the yield above raises
    # GeneratorExit if the client is gone, so an undelivered run is free (owner
    # decision 2026-09-04 — that loss is ours). A debit failure must not turn a
    # finished run into a partner-visible error.
    try:
        psvc.debit_run(
            sb,
            wallet_id=pool["wallet_id"],
            amount=charge,
            request_id=request_id,
            key_id=ctx.key_id,
            metadata=charge_meta,
            action=action,
        )
    except Exception:
        logging.exception("partner debit failed action=%s org=%s request_id=%s", action, ctx.org_id, request_id)
    analytics_capture(
        ctx.org_id,
        f"{prefix}_completed",
        {**completed_props(result), "charged": billing["credits"], "replayed": replayed},
    )


def _save_upload(upload: UploadFile, dest: FSPath, max_bytes: int, code: str, hasher) -> int:
    """Stream an upload to disk, 413 over max_bytes. Feeds hasher so the
    idempotency id is bound to the payload. Returns bytes written."""
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

    # ---- pre-stream validation: plain HTTP statuses ----
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
        # Hash the PARSED payload: whitespace-different JSON of the same terms
        # is the same deliverable.
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
        # Started HERE, not in the generator: a client that disconnects first
        # would never reach it, and run_partner_calc's finally is the only
        # thing that deletes tmpdir. Must stay LAST in this try.
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
        # The worker never started, so the router still owns tmpdir.
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    def on_error(exc):
        if not isinstance(exc, CalculationError):
            return None
        # NB: the attribute is user_message — .message would AttributeError
        # mid-stream. `details` is the structured context (available_columns,
        # statement_songs…) a partner needs to fix the input unaided.
        return {
            "type": "error",
            "code": exc.code,
            "message": exc.user_message,
            "suggestion": exc.suggestion,
            "details": exc.details,
        }

    return StreamingResponse(
        stream_billed_run(
            sb,
            ctx,
            task=task,
            action=psvc.ONECLICK_ACTION,
            price=price,
            pool=pool,
            request_id=request_id,
            idempotency_key=idempotency_key,
            on_error=on_error,
            completed_props=lambda result: {"total_payments": result["summary"]["payments"]},
        ),
        media_type="text/event-stream",
    )
