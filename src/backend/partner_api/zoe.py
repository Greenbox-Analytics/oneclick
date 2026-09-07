"""Zoe on the API — OpenAI-compatible chat completions under /zoe/v1.

Point any OpenAI SDK at base_url=<host>/zoe/v1 with a partner key and it
works: POST /chat/completions (JSON in; JSON out, or SSE `chat.completion.chunk`
frames ending in `data: [DONE]` with stream=true) and GET /models.

STATELESS by design — no stored contracts, no conversation memory, no RAG.
The partner supplies the whole context in `messages`, exactly as with
OpenAI. A key spends credits; it must never widen into an org's stored
documents, which belong to members and their artists.

Billing mirrors the calculation endpoint: pool-checked before any work,
priced by credit_prices.partner_zoe_message through ai_pricing.compute_charge
(base / metered tail), and debited ONLY after the answer is delivered — after
the JSON body is yielded, or after the last content chunk and before the
terminal `stop` + [DONE] frames on a stream (a client that drops mid-answer
closes the generator there and is never charged). Every completion pays at
least the base: the product's free "conversational" fast path is a UI nicety
the API does not have. Every delivered body, and the stream's final `stop`
frame, carries `billing` (credits + request id). OpenAI's `usage` token block
is not sent — partners are told what a call cost, not how many tokens it
burned.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from analytics import capture as analytics_capture
from partner_api import service as psvc
from partner_api.router import _get_supabase, _sse, get_partner_context, require_partner_api
from partner_api.service import PartnerContext
from subscriptions.ai_pricing import compute_charge
from utils.llm.tracking import credits_for_llm_usage, iter_with_llm_context, llm_usage_snapshot, set_llm_context

ZOE_ACTION = "partner_zoe_message"
MODEL_ID = "zoe"
TRACKING_TOOL = "zoe_partner"
# Money path with a metered tail — bound both ends of the prompt.
MAX_INPUT_CHARS = 100_000
MAX_OUTPUT_TOKENS = 4_000
# The in-app general-knowledge persona (contract_chatbot), minus the
# per-account bits. Topical guard is prompt-only, same as the product.
ZOE_SYSTEM_PROMPT = (
    "You are Zoe, a knowledgeable music-business assistant for the Msanii platform. Only answer "
    "questions about the music industry (deals, royalties, rights, publishing, management, etc.). "
    "If the question is not related to the music business, politely decline and redirect. Answer "
    "from general music-business knowledge and from any documents the user includes; do NOT name "
    "sources or cite page numbers. Be concise — lead with a direct one- or two-sentence answer, "
    "then only the few most important points; avoid jargon."
)

zoe_router = APIRouter(prefix="/zoe/v1", dependencies=[Depends(require_partner_api)])


# ---- request DTO (the OpenAI subset we honour) ------------------------------


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]

    def text(self) -> str:
        return self.content if isinstance(self.content, str) else "\n".join(p.text for p in self.content)


class ChatCompletionRequest(BaseModel):
    # Other OpenAI fields (n, top_p, stop, user, tools…) are ignored, not
    # rejected, so a stock SDK call works. Only these four are honoured.
    model_config = ConfigDict(extra="ignore")
    model: str = MODEL_ID
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=MAX_OUTPUT_TOKENS)

    @model_validator(mode="after")
    def _bounded_input(self):
        if sum(len(m.text()) for m in self.messages) > MAX_INPUT_CHARS:
            raise ValueError(f"messages exceed {MAX_INPUT_CHARS} characters in total")
        return self


def _openai_messages(req: ChatCompletionRequest) -> list[dict]:
    # Ours first; a partner's own system message follows and may add context
    # (a contract, house style) — it cannot remove the persona above it.
    return [{"role": "system", "content": ZOE_SYSTEM_PROMPT}] + [
        {"role": m.role, "content": m.text()} for m in req.messages
    ]


def _llm_kwargs(req: ChatCompletionRequest) -> dict:
    kw = {"max_completion_tokens": req.max_tokens or MAX_OUTPUT_TOKENS}
    if req.temperature is not None:
        kw["temperature"] = req.temperature
    return kw


def _charge_for(price: int, measured, usage) -> tuple[int, dict]:
    """THE charge formula (ai_pricing.compute_charge), computed before the
    frame that reports it. Never a local max()."""
    return compute_charge(ZOE_ACTION, price, measured, usage)


def _debit(sb, ctx: PartnerContext, pool: dict, charge: int, meta: dict, request_id: str) -> None:
    """Charge-on-delivery. No idempotency: a chat completion has no natural
    request identity (OpenAI has none either), so every delivered answer pays.
    A debit failure never turns a delivered answer into a partner-visible
    error — log loudly. The number the partner was shown is the charge as
    computed; if the RPC fails the ledger never records it — fail-open on the
    request path, by design."""
    try:
        psvc.debit_run(
            sb,
            wallet_id=pool["wallet_id"],
            amount=charge,
            request_id=request_id,
            key_id=ctx.key_id,
            metadata=meta,
            action=ZOE_ACTION,
        )
    except Exception:
        logging.exception("partner zoe debit failed org=%s key=%s", ctx.org_id, ctx.key_id)


# ---- non-streaming -----------------------------------------------------------


def complete(org_id: str, req: ChatCompletionRequest) -> tuple[dict, int | None, dict | None]:
    """One completion in a worker thread. Returns the OpenAI-shaped body plus
    the two pricing inputs, both read INSIDE the tracking scope."""
    from utils.llm.client import get_openai_client
    from utils.llm.model_garden import model_for

    with set_llm_context(org_id, TRACKING_TOOL):
        resp = get_openai_client().chat.completions.create(
            model=model_for("zoe"), messages=_openai_messages(req), **_llm_kwargs(req)
        )
        measured, usage = credits_for_llm_usage(), llm_usage_snapshot()
    choice = resp.choices[0]
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": choice.message.content or ""},
                "finish_reason": choice.finish_reason or "stop",
            }
        ],
    }
    return body, measured, usage


# ---- streaming ---------------------------------------------------------------


def _stream(sb, ctx: PartnerContext, pool: dict, price: int, req: ChatCompletionRequest):
    """SYNC generator (Starlette steps it in the threadpool); main wraps it in
    iter_with_llm_context so every step sees the tracking scope and ONE
    accumulator spans the whole stream."""
    from utils.llm.client import get_openai_client
    from utils.llm.model_garden import model_for

    cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

    def frame(delta: dict, finish: str | None = None, billing: dict | None = None) -> str:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if billing is not None:
            payload["billing"] = billing
        return _sse(payload)

    try:
        stream = get_openai_client().chat.completions.create(
            model=model_for("zoe"), messages=_openai_messages(req), stream=True, **_llm_kwargs(req)
        )
        yield frame({"role": "assistant", "content": ""})
        for chunk in stream:
            if not chunk.choices:
                continue  # the terminal usage-only chunk (stream_options.include_usage)
            delta = chunk.choices[0].delta
            if delta is not None and delta.content:
                yield frame({"content": delta.content})
    except Exception:
        logging.exception("partner zoe stream failed org=%s key=%s", ctx.org_id, ctx.key_id)
        analytics_capture(ctx.org_id, "partner_zoe_failed", {"error_code": "internal_error", "stream": True})
        yield _sse(
            {
                "error": {
                    "message": "Zoe couldn't answer just now. Try again.",
                    "type": "server_error",
                    "code": "zoe_failed",
                },
                "billing": psvc.unbilled(),
            }
        )
        yield "data: [DONE]\n\n"
        return
    # The whole answer is on the wire: bill, then close. A client that dropped
    # mid-answer raised GeneratorExit at a yield above and never reaches here.
    request_id = str(uuid.uuid4())
    charge, meta = _charge_for(price, credits_for_llm_usage(), llm_usage_snapshot())
    _debit(sb, ctx, pool, charge, meta, request_id)
    analytics_capture(ctx.org_id, "partner_zoe_completed", {"charged": charge, "stream": True})
    yield frame({}, finish="stop", billing=psvc.billing_block(charge, request_id))
    yield "data: [DONE]\n\n"


# ---- routes ------------------------------------------------------------------


@zoe_router.get("/models")
async def zoe_models(ctx: PartnerContext = Depends(get_partner_context)):
    """OpenAI's model listing, so SDKs and gateways that probe it don't 404."""
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "msanii"}]}


@zoe_router.post("/chat/completions")
async def zoe_chat_completions(req: ChatCompletionRequest, ctx: PartnerContext = Depends(get_partner_context)):
    if req.model != MODEL_ID:
        # A caller asking for "gpt-4o" must not silently get Zoe.
        raise HTTPException(status_code=404, detail={"code": "model_not_found", "message": f"Use model '{MODEL_ID}'."})
    sb = _get_supabase()
    price = psvc.get_price(sb, ZOE_ACTION)
    pool = psvc.check_pool(sb, ctx.org_id, price)
    if not pool["ok"]:
        raise HTTPException(
            status_code=402, detail={"code": "insufficient_credits", "price": price, "balance": pool["balance"]}
        )
    analytics_capture(ctx.org_id, "partner_zoe_started", {"stream": req.stream, "messages": len(req.messages)})

    if req.stream:
        return StreamingResponse(
            iter_with_llm_context(ctx.org_id, TRACKING_TOOL, _stream(sb, ctx, pool, price, req)),
            media_type="text/event-stream",
        )

    try:
        body, measured, usage = await asyncio.to_thread(complete, ctx.org_id, req)
    except Exception:
        logging.exception("partner zoe completion failed org=%s key=%s", ctx.org_id, ctx.key_id)
        analytics_capture(ctx.org_id, "partner_zoe_failed", {"error_code": "internal_error", "stream": False})
        raise HTTPException(status_code=502, detail={"code": "zoe_failed"}) from None

    request_id = str(uuid.uuid4())
    charge, meta = _charge_for(price, measured, usage)
    body["billing"] = psvc.billing_block(charge, request_id)

    async def deliver():
        # The body is ONE frame: yield it, then bill — same rule as the
        # calculation stream. Nothing below runs for a client that is gone.
        yield json.dumps(body)
        _debit(sb, ctx, pool, charge, meta, request_id)
        analytics_capture(ctx.org_id, "partner_zoe_completed", {"charged": charge, "stream": False})

    return StreamingResponse(deliver(), media_type="application/json")
