"""LLM usage tracking: contextvar + OpenAI client proxy → ai_usage_log.

Endpoints set (user_id, tool) via set_llm_context(); TrackedOpenAI intercepts
chat.completions.create / embeddings.create, records response.usage with real
cost (subscriptions/ai_pricing.py), and delegates everything else untouched.

Best-effort by design: a logging failure must never break a user-facing call.
No context set → no row (covers scripts/tests/background jobs).
"""

import logging
import threading
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import partial
from typing import Any

from subscriptions.ai_pricing import credits_for_cost, estimate_cost_usd

logger = logging.getLogger(__name__)

# (user_id, tool) for the current request; None outside a tracked endpoint.
llm_call_context: ContextVar[tuple[str, str] | None] = ContextVar("llm_call_context", default=None)

# Running LLM spend for the current request, accumulated by _record_with_ctx.
# This is what makes a credit charge proportional to tokens actually burned
# rather than a flat per-action price. None outside a tracked scope (scripts,
# background jobs, tests) — callers then fall back to the flat estimate.
llm_usage_totals: ContextVar[dict | None] = ContextVar("llm_usage_totals", default=None)

# ponytail: one global lock, not per-accumulator. `d[k] += n` is a
# read-modify-write that OneClick's submit_with_context worker threads can
# genuinely interleave on, and this is a money path. Contention is a handful of
# LLM calls per request — go per-accumulator only if that ever shows up.
_totals_lock = threading.Lock()


def new_usage_totals() -> dict:
    """A fresh per-request LLM spend accumulator.

    `unpriced` counts calls whose model is missing from MODEL_RATES: their cost
    is unknowable, so the measured total is incomplete and callers must fall
    back to the flat estimate rather than under-charging.
    """
    return {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0, "unpriced": 0}


def _accumulate(totals: dict | None, *, cost: float | None, input_tokens: int, output_tokens: int) -> None:
    if totals is None:
        return
    with _totals_lock:
        totals["calls"] += 1
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        if cost is None:
            totals["unpriced"] += 1
        else:
            totals["cost_usd"] += cost


def llm_usage_snapshot() -> dict | None:
    """Copy of the current scope's accumulated spend, or None outside a scope."""
    totals = llm_usage_totals.get()
    if totals is None:
        return None
    with _totals_lock:
        return dict(totals)


def credits_for_llm_usage() -> int | None:
    """Credits owed for the LLM work done so far in the current tracked scope.

    None means "not measurable" — no scope active, or at least one call used a
    model absent from MODEL_RATES — and the caller must charge the flat
    estimate instead. A scope that made no LLM calls returns 0, which is how a
    cache hit becomes free without any per-tool special-casing.
    """
    totals = llm_usage_snapshot()
    if totals is None or totals["unpriced"]:
        return None
    return credits_for_cost(totals["cost_usd"])


@contextmanager
def set_llm_context(user_id: str, tool: str):
    """Scope LLM usage attribution to (user_id, tool) for the duration.

    Yields the spend accumulator, so a caller that needs the totals AFTER the
    block can hold onto it. Everything inside the block can read it via
    credits_for_llm_usage() instead.

    WORKS for: plain request/response handlers, and ASYNC generator bodies
    (StreamingResponse consumes an async generator inside one task, so a set()
    during the first __anext__ persists for later steps).
    DOES NOT survive SYNC generators under StreamingResponse — Starlette
    advances those in a threadpool where each next() gets a fresh context
    copy. Use iter_with_llm_context() for sync generators instead.
    """
    totals = new_usage_totals()
    token = llm_call_context.set((user_id, tool))
    totals_token = llm_usage_totals.set(totals)
    try:
        yield totals
    finally:
        llm_usage_totals.reset(totals_token)
        llm_call_context.reset(token)


def iter_with_llm_context(user_id: str, tool: str, inner):
    """Wrap a SYNC generator so the LLM context is live during EVERY resumption.

    Each next() re-binds the contextvars around the inner step, so LLM calls
    made anywhere inside that step see them — regardless of which threadpool
    thread Starlette runs the step on. ONE accumulator spans the whole stream,
    so a charge taken at the terminal event covers every call the stream made.
    """
    it = iter(inner)
    totals = new_usage_totals()
    while True:
        token = llm_call_context.set((user_id, tool))
        totals_token = llm_usage_totals.set(totals)
        try:
            item = next(it)
        except StopIteration:
            return
        finally:
            llm_usage_totals.reset(totals_token)
            llm_call_context.reset(token)
        yield item


def submit_with_context(executor, fn, /, *args, **kwargs):
    """executor.submit() that propagates contextvars into the worker thread.

    ThreadPoolExecutor.submit does NOT copy the caller's context, so LLM calls
    made inside workers would read llm_call_context as None and go unlogged.
    The context is copied HERE, at submit time in the parent thread.
    """
    ctx = copy_context()
    return executor.submit(ctx.run, partial(fn, *args, **kwargs))


def _default_get_supabase():
    from main import get_supabase_client  # lazy: avoid import cycle

    return get_supabase_client()


def _record(get_supabase: Callable, *, model: str, usage: Any, success: bool) -> None:
    _record_with_ctx(
        get_supabase,
        ctx=llm_call_context.get(),
        totals=llm_usage_totals.get(),
        model=model,
        usage=usage,
        success=success,
    )


def _record_with_ctx(
    get_supabase: Callable,
    *,
    ctx: tuple[str, str] | None,
    model: str,
    usage: Any,
    success: bool,
    totals: dict | None = None,
) -> None:
    if ctx is None:
        return
    user_id, tool = ctx
    try:
        input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
        cached_tokens = getattr(details, "cached_tokens", 0) or 0 if details is not None else 0
        cost = (
            estimate_cost_usd(model, input_tokens or 0, output_tokens or 0, cached_tokens)
            if input_tokens is not None
            else None
        )
        # Accumulate BEFORE the insert: the credit charge must not depend on
        # the analytics write succeeding. A call that reported no usage at all
        # (a failure before the API answered) contributes nothing but is NOT
        # counted unpriced — it burned no tokens, so it owes no credits.
        if input_tokens is not None or output_tokens is not None:
            _accumulate(totals, cost=cost, input_tokens=input_tokens or 0, output_tokens=output_tokens or 0)
        get_supabase().table("ai_usage_log").insert(
            {
                "user_id": user_id,
                "tool": tool,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached_tokens,
                "cost_usd": cost,
                "success": success,
            }
        ).execute()
    except Exception:
        logger.exception("ai_usage_log write failed (user=%s tool=%s model=%s)", user_id, tool, model)


class _UsageCapturingStream:
    """Wraps an OpenAI streaming response; records usage from the terminal chunk.

    stream=True calls return usage only on the FINAL chunk, and only when
    stream_options={"include_usage": True} was requested (the final chunk has
    empty `choices` — existing consumer loops that guard on `chunk.choices`
    tolerate it). Identity (user_id, tool) AND the spend accumulator are both
    SNAPSHOTTED at create() time because sync generators under StreamingResponse
    advance in threadpool steps where the contextvars may no longer be set.
    """

    def __init__(
        self,
        inner_stream,
        get_supabase: Callable,
        model: str,
        ctx: tuple[str, str] | None,
        totals: dict | None = None,
    ):
        self._inner = inner_stream
        self._get_supabase = get_supabase
        self._model = model
        self._ctx = ctx
        self._totals = totals
        self._usage = None
        self._recorded = False
        self._gen = None

    def _iterate(self):
        try:
            for chunk in self._inner:
                if getattr(chunk, "usage", None) is not None:
                    self._usage = chunk.usage
                    # self._model stays the REQUESTED model — chunks echo
                    # versioned snapshot ids that would miss MODEL_RATES
                    # and null out cost_usd.
                yield chunk
            self._finish(success=True)
        finally:
            # Client disconnects raise GeneratorExit (a BaseException) at the
            # yield — an `except Exception` never sees it. The guarded finally
            # records best-effort in EVERY exit path: normal completion already
            # recorded success=True above (this no-ops); any abnormal exit
            # (exception OR disconnect) records success=False with whatever
            # usage arrived before the cut.
            self._finish(success=False)

    def _finish(self, *, success: bool) -> None:
        if self._recorded:
            return
        self._recorded = True
        _record_with_ctx(
            self._get_supabase,
            ctx=self._ctx,
            totals=self._totals,
            model=self._model,
            usage=self._usage,
            success=success,
        )

    # __iter__ resolves on the TYPE, not through __getattr__. The sole consumer
    # iterates (`for chunk in response`), routing through ONE capturing generator
    # so usage is never silently skipped.
    def __iter__(self):
        if self._gen is None:
            self._gen = self._iterate()
        return self._gen

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TrackedCompletions:
    def __init__(self, inner_completions, get_supabase: Callable):
        self._inner = inner_completions
        self._get_supabase = get_supabase

    def create(self, *args, **kwargs):
        model = kwargs.get("model", "unknown")
        if kwargs.get("stream"):
            # Ask OpenAI for the terminal usage chunk (merge, don't clobber).
            opts = dict(kwargs.get("stream_options") or {})
            opts.setdefault("include_usage", True)
            kwargs["stream_options"] = opts
            # Snapshot NOW — see _UsageCapturingStream.
            ctx = llm_call_context.get()
            totals = llm_usage_totals.get()
            try:
                stream = self._inner.create(*args, **kwargs)
            except Exception:
                _record_with_ctx(self._get_supabase, ctx=ctx, totals=totals, model=model, usage=None, success=False)
                raise
            return _UsageCapturingStream(stream, self._get_supabase, model, ctx, totals)
        try:
            resp = self._inner.create(*args, **kwargs)
        except Exception:
            _record(self._get_supabase, model=model, usage=None, success=False)
            raise
        # Rate lookup keys on the REQUESTED model: the API echoes versioned
        # snapshot ids (e.g. gpt-5-mini-2026-…) that would miss MODEL_RATES
        # and null out cost_usd.
        _record(self._get_supabase, model=model, usage=getattr(resp, "usage", None), success=True)
        return resp


class _TrackedChat:
    def __init__(self, inner_chat, get_supabase: Callable):
        self._inner = inner_chat
        self.completions = _TrackedCompletions(inner_chat.completions, get_supabase)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TrackedEmbeddings:
    def __init__(self, inner_embeddings, get_supabase: Callable):
        self._inner = inner_embeddings
        self._get_supabase = get_supabase

    def create(self, *args, **kwargs):
        model = kwargs.get("model", "unknown")
        try:
            resp = self._inner.create(*args, **kwargs)
        except Exception:
            _record(self._get_supabase, model=model, usage=None, success=False)
            raise
        # Embeddings usage has prompt_tokens only; completion defaults handled in _record.
        # Rate lookup keys on the REQUESTED model: the API echoes versioned
        # snapshot ids that would miss MODEL_RATES and null out cost_usd.
        _record(self._get_supabase, model=model, usage=getattr(resp, "usage", None), success=True)
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TrackedOpenAI:
    """Transparent proxy around an OpenAI client that logs usage per call.

    Wrap at client-construction seams (utils/llm/client.py, zoe_chatbot) so
    existing call sites need zero changes.
    """

    def __init__(self, inner, get_supabase: Callable | None = None):
        self._inner = inner
        gs = get_supabase or _default_get_supabase
        self.chat = _TrackedChat(inner.chat, gs)
        self.embeddings = _TrackedEmbeddings(inner.embeddings, gs)

    def __getattr__(self, name):
        return getattr(self._inner, name)
