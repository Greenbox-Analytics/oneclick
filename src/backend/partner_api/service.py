"""API-key lifecycle + request auth for the partner API.

ONE key type (2026-09-04, owner decision — the backend/license hierarchy is
gone): a key is an org credential. It resolves to its org, spends that org's
pool, and can do everything the API offers. Only SHA-256 hashes are stored;
the plaintext leaves this module exactly once, in mint_key's return value.
"""

import hashlib
import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

KEY_PREFIX = "mk_live_"
# The API's own credit_prices rows — never the product's. One per tool; the
# partner deliverable is priced independently of the in-app one.
ONECLICK_ACTION = "partner_oneclick_run"
REGISTRY_ACTION = "partner_registry_parse"
SPLIT_SHEET_ACTION = "partner_split_sheet"


def partner_api_enabled() -> bool:
    # Only "true", exactly like credits_enabled / licensing_enabled.
    return os.getenv("PARTNER_API_ENABLED", "").strip().lower() == "true"


def partner_surface_enabled() -> bool:
    """The router gate: the partner surface depends on prices/wallets (credits)
    and orgs (licensing) as much as on its own flag."""
    from subscriptions.service import credits_enabled, licensing_enabled

    return partner_api_enabled() and credits_enabled() and licensing_enabled()


@dataclass
class PartnerContext:
    org_id: str
    key_id: str


def _hash_key(secret_key: str) -> str:
    return hashlib.sha256(secret_key.encode()).hexdigest()


def mint_key(
    sb,
    org_id: str,
    *,
    label: str,
    created_by: str | None = None,
    expires_at: str | None = None,
) -> dict:
    """Insert a key row; the returned dict carries the plaintext under 'secret'
    (shown once, never stored)."""
    secret_key = KEY_PREFIX + secrets.token_urlsafe(32)
    row = {
        "org_id": org_id,
        "label": label,
        "key_hash": _hash_key(secret_key),
        "key_prefix": secret_key[:12],
        "created_by": created_by,
        "expires_at": expires_at,
    }
    res = sb.table("partner_api_keys").insert(row).execute()
    stored = dict(res.data[0]) if res.data else dict(row)
    stored["secret"] = secret_key
    stored.pop("key_hash", None)
    return stored


def list_keys(sb, org_id: str) -> list[dict]:
    """An org's keys — explicit columns, so secrets and hashes can't appear."""
    return (
        sb.table("partner_api_keys")
        .select("id, org_id, label, key_prefix, status, expires_at, created_by, created_at, last_used_at")
        .eq("org_id", org_id)
        .execute()
        .data
        or []
    )


def revoke_key(sb, org_id: str, key_id: str) -> bool:
    """Returns False when nothing matched (unknown id, or another org's key)
    so the router can 404 instead of reporting a revoke that never happened."""
    res = sb.table("partner_api_keys").update({"status": "revoked"}).eq("org_id", org_id).eq("id", key_id).execute()
    return bool(res.data)


def _expired(row: dict) -> bool:
    # TIMESTAMPTZ arrives as ISO-8601 with an offset; 3.11's fromisoformat
    # reads it (a trailing "Z" included), so there is nothing to catch.
    raw = row.get("expires_at")
    return bool(raw) and datetime.fromisoformat(raw) <= datetime.now(UTC)


def resolve_key(sb, bearer: str | None) -> PartnerContext | None:
    """Bearer secret -> PartnerContext, or None (caller maps None to 401/403).
    Checks: key active + unexpired; org active, not archived, partner_api_enabled."""
    if not bearer or not bearer.startswith(KEY_PREFIX):
        return None
    res = (
        sb.table("partner_api_keys")
        .select("id, org_id, status, expires_at")
        .eq("key_hash", _hash_key(bearer))
        .eq("status", "active")
        .execute()
    )
    if not res.data:
        return None
    row = res.data[0]
    if _expired(row):
        return None
    org_res = (
        sb.table("organizations")
        .select("id, status, archived_at, partner_api_enabled")
        .eq("id", row["org_id"])
        .execute()
    )
    if not org_res.data:
        return None
    org = org_res.data[0]
    if org.get("archived_at") or org.get("status") != "active" or not org.get("partner_api_enabled"):
        return None
    # Best-effort usage stamp — never let it fail a request.
    try:
        sb.table("partner_api_keys").update({"last_used_at": datetime.now(UTC).isoformat()}).eq(
            "id", row["id"]
        ).execute()
    except Exception:
        pass
    return PartnerContext(org_id=row["org_id"], key_id=row["id"])


# ---- billing ----------------------------------------------------------------


def get_price(sb, action: str = ONECLICK_ACTION) -> int:
    """The BASE for a partner action (partner_oneclick_run, partner_registry_parse,
    partner_split_sheet, partner_zoe_message)
    from credit_prices (public-read table). The base is the floor; the charge
    itself comes from ai_pricing.compute_charge. A missing row is a deploy
    error, never a free run."""
    res = sb.table("credit_prices").select("credits").eq("action", action).execute()
    if not res.data:
        raise RuntimeError(f"no credit price seeded for {action}")
    return int(res.data[0]["credits"])


def check_pool(sb, org_id: str, price: int) -> dict:
    """THE billing gate. debit_credits deliberately tolerates overdraft
    (concurrency drift lands on the bundle, which may go negative) and no
    member cap applies on the partner path — so this pre-check is the only
    authority. Mirrors _check_credits_org's balance >= price comparison.

    `period_end` rides along for derive_request_id — the dedupe id must be
    scoped to the pool's billing period, and this is the read that has it."""
    from orgs.wallets import read_or_create_org_wallet

    pool = read_or_create_org_wallet(sb, org_id)
    balance = (pool.get("bundle_balance") or 0) + (pool.get("reserve_balance") or 0)
    return {
        "ok": balance >= price,
        "balance": balance,
        "wallet_id": pool.get("id"),
        "period_end": pool.get("period_end"),
    }


def derive_request_id(
    key_id: str, idempotency_key: str | None, payload_fingerprint: str, period_end: str | None
) -> str:
    """debit_credits dedupes on p_request_id. Namespaced by key id (one
    partner's retry header can never collide with another's, or with internal
    uuid4 debits), bound to the payload fingerprint — otherwise one header
    value ridden across DIFFERENT payloads would make every run after the
    first a free duplicate debit — AND to the pool's billing period, matching
    the product path (enforcement.gated_credits). Without the period term
    idx_credit_ledger_request_id, a global never-expiring unique index, would
    keep matching that first row forever: one pinned header would buy a year
    of runs for one charge. Same key + same deliverable + same period: charged
    once. New deliverable, or a new period: pays. No header => fresh uuid4:
    each retry pays."""
    if idempotency_key:
        period = period_end or "noperiod"
        return str(uuid.uuid5(uuid.UUID(key_id), f"{idempotency_key}:{payload_fingerprint}:{period}"))
    return str(uuid.uuid4())


def debit_run(
    sb,
    *,
    wallet_id: str,
    amount: int,
    request_id: str,
    key_id: str,
    metadata: dict | None = None,
    action: str = ONECLICK_ACTION,
):
    """Direct RPC on purpose: debit_for_action builds its own metadata from a
    CreditGrant and has no hook for partner attribution. `metadata` is the
    compute_charge explanation (base / metered / tail / tokens) — the SAME
    keys a product row carries — with the key id layered on top (what
    get_org_usage.byKey groups on). No p_member_id — a key is the ORG's
    credential, not a member's, so no cap counter moves.

    `amount` is compute_charge's result, NEVER get_price's base — the name
    says so because the base is the floor, not the charge. Returns the RPC's
    result so a caller can tell a real debit from {"duplicate": true}."""
    res = sb.rpc(
        "debit_credits",
        {
            "p_wallet_id": wallet_id,
            "p_amount": amount,
            "p_action": action,
            "p_request_id": request_id,
            "p_kind": "debit",
            "p_metadata": {
                **(metadata or {}),
                "source": "partner_api",
                "partner_key_id": key_id,
            },
        },
    ).execute()
    return res.data


# ---- phase-2 portal: Created-by labels --------------------------------------


def created_by_labels(sb, org_id: str, keys: list[dict]) -> dict[str, str]:
    """user_id -> display email for the Created by column.

    One org_members read for the org (creators are normally members, and a
    REMOVED member's row still carries the email — which is exactly the case
    the column exists for), then the auth-admin lookup ONCE per id that read
    did not cover. Best-effort, never raises: an unlabelled column must not
    fail the key list.
    """
    from orgs.service import _resolve_user_email

    ids = {k["created_by"] for k in keys if k.get("created_by")}
    if not ids:
        return {}
    labels: dict[str, str] = {}
    try:
        rows = (
            sb.table("org_members")
            .select("user_id, email")
            .eq("org_id", org_id)
            .in_("user_id", sorted(ids))
            .execute()
            .data
            or []
        )
        for r in rows:
            if r.get("user_id") and r.get("email"):
                labels[r["user_id"]] = r["email"]
        for uid in sorted(ids - set(labels)):
            email = _resolve_user_email(sb, uid)
            if email:
                labels[uid] = email
    except Exception:
        logging.exception("created_by_labels failed org=%s", org_id)
    return labels


# ---- calculation ------------------------------------------------------------


def run_partner_calc(
    sb,
    *,
    org_id: str,
    tmpdir: str,
    statement_path: str,
    contract_paths: list[str],
    contract_terms,  # PartnerContractTerms | None
    expenses: list[dict] | None,
) -> tuple[dict, int | None, dict | None]:
    """Synchronous calc pipeline (runs in a worker thread). Raises
    CalculationError for partner-visible failures.

    OWNS tmpdir cleanup: asyncio.to_thread can't be cancelled, so on client
    disconnect the generator dies while this thread still reads the files —
    cleanup must live where the files are used, in this finally.

    File mode: pdf -> markdown -> get_or_parse (shared cache — deliberately
    cross-tenant, keyed by SHA-256 of the parse text) -> merge -> calc.
    Terms mode: DTO -> ContractData -> calc. No LLM touched.

    Returns (result, measured_credits, usage): the frozen v1 result dict plus
    the two pricing inputs, both read INSIDE the tracking scope (the
    accumulator is a contextvar). Terms mode never opens a scope, so both are
    None = "unmeasured" and the caller charges the base. The caller prices
    with ai_pricing.compute_charge — the product's three-term formula (base /
    metered / base + size tail) — never a local max().
    """
    import shutil
    from dataclasses import asdict

    from oneclick.royalty_calculator import RoyaltyCalculator
    from partner_api.models import payment_to_dto, to_contract_data
    from utils.contract_parsing.cache import get_or_parse
    from utils.ingestion.pdf_markdown import pdf_to_markdown
    from utils.llm.tracking import credits_for_llm_usage, llm_usage_snapshot, set_llm_context

    try:
        calc = RoyaltyCalculator()
        measured = None
        usage = None

        # Order is load-bearing: cheap validation before billable work. Every
        # statement failure is a CalculationError, and the router returns
        # without debiting — parsing first would hand out free LLM runs.
        song_totals = calc.read_royalty_statement(statement_path)

        if contract_terms is not None:
            merged = to_contract_data(contract_terms)
        else:
            with set_llm_context(org_id, "oneclick_partner"):
                datas = []
                for path in contract_paths:
                    md = pdf_to_markdown(path)
                    datas.append(get_or_parse(sb, (lambda m=md: m)))
                # Both reads MUST happen inside the scope: the accumulator
                # resets when it exits. None from credits_for_llm_usage means
                # a model missing from MODEL_RATES — unmeasurable, base only.
                measured = credits_for_llm_usage()
                usage = llm_usage_snapshot()
            merged = calc.merge_contracts(datas) if len(datas) > 1 else datas[0]

        payments = calc._calculate_payments_from_data(merged, song_totals, expenses=expenses)

        dtos = [payment_to_dto(asdict(p)) for p in payments]
        result = {
            "payments": [d.model_dump() for d in dtos],
            "total_payments": len(dtos),
            "expense_review_required": any(d.basis == "net" for d in dtos),
        }
        return result, measured, usage
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---- contract parse ----------------------------------------------------------


def run_partner_parse(
    sb,
    *,
    org_id: str,
    tmpdir: str,
    contract_paths: list[str],
    main_artist_name: str = "",
) -> tuple[dict, int | None, dict | None]:
    """Synchronous parse pipeline for /registry/v1/splits (worker
    thread). OWNS tmpdir cleanup, for the same reason run_partner_calc does.

    pdf -> markdown -> get_or_parse (the shared parse cache — deliberately
    cross-tenant, keyed by SHA-256 of the contract text, holding no org data)
    -> merge -> two views of one contract: `contract_terms`, the frozen v1
    terms DTO that /oneclick/v1/royalties accepts verbatim, and `splits`, the
    Registry's per-party master / publishing / SoundExchange pivot.

    ValueError = the contract could not be read (no text, not a PDF) and is
    partner-visible; anything else is ours. Returns (result, measured, usage)
    with both pricing inputs read INSIDE the tracking scope, like the calc.
    """
    import shutil

    from oneclick.royalty_calculator import RoyaltyCalculator
    from partner_api.models import from_contract_data
    from registry import contract_splits
    from utils.contract_parsing.cache import get_or_parse
    from utils.ingestion.pdf_markdown import pdf_to_markdown
    from utils.llm.tracking import credits_for_llm_usage, llm_usage_snapshot, set_llm_context

    try:
        with set_llm_context(org_id, "registry_partner"):
            datas = []
            for path in contract_paths:
                md = pdf_to_markdown(path)
                datas.append(get_or_parse(sb, (lambda m=md: m)))
            measured = credits_for_llm_usage()
            usage = llm_usage_snapshot()
        merged = RoyaltyCalculator().merge_contracts(datas) if len(datas) > 1 else datas[0]
        result = {
            "contract_terms": from_contract_data(merged).model_dump(),
            "splits": contract_splits.parse_royalty_splits(
                contract_data=merged, main_artist_name=main_artist_name or ""
            ),
        }
        return result, measured, usage
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
