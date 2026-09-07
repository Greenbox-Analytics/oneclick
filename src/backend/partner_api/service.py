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
from datetime import UTC, datetime, timedelta

KEY_PREFIX = "mk_live_"
# An inactive key (revoked or expired) drops off the console this long after
# it went inactive. HIDDEN, never deleted: the row stays, so its spend keeps
# counting in the org's totals, series and folder rollups.
INACTIVE_KEY_TTL = timedelta(days=30)
# Explicit columns, so a secret or a hash can never ride along in a list.
KEY_COLUMNS = (
    "id, org_id, label, key_prefix, status, expires_at, revoked_at, folder_id, created_by, created_at, last_used_at"
)
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
    folder_id: str | None = None,
) -> dict:
    """Insert a key row; the returned dict carries the plaintext under 'secret'
    (shown once, never stored). An unknown folder raises ValueError — a key
    filed into another org's folder would leak that folder's name back through
    the usage rollup."""
    if folder_id is not None and not folder_belongs(sb, org_id, folder_id):
        raise ValueError("unknown folder")
    secret_key = KEY_PREFIX + secrets.token_urlsafe(32)
    row = {
        "org_id": org_id,
        "label": label,
        "key_hash": _hash_key(secret_key),
        "key_prefix": secret_key[:12],
        "created_by": created_by,
        "expires_at": expires_at,
        "folder_id": folder_id,
    }
    res = sb.table("partner_api_keys").insert(row).execute()
    stored = dict(res.data[0]) if res.data else dict(row)
    stored["secret"] = secret_key
    stored.pop("key_hash", None)
    return stored


def list_keys(sb, org_id: str) -> list[dict]:
    """An org's LISTED keys — explicit columns, so secrets and hashes can't
    appear, minus the long-inactive ones (is_hidden). Filtered in Python
    because the predicate is two nullable timestamps and a status, and the
    row count per org is tiny."""
    rows = sb.table("partner_api_keys").select(KEY_COLUMNS).eq("org_id", org_id).execute().data or []
    return [r for r in rows if not is_hidden(r)]


def revoke_key(sb, org_id: str, key_id: str) -> bool:
    """Returns False when nothing matched (unknown id, or another org's key)
    so the router can 404 instead of reporting a revoke that never happened.
    Stamps revoked_at: it is the clock the 30-day hiding runs on."""
    res = (
        sb.table("partner_api_keys")
        .update({"status": "revoked", "revoked_at": datetime.now(UTC).isoformat()})
        .eq("org_id", org_id)
        .eq("id", key_id)
        .execute()
    )
    return bool(res.data)


def _expired(row: dict, now: datetime | None = None) -> bool:
    # TIMESTAMPTZ arrives as ISO-8601 with an offset; 3.11's fromisoformat
    # reads it (a trailing "Z" included), so there is nothing to catch.
    raw = row.get("expires_at")
    return bool(raw) and datetime.fromisoformat(raw) <= (now or datetime.now(UTC))


def _stale(raw: str | None, cutoff: datetime) -> bool:
    return bool(raw) and datetime.fromisoformat(raw) < cutoff


def is_hidden(row: dict, now: datetime | None = None) -> bool:
    """Has this key been inactive long enough to drop off the console?

    A revoked row with NO revoked_at (pre-migration data) is never hidden: the
    only honest answer to "how long ago?" is "unknown", and hiding a key an
    admin can still see spend for is worse than showing one too many."""
    cutoff = (now or datetime.now(UTC)) - INACTIVE_KEY_TTL
    if row.get("status") == "revoked" and _stale(row.get("revoked_at"), cutoff):
        return True
    return _stale(row.get("expires_at"), cutoff)


def key_status(row: dict, now: datetime | None = None) -> str:
    """ "active" | "revoked" | "expired" — what the console shows. Revoked wins:
    a key revoked before its expiry was killed, not left to lapse."""
    if row.get("status") == "revoked":
        return "revoked"
    return "expired" if _expired(row, now) else "active"


# ---- folders ----------------------------------------------------------------


def folder_belongs(sb, org_id: str, folder_id: str) -> bool:
    res = sb.table("partner_key_folders").select("id").eq("org_id", org_id).eq("id", folder_id).execute()
    return bool(res.data)


def list_folders(sb, org_id: str) -> list[dict]:
    return (
        sb.table("partner_key_folders")
        .select("id, org_id, name, created_at")
        .eq("org_id", org_id)
        .order("name")
        .execute()
        .data
        or []
    )


def create_folder(sb, org_id: str, name: str) -> dict:
    """Idempotent on the name (case-insensitively): the UI is "type a folder
    name", so typing one that already exists must file the key there rather
    than 409 at someone who did nothing wrong. The returned row carries a
    transient `created` flag — the router pops it and only fires analytics on
    a real insert."""
    name = (name or "").strip()
    if not 1 <= len(name) <= 80:
        raise ValueError("folder name must be 1-80 characters")

    def _match():
        return next((f for f in list_folders(sb, org_id) if (f.get("name") or "").casefold() == name.casefold()), None)

    existing = _match()
    if existing:
        return {**existing, "created": False}
    try:
        res = sb.table("partner_key_folders").insert({"org_id": org_id, "name": name}).execute()
    except Exception as exc:
        # Two admins typing the same name at once: the (org_id, name) unique
        # index is the real arbiter, so lose the race by re-reading, never 500.
        # Same idiom as orgs.service / projects.service.
        if "23505" not in str(exc) and "duplicate key" not in str(exc).lower():
            raise
        raced = _match()
        if raced is None:
            raise
        return {**raced, "created": False}
    row = dict(res.data[0]) if res.data else {"org_id": org_id, "name": name}
    return {**row, "created": True}


def set_key_folder(sb, org_id: str, key_id: str, folder_id: str | None) -> bool:
    """False = nothing moved: the folder is not this org's, or the key isn't
    (unknown id, or another org's). Both scoped by org_id, so a caller can
    never file someone else's key or read a foreign folder's existence."""
    if folder_id is not None and not folder_belongs(sb, org_id, folder_id):
        return False
    res = sb.table("partner_api_keys").update({"folder_id": folder_id}).eq("org_id", org_id).eq("id", key_id).execute()
    return bool(res.data)


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


def already_charged(sb, request_id: str) -> bool:
    """Has this request id been debited already? ONE indexed read on
    idx_credit_ledger_request_id, so a replay under an Idempotency-Key can say
    `credits: 0, replayed: true` (spec 2026-09-06 §3.1). Only meaningful for a
    DERIVED id — a fresh uuid4 can never match, so callers skip the read when
    no header was sent. Advisory only — `debit_credits`'s own p_request_id
    check is the authority. Two identical calls racing, or a failed read,
    both report the price while one is charged: over-reports, never
    under-reports, never fails a delivered run."""
    try:
        res = sb.table("credit_ledger").select("id").eq("request_id", request_id).limit(1).execute()
        return bool(res.data)
    except Exception:
        logging.exception("already_charged read failed request_id=%s", request_id)
        return False


def billing_block(charge: int, request_id: str, *, replayed: bool = False) -> dict:
    """The `billing` object every delivered response carries."""
    block: dict = {"credits": 0 if replayed else charge, "request_id": request_id}
    if replayed:
        block["replayed"] = True
    return block


def unbilled() -> dict:
    """`billing` for an error event or frame: nothing delivered, nothing charged."""
    return {"credits": 0}


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


def key_console(sb, org_id: str) -> dict:
    """The key-console payload both surfaces return — the org admin's console
    and the Msanii admin's. One helper so the two can't drift."""
    keys = list_keys(sb, org_id)
    labels = created_by_labels(sb, org_id, keys)
    for k in keys:
        k["created_by_label"] = labels.get(k["created_by"]) if k.get("created_by") else None
    return {"keys": keys, "folders": list_folders(sb, org_id)}


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

    Returns (result, measured_credits, usage): the sectioned v1 result dict
    (`summary` + `payments`) plus the two pricing inputs, both read INSIDE the
    tracking scope (the accumulator is a contextvar). Terms mode never opens a
    scope, so both are None = "unmeasured" and the caller charges the base.
    The caller prices with ai_pricing.compute_charge — the product's
    three-term formula (base / metered / base + size tail) — never a local
    max().
    """
    import shutil
    from dataclasses import asdict

    from oneclick.royalty_calculator import RoyaltyCalculator
    from partner_api.models import calc_result, to_contract_data
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

        return calc_result([asdict(p) for p in payments]), measured, usage
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
    partner-visible; anything else is ours. Returns (result, measured, usage),
    the result being the sectioned v1 shape (`contract_terms` + `splits`),
    with both pricing inputs read INSIDE the tracking scope, like the calc.
    """
    import shutil

    from oneclick.royalty_calculator import RoyaltyCalculator
    from partner_api.models import from_contract_data, to_partner_splits
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
            "splits": to_partner_splits(
                contract_splits.parse_royalty_splits(contract_data=merged, main_artist_name=main_artist_name or "")
            ),
        }
        return result, measured, usage
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
