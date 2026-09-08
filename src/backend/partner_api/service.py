"""API-key lifecycle + request auth for the partner API.

ONE key type: a key is an org credential — it resolves to its org, spends that
org's pool, and can do everything the API offers. Only SHA-256 hashes are
stored; the plaintext leaves this module once, in mint_key's return value.
"""

import hashlib
import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

KEY_PREFIX = "mk_live_"
# An inactive key is HIDDEN from the console after this, never deleted: the row
# stays, so its spend keeps counting in the org's totals and rollups.
INACTIVE_KEY_TTL = timedelta(days=30)
# Explicit columns, so a secret or a hash can never ride along in a list.
KEY_COLUMNS = (
    "id, org_id, label, key_prefix, status, expires_at, revoked_at, folder_id, created_by, created_at, last_used_at"
)
# The API's own credit_prices rows — each partner deliverable is priced
# independently of the in-app one.
ONECLICK_ACTION = "partner_oneclick_run"
REGISTRY_ACTION = "partner_registry_parse"
SPLIT_SHEET_ACTION = "partner_split_sheet"


def partner_api_enabled() -> bool:
    # Only "true", exactly like credits_enabled / licensing_enabled.
    return os.getenv("PARTNER_API_ENABLED", "").strip().lower() == "true"


def partner_surface_enabled() -> bool:
    """Router gate: the surface needs prices/wallets (credits) and orgs
    (licensing) as much as its own flag."""
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
    (shown once, never stored). Unknown folder -> ValueError: filing a key into
    another org's folder would leak that folder's name via the usage rollup."""
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
    """An org's LISTED keys, minus the long-inactive ones. Filtered in Python:
    the predicate is two nullable timestamps and a status, over few rows."""
    rows = sb.table("partner_api_keys").select(KEY_COLUMNS).eq("org_id", org_id).execute().data or []
    return [r for r in rows if not is_hidden(r)]


def revoke_key(sb, org_id: str, key_id: str) -> bool:
    """False when nothing matched (unknown id, or another org's key), so the
    router 404s instead of reporting a revoke that never happened. revoked_at
    is the clock the 30-day hiding runs on."""
    res = (
        sb.table("partner_api_keys")
        .update({"status": "revoked", "revoked_at": datetime.now(UTC).isoformat()})
        .eq("org_id", org_id)
        .eq("id", key_id)
        .execute()
    )
    return bool(res.data)


def _expired(row: dict, now: datetime | None = None) -> bool:
    raw = row.get("expires_at")
    return bool(raw) and datetime.fromisoformat(raw) <= (now or datetime.now(UTC))


def _stale(raw: str | None, cutoff: datetime) -> bool:
    return bool(raw) and datetime.fromisoformat(raw) < cutoff


def is_hidden(row: dict, now: datetime | None = None) -> bool:
    """Inactive long enough to drop off the console? A revoked row with no
    revoked_at (pre-migration) is never hidden — "how long ago?" has no honest
    answer, and showing one key too many beats hiding one."""
    cutoff = (now or datetime.now(UTC)) - INACTIVE_KEY_TTL
    if row.get("status") == "revoked" and _stale(row.get("revoked_at"), cutoff):
        return True
    return _stale(row.get("expires_at"), cutoff)


def key_status(row: dict, now: datetime | None = None) -> str:
    """What the console shows. Revoked wins: a key revoked before its expiry
    was killed, not left to lapse."""
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
    """Idempotent on the name, case-insensitively: the UI is "type a folder
    name", so re-typing an existing one files the key there rather than 409ing.
    The transient `created` flag lets the router fire analytics on a real
    insert only."""
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
        # index is the arbiter, so lose the race by re-reading, never 500.
        if "23505" not in str(exc) and "duplicate key" not in str(exc).lower():
            raise
        raced = _match()
        if raced is None:
            raise
        return {**raced, "created": False}
    row = dict(res.data[0]) if res.data else {"org_id": org_id, "name": name}
    return {**row, "created": True}


def set_key_folder(sb, org_id: str, key_id: str, folder_id: str | None) -> bool:
    """False = nothing moved: the folder or the key is not this org's. Both
    scoped by org_id, so a caller can never touch another org's rows."""
    if folder_id is not None and not folder_belongs(sb, org_id, folder_id):
        return False
    res = sb.table("partner_api_keys").update({"folder_id": folder_id}).eq("org_id", org_id).eq("id", key_id).execute()
    return bool(res.data)


def resolve_key(sb, bearer: str | None) -> PartnerContext | None:
    """Bearer secret -> PartnerContext, or None (the caller 401s). Checks: key
    active + unexpired; org active, not archived, partner_api_enabled."""
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


# ---- request log + rate limit -----------------------------------------------

# Per-KEY, because a key is the unit of compromise: a leaked one drains the org
# pool as fast as it can dial, and until now the pool balance was the only
# ceiling. The count covers EVERY logged attempt in the window, refusals
# included, so a flood stays refused instead of oscillating at the limit.
RATE_LIMIT_WINDOW = timedelta(seconds=60)
DEFAULT_RATE_LIMIT_PER_MIN = 60
DEFAULT_LOG_RETENTION_DAYS = 30
# user_agent is caller-controlled and unbounded.
MAX_USER_AGENT = 200


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        logging.warning("%s is not an integer; using %d", name, default)
        return default


def rate_limit_per_min() -> int:
    """Requests per key per minute. 0 disables the limiter — an ops escape
    hatch, the same posture as BYPASS_PAYWALLS."""
    return _env_int("PARTNER_RATE_LIMIT_PER_MIN", DEFAULT_RATE_LIMIT_PER_MIN)


def presented_prefix(bearer: str | None) -> str | None:
    """The 12 chars a failed auth may record. None unless the value is one of
    OURS: a secret mis-pasted from another system must not leave a fragment in
    our database."""
    if not bearer or not bearer.startswith(KEY_PREFIX):
        return None
    return bearer[:12]


def client_ip(request) -> str | None:
    """The caller's IP. Cloud Run APPENDS the real client to whatever
    X-Forwarded-For the caller sent, so the LAST entry is the trustworthy one
    and everything before it is caller-controlled. (Behind a global load
    balancer Google adds its own hop and the real client moves to
    second-to-last — revisit here if an LB is ever put in front.)"""
    if request is None:
        return None
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[-1].strip()[:64] or None
    return getattr(getattr(request, "client", None), "host", None)


def log_request(sb, request, *, outcome: str, key_id=None, org_id=None, key_prefix=None) -> None:
    """One row per auth attempt: the rate-limit counter AND the only record of
    where a key is used from. Best-effort — an audit write must never fail a
    paid request."""
    try:
        sb.table("partner_api_requests").insert(
            {
                "key_id": key_id,
                "org_id": org_id,
                "key_prefix": key_prefix,
                "client_ip": client_ip(request),
                "user_agent": ((request.headers.get("user-agent") or "")[:MAX_USER_AGENT] or None) if request else None,
                "path": request.url.path if request else None,
                "outcome": outcome,
            }
        ).execute()
    except Exception:
        logging.exception("partner request log failed outcome=%s key=%s", outcome, key_id)


def rate_limited(sb, key_id: str) -> bool:
    """Is this key over its per-minute ceiling? A rolling 60s count, not a fixed
    bucket, so a burst cannot straddle a boundary and land 2x the limit.

    Fails OPEN: a broken counter must not take the API down for every partner.
    The pool balance is still a hard ceiling underneath."""
    limit = rate_limit_per_min()
    if limit <= 0:
        return False
    try:
        since = (datetime.now(UTC) - RATE_LIMIT_WINDOW).isoformat()
        res = (
            sb.table("partner_api_requests")
            .select("id", count="exact")
            .eq("key_id", key_id)
            .gte("created_at", since)
            .execute()
        )
        return (res.count or 0) >= limit
    except Exception:
        logging.exception("partner rate-limit check failed key=%s; allowing", key_id)
        return False


def purge_request_log(sb) -> int:
    """Drop aged-out rows; called by the daily billing sweep. The log is a
    rolling window — per-key SPEND history lives on credit_ledger.metadata and
    is never purged."""
    days = _env_int("PARTNER_LOG_RETENTION_DAYS", DEFAULT_LOG_RETENTION_DAYS)
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    res = sb.table("partner_api_requests").delete().lt("created_at", cutoff).execute()
    return len(res.data or [])


def recent_client_ips(sb, key_id: str, since: str, limit: int = 20) -> list[dict]:
    """Distinct source IPs for one key since `since`, busiest first. Empty on a
    read failure — a lookup that cannot show IPs is still worth answering."""
    try:
        rows = (
            sb.table("partner_api_requests")
            .select("client_ip, created_at")
            .eq("key_id", key_id)
            .gte("created_at", since)
            .order("created_at", desc=True)
            .limit(2000)
            .execute()
            .data
            or []
        )
    except Exception:
        logging.exception("recent_client_ips failed key=%s", key_id)
        return []
    seen: dict[str, dict] = {}
    for r in rows:
        ip = r.get("client_ip")
        if not ip:
            continue
        # Newest-first, so the first sighting of an IP IS its last_seen.
        hit = seen.setdefault(ip, {"ip": ip, "requests": 0, "last_seen": r.get("created_at")})
        hit["requests"] += 1
    return sorted(seen.values(), key=lambda h: h["requests"], reverse=True)[:limit]


def lookup_by_prefix(sb, presented: str, *, ip_window_days: int = 7) -> list[dict]:
    """ "We found a key in the wild — whose is it?" Matches on the 12-char
    prefix, so an admin can paste a whole key and the secret still never
    reaches a query string, a browser history or an access log.

    A LIST, not a row: after mk_live_ the prefix is 4 characters, so a collision
    is unlikely but possible — org and label disambiguate. Long-inactive keys
    are INCLUDED (is_hidden is deliberately not applied): a key leaked a year
    ago and already revoked is exactly what someone looks up.

    Each hit carries its recent distinct client IPs, which is the actual leak
    signal — one key answering from two continents is not a busy partner."""
    prefix = presented_prefix((presented or "").strip())
    if prefix is None:
        raise ValueError("not a Msanii API key prefix")
    keys = sb.table("partner_api_keys").select(KEY_COLUMNS).eq("key_prefix", prefix).execute().data or []
    if not keys:
        return []
    org_ids = sorted({k["org_id"] for k in keys if k.get("org_id")})
    orgs = sb.table("organizations").select("id, name").in_("id", org_ids).execute().data or []
    names = {o["id"]: o.get("name") for o in orgs}
    since = (datetime.now(UTC) - timedelta(days=ip_window_days)).isoformat()
    for k in keys:
        k["org_name"] = names.get(k.get("org_id"))
        # Derived, like the usage rollup: the column says nothing about expiry.
        k["status"] = key_status(k)
        k["recent_ips"] = recent_client_ips(sb, k["id"], since)
    return keys


# ---- billing ----------------------------------------------------------------


def get_price(sb, action: str) -> int:
    """The BASE for a partner action, from credit_prices. A floor, not the
    charge — that comes from ai_pricing.compute_charge. A missing row is a
    deploy error, never a free run."""
    res = sb.table("credit_prices").select("credits").eq("action", action).execute()
    if not res.data:
        raise RuntimeError(f"no credit price seeded for {action}")
    return int(res.data[0]["credits"])


def check_pool(sb, org_id: str, price: int) -> dict:
    """THE billing gate: debit_credits tolerates overdraft and no member cap
    applies to a key, so this pre-check is the only authority. `period_end`
    rides along for derive_request_id, which scopes the dedupe id to the
    pool's billing period."""
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
    """The id debit_credits dedupes on. Same key + same deliverable + same
    period is charged once; a new deliverable or a new period pays; no header
    means a fresh uuid4, so every retry pays.

    All three terms are load-bearing. The key id keeps one partner's header
    from colliding with another's. The fingerprint stops one header ridden
    across different payloads making every later run a free duplicate. The
    period stops idx_credit_ledger_request_id — global and never-expiring —
    matching that first row forever, which would buy a year of runs for one
    charge."""
    if idempotency_key:
        period = period_end or "noperiod"
        return str(uuid.uuid5(uuid.UUID(key_id), f"{idempotency_key}:{payload_fingerprint}:{period}"))
    return str(uuid.uuid4())


def already_charged(sb, request_id: str) -> bool:
    """Already debited? One indexed read, so a replay under an Idempotency-Key
    can report `credits: 0, replayed: true`. Only meaningful for a DERIVED id.
    Advisory — debit_credits' own check is the authority — and it fails safe:
    a race or a failed read over-reports the price, never under-reports."""
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
    """Direct RPC on purpose: debit_for_action builds metadata from a
    CreditGrant and has no hook for partner attribution. `metadata` is
    compute_charge's explanation plus the key id (what byKey groups on). No
    p_member_id: a key is the ORG's credential, so no cap counter moves.
    `amount` is compute_charge's result, NEVER get_price's base."""
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
    """user_id -> display email for the Created by column. One org_members read
    (a REMOVED member's row still carries the email, which is the case the
    column exists for), then one auth-admin lookup per id it missed.
    Best-effort: an unlabelled column must not fail the key list."""
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
    """The key-console payload both the org admin's console and the Msanii
    admin's return, so the two can't drift."""
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
    """Synchronous calc pipeline (worker thread). Raises CalculationError for
    partner-visible failures.

    File mode: pdf -> markdown -> get_or_parse (shared cross-tenant cache,
    keyed by SHA-256 of the parse text) -> merge -> calc. Terms mode: DTO ->
    ContractData -> calc, no LLM.

    OWNS tmpdir cleanup: to_thread can't be cancelled, so on client disconnect
    the generator dies while this thread is still reading the files.

    Returns (result, measured, usage) — the two pricing inputs read INSIDE the
    tracking scope. Terms mode opens no scope, so both are None = unmeasured
    and the caller charges the base.
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

        # Order is load-bearing: cheap validation before billable work, or a
        # bad statement buys a free LLM run.
        song_totals = calc.read_royalty_statement(statement_path)

        if contract_terms is not None:
            merged = to_contract_data(contract_terms)
        else:
            with set_llm_context(org_id, "oneclick_partner"):
                datas = []
                for path in contract_paths:
                    md = pdf_to_markdown(path)
                    datas.append(get_or_parse(sb, (lambda m=md: m)))
                # Inside the scope: the accumulator resets when it exits.
                # None = a model missing from MODEL_RATES, so base only.
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
    """Parse pipeline for /registry/v1/splits (worker thread). Owns tmpdir
    cleanup and reads its pricing inputs in-scope, like run_partner_calc.

    pdf -> markdown -> get_or_parse -> merge -> two views of one contract:
    `contract_terms`, the frozen DTO /oneclick/v1/royalties takes verbatim,
    and `splits`, the Registry's per-party pivot.

    ValueError = unreadable contract, and is partner-visible; anything else
    is ours.
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
