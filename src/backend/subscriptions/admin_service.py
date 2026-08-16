"""Admin operations on subscriptions, overrides, and pro_requests.

Uses service-role Supabase client to bypass RLS on cross-user reads/writes.
The require_admin FastAPI dependency is the security gate; this service
itself does NOT re-check admin permissions.
"""

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Literal

from supabase import Client

from analytics import identify as analytics_identify
from subscriptions.admin_auth import env_admin_emails, is_env_admin
from subscriptions.service import EntitlementsService

logger = logging.getLogger(__name__)


def _normalize_tester_reason(raw: str | None) -> str:
    """Ensure tester-grant reasons satisfy the `LIKE 'tester%'` convention used
    by list_tester_grants and is_active_tester_row. Forces the leading `tester`
    prefix to lowercase so case-sensitive SQL LIKE matches; preserves any
    admin-typed suffix as-is. Empty input becomes the bare `tester` default.
    """
    r = (raw or "").strip()
    if not r:
        return "tester"
    if r.lower().startswith("tester"):
        # Force the leading 'tester' prefix to lowercase. Preserves the rest
        # (case + spacing) for admin readability.
        return "tester" + r[6:]
    return f"tester ({r})"


class AdminService:
    def __init__(self, supabase: Client, entitlements_service: EntitlementsService):
        self.supabase = supabase
        self.entitlements_service = entitlements_service

    def list_users(self, search: str = "", page: int = 1, per_page: int = 25) -> dict:
        """Returns { users: [{id, email, tier, has_override, created_at}], page, per_page, has_more }.

        Lists users via the auth admin API (paginated), filtered by email substring.
        Joins each with their subscriptions tier and whether they have a tier_overrides row.

        NOTE: pagination requires `list_users(page=, per_page=)` kwargs. If the
        installed supabase-py version raises TypeError on these kwargs, we let it
        propagate — silently degrading to "no pagination" would return page 1
        forever and the operator wouldn't notice. Upgrade supabase-py if hit.

        The auth admin API doesn't expose a true total count; we return `has_more`
        based on whether this page filled per_page.
        """
        if search:
            # Server-side search via SECURITY DEFINER RPC — see migration
            # 20260520000000_admin_search_users_by_email.sql. Avoids paginating
            # through every auth page when looking for a specific email substring.
            rpc_res = self.supabase.rpc(
                "admin_search_users_by_email",
                {"p_search": search, "p_limit": per_page},
            ).execute()
            rpc_rows = rpc_res.data or []

            # Adapt RPC rows to the same shape downstream code expects from
            # auth.admin.list_users() — objects with .id, .email, .created_at attrs.
            class _RpcUser:
                __slots__ = ("id", "email", "created_at")

                def __init__(self, row):
                    self.id = row.get("id")
                    self.email = row.get("email")
                    self.created_at = row.get("created_at")

            auth_users = [_RpcUser(r) for r in rpc_rows]
        else:
            auth_users = self.supabase.auth.admin.list_users(page=page, per_page=per_page)

        if not auth_users:
            return {"users": [], "page": page, "per_page": per_page, "has_more": False}

        user_ids = [getattr(u, "id", None) for u in auth_users if getattr(u, "id", None)]

        subs_res = self.supabase.table("subscriptions").select("user_id, tier").in_("user_id", user_ids).execute()
        subs_by_uid = {row["user_id"]: row for row in (subs_res.data or [])}

        # Fetch reason so we can exclude `tester_revoked` markers — they're not
        # real overrides (see _read_override in service.py for the same logic).
        overrides_res = (
            self.supabase.table("tier_overrides").select("user_id, reason").in_("user_id", user_ids).execute()
        )
        override_uids = {row["user_id"] for row in (overrides_res.data or []) if row.get("reason") != "tester_revoked"}

        # Bulk fetch profiles.is_admin + full_name for these user_ids (no N+1).
        # full_name rides the SAME select as is_admin — same reason
        # list_tester_grants reads it: profiles is the only place a display
        # name lives (auth.users has none), and it may be null for an
        # unfinished onboarding.
        profiles_res = self.supabase.table("profiles").select("id, is_admin, full_name").in_("id", user_ids).execute()
        profile_rows = profiles_res.data or []
        admin_uids = {row["id"] for row in profile_rows if row.get("is_admin") is True}
        name_by_uid = {row["id"]: row.get("full_name") for row in profile_rows if row.get("full_name")}

        env_emails = env_admin_emails()

        users = []
        for u in auth_users:
            uid = getattr(u, "id", None)
            email = getattr(u, "email", None)
            created_at = getattr(u, "created_at", None)
            if not uid:
                continue
            sub = subs_by_uid.get(uid, {})
            users.append(
                {
                    "id": uid,
                    "email": email,
                    "name": name_by_uid.get(uid),
                    "tier": sub.get("tier", "free"),
                    "has_override": uid in override_uids,
                    "is_admin": uid in admin_uids,
                    "is_env_admin": bool(email and email.strip().lower() in env_emails),
                    "created_at": str(created_at) if created_at else None,
                }
            )

        return {
            "users": users,
            "page": page,
            "per_page": per_page,
            "has_more": len(auth_users) >= per_page,
        }

    def get_user_detail(self, user_id: str) -> dict:
        """Returns { user, entitlements, override }.

        The raw `override` row is included so the admin override-editor can
        pre-fill with current values (vs starting empty and accidentally clearing
        existing overrides via incomplete re-submit).
        """
        try:
            user_res = self.supabase.auth.admin.get_user_by_id(user_id)
            user_obj = user_res.user
            user = {
                "id": getattr(user_obj, "id", user_id),
                "email": getattr(user_obj, "email", None),
                "created_at": str(getattr(user_obj, "created_at", "")) or None,
            }
        except Exception:
            user = {"id": user_id, "email": None, "created_at": None}

        # Enrich with admin flags (defensive: missing profile row → False)
        try:
            pr = self.supabase.table("profiles").select("is_admin, full_name").eq("id", user_id).limit(1).execute()
            user["is_admin"] = bool(pr.data and pr.data[0].get("is_admin") is True)
            user["name"] = (pr.data[0].get("full_name") if pr.data else None) or None
        except Exception as exc:
            logger.warning("get_user_detail profiles lookup failed for %s: %s", user_id, exc)
            user["is_admin"] = False
            user["name"] = None

        user["is_env_admin"] = is_env_admin(user.get("email"))

        ent = self.entitlements_service.get_for_user(user_id)

        ovr_res = self.supabase.table("tier_overrides").select("*").eq("user_id", user_id).execute()
        override = ovr_res.data[0] if ovr_res.data else None
        # `tester_revoked` is a marker, not a real override — hide it from the
        # admin's OverrideEditor pre-fill so the form doesn't show stale fields.
        if override and override.get("reason") == "tester_revoked":
            override = None

        return {"user": user, "entitlements": ent.to_dict(), "override": override}

    def set_tier(self, user_id: str, tier: Literal["free", "basic", "pro"]) -> None:
        self.supabase.table("subscriptions").upsert(
            {
                "user_id": user_id,
                "tier": tier,
                "status": "active",
                "updated_at": datetime.now(UTC).isoformat(),
            },
            on_conflict="user_id",
        ).execute()

    def apply_override(self, user_id: str, payload: dict) -> None:
        """Sparse upsert: only fields present in `payload` are written.

        `payload` is the OverridePayload.model_dump(exclude_none=True) shape;
        special-cases:
          - `expires_days`: converted to expires_at = now + N days, then dropped
          - `granted_by`: NEVER included (column was removed in SP1)
        """
        write = {
            "user_id": user_id,
            "granted_at": datetime.now(UTC).isoformat(),
        }

        for k in (
            "max_artists",
            "max_projects",
            "max_tasks",
            "max_storage_bytes",
            "max_split_sheets_per_month",
            "monthly_credits",
            "max_works",
            "zoe_enabled",
            "oneclick_enabled",
            "registry_enabled",
            "integrations_allowed",
            "reason",
        ):
            if k in payload:
                write[k] = payload[k]

        if "expires_days" in payload and payload["expires_days"] is not None:
            write["expires_at"] = (datetime.now(UTC) + timedelta(days=int(payload["expires_days"]))).isoformat()

        write.pop("granted_by", None)

        self.supabase.table("tier_overrides").upsert(write, on_conflict="user_id").execute()

    def clear_override(self, user_id: str) -> None:
        self.supabase.table("tier_overrides").delete().eq("user_id", user_id).execute()

    def list_pro_requests(self, status: str | None = None) -> list[dict]:
        q = self.supabase.table("pro_requests").select("*").order("created_at", desc=True)
        if status:
            q = q.eq("status", status)
        res = q.execute()
        return res.data or []

    # ------------------------------------------------------------------
    # Tester grants — "tester" reason tier_overrides rows
    # ------------------------------------------------------------------

    def _pending_tester_rows(self) -> list[dict]:
        """Unclaimed pre-signup designations, keyed by email (no user_id yet)."""
        pending_res = self.supabase.table("pending_tester_grants").select("*").is_("claimed_at", "null").execute()
        return [
            {
                "user_id": None,
                "email": row["email"],
                "name": None,
                "reason": row.get("reason"),
                "expires_at": None,
                "granted_at": row.get("created_at"),
                "grant_duration_days": row.get("grant_duration_days"),
                "pending": True,
            }
            for row in pending_res.data or []
        ]

    def list_tester_grants(self) -> list[dict]:
        """Returns active tier_overrides rows where reason LIKE 'tester%',
        enriched with the user's `email` (from auth.users) and `name`
        (from profiles.full_name), PLUS unclaimed pre-signup pending rows
        (flagged `pending: true`).

        Filters expired rows in Python (expires_at < now) rather than
        composing a Supabase OR filter. Suitable for small admin-facing lists.
        """
        pending_rows = self._pending_tester_rows()

        res = self.supabase.table("tier_overrides").select("*").ilike("reason", "tester%").execute()
        now = datetime.now(UTC).isoformat()
        active = [
            row
            for row in (res.data or [])
            if row.get("reason") != "tester_revoked" and (row.get("expires_at") is None or row["expires_at"] > now)
        ]
        if not active:
            return pending_rows

        user_ids = [row["user_id"] for row in active if row.get("user_id")]

        # Email comes from auth.users (profiles doesn't store email).
        # Per-user lookup avoids list_users() pagination edge cases — tester
        # grants are always a small list so N round-trips is acceptable.
        email_by_id: dict[str, str] = {}
        for uid in user_ids:
            try:
                resp = self.supabase.auth.admin.get_user_by_id(uid)
                # supabase-py wraps the user in `.user` (object with `.email` attribute)
                u = getattr(resp, "user", resp)
                email = getattr(u, "email", None) or ""
                if email:
                    email_by_id[uid] = email
            except Exception as e:
                logger.warning("list_tester_grants: get_user_by_id failed for %s: %s", uid, e)

        # Name comes from profiles.full_name (may be null for unfinished onboarding).
        name_by_id: dict[str, str] = {}
        try:
            profs = self.supabase.table("profiles").select("id, full_name").in_("id", user_ids).execute()
            for p in profs.data or []:
                if p.get("id"):
                    name_by_id[p["id"]] = p.get("full_name") or ""
        except Exception as e:
            logger.warning("list_tester_grants: profiles lookup failed: %s", e)

        for row in active:
            uid = row.get("user_id")
            row["email"] = email_by_id.get(uid) or None
            row["name"] = name_by_id.get(uid) or None
        return active + pending_rows

    def _activate_tester_grant(
        self,
        user_id: str,
        email_norm: str,
        expires_at: str | None,
        normalized_reason: str,
        credits: int | None,
        grant_duration_days: int | None,
    ) -> dict:
        """Write the live tester tier_overrides row for `user_id` and return the
        create_tester_grant response shape. Shared by the normal matched-email
        path and the not-matched branch's "claimed pending row, but the search
        RPC false-negatived and the claimed user is still alive" recovery.
        """
        # A designation may already exist as a pending row (e.g. designated
        # pre-signup, then the claim was skipped on an unverified sign-in).
        # The active grant we're about to write supersedes it — clear it so it
        # can't linger under "Pending" or re-claim later with stale terms.
        (
            self.supabase.table("pending_tester_grants")
            .delete()
            .eq("email", email_norm)
            .is_("claimed_at", "null")
            .execute()
        )

        if grant_duration_days is not None:
            # Days wins over legacy absolute expires_at; claim time == submit
            # time for an existing user.
            expires_at = (datetime.now(UTC) + timedelta(days=grant_duration_days)).isoformat()

        payload = {
            "user_id": user_id,
            "max_artists": -1,
            "max_projects": -1,
            "max_tasks": -1,
            "max_storage_bytes": -1,
            "max_split_sheets_per_month": -1,
            "max_oneclick_runs_per_month": -1,
            "zoe_enabled": True,
            "oneclick_enabled": True,
            "registry_enabled": True,
            "integrations_allowed": ["google_drive", "dropbox"],
            "reason": normalized_reason,
            "expires_at": expires_at,
            "granted_at": datetime.now(UTC).isoformat(),
        }
        self.supabase.table("tier_overrides").upsert(payload, on_conflict="user_id").execute()

        # Initial credit allocation — once-per-user idempotent, so an admin
        # re-grant (e.g. extending expiry) never double-fills. Unguarded on
        # purpose: a failure 500s the admin call, and a retry converges.
        grant_initial_tester_credits(self.supabase, user_id, credits)

        try:
            analytics_identify(
                user_id,
                {
                    "is_tester": True,
                    "tester_granted_at": payload.get("granted_at"),
                    "tester_expires_at": expires_at,
                },
            )
        except Exception as e:
            logger.warning("analytics identify on tester-grant create failed: %s", e)
        return {"user_id": user_id, "email": email_norm, "expires_at": expires_at, "reason": normalized_reason}

    def create_tester_grant(
        self,
        email: str,
        expires_at: str | None = None,
        reason: str = "tester",
        credits: int | None = None,
        grant_duration_days: int | None = None,
        created_by: str = "",
    ) -> dict:
        """Grant tester access to an existing user, or park a PENDING grant for
        an email with no account yet (claimed by /me/bootstrap-tester on the
        user's first verified sign-in).

        Lookup goes through the admin_search_users_by_email RPC with
        p_limit=100 + an exact match here: the RPC is substring-ILIKE with a
        default limit of 10, so a small window can push the exact match out
        (the old bare list_users() page-one scan had the same failure worse).
        """
        email_norm = email.strip().lower()
        rpc_res = self.supabase.rpc("admin_search_users_by_email", {"p_search": email_norm, "p_limit": 100}).execute()
        matched = [r for r in (rpc_res.data or []) if (r.get("email") or "").strip().lower() == email_norm]
        normalized_reason = _normalize_tester_reason(reason)

        if not matched:
            # No live user matched. Before parking a plain pending row, check
            # whether this email already has a CLAIMED row — a claimed row
            # with no RPC match means either the search false-negatived (the
            # user changed their email, or fell outside the 100-row window) or
            # the claimed user's account was deleted. The plain upsert below
            # deliberately never touches claim columns, so this is the only
            # place that can tell those two cases apart.
            existing_res = (
                self.supabase.table("pending_tester_grants")
                .select("claimed_at, claimed_user_id")
                .eq("email", email_norm)
                .execute()
            )
            existing_rows = existing_res.data or []
            existing = existing_rows[0] if existing_rows else None

            pending_extra: dict = {}
            if existing and existing.get("claimed_at"):
                claimed_user_id = existing.get("claimed_user_id")
                user_gone = True
                if claimed_user_id:
                    try:
                        auth_res = self.supabase.auth.admin.get_user_by_id(claimed_user_id)
                        # Tolerate both the `.user`-wrapped shape and a bare
                        # user object (mirrors list_tester_grants' lookup).
                        user_obj = getattr(auth_res, "user", auth_res)
                        user_gone = user_obj is None
                    except Exception:
                        user_gone = True

                if not user_gone:
                    # RPC false-negative — the claimed user is still alive.
                    # Treat this exactly like a normal active re-grant instead
                    # of parking a second, unreachable pending row.
                    return self._activate_tester_grant(
                        claimed_user_id, email_norm, expires_at, normalized_reason, credits, grant_duration_days
                    )

                # Claimed user's account was deleted. An admin re-designating
                # this email is a deliberate decision to re-open the claim for
                # a fresh signup — ONLY in this branch do we touch claim
                # columns; the normal pending upsert below must never clobber
                # a live claim.
                pending_extra = {"claimed_at": None, "claimed_user_id": None}

            # Pre-signup designation. Payload deliberately EXCLUDES
            # claimed_at/claimed_user_id (outside the deleted-user recovery
            # above) so a re-POST never clobbers a claim (PostgREST upsert
            # only touches the columns it's given).
            pending = {
                "email": email_norm,
                "reason": normalized_reason,
                "grant_duration_days": grant_duration_days,
                "credits": credits,
                "created_by": created_by,
                **pending_extra,
            }
            self.supabase.table("pending_tester_grants").upsert(pending, on_conflict="email").execute()
            return {
                "email": email_norm,
                "pending": True,
                "reason": normalized_reason,
                "grant_duration_days": grant_duration_days,
            }

        return self._activate_tester_grant(
            matched[0]["id"], email_norm, expires_at, normalized_reason, credits, grant_duration_days
        )

    def revoke_tester_grant(self, user_id: str) -> None:
        """Mark the tier_overrides row as 'tester_revoked' so bootstrap-tester
        won't auto-re-grant on next sign-in (for users in TESTER_EMAILS env).

        All caps and feature flags are nulled — the user reverts to tier defaults
        (free, in the absence of a different non-tester override). The row stays
        as a sticky marker; admin re-grant via create_tester_grant overwrites it.
        """
        payload = {
            "user_id": user_id,
            "max_artists": None,
            "max_projects": None,
            "max_tasks": None,
            "max_storage_bytes": None,
            "max_split_sheets_per_month": None,
            "max_oneclick_runs_per_month": None,
            "zoe_enabled": None,
            "oneclick_enabled": None,
            "registry_enabled": None,
            "integrations_allowed": None,
            "reason": "tester_revoked",
            "expires_at": None,
            "granted_at": datetime.now(UTC).isoformat(),
        }
        self.supabase.table("tier_overrides").upsert(payload, on_conflict="user_id").execute()
        try:
            analytics_identify(
                user_id,
                {
                    "is_tester": False,
                    "tester_granted_at": None,
                    "tester_expires_at": None,
                },
            )
        except Exception as e:
            logger.warning("analytics identify on tester-grant revoke failed: %s", e)

    def revoke_pending_tester_grant(self, email: str) -> None:
        """Delete an UNCLAIMED pre-signup designation. Claimed rows are
        immutable history (revoke the live grant via revoke_tester_grant)."""
        (
            self.supabase.table("pending_tester_grants")
            .delete()
            .eq("email", email.strip().lower())
            .is_("claimed_at", "null")
            .execute()
        )

    # ------------------------------------------------------------------
    # Admin role management (profiles.is_admin toggle)
    # ------------------------------------------------------------------

    def promote_user(self, user_id: str) -> None:
        """Set profiles.is_admin = true for user_id. Idempotent.

        Uses service-role client — bypasses RLS.
        """
        self.supabase.table("profiles").update({"is_admin": True}).eq("id", user_id).execute()

    def demote_user(self, user_id: str) -> None:
        """Set profiles.is_admin = false for user_id. Idempotent.

        Note: this only affects DB-backed admin status. If the target is
        ALSO in ADMIN_EMAILS env-var, they remain admin via the env path
        until removed from the allowlist — the admin_router endpoint
        rejects demote requests against env-admins so the operator sees
        the correct next step.
        """
        self.supabase.table("profiles").update({"is_admin": False}).eq("id", user_id).execute()

    def get_email_for_user_id(self, user_id: str) -> str | None:
        """Look up an auth user's email by id. Returns None if not found
        or the lookup fails — callers should treat None as "unknown" and
        decide what's safe (e.g. promote may proceed without email, demote
        cannot verify env-admin status without it)."""
        try:
            res = self.supabase.auth.admin.get_user_by_id(user_id)
            user = getattr(res, "user", None)
            return getattr(user, "email", None) if user else None
        except Exception as exc:
            logger.warning("get_email_for_user_id lookup failed for %s: %s", user_id, exc)
            return None


# ------------------------------------------------------------------
# Admin credit tooling — manual credit grants + ledger inspection
# ------------------------------------------------------------------

TESTER_INITIAL_CREDITS_DEFAULT = 1000


def _tester_initial_credits_default() -> int:
    try:
        return int(os.getenv("TESTER_INITIAL_CREDITS", str(TESTER_INITIAL_CREDITS_DEFAULT)))
    except ValueError:
        return TESTER_INITIAL_CREDITS_DEFAULT


def grant_initial_tester_credits(supabase, user_id: str, amount: int | None = None) -> bool:
    """One-time reserve grant for a new tester (replaces the old idea of a
    monthly_credits override, which the sweep couldn't see — review 2026-07-19).

    Idempotent FOREVER per user: request_id `tester-init:{user_id}` hits the
    ledger's unique index, so bootstrap re-runs and admin re-grants can never
    double-fill. Top-ups go through POST /admin/users/{id}/credits/grant.
    Reserve bucket on purpose: tester credits must survive rollover (spec §6).

    Returns True when a grant RPC was issued; False when skipped (credits off,
    non-positive amount, or missing wallet — warn-logged, never raised).
    """
    from subscriptions.service import credits_enabled

    if not credits_enabled():
        return False
    amount = amount if amount is not None else _tester_initial_credits_default()
    if amount <= 0:
        return False
    wallet_res = (
        supabase.table("credit_wallets").select("id").eq("owner_type", "user").eq("owner_id", user_id).execute()
    )
    if not wallet_res.data:
        logger.warning("tester credits: no wallet for user %s — skipping initial grant", user_id)
        return False
    supabase.rpc(
        "grant_credits",
        {
            "p_wallet_id": wallet_res.data[0]["id"],
            "p_amount": amount,
            "p_kind": "admin_grant",
            "p_bucket": "reserve",
            "p_metadata": {"reason": "tester_initial"},
            "p_request_id": f"tester-init:{user_id}",
        },
    ).execute()
    return True


def claim_pending_tester_grant(supabase, pending: dict, user_id: str, email_norm: str) -> bool:
    """Convert a pending_tester_grants row into a live tester override for the
    user who just signed in. Returns True only when the claim belongs to this
    user (including losing a race to the SAME user's concurrent sign-in).

    Order is GRANT THEN CLAIM: the override upsert and tester-init credits are
    both idempotent, so a concurrent double-run is harmless — while
    claim-then-grant could stamp claimed_at and then fail the grant, leaving a
    claimed row and no tester (nothing would retry). The conditional update
    (WHERE claimed_at IS NULL) is the claim mutex; the race loser sees zero
    rows updated. Third failure mode: the claim UPDATE itself raises (e.g. a
    transient DB error) after the upsert already landed — the exception
    propagates out to main.py's catch (sign-in never 500s), the row stays
    unclaimed, and the next sign-in retries the whole (idempotent) grant +
    claim.

    Verified-email guard: GoTrue only leaves email_confirmed_at unset when
    "Confirm email" is ON (verified ON for this project 2026-08-08) — this is
    what stops a squatter who registered someone else's address from
    collecting the grant. Unverified → False, row stays pending, next
    sign-in retries.
    """
    auth_user = supabase.auth.admin.get_user_by_id(user_id)
    user_obj = getattr(auth_user, "user", auth_user)
    if not getattr(user_obj, "email_confirmed_at", None):
        logger.info("pending tester claim skipped (unverified email) for %s", user_id)
        return False

    days = pending.get("grant_duration_days")
    expires_at = (datetime.now(UTC) + timedelta(days=days)).isoformat() if days else None
    granted_at = datetime.now(UTC).isoformat()

    # Same shape as create_tester_grant / bootstrap env path — feature gating
    # must be identical. Upsert on user_id also OVERRIDES a tester_revoked
    # sticky marker, exactly like an admin re-grant: the admin designating
    # this email is the later, stronger intent.
    payload = {
        "user_id": user_id,
        "max_artists": -1,
        "max_projects": -1,
        "max_tasks": -1,
        "max_storage_bytes": -1,
        "max_split_sheets_per_month": -1,
        "max_oneclick_runs_per_month": -1,
        "zoe_enabled": True,
        "oneclick_enabled": True,
        "registry_enabled": True,
        "integrations_allowed": ["google_drive", "dropbox"],
        "reason": _normalize_tester_reason(pending.get("reason")),
        "expires_at": expires_at,
        "granted_at": granted_at,
    }
    supabase.table("tier_overrides").upsert(payload, on_conflict="user_id").execute()

    # Best-effort, at parity with every other tester path: skips (warn) when
    # CREDITS_ENABLED is off, tester-init:{user_id} is once-ever, and a
    # transient credits failure must not block the claim — the override is
    # already live, and an unclaimed row would desync the pending list while
    # misreporting granted=False (env path in main.py wraps identically).
    try:
        grant_initial_tester_credits(supabase, user_id, pending.get("credits"))
    except Exception as exc:
        logger.warning("pending tester claim: initial credits failed for %s: %s", user_id, exc)

    claim_res = (
        supabase.table("pending_tester_grants")
        .update({"claimed_at": granted_at, "claimed_user_id": user_id})
        .eq("email", email_norm)
        .is_("claimed_at", "null")
        .execute()
    )
    if not claim_res.data:
        # Lost a concurrent race AFTER granting. "Byte-identical, nothing to
        # undo" only holds for the SAME user's other sign-in — the claim did
        # happen for them, so report success and let AuthContext refresh. A
        # cross-user loser is different: THIS call's own tier_overrides
        # upsert for `user_id` already landed too, so returning False here is
        # correct (the pending row belongs to the other user) but leaves that
        # extra upsert behind — harmless since it mirrors what a real grant
        # would write, and practically unreachable anyway (pending rows are
        # keyed by email, and GoTrue enforces unique emails).
        row_res = supabase.table("pending_tester_grants").select("claimed_user_id").eq("email", email_norm).execute()
        row = (row_res.data or [None])[0]
        return bool(row and row.get("claimed_user_id") == user_id)

    try:
        analytics_identify(
            user_id,
            {"is_tester": True, "tester_granted_at": granted_at, "tester_expires_at": expires_at},
        )
    except Exception as e:
        logger.warning("analytics identify on pending tester claim failed: %s", e)
    return True


def grant_user_credits(
    supabase, user_id: str, amount: int, reason: str, granted_by: str, request_id: str | None = None
) -> dict:
    """Admin grant → reserve bucket (never expires; spec §6 two-bucket rationale).

    `request_id` is a client-supplied idempotency key. The frontend supplies a
    stable key per user-initiated grant action, so a retry / double-submit of the
    SAME action dedupes at the RPC (grant_credits keys on p_request_id) instead of
    creating a second ledger row. Two DELIBERATE separate grants use different keys
    and both apply. When None, no dedupe key is passed (each call is distinct).

    NOTE: there is no admin revoke/debit endpoint yet, so an over-grant (e.g. a
    fat-fingered amount) currently requires a follow-up revoke endpoint or a manual
    support process to unwind. That companion endpoint is a tracked follow-up.
    """
    wallet_res = (
        supabase.table("credit_wallets").select("id").eq("owner_type", "user").eq("owner_id", user_id).execute()
    )
    if not wallet_res.data:
        raise ValueError("User has no credit wallet")
    res = supabase.rpc(
        "grant_credits",
        {
            "p_wallet_id": wallet_res.data[0]["id"],
            "p_amount": amount,
            "p_kind": "admin_grant",
            "p_bucket": "reserve",
            "p_metadata": {"reason": reason, "granted_by": granted_by},
            "p_request_id": f"admin-grant:{request_id}" if request_id else None,
        },
    ).execute()
    return res.data if isinstance(res.data, dict) else {"granted": amount}


def grant_org_credits(supabase, org_id: str, amount: int, reason: str, granted_by: str, request_id: str) -> dict:
    """Admin comped-pack grant into an org's POOL (reserve bucket).

    Mirrors stripe_events._handle_org_topup_grant minus Stripe: same wallet
    helper, same RPC, then the SAME shared activation check — so a comped
    grant is behaviorally identical to a purchase (counts toward the
    activation floor via PAID_IN_KINDS, never expires, reclaimable by the
    reserve-only clawback) while keeping provenance honest
    (kind='admin_grant', never 'purchase').

    'admin-org-grant:' is its own idempotency namespace — credit_ledger's
    request_id index is GLOBAL and 'admin-grant:' belongs to user grants; a
    shared prefix would let one operator key silently no-op the other grant.

    Raises ValueError (-> 404 at the router) when the org doesn't exist.
    """
    from orgs.wallets import maybe_activate_org, read_or_create_org_wallet

    # Existence-only check (not status): granting to an archived/suspended
    # org is deliberate — support use cases like a settled dispute or a
    # goodwill comp on a paused account. Activation itself still only ever
    # flips pending -> active, inside maybe_activate_org.
    org_res = supabase.table("organizations").select("id").eq("id", org_id).execute()
    if not org_res.data:
        raise ValueError(f"Organization not found: {org_id}")

    wallet = read_or_create_org_wallet(supabase, org_id)
    res = supabase.rpc(
        "grant_credits",
        {
            "p_wallet_id": wallet["id"],
            "p_amount": amount,
            "p_kind": "admin_grant",
            "p_bucket": "reserve",
            "p_metadata": {
                "reason": reason,
                "granted_by": granted_by,
                "org_id": org_id,
                "source": "admin_grant",
            },
            "p_request_id": f"admin-org-grant:{request_id}",
        },
    ).execute()
    activated = maybe_activate_org(supabase, org_id, wallet["id"])
    result = res.data if isinstance(res.data, dict) else {"granted": amount}
    result["activated"] = activated
    return result


# Per-owner-type clawback config — the ONLY things that differ between the
# user-wallet and org-pool clawback paths: the idempotency-namespace prefix, the
# no-wallet ValueError text, and the bad-RPC-shape RuntimeError text. `{id}` is
# filled with the owner id (the user no-wallet message has no placeholder, which
# .format harmlessly ignores). Everything else is the shared debit_credits invariant.
_CLAWBACK_CONF = {
    "user": {
        "prefix": "admin-adjust",
        "no_wallet": "User has no credit wallet",
        "bad_shape": "clawback: debit_credits returned unexpected shape for user {id}",
    },
    "org": {
        "prefix": "admin-org-adjust",
        "no_wallet": "Organization has no pool wallet: {id}",
        "bad_shape": "org clawback: debit_credits returned unexpected shape for org {id}",
    },
}


def _admin_clawback(supabase, owner_type: str, owner_id: str, amount: int, metadata: dict, request_id: str) -> dict:
    """Shared clawback core for adjust_user_credits / adjust_org_pool. Per-path
    strings come from _CLAWBACK_CONF; the wallet lookup, the debit_credits RPC
    shape, and the never-fabricate-a-removal shape guard are IDENTICAL and live
    here once. Callers own the `metadata` dict (the only structural difference)."""
    conf = _CLAWBACK_CONF[owner_type]
    wallet_res = (
        supabase.table("credit_wallets").select("id").eq("owner_type", owner_type).eq("owner_id", owner_id).execute()
    )
    if not wallet_res.data:
        raise ValueError(conf["no_wallet"].format(id=owner_id))
    res = supabase.rpc(
        "debit_credits",
        {
            "p_wallet_id": wallet_res.data[0]["id"],
            "p_amount": amount,
            "p_action": "admin_adjust",
            "p_request_id": f"{conf['prefix']}:{request_id}",
            "p_kind": "clawback",
            "p_metadata": metadata,
        },
    ).execute()
    if not isinstance(res.data, dict):
        # Never claim a removal we can't confirm — an unexpected RPC shape is
        # an error to surface, not a success to fabricate.
        raise RuntimeError(conf["bad_shape"].format(id=owner_id))
    return res.data


def adjust_user_credits(supabase, user_id: str, amount: int, reason: str, adjusted_by: str, request_id: str) -> dict:
    """Admin clawback (pack refunds / chargebacks — spec 2026-07-19 §3).

    This is the ONLY sanctioned way to remove credits; a hand-written UPDATE
    against wallet tables would break the sum(delta) == bundle + reserve
    reconciliation invariant that the ledger exists to guarantee.

    Removes up to `amount` credits with a ledger row of kind 'clawback'.
    RESERVE-ONLY and CLAMPED in the RPC — purchased credits live in reserve,
    and touching the bundle would be undone by the next rollover. Never
    creates a negative balance; the RPC returns {removed, shortfall,
    balance_after} on a fresh clawback, and a shortfall is a written-off cost
    (spec §3) — the refund exceeded what was recoverable from the wallet.

    Reserve is FUNGIBLE: admin/tester goodwill grants share the same bucket
    as purchased credits, so a clawback may end up consuming goodwill credits
    once the purchase remnant is spent. That's accepted — it's how prepaid
    balances work — and the ledger `reason` records why, so support isn't
    surprised when a clawback removes more than "just the purchase".

    The clamp/bucket semantics themselves live in the RPC and can't be
    verified against a mocked Supabase client — see the spec's real-DB
    launch-gate list (spec §10) for the concrete scenarios to check there.

    `request_id` is REQUIRED (unlike grant_user_credits' optional key): a
    double-submitted clawback removes a real customer's credits twice, not
    merely double-grants goodwill. Support has the Stripe refund/dispute id
    in hand at exactly this moment — use it.

    Deliberately NO analytics event: this is an admin support action, not
    user behavior, and PostHog is not the billing record (Stripe + the
    ledger are).
    """
    return _admin_clawback(
        supabase, "user", user_id, amount, {"reason": reason, "adjusted_by": adjusted_by}, request_id
    )


def adjust_org_pool(supabase, org_id: str, amount: int, reason: str, adjusted_by: str, request_id: str) -> dict:
    """Admin clawback on an org's POOL wallet — archived/live org pool
    disposition tooling (follow-ups plan 2026-07-22, Task 2).

    RUNBOOK — support executes ONE of two pairs, both halves idempotent and
    ledger-audited (neither half alone is "done"):
      - REFUND: this clawback (removes the pool balance being refunded) +
        a manual Stripe refund issued separately by support, using the same
        refund/dispute id as this call's `idempotency_key`/`reason`.
      - GOODWILL MIGRATION: this clawback on the org pool + the EXISTING
        per-user admin grant (`grant_user_credits`) re-issuing the same (or a
        partial) amount into a destination user's PERSONAL wallet.
    This is the ONLY sanctioned way to remove credits from an org pool; a
    hand-written UPDATE against credit_wallets/credit_ledger would break the
    sum(delta) == bundle_balance + reserve_balance reconciliation invariant
    the ledger exists to guarantee — see adjust_user_credits above, which
    this mirrors for the org-wallet case.

    BUCKETS — an org pool holds BOTH, with different provenance:
    `reserve_balance` is purchased packs, `bundle_balance` is the current
    period's contract dispersal (granted by the sweep's rollover, spent by
    members, expired at each period end). `debit_credits`'s clawback branch is
    deliberately RESERVE-ONLY, so this recovers purchased credits and never the
    dispersal — correct in both directions: a refund is against money the org
    actually paid for a pack, and clawing back dispersed credits would be undone
    by the next rollover anyway. A shortfall means the purchased remnant had
    already been spent; the RPC returns it so support can see the refund exceeded
    what was recoverable.

    Raises ValueError (-> 404 at the router) when the org has no pool
    wallet yet — there is nothing to claw back. Raises RuntimeError on a
    non-dict RPC result — never fabricate a removal we can't confirm (mirrors
    adjust_user_credits).

    Deliberately NO analytics event: this is an admin support action, not
    user behavior, and PostHog is not the billing record (Stripe + the
    ledger are) — same stance as the per-user clawback.
    """
    return _admin_clawback(
        supabase, "org", org_id, amount, {"reason": reason, "adjusted_by": adjusted_by, "org_id": org_id}, request_id
    )


def get_org_pool(supabase, org_id: str) -> dict:
    """Support-visibility snapshot of an org's pool, read BEFORE deciding how
    to dispose of it via `adjust_org_pool` (see that function's runbook).

    Returns {orgId, status, archivedAt, poolBalance, cumulativePaidIn,
    ledger}. `cumulativePaidIn` is computed via the SAME
    `orgs.wallets.cumulative_paid_in` helper the Phase B activation check
    (`stripe_events._handle_org_topup_grant`) uses — support's "did this org
    cross the minimum" must be definitionally identical to what activation
    summed, not a second reimplementation that can drift.

    Raises ValueError (-> 404 at the router) when the org itself doesn't
    exist. An org that exists but has no pool wallet yet (never topped up —
    a normal state for a fresh 'pending' org) is NOT an error: it reports
    zero balance/cumulative and an empty ledger.
    """
    from orgs.wallets import cumulative_paid_in

    org_res = supabase.table("organizations").select("id, status, archived_at").eq("id", org_id).execute()
    org_rows = org_res.data or []
    if not org_rows:
        raise ValueError(f"Organization not found: {org_id}")
    org = org_rows[0]

    wallet_res = supabase.table("credit_wallets").select("*").eq("owner_type", "org").eq("owner_id", org_id).execute()
    wallet_rows = wallet_res.data or []
    if not wallet_rows:
        return {
            "orgId": org_id,
            "status": org.get("status"),
            "archivedAt": org.get("archived_at"),
            "poolBalance": 0,
            "cumulativePaidIn": 0,
            "ledger": [],
        }
    wallet = wallet_rows[0]
    wallet_id = wallet["id"]
    pool_balance = (wallet.get("bundle_balance") or 0) + (wallet.get("reserve_balance") or 0)

    ledger_res = (
        supabase.table("credit_ledger")
        .select("*")
        .eq("wallet_id", wallet_id)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )

    return {
        "orgId": org_id,
        "status": org.get("status"),
        "archivedAt": org.get("archived_at"),
        "poolBalance": pool_balance,
        "cumulativePaidIn": cumulative_paid_in(supabase, wallet_id),
        "ledger": ledger_res.data or [],
    }


# ------------------------------------------------------------------
# Enterprise org creation + kind flip (spec 2026-08-15 §2, review r2 hole 5).
# Post-migration, these two are the ONLY producers of kind='enterprise' rows
# — orgs.service.create_org is self-serve + slot-gated for everyone else.
# ------------------------------------------------------------------


def create_enterprise_org(supabase: Client, name: str, admin_email: str) -> dict:
    """Msanii-admin-only: create an ENTERPRISE org for an EXISTING customer
    account. `created_by` is the CUSTOMER's user id, never the operator's —
    the auto_create_org_admin DB trigger seats `created_by` as the org's
    sole admin, so creating under the Msanii admin's own id would make
    MSANII the customer's admin, and a NULL would leave nobody able to
    invite. `kind='enterprise'` is written explicitly here.

    Status defaults to 'pending' (DB CHECK default) — the activation floor
    still applies, same as today.

    Raises ValueError (-> 404 at the router) when `admin_email` has no
    account — enterprise onboarding requires an existing account, unlike
    orgs.service.invite_member's invite-by-email path.
    """
    from orgs.service import _find_user_id_by_email

    customer_id = _find_user_id_by_email(supabase, admin_email)
    if not customer_id:
        raise ValueError(f"No account found for email: {admin_email}")

    res = (
        supabase.table("organizations")
        .insert({"name": name, "created_by": customer_id, "kind": "enterprise"})
        .execute()
    )
    org = res.data[0] if res.data else None
    if not org:
        raise RuntimeError("Failed to create organization")
    return org


def set_org_kind(supabase: Client, org_id: str, kind: str, covered_by_user_id: str | None = None) -> dict:
    """Msanii-admin-only: flip an org's `kind` between 'enterprise' and
    'self_serve'.

    Flipping TO 'self_serve' REQUIRES `covered_by_user_id` — the router
    enforces this with a 422 before calling in; this function re-checks
    (defense in depth, since it takes no authz dependency of its own). Mirrors
    orgs.service.claim_coverage's guard block:
      - refuses an archived or dissolved org (ValueError -> 400 at the
        router), same as claim_coverage — coverage should never be assigned
        onto a team that can't be reactivated this way.
      - the coverer must be an ACTIVE admin of `org_id` (orgs.authz.is_org_admin)
      - the coverer must have a free team slot on their own plan
        (orgs.standing.require_free_slot, which raises NoSlotError -> 402 at
        the router)
    Without all three, the org would sit uncovered and lapse in the grace
    window with nobody able to fix it. On a clean flip, stamps
    `covered_by`/`covered_at`, and additionally:
      - sets `status='active'` — create_org's invariant is that the slot IS
        the activation; a flip must land in the same state a fresh
        self-serve create would.
      - zeroes `monthly_dispersal_credits` — an enterprise dispersal
        surviving the flip would mint free monthly credits into a self-serve
        pool forever after: the sweep's rollover grants
        `monthly_dispersal_credits` to ANY org with a positive value
        regardless of `kind` (subscriptions/sweep.py), so a stale enterprise
        contract volume left on a self-serve row is a standing credit leak,
        not a no-op.

    Flipping TO 'enterprise' clears `grace_started_at` — standing.py (the
    sweep) ignores enterprise orgs entirely, so a stale grace-start
    timestamp left over from a prior self_serve stint must not linger and
    confuse a later flip back.

    Raises ValueError for an invalid `kind`, a missing coverer, an archived/
    dissolved org, or a coverer who isn't an active admin of this org (-> 400
    at the router, mirroring set_org_dispersal). Raises RuntimeError if the
    update affects no row (org deleted between the router's existence check
    and this write) — never fabricate a success body for a write we can't
    confirm landed, mirroring create_enterprise_org. Org existence is
    otherwise checked at the router (mirrors the dispersal endpoint's
    404-then-400 split) — this function assumes `org_id` is real going in.
    """
    from orgs import authz as org_authz
    from orgs.standing import require_free_slot

    if kind not in ("self_serve", "enterprise"):
        raise ValueError(f"invalid kind {kind!r}")

    payload: dict = {"kind": kind}
    if kind == "self_serve":
        if not covered_by_user_id:
            raise ValueError("covered_by_user_id is required when flipping to self_serve")

        org_res = supabase.table("organizations").select("archived_at, dissolved_at").eq("id", org_id).execute()
        org_row = (org_res.data or [None])[0]
        if org_row and (org_row.get("archived_at") or org_row.get("dissolved_at")):
            raise ValueError("Cannot flip an archived or dissolved organization to self_serve")

        if not org_authz.is_org_admin(supabase, covered_by_user_id, org_id):
            raise ValueError("covered_by_user_id must be an active admin of this organization")
        require_free_slot(supabase, covered_by_user_id)  # raises NoSlotError -> 402 at the router

        payload["covered_by"] = covered_by_user_id
        payload["covered_at"] = datetime.now(UTC).isoformat()
        payload["status"] = "active"
        payload["monthly_dispersal_credits"] = 0
    else:
        payload["grace_started_at"] = None

    res = supabase.table("organizations").update(payload).eq("id", org_id).execute()
    if not res.data:
        raise RuntimeError(f"Failed to update organization kind: {org_id}")
    return res.data[0]


def list_orgs_admin(supabase) -> list[dict]:
    """Every org for the admin Organizations tab — INCLUDING suspended and
    archived (labeled, never hidden: the tooling must not vanish for exactly
    the orgs an operator needs to inspect).

    Per-org reads are N+1 (wallet + member count + paid-in sum) — deliberate:
    org count is tens, and reusing orgs.wallets.cumulative_paid_in keeps the
    activation number definitionally identical to what maybe_activate_org
    enforces (get_org_pool's docstring makes the same argument).
    """
    from orgs.wallets import cumulative_paid_in

    default_floor = int(os.getenv("ENTERPRISE_MIN_INITIAL_CREDITS", "10000"))
    orgs_res = (
        supabase.table("organizations")
        .select("id, name, status, archived_at, monthly_dispersal_credits, min_initial_purchase_credits, created_at")
        .order("created_at")
        .execute()
    )
    out: list[dict] = []
    for org in orgs_res.data or []:
        org_id = org["id"]
        wallet_res = (
            supabase.table("credit_wallets")
            .select("id, bundle_balance, reserve_balance")
            .eq("owner_type", "org")
            .eq("owner_id", org_id)
            .execute()
        )
        wallet = (wallet_res.data or [None])[0]
        members_res = (
            supabase.table("org_members")
            .select("id", count="exact")
            .eq("org_id", org_id)
            .eq("status", "active")
            .execute()
        )
        out.append(
            {
                "id": org_id,
                "name": org.get("name"),
                "status": org.get("status"),
                "archivedAt": org.get("archived_at"),
                "memberCount": members_res.count or 0,
                "bundleBalance": (wallet or {}).get("bundle_balance") or 0,
                "reserveBalance": (wallet or {}).get("reserve_balance") or 0,
                "monthlyDispersalCredits": org.get("monthly_dispersal_credits") or 0,
                "activationFloor": org.get("min_initial_purchase_credits") or default_floor,
                "cumulativePaidIn": cumulative_paid_in(supabase, wallet["id"]) if wallet else 0,
            }
        )
    return out


def get_user_credit_ledger(supabase, user_id: str, limit: int = 100) -> list[dict]:
    wallet_res = (
        supabase.table("credit_wallets").select("id").eq("owner_type", "user").eq("owner_id", user_id).execute()
    )
    if not wallet_res.data:
        return []
    res = (
        supabase.table("credit_ledger")
        .select("*")
        .eq("wallet_id", wallet_res.data[0]["id"])
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []
