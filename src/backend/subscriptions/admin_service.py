"""Admin operations on subscriptions, overrides, and pro_requests.

Uses service-role Supabase client to bypass RLS on cross-user reads/writes.
The require_admin FastAPI dependency is the security gate; this service
itself does NOT re-check admin permissions.
"""

import logging
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

        # Bulk fetch profiles.is_admin for these user_ids (no N+1)
        profiles_res = self.supabase.table("profiles").select("id, is_admin").in_("id", user_ids).execute()
        admin_uids = {row["id"] for row in (profiles_res.data or []) if row.get("is_admin") is True}

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
            pr = self.supabase.table("profiles").select("is_admin").eq("id", user_id).limit(1).execute()
            user["is_admin"] = bool(pr.data and pr.data[0].get("is_admin") is True)
        except Exception as exc:
            logger.warning("get_user_detail profiles lookup failed for %s: %s", user_id, exc)
            user["is_admin"] = False

        user["is_env_admin"] = is_env_admin(user.get("email"))

        ent = self.entitlements_service.get_for_user(user_id)

        ovr_res = self.supabase.table("tier_overrides").select("*").eq("user_id", user_id).execute()
        override = ovr_res.data[0] if ovr_res.data else None
        # `tester_revoked` is a marker, not a real override — hide it from the
        # admin's OverrideEditor pre-fill so the form doesn't show stale fields.
        if override and override.get("reason") == "tester_revoked":
            override = None

        return {"user": user, "entitlements": ent.to_dict(), "override": override}

    def set_tier(self, user_id: str, tier: Literal["free", "pro"]) -> None:
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

    def list_tester_grants(self) -> list[dict]:
        """Returns active tier_overrides rows where reason LIKE 'tester%',
        enriched with the user's `email` (from auth.users) and `name`
        (from profiles.full_name).

        Filters expired rows in Python (expires_at < now) rather than
        composing a Supabase OR filter. Suitable for small admin-facing lists.
        """
        res = self.supabase.table("tier_overrides").select("*").ilike("reason", "tester%").execute()
        now = datetime.now(UTC).isoformat()
        active = [
            row
            for row in (res.data or [])
            if row.get("reason") != "tester_revoked" and (row.get("expires_at") is None or row["expires_at"] > now)
        ]
        if not active:
            return []

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
        return active

    def create_tester_grant(
        self,
        email: str,
        expires_at: str | None = None,
        reason: str = "tester",
    ) -> dict:
        """Look up user by email, upsert a full-Pro override with reason='tester*'.

        Raises ValueError if no auth user matches the email.
        """
        auth_users = self.supabase.auth.admin.list_users()
        matched = [u for u in auth_users if (getattr(u, "email", "") or "").lower() == email.lower()]
        if not matched:
            raise ValueError(f"User not found: {email}")
        user = matched[0]
        user_id = getattr(user, "id", None)

        normalized_reason = _normalize_tester_reason(reason)
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
            "integrations_allowed": ["google_drive", "slack"],
            "reason": normalized_reason,
            "expires_at": expires_at,
            "granted_at": datetime.now(UTC).isoformat(),
        }
        self.supabase.table("tier_overrides").upsert(payload, on_conflict="user_id").execute()
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
        return {"user_id": user_id, "email": email, "expires_at": expires_at, "reason": normalized_reason}

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
