"""EntitlementsService — single chokepoint for tier/cap/feature checks.

Reads four tables on every request (no cache; see spec §3.1 / §4.1 #4).
Per-field override merge with integrations_allowed replace-semantics.
Lazy period rollover and the can() chokepoint are added in Tasks 4 and 5.
"""

import os
from dataclasses import replace
from datetime import UTC, datetime

from dateutil.relativedelta import relativedelta
from supabase import Client

from subscriptions.models import (
    Caps,
    Entitlements,
    Features,
    Usage,
)

# ---------------------------------------------------------------------------
# Beta bypass + admin helpers
# ---------------------------------------------------------------------------


def _bypass_paywalls_enabled() -> bool:
    """Returns True when BYPASS_PAYWALLS env var is set to 'true' (case-insensitive).

    Defaults to False if unset so that toggling behaviour is explicit.
    """
    return os.getenv("BYPASS_PAYWALLS", "").strip().lower() == "true"


def _is_admin_email(email: str | None) -> bool:
    """Returns True if *email* is in the ADMIN_EMAILS allowlist (case-insensitive).

    Replicates the same lookup logic used in subscriptions/admin_auth.py so
    that the service layer can check admin status without depending on FastAPI
    request machinery.  Returns False if email is None or ADMIN_EMAILS is unset.
    """
    if not email:
        return False
    raw = os.getenv("ADMIN_EMAILS", "")
    admin_emails = {e.strip().lower() for e in raw.split(",") if e.strip()}
    return email.strip().lower() in admin_emails


def _patch_for_max_pro(ent: Entitlements) -> Entitlements:
    """Return a copy of *ent* with all caps set to -1 (unlimited) and all features enabled.

    Preserves: usage, tier string, status, user_id, has_overrides, and all
    Stripe billing fields.  Only caps and features are replaced.
    """
    max_caps = Caps(
        max_artists=-1,
        max_projects=-1,
        max_tasks=-1,
        max_storage_bytes=-1,
        max_split_sheets_per_month=-1,
        max_oneclick_runs_per_month=-1,
    )
    max_features = Features(
        zoe_enabled=True,
        oneclick_enabled=True,
        registry_enabled=True,
        integrations_allowed=["google_drive", "slack", "notion"],
    )
    return replace(ent, caps=max_caps, features=max_features)


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp from Supabase. Returns None if input is None/empty."""
    if not ts:
        return None
    return datetime.fromisoformat(ts)


class EntitlementsService:
    """Reads merged entitlements for a user. NO cache — each call hits DB."""

    def __init__(self, supabase: Client):
        self.supabase = supabase

    def get_for_user(self, user_id: str, *, is_admin: bool = False) -> Entitlements:
        """Returns merged entitlements (subscription + tier defaults + overrides + usage).

        If BYPASS_PAYWALLS env var is 'true' OR *is_admin* is True, all caps are
        patched to -1 (unlimited) and all features are enabled — effectively giving
        every affected user Pro-shaped entitlements without touching the DB.
        """
        sub = self._read_or_create_subscription(user_id)
        tier_row = self._read_tier_entitlements(sub["tier"])
        if tier_row is None:
            raise RuntimeError(
                f"Missing tier_entitlements row for tier='{sub['tier']}' "
                "— operator misconfiguration. Run the seed migration."
            )
        override = self._read_override(user_id)
        usage_row = self._read_or_create_usage_counter(user_id)
        usage_row = self._maybe_rollover_period(user_id, usage_row)

        caps, features, has_overrides = self._merge(tier_row, override)
        usage = Usage(
            total_storage_bytes=usage_row.get("total_storage_bytes", 0),
            split_sheets_this_period=usage_row.get("split_sheets_this_period", 0),
            zoe_queries_this_period=usage_row.get("zoe_queries_this_period", 0),
            oneclick_runs_this_period=usage_row.get("oneclick_runs_this_period", 0),
            period_end=_parse_iso(usage_row["period_end"]),
        )

        ent = Entitlements(
            user_id=user_id,
            tier=sub["tier"],
            status=sub["status"],
            caps=caps,
            features=features,
            usage=usage,
            has_overrides=has_overrides,
            stripe_subscription_id=sub.get("stripe_subscription_id"),
            stripe_price_id=sub.get("stripe_price_id"),
            current_period_end=_parse_iso(sub.get("current_period_end")),
            cancel_at_period_end=bool(sub.get("cancel_at_period_end", False)),
        )

        if _bypass_paywalls_enabled() or is_admin:
            ent = _patch_for_max_pro(ent)

        return ent

    # -----------------------------------------------------------------------
    # DB reads (with auto-create-on-miss)
    # -----------------------------------------------------------------------

    def _read_or_create_subscription(self, user_id: str) -> dict:
        res = self.supabase.table("subscriptions").select("*").eq("user_id", user_id).execute()
        if res.data:
            return res.data[0]
        self.supabase.table("subscriptions").upsert(
            {"user_id": user_id, "tier": "free", "status": "active"}, on_conflict="user_id"
        ).execute()
        res = self.supabase.table("subscriptions").select("*").eq("user_id", user_id).execute()
        return res.data[0] if res.data else {"user_id": user_id, "tier": "free", "status": "active"}

    def _read_tier_entitlements(self, tier: str) -> dict | None:
        res = self.supabase.table("tier_entitlements").select("*").eq("tier", tier).execute()
        return res.data[0] if res.data else None

    def _read_override(self, user_id: str) -> dict | None:
        res = self.supabase.table("tier_overrides").select("*").eq("user_id", user_id).execute()
        if not res.data:
            return None
        ovr = res.data[0]
        expires_at = _parse_iso(ovr.get("expires_at"))
        if expires_at is not None and expires_at < datetime.now(UTC):
            return None
        return ovr

    def _read_or_create_usage_counter(self, user_id: str) -> dict:
        res = self.supabase.table("usage_counters").select("*").eq("user_id", user_id).execute()
        if res.data:
            return res.data[0]
        now = datetime.now(UTC)
        self.supabase.table("usage_counters").upsert(
            {
                "user_id": user_id,
                "total_storage_bytes": 0,
                "split_sheets_this_period": 0,
                "period_start": now.isoformat(),
                "period_end": (now + relativedelta(months=1)).isoformat(),
            },
            on_conflict="user_id",
        ).execute()
        res = self.supabase.table("usage_counters").select("*").eq("user_id", user_id).execute()
        return (
            res.data[0]
            if res.data
            else {
                "user_id": user_id,
                "total_storage_bytes": 0,
                "split_sheets_this_period": 0,
                "period_start": now.isoformat(),
                "period_end": (now + relativedelta(months=1)).isoformat(),
            }
        )

    # -----------------------------------------------------------------------
    # Lazy period rollover (race-fixed)
    # -----------------------------------------------------------------------

    def _maybe_rollover_period(self, user_id: str, usage_row: dict) -> dict:
        """If period_end < now, advance period (one calendar month at a time) and reset counter.

        Race fix: the UPDATE includes WHERE period_end < new_period_end so concurrent
        racers don't both succeed — only the first UPDATE matches. The losing racer's
        UPDATE no-ops; we re-read to get the freshly-rolled row either way.
        """
        period_end = _parse_iso(usage_row.get("period_end"))
        now = datetime.now(UTC)
        if period_end is None or period_end >= now:
            return usage_row

        new_period_end = period_end
        while new_period_end < now:
            new_period_end = new_period_end + relativedelta(months=1)
        new_period_start = new_period_end - relativedelta(months=1)

        # Race fix: only the first racer's UPDATE finds period_end < new_period_end.
        self.supabase.table("usage_counters").update(
            {
                "split_sheets_this_period": 0,
                "zoe_queries_this_period": 0,
                "oneclick_runs_this_period": 0,
                "period_start": new_period_start.isoformat(),
                "period_end": new_period_end.isoformat(),
                "updated_at": now.isoformat(),
            }
        ).eq("user_id", user_id).lt("period_end", new_period_end.isoformat()).execute()

        # Re-read to get whichever row won (ours, or a concurrent racer's).
        res = self.supabase.table("usage_counters").select("*").eq("user_id", user_id).execute()
        return (
            res.data[0]
            if res.data
            else {
                **usage_row,
                "split_sheets_this_period": 0,
                "period_start": new_period_start.isoformat(),
                "period_end": new_period_end.isoformat(),
                "updated_at": now.isoformat(),
            }
        )

    # -----------------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------------

    @staticmethod
    def _merge(tier_row: dict, override: dict | None) -> tuple[Caps, Features, bool]:
        def pick(field: str):
            if override is not None and override.get(field) is not None:
                return override[field]
            return tier_row[field]

        caps = Caps(
            max_artists=pick("max_artists"),
            max_projects=pick("max_projects"),
            max_tasks=pick("max_tasks"),
            max_storage_bytes=pick("max_storage_bytes"),
            max_split_sheets_per_month=pick("max_split_sheets_per_month"),
            max_oneclick_runs_per_month=pick("max_oneclick_runs_per_month"),
        )
        features = Features(
            zoe_enabled=pick("zoe_enabled"),
            oneclick_enabled=pick("oneclick_enabled"),
            registry_enabled=pick("registry_enabled"),
            integrations_allowed=list(pick("integrations_allowed") or []),
        )
        return caps, features, override is not None

    # -----------------------------------------------------------------------
    # can() — single chokepoint with host-wins resolution
    # -----------------------------------------------------------------------

    def can(self, user_id: str, action, host_user_id: str | None = None, **ctx):
        """Returns CheckResult(allowed, reason, upgrade_required) for the given action.

        host_user_id semantics:
          - Cap actions (CREATE_*, GENERATE_SPLIT_SHEET): host_user_id is ignored.
            Caps are about how many of YOUR OWN resources you can create.
          - Feature actions (USE_ZOE/ONECLICK/REGISTRY/INTEGRATION): if host provided,
            allow if either acting user OR host has the feature.
          - UPLOAD_BYTES: storage is owner-scoped; check the host's cap if provided,
            else acting user's cap.
        """
        from subscriptions.models import Action, CheckResult

        ent = self.get_for_user(user_id)

        def deny(reason: str) -> CheckResult:
            return CheckResult(allowed=False, reason=reason, upgrade_required=True)

        def allow() -> CheckResult:
            return CheckResult(allowed=True, reason=None, upgrade_required=False)

        # Cap actions — always check acting user
        if action == Action.CREATE_ARTIST:
            return self._check_count_cap(ctx.get("current_count", 0), ent.caps.max_artists, "artists")
        if action == Action.CREATE_PROJECT:
            return self._check_count_cap(ctx.get("current_count", 0), ent.caps.max_projects, "projects")
        if action == Action.CREATE_TASK:
            return self._check_count_cap(ctx.get("current_count", 0), ent.caps.max_tasks, "tasks")

        if action == Action.GENERATE_SPLIT_SHEET:
            cap = ent.caps.max_split_sheets_per_month
            if cap == -1:
                return allow()
            if ent.usage.split_sheets_this_period >= cap:
                return deny(f"You've used your {cap} split sheet(s) for this period.")
            return allow()

        # UPLOAD_BYTES — owner-scoped storage; check host's cap if provided
        if action == Action.UPLOAD_BYTES:
            size = int(ctx.get("size", 0))
            owner_ent = ent
            if host_user_id and host_user_id != user_id:
                owner_ent = self.get_for_user(host_user_id)
            cap = owner_ent.caps.max_storage_bytes
            if cap == -1:
                return allow()
            if owner_ent.usage.total_storage_bytes + size > cap:
                return deny(
                    f"Upload would exceed the project owner's storage limit "
                    f"({owner_ent.usage.total_storage_bytes} + {size} > {cap} bytes)."
                )
            return allow()

        # Feature actions — host-wins
        if action == Action.USE_ZOE:
            if ent.features.zoe_enabled:
                return allow()
            if host_user_id and host_user_id != user_id:
                host_ent = self.get_for_user(host_user_id)
                if host_ent.features.zoe_enabled:
                    return allow()
            return deny("Zoe is a Pro feature.")

        if action == Action.USE_ONECLICK:
            # Feature flag check (with host-wins resolution)
            feature_ok = ent.features.oneclick_enabled
            resolved_ent = ent  # the entitlements whose caps we'll enforce
            if not feature_ok and host_user_id and host_user_id != user_id:
                host_ent = self.get_for_user(host_user_id)
                if host_ent.features.oneclick_enabled:
                    feature_ok = True
                    resolved_ent = host_ent
            if not feature_ok:
                return deny("OneClick is a Pro feature.")
            # Per-period cap check (against the entitlements that granted access)
            cap = resolved_ent.caps.max_oneclick_runs_per_month
            if cap != -1 and ent.usage.oneclick_runs_this_period >= cap:
                return deny(f"You've used your {cap} OneClick run(s) for this period.")
            return allow()

        if action == Action.USE_REGISTRY:
            if ent.features.registry_enabled:
                return allow()
            if host_user_id and host_user_id != user_id:
                host_ent = self.get_for_user(host_user_id)
                if host_ent.features.registry_enabled:
                    return allow()
            return deny("The Rights Registry is a Pro feature.")

        if action == Action.USE_INTEGRATION:
            name = ctx.get("name", "")
            if name in ent.features.integrations_allowed:
                return allow()
            if host_user_id and host_user_id != user_id:
                host_ent = self.get_for_user(host_user_id)
                if name in host_ent.features.integrations_allowed:
                    return allow()
            return deny(f"The {name} integration is not available on your plan.")

        # Defensive — unknown action
        return deny(f"Unknown action: {action!r}")

    @staticmethod
    def _check_count_cap(current: int, cap: int, label: str):
        from subscriptions.models import CheckResult

        if cap == -1 or current < cap:
            return CheckResult(allowed=True, reason=None, upgrade_required=False)
        return CheckResult(
            allowed=False,
            reason=f"You've reached your limit of {cap} {label}.",
            upgrade_required=True,
        )

    # -----------------------------------------------------------------------
    # Atomic counter increments (called by Zoe / OneClick endpoints)
    # -----------------------------------------------------------------------

    def increment_usage(self, user_id: str, counter_name: str) -> None:
        """Atomically increment one usage counter via the increment_usage_counter RPC.

        Best-effort — logs but does not raise on failure (so user-facing actions
        don't fail because of usage tracking).

        counter_name must be one of: 'zoe_queries_this_period',
        'oneclick_runs_this_period', 'split_sheets_this_period'.
        """
        import logging

        ALLOWED = {
            "zoe_queries_this_period",
            "oneclick_runs_this_period",
            "split_sheets_this_period",
        }
        if counter_name not in ALLOWED:
            logging.error("increment_usage: invalid counter %r", counter_name)
            return
        try:
            self.supabase.rpc(
                "increment_usage_counter",
                {"p_user_id": user_id, "p_counter_name": counter_name},
            ).execute()
        except Exception:
            logging.exception("increment_usage failed user_id=%s counter=%s", user_id, counter_name)

    # -----------------------------------------------------------------------
    # Bulk-resolve for list endpoints (no period rollover side effects)
    # -----------------------------------------------------------------------

    def bulk_get_for_users(self, user_ids: list[str]) -> dict[str, Entitlements]:
        """Returns {user_id: Entitlements} for a batch of users in 4 round-trips.

        Used by list endpoints that need host-wins resolution across many rows
        (e.g., listing 50 tasks across mixed-owner projects). Without this,
        the naive pattern would call get_for_user N times.

        NOTE: skips lazy period rollover (per-user concern; expensive in batch).
        Single-user reads via get_for_user still trigger rollover correctly.
        """
        if not user_ids:
            return {}

        subs = self.supabase.table("subscriptions").select("*").in_("user_id", user_ids).execute()
        subs_by_uid = {row["user_id"]: row for row in (subs.data or [])}

        # Always fetch the Free tier row alongside any tiers actually held by
        # users in this batch — users with no subscriptions row default to
        # Free, and we need that tier_row to materialize their Entitlements.
        # Using `or {"free"}` would only fire when the comprehension is empty,
        # silently dropping default-Free users from a mixed-tier batch.
        tiers = {row["tier"] for row in subs_by_uid.values()} | {"free"}
        tier_rows = self.supabase.table("tier_entitlements").select("*").in_("tier", list(tiers)).execute()
        tier_by_name = {row["tier"]: row for row in (tier_rows.data or [])}

        overrides = self.supabase.table("tier_overrides").select("*").in_("user_id", user_ids).execute()
        overrides_by_uid = {row["user_id"]: row for row in (overrides.data or [])}

        usages = self.supabase.table("usage_counters").select("*").in_("user_id", user_ids).execute()
        usage_by_uid = {row["user_id"]: row for row in (usages.data or [])}

        result: dict[str, Entitlements] = {}
        for uid in user_ids:
            sub = subs_by_uid.get(uid, {"tier": "free", "status": "active"})
            tier_row = tier_by_name.get(sub["tier"])
            if tier_row is None:
                # Operator misconfig — silently skip this user
                continue
            ovr = overrides_by_uid.get(uid)
            if ovr is not None:
                expires = _parse_iso(ovr.get("expires_at"))
                if expires is not None and expires < datetime.now(UTC):
                    ovr = None
            usage_row = usage_by_uid.get(uid, {})
            caps, features, has_overrides = self._merge(tier_row, ovr)
            usage = Usage(
                total_storage_bytes=usage_row.get("total_storage_bytes", 0),
                split_sheets_this_period=usage_row.get("split_sheets_this_period", 0),
                zoe_queries_this_period=usage_row.get("zoe_queries_this_period", 0),
                oneclick_runs_this_period=usage_row.get("oneclick_runs_this_period", 0),
                # If no usage_counters row yet (user hasn't made a metered request),
                # synthesize a future period_end matching what _read_or_create_usage_counter
                # would set on first write — keeps gating decisions consistent.
                period_end=_parse_iso(usage_row.get("period_end")) or (datetime.now(UTC) + relativedelta(months=1)),
            )
            result[uid] = Entitlements(
                user_id=uid,
                tier=sub.get("tier", "free"),
                status=sub.get("status", "active"),
                caps=caps,
                features=features,
                usage=usage,
                has_overrides=has_overrides,
            )
        return result

    # -----------------------------------------------------------------------
    # Safe variant — never raises, returns degraded=True Free defaults on error
    # -----------------------------------------------------------------------

    def get_for_user_safe(self, user_id: str, *, is_admin: bool = False) -> Entitlements:
        """Endpoint-friendly wrapper. Logs and returns degraded Free entitlements on error.

        Accepts the same *is_admin* flag as get_for_user; passes it through so that
        admin users (or all users when BYPASS_PAYWALLS=true) receive Pro-shaped
        entitlements even on the safe path.

        IMPORTANT: the hardcoded fallback caps/features below MUST be kept in sync with
        the seed values in supabase/migrations/20260509000001_subscription_foundation.sql
        (the `INSERT INTO tier_entitlements ('free', ...)` row). If you ratchet the Free
        tier defaults via a SQL UPDATE later, also update these constants — otherwise
        users hitting the degraded path during a DB outage will see stale (likely tighter)
        limits than the live tier defaults. Acceptable for v1 because:
          (1) the degraded path only fires when DB is unreachable (rare and brief), and
          (2) returning OLDER (tighter) defaults than current is fail-safe — never
              accidentally over-permissive.
        Future work: try a separate read of just `tier_entitlements` here before falling
        back to hardcoded; cheap if the rest of the DB is down but tier rows are cached
        somewhere.
        """
        try:
            return self.get_for_user(user_id, is_admin=is_admin)
        except Exception:
            import logging

            logging.exception("entitlements_degraded user_id=%s", user_id)
            now = datetime.now(UTC)
            return Entitlements(
                user_id=user_id,
                tier="free",
                status="active",
                # Keep these in sync with tier_entitlements 'free' seed row.
                caps=Caps(
                    max_artists=3,
                    max_projects=3,
                    max_tasks=50,
                    max_storage_bytes=1073741824,
                    max_split_sheets_per_month=5,
                    max_oneclick_runs_per_month=1,
                ),
                features=Features(
                    zoe_enabled=False,
                    oneclick_enabled=True,
                    registry_enabled=False,
                    integrations_allowed=["google_drive"],
                ),
                usage=Usage(
                    total_storage_bytes=0,
                    split_sheets_this_period=0,
                    zoe_queries_this_period=0,
                    oneclick_runs_this_period=0,
                    period_end=now + relativedelta(months=1),
                ),
                has_overrides=False,
                degraded=True,
            )
