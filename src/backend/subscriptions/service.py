"""EntitlementsService — single chokepoint for tier/cap/feature checks.

Reads four tables on every request (six when CREDITS_ENABLED) (no cache; see spec §3.1 / §4.1 #4).
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


def credits_enabled() -> bool:
    """True when the credits system is live (CREDITS_ENABLED env var).

    Flag retirement is CODE-LEVEL (spec §9): when on, zoe/oneclick/registry
    feature flags resolve true for every tier; stored values are never
    mutated so flipping this off restores legacy gating exactly.
    """
    return os.getenv("CREDITS_ENABLED", "").strip().lower() == "true"


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
        # -1 = unlimited, consistent with every other cap sentinel; never used
        # as a wallet grant (grant is hoisted pre-patch in get_for_user).
        monthly_credits=-1,
        max_works=-1,
        included_storage_bytes=-1,
    )
    max_features = Features(
        zoe_enabled=True,
        oneclick_enabled=True,
        registry_enabled=True,
        integrations_allowed=["google_drive", "slack"],
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

        # Code-level flag retirement (spec §9): under CREDITS_ENABLED the credit
        # balance IS the AI gate; stored flags are preserved untouched as the
        # rollback path.
        if credits_enabled():
            features = replace(features, zoe_enabled=True, oneclick_enabled=True, registry_enabled=True)

        # captured BEFORE any admin/bypass caps patch — the wallet grant must
        # always be the tier's real grant, never a patched sentinel.
        merged_monthly_grant = caps.monthly_credits

        # Admin = implicit Pro. Admins (profiles.is_admin=True) get unlimited
        # caps + all features regardless of subscription tier or override state.
        # Cleaner than maintaining a synthetic 'admin' tier_overrides row that
        # would drift if admin status is later revoked.
        #
        # Trade-off: env-only admins (ADMIN_EMAILS users whose profiles.is_admin
        # is false) won't get this implicit upgrade — they can use the self-grant
        # Pro button if needed. Most admins are DB admins via the /admin/users
        # promote flow, which sets profiles.is_admin=True.
        from subscriptions.admin_auth import is_db_admin  # local: same package, avoid circular at import time

        if is_db_admin(self.supabase, user_id):
            caps = Caps(
                max_artists=-1,
                max_projects=-1,
                max_tasks=-1,
                max_storage_bytes=-1,
                max_split_sheets_per_month=-1,
                max_oneclick_runs_per_month=-1,
                # -1 = unlimited, consistent with every other cap sentinel; never
                # used as a wallet grant (grant is hoisted pre-patch above).
                monthly_credits=-1,
                max_works=-1,
                included_storage_bytes=-1,
            )
            features = Features(
                zoe_enabled=True,
                oneclick_enabled=True,
                registry_enabled=True,
                integrations_allowed=["google_drive", "slack"],
            )

        usage = Usage(
            total_storage_bytes=usage_row.get("total_storage_bytes", 0),
            split_sheets_this_period=usage_row.get("split_sheets_this_period", 0),
            zoe_queries_this_period=usage_row.get("zoe_queries_this_period", 0),
            oneclick_runs_this_period=usage_row.get("oneclick_runs_this_period", 0),
            period_end=_parse_iso(usage_row["period_end"]),
        )

        credits_info = None
        if credits_enabled():
            from subscriptions.models import CreditsInfo

            wallet = self._read_or_create_wallet(user_id)
            wallet = self._maybe_rollover_wallet(wallet, merged_monthly_grant)
            credits_info = CreditsInfo(
                bundle_balance=wallet.get("bundle_balance", 0),
                reserve_balance=wallet.get("reserve_balance", 0),
                monthly_grant=merged_monthly_grant,
                overage_this_period=wallet.get("overage_this_period", 0),
                overage_enabled=bool(sub.get("overage_enabled", False)),
                overage_cap_credits=sub.get("overage_cap_credits"),
                storage_overage_enabled=bool(sub.get("storage_overage_enabled", False)),
                period_end=_parse_iso(wallet.get("period_end")),
                prices=self._get_credit_prices(),
            )

        ent = Entitlements(
            user_id=user_id,
            tier=sub["tier"],
            status=sub["status"],
            caps=caps,
            features=features,
            usage=usage,
            has_overrides=has_overrides,
            credits=credits_info,
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
        # `tester_revoked` is a sticky marker that prevents bootstrap-tester from
        # auto-re-granting; functionally it's NOT an override (all caps are null).
        # Treat as absent so has_overrides=False and the user reverts to tier defaults.
        if ovr.get("reason") == "tester_revoked":
            return None
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
        # Defensive backfill: this is the first read after a counter row was missing.
        # If the user predates the storage trigger (or the row was deleted), they
        # may have project_files / audio_files that aren't reflected in
        # total_storage_bytes. Run the recompute Postgres function once. Best-effort.
        try:
            self.supabase.rpc("recalc_user_storage", {"p_user_id": user_id}).execute()
        except Exception as e:
            import logging

            logging.warning("recalc_user_storage backfill failed for %s: %s", user_id, e)
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

    def _read_or_create_wallet(self, user_id: str) -> dict:
        res = (
            self.supabase.table("credit_wallets").select("*").eq("owner_type", "user").eq("owner_id", user_id).execute()
        )
        if res.data:
            return res.data[0]
        now = datetime.now(UTC)
        # period_end = now: the caller's _maybe_rollover_wallet fires immediately
        # and grants the tier's monthly credits (same seeding trick as the
        # migration's trigger/backfill — no wallet ever starts un-granted).
        self.supabase.table("credit_wallets").upsert(
            {
                "owner_type": "user",
                "owner_id": user_id,
                "period_start": (now - relativedelta(months=1)).isoformat(),
                "period_end": now.isoformat(),
            },
            on_conflict="owner_type,owner_id",
        ).execute()
        res = (
            self.supabase.table("credit_wallets").select("*").eq("owner_type", "user").eq("owner_id", user_id).execute()
        )
        return (
            res.data[0]
            if res.data
            else {
                "id": None,
                "owner_type": "user",
                "owner_id": user_id,
                "bundle_balance": 0,
                "reserve_balance": 0,
                "overage_this_period": 0,
                "period_start": (now - relativedelta(months=1)).isoformat(),
                "period_end": now.isoformat(),
            }
        )

    def _get_credit_prices(self) -> dict[str, int]:
        res = self.supabase.table("credit_prices").select("*").execute()
        return {row["action"]: row["credits"] for row in (res.data or [])}

    def _maybe_rollover_wallet(self, wallet: dict, monthly_grant: int) -> dict:
        """Lazy monthly rollover: advance period (month steps), expire bundle, grant.

        NOTE: the boundary operators are DELIBERATELY flipped versus
        _maybe_rollover_period (guard `period_end > now` here vs `>= now` there;
        month-step loop `<= now` here vs `< now` there). Wallets are seeded with
        period_end = now() exactly, so an exactly-now bound must roll (and the
        loop must still advance it) for the seed-grant to fire on first read.
        Do NOT "fix" this asymmetry. The RPC is race-guarded (period must have
        actually ended AND new bound must advance it), so concurrent racers
        converge and clock-skewed callers can't double-roll.
        """
        period_end = _parse_iso(wallet.get("period_end"))
        now = datetime.now(UTC)
        if wallet.get("id") is None or period_end is None or period_end > now:
            return wallet
        new_period_end = period_end
        while new_period_end <= now:
            new_period_end = new_period_end + relativedelta(months=1)
        new_period_start = new_period_end - relativedelta(months=1)
        try:
            self.supabase.rpc(
                "rollover_wallet",
                {
                    "p_wallet_id": wallet["id"],
                    "p_monthly_grant": monthly_grant,
                    "p_new_period_start": new_period_start.isoformat(),
                    "p_new_period_end": new_period_end.isoformat(),
                },
            ).execute()
        except Exception:
            import logging

            logging.exception("rollover_wallet failed wallet=%s", wallet.get("id"))
        res = self.supabase.table("credit_wallets").select("*").eq("id", wallet["id"]).execute()
        return res.data[0] if res.data else wallet

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

        def pick_default(field: str, default):
            if override is not None and override.get(field) is not None:
                return override[field]
            return tier_row.get(field, default)

        caps = Caps(
            max_artists=pick("max_artists"),
            max_projects=pick("max_projects"),
            max_tasks=pick("max_tasks"),
            max_storage_bytes=pick("max_storage_bytes"),
            max_split_sheets_per_month=pick("max_split_sheets_per_month"),
            max_oneclick_runs_per_month=pick("max_oneclick_runs_per_month"),
            monthly_credits=pick_default("monthly_credits", 0),
            max_works=pick_default("max_works", -1),
            included_storage_bytes=pick_default("included_storage_bytes", -1),
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

        if action == Action.CREATE_WORK:
            return self._check_count_cap(ctx.get("current_count", 0), ent.caps.max_works, "registered works")

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
            projected = owner_ent.usage.total_storage_bytes + size
            if cap != -1 and projected > cap:
                return deny(
                    f"Upload would exceed the project owner's storage limit "
                    f"({owner_ent.usage.total_storage_bytes} + {size} > {cap} bytes)."
                )
            # Credits model: paid tiers have unlimited hard cap (-1) but an
            # included allowance; past it, uploads need storage pay-per-use
            # opt-in (spec §5). Existing files are never touched.
            if credits_enabled() and owner_ent.tier in self.PAID_TIERS:
                included = owner_ent.caps.included_storage_bytes
                storage_ok = owner_ent.credits.storage_overage_enabled if owner_ent.credits else False
                if included != -1 and projected > included and not storage_ok:
                    return deny(
                        "This upload goes past the storage included in your plan. "
                        "Turn on storage pay-per-use in Billing settings to continue."
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
            return deny("The Metadata Registry is a Pro feature.")

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
    # Credits: decision + debit (spec §3, §4, §7)
    # -----------------------------------------------------------------------

    PAID_TIERS = ("pro", "pro_max")

    def check_credits(self, user_id: str, action: str, *, is_admin: bool = False):
        """Can this user run `action` right now, and at what price?

        Order: bypass/admin → price lookup → wallet balance → opt-in overage →
        wall. Degraded policy (spec §12): paid fails open (uncharged), free
        fails closed. NOTE: re-reads tables per call like the rest of this
        service (no-cache philosophy) — optimization deferred deliberately.
        """
        import logging

        from subscriptions.admin_auth import is_db_admin
        from subscriptions.models import CreditCheckResult

        if not credits_enabled():
            return CreditCheckResult(allowed=True, price=0)
        if _bypass_paywalls_enabled() or is_admin or is_db_admin(self.supabase, user_id):
            # Short-circuit BEFORE wallet read/debit — no ledger rows for admins.
            return CreditCheckResult(allowed=True, price=0)

        try:
            sub = self._read_or_create_subscription(user_id)
        except Exception:
            # Full outage: the tier is unknowable without the subscription row,
            # so even genuinely paid users fail CLOSED here — a deliberate
            # conservative call (spec §12 tension: fail-open is reserved for
            # KNOWN paid tiers in the degraded handler below).
            logging.exception("check_credits: subscription read failed user=%s", user_id)
            return CreditCheckResult(
                allowed=False,
                price=0,
                degraded=True,
                reason="We couldn't check your plan just now — please try again in a moment.",
            )
        tier = sub.get("tier", "free")

        try:
            prices = self._get_credit_prices()
        except Exception:
            logging.exception("check_credits: price read failed user=%s", user_id)
            prices = None
        if prices is not None and action not in prices:
            # Config error (unseeded action), NOT an outage: deny every tier
            # with honest copy so the gap surfaces in the first QA run instead
            # of silently leaking COGS (paid) or lying about a retry (free).
            logging.error("check_credits: no credit price seeded for action %r", action)
            return CreditCheckResult(
                allowed=False,
                price=0,
                reason="This action isn't set up for credits yet. Please contact support.",
            )

        try:
            if prices is None:
                # A price READ failure (unlike the missing-key config error
                # above) is an outage — route into the degraded policy below.
                raise RuntimeError("credit price read failed")
            price = prices[action]
            if price <= 0:
                # Retuned-to-0 actions (or negative drift in seeded data) are
                # free — never wall a zero-price action behind the balance check.
                return CreditCheckResult(allowed=True, price=0, reset_date=None)
            tier_row = self._read_tier_entitlements(tier)
            if tier_row is None:
                # Operator misconfig — fail LOUD into the degraded policy
                # (paid open, free closed) with the wallet untouched, never
                # a destructive 0-grant rollover. Mirrors get_for_user.
                raise RuntimeError(f"Missing tier_entitlements row for tier={tier!r}")
            override = self._read_override(user_id)
            caps, _, _ = self._merge(tier_row, override)
            monthly_grant = caps.monthly_credits
            wallet = self._read_or_create_wallet(user_id)
            wallet = self._maybe_rollover_wallet(wallet, monthly_grant)
            balance = wallet.get("bundle_balance", 0) + wallet.get("reserve_balance", 0)
            reset_date = _parse_iso(wallet.get("period_end"))

            if balance >= price:
                return CreditCheckResult(allowed=True, price=price, reset_date=reset_date)

            if tier in self.PAID_TIERS:
                if sub.get("overage_enabled"):
                    cap = sub.get("overage_cap_credits")
                    used = wallet.get("overage_this_period", 0)
                    # Accepted bounded overshoot: N concurrent racers can each
                    # pass this check before any debit lands, exceeding the cap
                    # by at most (N-1) x price — same acceptance as the
                    # balance-drift concurrency policy.
                    if cap is not None and used + price > cap:
                        return CreditCheckResult(
                            allowed=False,
                            price=price,
                            reset_date=reset_date,
                            reason=(
                                "You've reached your pay-per-use limit for this month. "
                                "Raise it in Billing settings, or wait for your credits to reset."
                            ),
                        )
                    return CreditCheckResult(allowed=True, price=price, use_overage=True, reset_date=reset_date)
                return CreditCheckResult(
                    allowed=False,
                    price=price,
                    overage_available=True,
                    reset_date=reset_date,
                    reason="You've used your included credits for this month.",
                )

            return CreditCheckResult(
                allowed=False,
                price=price,
                upgrade_required=True,
                reset_date=reset_date,
                reason="You've used this month's credits.",
            )
        except Exception:
            logging.exception("check_credits degraded user=%s action=%s", user_id, action)
            if tier in self.PAID_TIERS:
                # Fail open for paying users; the action goes uncharged (accepted).
                return CreditCheckResult(allowed=True, price=0, degraded=True)
            return CreditCheckResult(
                allowed=False,
                price=0,
                degraded=True,
                reason="Credits are temporarily unavailable — please try again in a moment.",
            )

    def debit_for_action(self, user_id: str, grant) -> None:
        """Debit a CreditGrant after the action succeeded. Best-effort, never raises.

        NOTE: overage_debit rows are billed OFF the request path — daily
        sweep creates pending InvoiceItems; invoice.created attaches
        stragglers to the draft renewal invoice. Never call Stripe here.
        """
        import logging

        if grant is None or not grant.enabled or grant.price <= 0:
            return
        try:
            wallet = self._read_or_create_wallet(user_id)
            if wallet.get("id") is None:
                return
            self.supabase.rpc(
                "debit_credits",
                {
                    "p_wallet_id": wallet["id"],
                    "p_amount": grant.price,
                    "p_action": grant.action,
                    "p_request_id": grant.request_id,
                    "p_kind": grant.kind,
                    "p_metadata": {},
                },
            ).execute()
        except Exception:
            logging.exception(
                "debit_for_action failed user=%s action=%s request=%s",
                user_id,
                grant.action,
                grant.request_id,
            )

    # -----------------------------------------------------------------------
    # Per-tool credit usage (Account & Billing usage view)
    # -----------------------------------------------------------------------

    def get_credit_usage(self, user_id: str) -> dict:
        """Per-tool credit spend for the current wallet period (Account & Billing usage view).

        Returns {"enabled": False} when credits are off. Otherwise aggregates
        credit_ledger debit/overage rows since period_start, grouped by action.
        """
        if not credits_enabled():
            return {"enabled": False}

        sub = self._read_or_create_subscription(user_id)
        tier = sub.get("tier", "free")
        tier_row = self._read_tier_entitlements(tier)
        if tier_row is None:
            raise RuntimeError(f"Missing tier_entitlements row for tier={tier!r}")
        override = self._read_override(user_id)
        caps, _, _ = self._merge(tier_row, override)
        monthly_grant = caps.monthly_credits

        wallet = self._read_or_create_wallet(user_id)
        wallet = self._maybe_rollover_wallet(wallet, monthly_grant)
        period_start = wallet.get("period_start")
        prices = self._get_credit_prices()

        agg: dict[str, dict] = {}
        if wallet.get("id") is not None and period_start is not None:
            rows = (
                self.supabase.table("credit_ledger")
                .select("action, delta, kind, metadata")
                .eq("wallet_id", wallet["id"])
                .gte("created_at", period_start)
                .in_("kind", ["debit", "overage_debit"])
                .execute()
            )
            for r in rows.data or []:
                action = r.get("action")
                if not action:
                    continue
                if r.get("kind") == "overage_debit":
                    amt = (r.get("metadata") or {}).get("credits_billed") or 0
                else:
                    amt = -(r.get("delta") or 0)
                a = agg.setdefault(action, {"count": 0, "spent": 0})
                a["count"] += 1
                a["spent"] += amt

        tools = []
        for action in ("oneclick_run", "registry_parse", "zoe_message"):
            a = agg.get(action, {"count": 0, "spent": 0})
            tools.append({"action": action, "price": prices.get(action), "count": a["count"], "spent": a["spent"]})

        bundle = wallet.get("bundle_balance", 0)
        reserve = wallet.get("reserve_balance", 0)
        return {
            "enabled": True,
            "periodStart": period_start,
            "periodEnd": wallet.get("period_end"),
            "monthlyGrant": monthly_grant,
            "bundleBalance": bundle,
            "reserveBalance": reserve,
            "balance": bundle + reserve,
            "overageThisPeriod": wallet.get("overage_this_period", 0),
            "tools": tools,
        }

    def get_credit_usage_safe(self, user_id: str) -> dict:
        """Never-raises wrapper for the endpoint."""
        import logging

        try:
            return self.get_credit_usage(user_id)
        except Exception:
            logging.exception("get_credit_usage failed user=%s", user_id)
            return {"enabled": False}

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
            # Code-level flag retirement (spec §9) — keep in lockstep with
            # get_for_user so bulk resolution can never disagree with it.
            if credits_enabled():
                features = replace(features, zoe_enabled=True, oneclick_enabled=True, registry_enabled=True)
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
                    monthly_credits=50,
                    max_works=10,
                    included_storage_bytes=1073741824,
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
