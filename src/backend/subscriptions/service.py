"""EntitlementsService — single chokepoint for tier/cap/feature checks.

Reads four tables on every request (six when CREDITS_ENABLED) (no cache; see spec §3.1 / §4.1 #4).
Per-field override merge with integrations_allowed replace-semantics.
Lazy period rollover and the can() chokepoint are added in Tasks 4 and 5.
"""

import logging
import os
from dataclasses import replace
from datetime import UTC, datetime

from dateutil.relativedelta import relativedelta
from supabase import Client

from subscriptions.models import (
    Action,
    Caps,
    CheckResult,
    CreditCheckResult,
    CreditsInfo,
    Entitlements,
    Features,
    ManagedByOrg,
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


def licensing_enabled() -> bool:
    """True when org/seat licensing (Phase B) is live (LICENSING_ENABLED env var).

    Flag retirement is CODE-LEVEL, same true-rollback discipline as
    credits_enabled: the entire /orgs surface 404s (router-level dependency),
    and entitlements resolution short-circuits to personal billing, when
    this is off. No stored row is ever mutated by toggling it, so flipping
    it back off restores pre-licensing behavior exactly.
    """
    return os.getenv("LICENSING_ENABLED", "").strip().lower() == "true"


# ---------------------------------------------------------------------------
# ENTERPRISE_SHAPE — the entitlements an ACTIVE org seat synthesizes at
# resolution time (Licensing Phase B, spec §5, rules 11/12). NO tier_entitlements
# row, NO subscriptions.tier mutation: this is a code constant, same pattern as
# the admin implicit-Pro. Counts are UNLIMITED (-1: zero marginal cost), all
# features on, integrations both allowed — but STORAGE IS FINITE (rule 12): a
# one-time minimum pool purchase must never confer unlimited un-billed storage
# forever, and the storage-overage sweep cannot bill a -1 cap. Storage reads from
# ENTERPRISE_SEAT_STORAGE_BYTES (CALIBRATE 500 GB) at call time so operators can
# retune it without a redeploy of this constant.
# ---------------------------------------------------------------------------

ENTERPRISE_SHAPE_INTEGRATIONS = ["google_drive", "dropbox"]


def _enterprise_seat_storage_bytes() -> int:
    """Finite per-seat storage ceiling (rule 12). Never -1."""
    return int(os.getenv("ENTERPRISE_SEAT_STORAGE_BYTES", str(500 * 1024**3)))


def _enterprise_caps() -> Caps:
    storage = _enterprise_seat_storage_bytes()
    return Caps(
        max_artists=-1,
        max_projects=-1,
        max_tasks=-1,
        # FINITE storage (rule 12) — max == included, hard block, no seat overage.
        max_storage_bytes=storage,
        max_split_sheets_per_month=-1,
        max_oneclick_runs_per_month=-1,
        # Seats draw org allocations, not a personal monthly grant — 0, and it must
        # NEVER reach _maybe_rollover_wallet for the personal wallet (rule 11).
        monthly_credits=0,
        max_works=-1,
        included_storage_bytes=storage,
    )


def _enterprise_features() -> Features:
    return Features(
        zoe_enabled=True,
        oneclick_enabled=True,
        registry_enabled=True,
        integrations_allowed=list(ENTERPRISE_SHAPE_INTEGRATIONS),
    )


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
        integrations_allowed=["google_drive", "dropbox"],
    )
    return replace(ent, caps=max_caps, features=max_features)


def _parse_iso(ts: str | None) -> datetime | None:
    """Parse an ISO timestamp from Supabase. Returns None if input is None/empty."""
    if not ts:
        return None
    return datetime.fromisoformat(ts)


def _pool_visible_to(role: str | None) -> bool:
    """Whether an org member of this role may see the shared credit pool.

    Admins only. The pool balance — with the dispersal and activation figures
    that sit beside it — is a commercial fact about the ORG (what it bought,
    what it has left), not about the member's own access. A member's own
    ceiling is `member_cap`, which they always see; with no cap they draw
    straight from the pool and the honest thing to show is exactly that, with
    no number attached.

    Defaults CLOSED: an unknown or missing role sees nothing. This is the one
    predicate for pool visibility on the entitlements side — orgs.service has
    the matching one for the org row itself (`redact_org_for_role`).
    """
    return role == "admin"


class EntitlementsService:
    """Reads merged entitlements for a user. NO cache — each call hits DB."""

    def __init__(self, supabase: Client):
        self.supabase = supabase

    def get_for_user(self, user_id: str, *, is_admin: bool = False) -> Entitlements:
        """Returns merged entitlements (subscription + tier defaults + overrides + usage).

        If BYPASS_PAYWALLS env var is 'true' OR *is_admin* is True, all caps are
        patched to -1 (unlimited) and all features are enabled — effectively giving
        every affected user Pro-shaped entitlements without touching the DB.

        CONTEXT-AWARE (Licensing Phase B, spec §5): when the caller is in an ACTIVE
        org billing context (`_resolve_context` confers it), caps/features come from
        ENTERPRISE_SHAPE and the credits block reflects the SEAT wallet — the PERSONAL
        WALLET IS NEVER READ AND NEVER ROLLED (rule 11: monthly_credits=0 through the
        personal rollover call site would expire a subscriber's whole bundle). Personal
        context (or licensing off) is byte-identical to pre-licensing.

        KNOWN DIVERGENCE (spec §5): this is context-aware, but `bulk_get_for_users` is
        personal-only BY DESIGN — a cap checked through the bulk (host-wins) path in
        org context gets the PERSONAL answer, not the enterprise one. Accepted for v1;
        folded into the resource-derivation follow-up. Do not "fix" bulk to read context.
        """
        # Resolve billing context FIRST (spec §5). None = personal (licensing off, no
        # preference, or a dead/pending reference — pending returns non-None with
        # pending=True but does NOT confer enterprise). Only an ACTIVE org confers.
        ctx = self._resolve_context(user_id)
        in_org = ctx is not None and not ctx.get("pending")

        # Personal subscription is read in BOTH contexts — its tier/status/stripe
        # fields still surface (the org-context profile shows "you can keep or cancel
        # your personal plan"), but in org context its TIER never drives caps and its
        # WALLET is never touched.
        sub = self._read_or_create_subscription(user_id)

        # Storage/usage is a single per-user counter that follows the user across
        # contexts (spec §5 seat-storage aftermath) — read/rolled the same either way.
        usage_row = self._read_or_create_usage_counter(user_id)
        usage_row = self._maybe_rollover_period(user_id, usage_row)
        usage = Usage(
            total_storage_bytes=usage_row.get("total_storage_bytes", 0),
            split_sheets_this_period=usage_row.get("split_sheets_this_period", 0),
            zoe_queries_this_period=usage_row.get("zoe_queries_this_period", 0),
            oneclick_runs_this_period=usage_row.get("oneclick_runs_this_period", 0),
            period_end=_parse_iso(usage_row["period_end"]),
        )

        managed_by_org: ManagedByOrg | None = None
        credits_info = None

        if in_org:
            # ---- ORG CONTEXT (rule 11) ----------------------------------------
            # Enterprise shape from the code constant; NO tier row, NO merge, NO
            # personal wallet read/rollover. `managed_by_org` is set UNCONDITIONALLY
            # (independent of credits_enabled()) — the storage-wall support-copy
            # detection in can() (UPLOAD_BYTES, rule 13) and the top-level
            # billingContext payload field (Task 3 follow-up) both key off its mere
            # presence, not off the credits flag.
            caps = _enterprise_caps()
            features = _enterprise_features()
            has_overrides = False
            managed_by_org = ManagedByOrg(
                org_id=ctx["org_id"], org_name=ctx["org_name"], role=ctx["role"], kind=ctx.get("kind")
            )

            # The credits block is the ORG POOL, never rolled from here (the
            # dispersal sweep owns the pool's period). Built ONLY when
            # credits_enabled(), mirroring the personal branch below, so a
            # credits-off org read never lazily creates a pool nobody will use.
            #
            # `monthly_grant` is the member's CAP, not a grant: nothing is
            # allocated to them, so what the UI must show is "your ceiling on
            # the shared pool".
            #
            # POOL VISIBILITY (see _pool_visible_to): the balance is the ORG's
            # money, so only an org ADMIN sees it. A plain member sees their own
            # cap and nothing else; with no cap they see that they draw on the
            # org pool, without a number. Redaction happens HERE, at the read,
            # not in the UI — a number never sent cannot leak.
            if credits_enabled():
                from orgs.wallets import read_or_create_org_wallet

                pool = read_or_create_org_wallet(self.supabase, ctx["org_id"])
                cap, cap_used = self._member_cap(ctx["org_id"], ctx["org_member_id"])
                show_pool = _pool_visible_to(ctx.get("role"))
                credits_info = CreditsInfo(
                    bundle_balance=pool.get("bundle_balance", 0) if show_pool else None,
                    reserve_balance=pool.get("reserve_balance", 0) if show_pool else None,
                    monthly_grant=cap if cap is not None else 0,
                    overage_this_period=pool.get("overage_this_period", 0),
                    # NO pay-as-you-go on a pool — a member past their cap asks
                    # for a raise; a dry pool is the admin's to top up.
                    overage_enabled=False,
                    overage_cap_credits=cap,
                    period_end=_parse_iso(pool.get("period_end")),
                    prices=self._get_credit_prices(),
                    member_cap=cap,
                    member_cap_used=cap_used,
                )
        else:
            # ---- PERSONAL CONTEXT (byte-identical to pre-licensing) ------------
            tier_row = self._read_tier_entitlements(sub["tier"])
            if tier_row is None:
                raise RuntimeError(
                    f"Missing tier_entitlements row for tier='{sub['tier']}' "
                    "— operator misconfiguration. Run the seed migration."
                )
            override = self._read_override(user_id)
            caps, features, has_overrides = self._merge(tier_row, override)

            # captured BEFORE any admin/bypass caps patch — the wallet grant must
            # always be the tier's real grant, never a patched sentinel. Precedence
            # (Task 1, spec §1): explicit override > grandfathered > tier value.
            merged_monthly_grant = self._resolve_monthly_grant(sub, override, caps.monthly_credits)

            if credits_enabled():
                wallet = self._read_or_create_wallet(user_id)
                wallet = self._maybe_rollover_wallet(wallet, merged_monthly_grant)
                credits_info = CreditsInfo(
                    bundle_balance=wallet.get("bundle_balance", 0),
                    reserve_balance=wallet.get("reserve_balance", 0),
                    monthly_grant=merged_monthly_grant,
                    overage_this_period=wallet.get("overage_this_period", 0),
                    overage_enabled=bool(sub.get("overage_enabled", False)),
                    overage_cap_credits=sub.get("overage_cap_credits"),
                    period_end=_parse_iso(wallet.get("period_end")),
                    prices=self._get_credit_prices(),
                )

        # Admin = implicit Pro. Admins (profiles.is_admin=True) get unlimited
        # caps + all features regardless of subscription tier, override, OR org
        # context. Applied AFTER the branch so it keeps today's precedence.
        # Cleaner than maintaining a synthetic 'admin' tier_overrides row that
        # would drift if admin status is later revoked.
        #
        # Trade-off: env-only admins (ADMIN_EMAILS users whose profiles.is_admin
        # is false) won't get this implicit upgrade — they can use the self-grant
        # Pro button if needed. Most admins are DB admins via the /admin/users
        # promote flow, which sets profiles.is_admin=True.
        from subscriptions.admin_auth import is_db_admin  # local: same package, avoid circular at import time

        if is_db_admin(self.supabase, user_id):
            # Team dials are not count caps (never -1) — an implicit-Pro admin
            # gets the PRO tier's real dials, read live rather than hardcoded,
            # so a dial retune doesn't require touching this patch too.
            try:
                pro_tier_row = self._read_tier_entitlements("pro") or {}
            except Exception:
                logging.exception("admin implicit-Pro: failed to read pro tier_entitlements row")
                pro_tier_row = {}
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
                max_teams=pro_tier_row.get("max_teams", 0),
                max_team_members=pro_tier_row.get("max_team_members", 0),
                team_storage_bytes=pro_tier_row.get("team_storage_bytes", 0),
            )
            features = Features(
                zoe_enabled=True,
                oneclick_enabled=True,
                registry_enabled=True,
                integrations_allowed=["google_drive", "dropbox"],
            )

        # availableContexts is included for ALL users when licensing is on (personal
        # users with no seats get just the personal entry). None when off → the
        # to_dict key is omitted entirely so the pre-licensing payload is identical.
        available_contexts = self._list_available_contexts(user_id) if licensing_enabled() else None

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
            managed_by_org=managed_by_org,
            available_contexts=available_contexts,
        )

        if _bypass_paywalls_enabled() or is_admin:
            ent = _patch_for_max_pro(ent)

        return ent

    # -----------------------------------------------------------------------
    # Billing context resolution (Licensing Phase B, spec §5)
    # -----------------------------------------------------------------------

    @staticmethod
    def _first_row(res) -> dict | None:
        """Normalize a Supabase result's `.data` (list from a plain execute, dict
        from maybe_single, None, or []) to a single row dict or None."""
        data = getattr(res, "data", None)
        if isinstance(data, list):
            return data[0] if data else None
        return data or None

    def _resolve_context(self, user_id: str) -> dict | None:
        """Resolve the caller's BILLING CONTEXT (spec §5, rules 7 + 11).

        Returns None for PERSONAL context — licensing off, no stored preference, or a
        genuinely DEAD reference (in which case the stale preference is lazily
        cleared). Returns a dict {org_id, org_name, org_member_id, role, pending}
        when `profiles.billing_context_org_id` points at an org where the caller
        holds an ACTIVE seat AND the org is non-archived with status in
        ('active','pending'):
          - org 'active'  -> pending=False — CONFERS the enterprise shape.
          - org 'pending' -> pending=True  — confers NOTHING yet (caller treats it as
                             personal), but the preference is DELIBERATELY KEPT
                             (rule 7: "not yet active" ≠ "stale"; the moment the org
                             activates, billing context just starts working — the
                             create-org → invite → buy onboarding order must not wipe
                             every member's preference before activation).

        A SUSPENDED org (status not in active/pending, but the org row still exists
        un-archived and the caller's seat is still active) gets the same rule-7
        "park, don't clear" treatment as pending: the caller resolves to PERSONAL
        while the org is suspended, but the stored preference survives, so org
        billing resumes automatically the moment the org is reactivated. Suspension
        is temporary by definition — clearing here would silently dump every member
        onto personal billing PERMANENTLY, with nobody switched back on reactivate.

        `billing_context_org_id` is ATTACKER-CONTROLLED (users can PATCH their own
        profiles row): this validation is the load-bearing security gate. Anything
        that is not the caller's own active seat in a live org falls closed to
        personal, and lazy-clear fires for genuinely DEAD references (no seat,
        revoked/suspended/removed seat, archived/nonexistent org) — NEVER for a
        pending or suspended org. The whole function short-circuits to None when
        licensing is off.
        """
        if not licensing_enabled():
            return None

        prof = self.supabase.table("profiles").select("billing_context_org_id").eq("id", user_id).execute()
        prof_row = self._first_row(prof)
        org_id = prof_row.get("billing_context_org_id") if prof_row else None
        if not org_id:
            return None

        seat = self._first_row(
            self.supabase.table("org_members")
            .select("id, role, status")
            .eq("org_id", org_id)
            .eq("user_id", user_id)
            .eq("status", "active")
            .execute()
        )
        if not seat:
            self._clear_billing_context(user_id)
            return None

        org = self._first_row(
            self.supabase.table("organizations")
            .select("id, name, status, archived_at, kind")
            .eq("id", org_id)
            .execute()
        )
        if not org or org.get("archived_at") is not None:
            self._clear_billing_context(user_id)
            return None
        if org.get("status") not in ("active", "pending"):
            # SUSPENDED org: park, don't clear (see docstring). Personal context
            # for now; org billing resumes automatically on reactivation.
            return None

        return {
            "org_id": org_id,
            "org_name": org.get("name"),
            "org_member_id": seat.get("id"),
            "role": seat.get("role"),
            "pending": org.get("status") == "pending",
            "kind": org.get("kind"),
        }

    def _clear_billing_context(self, user_id: str) -> None:
        """Best-effort lazy-clear of a dead billing-context preference (spec §5).

        NEVER raises — a failed clear must not break an entitlements read; the next
        read simply re-attempts. Fires only for genuinely dead references (see
        _resolve_context), never for a pending or suspended org (rule 7)."""
        try:
            self.supabase.table("profiles").update({"billing_context_org_id": None}).eq("id", user_id).execute()
        except Exception:
            logging.warning("failed to lazy-clear billing_context_org_id for %s", user_id)

    def _list_available_contexts(self, user_id: str) -> list[dict]:
        """The `availableContexts` payload (spec §5): personal + every ACTIVE seat
        whose org is non-archived and status in ('active','pending').

        INDEPENDENT of the stored preference — it lists what the caller COULD switch
        to (the switcher renders from it). Pending-org seats ARE included, marked
        {pending: True}. Best-effort: a broken org read must never degrade the whole
        entitlements payload, so it falls back to personal-only."""
        contexts: list[dict] = [{"type": "personal"}]
        try:
            members = (
                self.supabase.table("org_members")
                .select("org_id, role, status")
                .eq("user_id", user_id)
                .eq("status", "active")
                .execute()
            )
            rows = members.data or []
            if not rows:
                return contexts
            org_ids = [r["org_id"] for r in rows]
            orgs = (
                self.supabase.table("organizations")
                .select("id, name, status, archived_at, kind")
                .in_("id", org_ids)
                .execute()
            )
            org_by_id = {o["id"]: o for o in (orgs.data or [])}
            for r in rows:
                org = org_by_id.get(r.get("org_id"))
                if not org or org.get("archived_at") is not None:
                    continue
                if org.get("status") not in ("active", "pending"):
                    continue
                contexts.append(
                    {
                        "type": "org",
                        "orgId": org["id"],
                        "orgName": org.get("name"),
                        "role": r.get("role"),
                        "pending": org.get("status") == "pending",
                        "kind": org.get("kind"),
                    }
                )
        except Exception:
            logging.warning("failed to build availableContexts for %s", user_id)
            return [{"type": "personal"}]
        return contexts

    # -----------------------------------------------------------------------
    # Resource -> billing-org resolution (Licensing Phase C, spec Sections 6/11)
    #
    # These two resolvers answer "does this RESOURCE live in a project linked
    # to an org where the CALLER holds an active seat?" — independent of the
    # caller's ambient `_resolve_context` preference. Task 6 (check_credits)
    # and Task 7 (can()) are the only callers; this file's chokepoints
    # themselves are untouched here (pure additions).
    # -----------------------------------------------------------------------

    def _org_context_for(self, user_id: str, org_id: str) -> dict | None:
        """The `_resolve_context` ctx shape for one org, or None when the org is
        not live or the caller holds no ACTIVE seat in it. Shared by the artist
        and project-link derivation paths so they cannot diverge on what
        'billable org' means."""
        org = self._first_row(
            self.supabase.table("organizations")
            .select("id, name, status, archived_at, kind")
            .eq("id", org_id)
            .execute()
        )
        if not org or org.get("status") != "active" or org.get("archived_at") is not None:
            return None
        member = self._first_row(
            self.supabase.table("org_members")
            .select("id, role")
            .eq("org_id", org_id)
            .eq("user_id", user_id)
            .eq("status", "active")
            .execute()
        )
        if not member:
            return None
        return {
            "org_id": org["id"],
            "org_name": org.get("name"),
            "org_member_id": member["id"],
            "role": member.get("role"),
            "kind": org.get("kind"),
        }

    def resolve_billing_org_for_project(self, user_id: str, project_id: str) -> dict | None:
        """Resolve the ACTIVE org billing context for a specific PROJECT.

        ONE edge: ARTIST OWNERSHIP (`projects.artist_id -> artists.team_id`).
        An artist belongs to at most one team, and everything below it —
        projects, works, files, audio — belongs to that team by construction,
        so a single two-hop read settles who pays with no ambiguity to resolve.
        `org_project_links`, which answered the same question one project at a
        time, was retired in 20260804000001; two sources of truth for the payer
        is exactly what this edge exists to eliminate.

        Returns the `_resolve_context` ctx shape (`org_id`, `org_name`,
        `org_member_id`, `role`) **plus `project_id`** (round-4 pin: Task 6's
        deny branch needs it for the lazy owner-check, and neither the ctx's org
        fields nor the contract-derived call path in
        `resolve_billing_org_for_resource` can otherwise recover it) when:

          - the project's artist is owned by an org,
          - that org is ACTIVE (not pending/suspended) and not archived, and
          - the caller holds an ACTIVE seat in it.

        Else None — including when licensing is off (short-circuits before any
        query — no existence oracle, no cost), the project is unlinked, the
        org is pending/suspended/archived, or the caller has no active seat in
        it (rule 4: derivation only ever UPGRADES, never restricts — anything
        short of a live seat in a live org falls through to today's ambient
        behavior, byte-identical).

        Rule 10 (deliberate, stated here so no admin discovers it as a
        surprise): this keys on (project linked to org) AND (caller holds a
        seat in that org) — NOT on (an admin granted THIS member THIS
        project). A seat-holder with ORGANIC access to a linked project spends
        the org's credits on it even though no admin ever assigned them to
        it. That is the owner's consent-by-linking working as designed —
        billing population is a superset of the access-granted population.

        Deliberately NOT computed here: `is_project_owner`. That would tax
        every derivation, including every ALLOW, with an extra read; ownership
        only matters on the DENY branch (Task 6's owner-aware dry-seat wall),
        where a wall is already being built and one more indexed read is off
        the happy path.

        Any exception (a broken read on any of the three tables) returns None
        and logs rather than raising — derivation must NEVER break a request
        (rule 4's fall-through discipline).
        """
        if not licensing_enabled():
            return None
        try:
            project = self._first_row(
                self.supabase.table("projects").select("artist_id").eq("id", project_id).execute()
            )
            if not project or not project.get("artist_id"):
                return None

            artist = self._first_row(
                self.supabase.table("artists").select("team_id").eq("id", project["artist_id"]).execute()
            )
            team_id = (artist or {}).get("team_id")
            if not team_id:
                # No artist team -> personal billing. There is no second source
                # of truth: org_project_links was retired in 20260804000001.
                return None

            # A team the caller has no live seat in resolves to None rather than
            # billing a team they are not part of (rule 4 — derivation only ever
            # UPGRADES, never restricts).
            ctx = self._org_context_for(user_id, team_id)
            if ctx is None:
                return None
            return {**ctx, "project_id": project_id}
        except Exception:
            logging.exception("resolve_billing_org_for_project failed user_id=%s project_id=%s", user_id, project_id)
            return None

    def resolve_billing_org_for_resource(
        self,
        user_id: str,
        *,
        project_id: str | None = None,
        contract_file_ids: list[str] | None = None,
    ) -> dict | None:
        """Resolve the ACTIVE org billing context for a RESOURCE.

        `project_id` (direct — e.g. OneClick, registry) delegates straight to
        `resolve_billing_org_for_project`.

        `contract_file_ids` (a LIST — e.g. Zoe's `contract_ids`) resolves on
        **unanimity** (rule 5, round 2): ONE batched read
        (`project_files.select("id, project_id").in_("id", ids)`) — derivation
        fires only when EVERY id resolves to a row AND all of them share
        EXACTLY ONE project, which is then delegated to
        `resolve_billing_org_for_project`. ANY spread across more than one
        project, any id the batched read doesn't return (deleted / wrong id /
        no access), or any row with a NULL `project_id` → None (ambient) — a
        single mixed-project or unresolvable id must never let one contract's
        project silently win ("first contract wins" is explicitly forbidden
        by rule 5: non-deterministic money attribution).

        No resource at all (`project_id` and `contract_file_ids` both falsy) ->
        None (ambient context, unchanged) — e.g. Zoe's general mode.

        Short-circuits None when licensing is off (no queries). Any exception
        (in the batched read or in the delegated project resolution) returns
        None and logs, never raises — derivation must NEVER break a request
        (rule 4's fall-through discipline). No org identity is ever returned
        for a non-seat-holder (rule 4's no-oracle clause) — this resolver's
        output is internal only, never reflected in a response payload.
        """
        if not licensing_enabled():
            return None
        if project_id is not None:
            return self.resolve_billing_org_for_project(user_id, project_id)
        if not contract_file_ids:
            return None
        try:
            ids = list(contract_file_ids)
            rows = self.supabase.table("project_files").select("id, project_id").in_("id", ids).execute()
            by_id = {row.get("id"): row.get("project_id") for row in (rows.data or [])}

            resolved_project_ids = set()
            for contract_id in ids:
                resolved = by_id.get(contract_id)
                if not resolved:
                    # Unresolvable id or NULL project_id — ambient (rule 5 unanimity).
                    return None
                resolved_project_ids.add(resolved)

            if len(resolved_project_ids) != 1:
                # Spread across >1 project — no deterministic attribution (rule 5).
                return None

            return self.resolve_billing_org_for_project(user_id, resolved_project_ids.pop())
        except Exception:
            logging.exception(
                "resolve_billing_org_for_resource failed user_id=%s contract_file_ids=%s",
                user_id,
                contract_file_ids,
            )
            return None

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
        """Delegates to `orgs.wallets.read_or_create_user_wallet` (moved there
        2026-08-15 so both owner types — user and org — share one
        read-or-create implementation instead of two siblings). Lazy import:
        `orgs.wallets` has no subscriptions imports, so this stays a one-way
        dependency (mirrors this file's other `from orgs...` call sites)."""
        from orgs.wallets import read_or_create_user_wallet

        return read_or_create_user_wallet(self.supabase, user_id)

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
        _REQUIRED = object()

        def pick(field: str, default=_REQUIRED):
            if override is not None and override.get(field) is not None:
                return override[field]
            if default is _REQUIRED:
                return tier_row[field]
            return tier_row.get(field, default)

        caps = Caps(
            max_artists=pick("max_artists"),
            max_projects=pick("max_projects"),
            max_tasks=pick("max_tasks"),
            max_storage_bytes=pick("max_storage_bytes"),
            max_split_sheets_per_month=pick("max_split_sheets_per_month"),
            max_oneclick_runs_per_month=pick("max_oneclick_runs_per_month"),
            monthly_credits=pick("monthly_credits", 0),
            max_works=pick("max_works", -1),
            included_storage_bytes=pick("included_storage_bytes", -1),
            # Team dials (spec 2026-08-15 §1) come from the tier row only —
            # overrides deliberately do NOT carry them (YAGNI: no admin-override
            # UI for team caps yet).
            max_teams=tier_row.get("max_teams", 0),
            max_team_members=tier_row.get("max_team_members", 0),
            team_storage_bytes=tier_row.get("team_storage_bytes", 0),
        )
        features = Features(
            zoe_enabled=pick("zoe_enabled"),
            oneclick_enabled=pick("oneclick_enabled"),
            registry_enabled=pick("registry_enabled"),
            integrations_allowed=list(pick("integrations_allowed") or []),
        )
        if credits_enabled():
            # Credits ARE the gate for the metered AI tools, on every tier: the
            # wallet decides, so the per-tool doors are open and the legacy
            # per-period OneClick counter goes unlimited. Without this, Free is
            # walled twice — "Zoe is a Pro feature" before the user ever sees what
            # Zoe does, which paywalls the thing that sells the plan. The stored
            # flags/caps are NOT mutated (spec §9 rollback): flag off restores
            # legacy gating exactly. Split sheets are NOT credit-priced, so their
            # per-period cap stays; storage and resource counts are a separate
            # dimension and stay as configured.
            features = replace(features, zoe_enabled=True, oneclick_enabled=True, registry_enabled=True)
            caps = replace(caps, max_oneclick_runs_per_month=-1)
        return caps, features, override is not None

    @staticmethod
    def _resolve_monthly_grant(sub: dict, override: dict | None, tier_monthly_credits: int) -> int:
        """Grant precedence (spec §1): explicit admin override > grandfathered
        > tier value.

        `tier_monthly_credits` is the MERGED value from `_merge` (tier row with
        any per-field override already folded in) — branch order below matters
        BECAUSE of that: the override check must run FIRST, or a grandfathered
        grant would beat an explicit admin override instead of losing to it.
        `override` is None both when no `tier_overrides` row exists at all AND
        when one exists but its `monthly_credits` field is NULL (the admin left
        it unset) — either way `_merge` already fell back to the tier default
        for that value, so this helper has nothing extra to add on that path.

        Grandfathering is NOT indefinite (spec §1, owner policy clarification):
        it expires with the subscriber's already-paid billing period, not on a
        calendar date chosen at merge time. It is honored ONLY when BOTH
        `grandfathered_monthly_credits` and `grandfathered_until` are set AND
        `grandfathered_until` is still in the future; a grant with no expiry
        stamped is treated as already expired (defensive — the migration
        backfill always stamps both together, so this should never happen in
        practice) rather than as indefinite. Expiry is silent and read-time:
        nothing fires an event or NULLs the columns at the boundary, a read
        after it simply falls through to the tier value. `grandfathered_until`
        is stamped ONCE, at grant time, and is never extended afterward —
        webhooks NULL both columns together on any tier change (see
        stripe_events.py), they never push the expiry forward.
        """
        if override is not None and override.get("monthly_credits") is not None:
            return override["monthly_credits"]
        gf = sub.get("grandfathered_monthly_credits")
        until = _parse_iso(sub.get("grandfathered_until"))
        if gf is not None and until is not None and until > datetime.now(UTC):
            return gf
        return tier_monthly_credits

    # -----------------------------------------------------------------------
    # can() — single chokepoint with host-wins resolution
    # -----------------------------------------------------------------------

    def can(
        self,
        user_id: str,
        action,
        host_user_id: str | None = None,
        resource_project_id: str | None = None,
        **ctx,
    ):
        """Returns CheckResult(allowed, reason, upgrade_required) for the given action.

        host_user_id semantics:
          - Cap actions (CREATE_*, GENERATE_SPLIT_SHEET): host_user_id is ignored.
            Caps are about how many of YOUR OWN resources you can create.
          - Feature actions (USE_ZOE/ONECLICK/REGISTRY/INTEGRATION): if host provided,
            allow if either acting user OR host has the feature.
          - UPLOAD_BYTES: storage is owner-scoped; check the host's cap if provided,
            else acting user's cap.

        resource_project_id (Licensing Phase C, spec §6/§11, rules 4/9): the
        project the created/uploaded resource lives in. When that project is
        linked to an ACTIVE org where the caller holds an ACTIVE seat, the org's
        caps apply even from personal ambient context — UPGRADE-ONLY (rule 4):
        a derived cap NEVER shrinks a more permissive personal answer. Default
        None → no derivation → byte-identical to pre-Phase-C. Only CREATE_WORK
        (counts) and UPLOAD_BYTES (storage) derive; CREATE_WORK derives its count
        cap unconditionally (enterprise -1 is unlimited under ANY scoping),
        UPLOAD_BYTES derives ONLY when the caller IS the storage-counter owner
        (rule 9 — the collision fix, documented in that branch).
        """
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
            cap = ent.caps.max_works
            # Caps derivation (Licensing Phase C, rule 9): a work created in a
            # project linked to an org where the caller holds an ACTIVE seat uses
            # the org's caps. COUNTS ARE EXEMPT from the storage owner-scoping
            # precedence (rule 9) precisely because enterprise makes them -1
            # (unlimited) — an unlimited cap is safe to substitute under ANY
            # scoping, so no owner check is needed and no one else's counter is
            # ever consulted; THAT is the load-bearing reason the messy
            # owner-scoping interaction only matters for storage (UPLOAD_BYTES).
            # Upgrade-only (rule 4): take the MORE PERMISSIVE of the personal and
            # enterprise caps (enterprise -1 wins over any finite personal cap; a
            # personal -1 is already unlimited). Derivation is skipped entirely
            # when resource_project_id is None or licensing is off (the resolver
            # short-circuits) → byte-identical to today.
            if resource_project_id is not None and self.resolve_billing_org_for_project(user_id, resource_project_id):
                cap = -1  # enterprise counts are unlimited
            return self._check_count_cap(ctx.get("current_count", 0), cap, "registered works")

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

            # Storage-caps derivation (Licensing Phase C, rule 9 — THE collision
            # fix). Storage accrues to the OWNER's counter (recalc_user_storage),
            # so this check is owner-scoped: the host's cap against the host's
            # counter. Deriving an enterprise PER-SEAT cap and checking it against
            # SOMEONE ELSE'S counter is incoherent — so caps derivation applies
            # ONLY when the caller IS the storage-counter owner (host_user_id is
            # None or equals the caller). A seat-holding COLLABORATOR uploading to
            # another owner's linked project keeps today's host-scoped check
            # UNTOUCHED — no derivation attempt, no artist-ownership query at all.
            # (Counts, CREATE_WORK, are exempt from this precedence because
            # enterprise makes them -1, unlimited under any scoping; storage is
            # finite, which is the whole reason the collision only bites here.)
            # Derivation now only ever resolves a TEAM-OWNED artist (the link
            # table is gone), so the bytes are always on the ORG's counter —
            # _bump_storage put them there. Measuring the caller's personal
            # counter would compare the wrong number against the wrong cap.
            derived_storage = False
            team_deny_reason = None
            if (host_user_id is None or host_user_id == user_id) and resource_project_id is not None:
                org_ctx = self.resolve_billing_org_for_project(user_id, resource_project_id)
                if org_ctx is not None:
                    derived_storage = True
                    used, cap, team_deny_reason = self._team_storage(org_ctx["org_id"])
                    projected = used + size

            if cap != -1 and projected > cap:
                # An org storage wall — whether from ambient org context (rule 13)
                # or a DERIVED team (rule 9) — points at SUPPORT for an enterprise
                # seat (the admin has no storage lever there) or at the org's own
                # admin for a self-serve pool (Task 12: they DO have a lever —
                # buy more, or free up space); _team_storage picks the copy.
                if derived_storage:
                    return deny(team_deny_reason)
                if getattr(owner_ent, "managed_by_org", None) is not None:
                    return deny("Your organization seat's storage is full. Contact support to discuss options.")
                return deny(
                    f"Upload would exceed the project owner's storage limit "
                    f"({owner_ent.usage.total_storage_bytes} + {size} > {cap} bytes)."
                )
            # Credits model: paid tiers have unlimited hard cap (-1) but a finite
            # included allowance, and uploads BLOCK at it — there is no storage
            # pay-per-use (that path billed nobody, so it was cut; storage is a
            # hard cap on every tier). Existing files are never touched. SKIP this
            # personal allowance gate when storage was DERIVED to an org seat
            # (rule 9): a personal per-plan included-allowance prompt is
            # incoherent for org-billed storage. What bounds the upload then
            # depends on the caller's PERSONAL cap (rule 4, upgrade-only max):
            # a finite personal cap (free tier) upgrades to the finite seat cap,
            # which the hard-cap check above enforces; an unlimited personal cap
            # (paid tiers) STAYS -1 — derivation never shrinks an entitlement —
            # so NO byte ceiling fires for them here. Deliberate, and not a
            # revenue leak: seat-holders are already exempt from personal
            # storage billing (rule 13's sweep grandfather), and it mirrors
            # ambient org context (where included == max == seat storage, so
            # this gate never independently fires).
            if not derived_storage and credits_enabled() and owner_ent.tier in self.PAID_TIERS:
                included = owner_ent.caps.included_storage_bytes
                if included != -1 and projected > included:
                    return deny(
                        "This upload goes past the storage included in your plan. "
                        "Upgrade for more storage, or remove some files to free up space."
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
        if cap == -1 or current < cap:
            return CheckResult(allowed=True, reason=None, upgrade_required=False)
        return CheckResult(
            allowed=False,
            reason=f"You've reached your limit of {cap} {label}.",
            upgrade_required=True,
        )

    def _team_storage(self, org_id: str) -> tuple[int, int, str]:
        """(bytes used, cap, member-facing deny reason) for a team.

        Task 12: the org row's `kind` branches this — self_serve delegates to
        the owner's pool (orgs/storage_guard.upload_allowed's math, inlined
        here as a used/cap pair so the existing cap check above stays
        untouched); anything else (enterprise) keeps the ORIGINAL per-seat
        calc below byte-identical, `kind`/`covered_by` just ride along on the
        same row read — no new query.

        Enterprise: used comes from the cached `organizations.storage_bytes`,
        which the storage triggers in 20260803000003 keep live — an aggregate
        over two file tables is far too expensive to run on the gate for every
        upload. Cap is the per-seat allowance times the number of ACTIVE
        seats. The env value is per person; charging a ten-person team one
        seat's worth of space would make the feature unusable the day it
        shipped.

        ponytail: cap moves when the roster does, so an org that loses members
        can sit over its cap with files already uploaded (existing files are
        never touched — only further uploads block). An explicit
        organizations.storage_cap_bytes set alongside the credit dispersal is
        the upgrade path if that becomes a real complaint.
        """
        org = self._first_row(
            self.supabase.table("organizations").select("storage_bytes, kind, covered_by").eq("id", org_id).execute()
        )
        if (org or {}).get("kind") == "self_serve":
            from orgs.storage_guard import TEAM_STORAGE_FULL_MSG, pool_state

            used, pool, is_pro_like = pool_state(self.supabase, org.get("covered_by"))
            cap = -1 if is_pro_like else pool
            return used, cap, TEAM_STORAGE_FULL_MSG

        seats = (
            self.supabase.table("org_members")
            .select("id", count="exact")
            .eq("org_id", org_id)
            .eq("status", "active")
            .execute()
            .count
        ) or 0
        used = (org or {}).get("storage_bytes") or 0
        return (
            used,
            _enterprise_seat_storage_bytes() * max(seats, 1),
            "Your team's storage is full. Contact support to discuss more space.",
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

    PAID_TIERS = ("basic", "pro")

    def check_credits(
        self,
        user_id: str,
        action: str,
        *,
        resource_project_id: str | None = None,
        resource_contract_ids: list[str] | None = None,
    ):
        """Can this user run `action` right now, and at what ESTIMATED price?

        `credit_prices` is a pre-flight ESTIMATE, not the charge: the real cost
        of an AI action is unknowable until its tokens are burned, so this
        reserves a plausible amount and `debit_for_action` bills what was
        actually metered. A run that measures above its estimate is allowed to
        overshoot (the debit_credits RPC absorbs it as bundle drift, exactly
        like a raced concurrent debit) — the estimate exists to keep a broke
        user from starting expensive work, not to cap the bill.

        Order: bypass/admin → resource-derived org (Phase C) → ambient org
        (Phase B) → estimate lookup → wallet balance → opt-in overage → wall.
        Degraded policy (spec §12): paid fails open (uncharged), free fails
        closed. NOTE: re-reads tables per call like the rest of this service
        (no-cache philosophy) — optimization deferred deliberately.

        `resource_project_id` / `resource_contract_ids` (Licensing Phase C) are
        the resource the action operates on; a resource in a project linked to an
        ACTIVE org where the caller holds an ACTIVE seat bills that org's seat,
        winning over ambient context (rule 5). Both default None → no resource →
        the pre-Phase-C ambient/personal path, byte-identical.
        """
        if not credits_enabled():
            return CreditCheckResult(allowed=True, price=0)
        if _bypass_paywalls_enabled():
            # Ops escape hatch only. ADMINS ARE METERED like everyone else
            # (owner decision, 2026-08-15): admin ledger rows are what make the
            # credit system testable by its own operators, and admins can
            # self-grant when they run low. The old is_admin/is_db_admin
            # short-circuit ("no ledger rows for admins") is deliberately gone —
            # do not reintroduce it here; BYPASS_PAYWALLS covers the blanket-off
            # case explicitly.
            return CreditCheckResult(allowed=True, price=0)

        # Resource-derived org billing (Licensing Phase C, spec §6/§11, rules 4-6).
        # RESOLUTION ORDER: derived-resource org → ambient context → personal. A
        # resource (OneClick project, Zoe contract_ids, registry contract) living
        # in a project linked to an ACTIVE org where the caller holds an ACTIVE
        # seat WINS over ambient context (rule 5: "auto-bills that org regardless
        # of ambient context") and routes to `_check_credits_org` VERBATIM — the
        # same seat wallet, no overage, managedByOrg 402, and the seat wallet_id
        # threaded into the grant so the debit follows the check (rule 6). The
        # derived ctx additionally carries `project_id`, which the deny branch
        # needs for the owner-aware dry-seat wall (rule 11). Any miss (no resource,
        # unlinked project, no seat, pending/suspended/archived org, licensing off,
        # or a mixed-project contract list) falls through to the ambient/personal
        # flow below, byte-identical (rule 4). The resolver swallows its own
        # errors and returns None, so a derivation fault can never break a request.
        derived_ctx = self.resolve_billing_org_for_resource(
            user_id,
            project_id=resource_project_id,
            contract_file_ids=resource_contract_ids,
        )
        if derived_ctx is not None:
            return self._check_credits_org(user_id, action, derived_ctx)

        # Billing context (Licensing Phase B, spec §5, rules 8/9). An ACTIVE org
        # seat pays from the SEAT wallet ONLY — no personal subscription read, no
        # overage, no personal fallback. `_resolve_context` returns None when
        # licensing is off / no preference / a dead reference (so the personal
        # path below is byte-identical), and a pending-org marker (pending=True)
        # which confers nothing yet and therefore also routes to the personal
        # path. Only a live, ACTIVE org seat takes the seat branch.
        ctx = self._resolve_context(user_id)
        if ctx is not None and not ctx.get("pending"):
            return self._check_credits_org(user_id, action, ctx)

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
            price = self._price_or_wall(action, managed_by_org=False)
            if isinstance(price, CreditCheckResult):
                return price
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
            # Precedence (Task 1, spec §1): explicit override > grandfathered >
            # tier value — must match get_for_user and the sweep exactly.
            monthly_grant = self._resolve_monthly_grant(sub, override, caps.monthly_credits)
            wallet = self._read_or_create_wallet(user_id)
            wallet = self._maybe_rollover_wallet(wallet, monthly_grant)
            balance = wallet.get("bundle_balance", 0) + wallet.get("reserve_balance", 0)
            reset_date = _parse_iso(wallet.get("period_end"))
            # The wallet the check passes against — threaded into the grant so the
            # debit targets THIS wallet (rule 9), never a re-resolved one.
            wallet_id = wallet.get("id")

            if balance >= price:
                return CreditCheckResult(allowed=True, price=price, reset_date=reset_date, wallet_id=wallet_id)

            if tier in self.PAID_TIERS:
                if sub.get("status") == "past_due":
                    # A failing renewal must not accrue MORE debt via
                    # pay-per-use (Stripe retries can run for weeks before
                    # subscription.deleted fires). Balance spending above
                    # stays allowed; only overage pauses.
                    return CreditCheckResult(
                        allowed=False,
                        price=price,
                        reset_date=reset_date,
                        reason=(
                            "Your last payment didn't go through, so pay-per-use is paused. "
                            "Update your payment method in Billing to keep going."
                        ),
                    )
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
                    return CreditCheckResult(
                        allowed=True, price=price, use_overage=True, reset_date=reset_date, wallet_id=wallet_id
                    )
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

    def _price_or_wall(self, action: str, *, managed_by_org: bool) -> int | CreditCheckResult:
        """The seeded base price for `action`, or the config-error WALL when it
        isn't seeded — a missing key is a config error, not an outage, so every
        tier is denied with honest copy and the gap surfaces in the first QA run
        instead of leaking COGS. A price READ failure propagates: callers run
        this inside their degraded try, so an outage takes the degraded policy
        (paid open, free/org per spec §12), never a wall."""
        prices = self._get_credit_prices()
        if action not in prices:
            logging.error("check_credits: no credit price seeded for action %r", action)
            return CreditCheckResult(
                allowed=False,
                price=0,
                managed_by_org=managed_by_org,
                reason="This action isn't set up for credits yet. Please contact support.",
            )
        return prices[action]

    def _check_credits_org(self, user_id: str, action: str, ctx: dict):
        """check_credits for an ACTIVE org billing context (Licensing Phase B, rule 8).

        Pays from the SEAT wallet ONLY: no personal subscription read, no overage,
        no personal fallback, no upgrade path (the seat wall points the member at
        their admin, wired in enforcement.gated_credits). Seat wallets are
        NULL-period, so `reset_date` is always None.

        TWO ceilings apply, and they wall differently because the fix differs:
        the member's monthly CAP (they ask their admin for a raise) and the ORG
        POOL balance (the admin has to buy credits). `cap_reached` distinguishes
        them for the 402.

        Degraded policy mirrors the paid personal tier — a READ EXCEPTION (prices
        or the pool) fails OPEN uncharged (spec §12). A MISSING pool ROW is NOT an
        exception: the helper lazy-creates it at zero, which correctly walls (402)
        rather than failing open (rule 8's carve-out).
        """
        from orgs.wallets import read_or_create_org_wallet

        try:
            price = self._price_or_wall(action, managed_by_org=True)
            if isinstance(price, CreditCheckResult):
                return price
            if price <= 0:
                # Retuned-to-0 (or drifted-negative) prices are free — never wall.
                return CreditCheckResult(allowed=True, price=0, managed_by_org=True, reset_date=None)

            # Lazy-create at zero (rule 8): a missing row is a legitimate 402, not
            # an outage. Only a genuine READ EXCEPTION escapes into the except.
            pool = read_or_create_org_wallet(self.supabase, ctx["org_id"])
            balance = pool.get("bundle_balance", 0) + pool.get("reserve_balance", 0)
            cap, cap_used = self._member_cap(ctx["org_id"], ctx["org_member_id"])

            # The CAP is checked first: it is the member's own ceiling, and the
            # remedy (ask for a raise) is different from a dry pool (the admin
            # buys credits). This is the advisory pre-check — debit_credits holds
            # the authoritative counter under the wallet lock.
            cap_reached = cap is not None and cap_used + price > cap
            if not cap_reached and balance >= price:
                return CreditCheckResult(
                    allowed=True,
                    price=price,
                    wallet_id=pool.get("id"),
                    org_member_id=ctx["org_member_id"],
                    managed_by_org=True,
                )

            # NO overage, NO personal fallback, NO upgrade (rule 8).
            if cap_reached:
                reason = (
                    f"You've reached your monthly credit limit ({cap:,} credits). "
                    "Ask your organization's admin to raise it."
                )
            else:
                reason = "Your organization is out of credits. Ask your admin to top up."
            # No "unlink this project" CTA: a project belongs to the org because
            # its ARTIST does, and moving an artist back out is not self-serve.
            return CreditCheckResult(
                allowed=False,
                price=price,
                managed_by_org=True,
                cap_reached=cap_reached,
                reason=reason,
            )
        except Exception:
            # Org-path READ ERROR → fail open uncharged, like the paid personal
            # tier (spec §12). price=0 → the grant is disabled, so the debit no-ops.
            logging.exception("check_credits(org) degraded user=%s action=%s", user_id, action)
            return CreditCheckResult(allowed=True, price=0, managed_by_org=True, degraded=True)

    def debit_for_action(self, user_id: str, grant) -> None:
        """Debit a CreditGrant after the action succeeded. Best-effort, never raises.

        The amount is max(BASE, METERED). `grant.price` is the per-action BASE
        RATE from credit_prices — the published price of the deliverable, and a
        floor. `measured` is what the TrackedOpenAI proxy saw this request (real
        OpenAI cost x CREDIT_MARKUP / price-per-credit) and only wins when the
        run cost more than the base already covers. Unmeasurable spend (no
        tracked scope, or a model missing from MODEL_RATES) charges the base.

        Two consequences worth knowing: an action that made no LLM call — a cache
        hit — measures 0 and still pays the base, because the user received the
        same deliverable (cache-hit debits are deduped per period by a
        deterministic request_id, see enforcement.gated_credits' dedupe_key); and
        a heavy run can measure ABOVE the reserved estimate, which the
        debit_credits RPC absorbs as bundle drift exactly like a raced debit.

        NOTE: overage_debit rows are billed OFF the request path — daily
        sweep creates pending InvoiceItems; invoice.created attaches
        stragglers to the draft renewal invoice. Never call Stripe here.
        """
        from utils.llm.tracking import credits_for_llm_usage, llm_usage_snapshot

        if grant is None or not grant.enabled or grant.price <= 0:
            return
        measured = credits_for_llm_usage()
        # BASE RATE (spec 2026-08-17 §2): grant.price is the per-action base, not
        # an estimate. It is a FLOOR — the metered value only decides the charge
        # when the run cost more than the base already covers. A 30-credit base
        # covers ~$0.20 of COGS, roughly 20x a median run, so the tail fires only
        # on a pathological input (e.g. a derive over ~18 contracts).
        amount = grant.price if measured is None else max(grant.price, measured)
        if amount <= 0:
            return
        try:
            wallet_id = getattr(grant, "wallet_id", None)
            if wallet_id is None:
                # Legacy / free grant (no targeted wallet): resolve the caller's
                # PERSONAL wallet — today's behavior. A wallet_id-bearing grant
                # (rule 9) SKIPS this: the debit targets the exact wallet the check
                # passed against, resolving nothing, so a billing-context switch
                # between check and debit can never relocate the charge.
                wallet = self._read_or_create_wallet(user_id)
                wallet_id = wallet.get("id")
            if wallet_id is None:
                return
            # p_member_id is set ONLY for org-pool spend: it makes the RPC move
            # the member's cap counter in the same transaction as the debit, and
            # tags the ledger row with who spent it. None on personal wallets, so
            # the org_members read inside the RPC never happens for them.
            # The token counts and real cost behind this charge ride on the
            # ledger row: with a metered price, "why 9 credits?" is otherwise
            # unanswerable from the ledger alone. `estimated` records what the
            # gate reserved, so over/under-runs are measurable.
            usage = llm_usage_snapshot() or {}
            # `metered` answers "was this charge cost-driven?", NOT "was cost
            # readable?". Under base rates `measured is not None` is true on
            # nearly every LLM-touching row while the amount is still the base,
            # so the old meaning would make it impossible to find the rows where
            # metering actually decided the price. Keep both facts, separately.
            metadata = {
                "estimated": grant.price,
                "base": grant.price,
                "measurable": measured is not None,
                "metered_credits": measured,
                "metered": measured is not None and measured > grant.price,
            }
            if usage:
                metadata.update(
                    {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "llm_calls": usage.get("calls", 0),
                        "cost_usd": round(usage.get("cost_usd", 0.0), 6),
                    }
                )
            payload = {
                "p_wallet_id": wallet_id,
                "p_amount": amount,
                "p_action": grant.action,
                "p_request_id": grant.request_id,
                "p_kind": grant.kind,
                "p_metadata": metadata,
            }
            member_id = getattr(grant, "org_member_id", None)
            if member_id is not None:
                payload["p_member_id"] = member_id
            res = self.supabase.rpc("debit_credits", payload).execute()
            # A cap overshoot means the pre-check in check_credits was raced by
            # this member's own concurrent action. The work is already done and
            # paid for, so the debit stands — but it is worth knowing about.
            data = res.data if isinstance(getattr(res, "data", None), dict) else {}
            if data.get("cap_exceeded"):
                logging.warning(
                    "debit exceeded member cap (raced pre-check) user=%s member=%s action=%s cap=%s used=%s",
                    user_id,
                    member_id,
                    grant.action,
                    data.get("cap"),
                    data.get("cap_used"),
                )
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

    def _aggregate_tool_usage(
        self,
        wallet_id: str | None,
        prices: dict[str, int],
        *,
        since: str | None = None,
        org_member_id: str | None = None,
    ) -> list[dict]:
        """Shared per-action ledger aggregation for the usage view (personal AND
        org-context paths, Task 7). `since=None` means ALL-TIME.

        `org_member_id` scopes the scan to ONE member's rows on a shared org
        pool, matching on the `metadata.org_member_id` that debit_credits
        stamps — the same attribution key the admin console groups by. Without
        it, a pool wallet aggregates every member's spend. Filtered in Python
        rather than as a PostgREST JSONB filter to keep one code path with the
        personal case; the scan is already bounded by wallet + period.
        """
        agg: dict[str, dict] = {}
        if wallet_id is not None:
            query = (
                self.supabase.table("credit_ledger")
                .select("action, delta, kind, metadata")
                .eq("wallet_id", wallet_id)
                .in_("kind", ["debit", "overage_debit"])
            )
            if since is not None:
                query = query.gte("created_at", since)
            rows = query.execute()
            for r in rows.data or []:
                action = r.get("action")
                if not action:
                    continue
                if org_member_id is not None and (r.get("metadata") or {}).get("org_member_id") != org_member_id:
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
        return tools

    def get_credit_usage(self, user_id: str) -> dict:
        """Per-tool credit spend for the Account & Billing usage view.

        Returns {"enabled": False} when credits are off. CONTEXT-AWARE (Licensing
        Phase B, Task 7): an ACTIVE org billing context (not pending) reflects the
        SEAT wallet's ledger, ALL-TIME — the personal wallet/subscription/tier rows
        are NEVER read in that branch (mirrors get_for_user's rule 11). Pending-org,
        personal, and licensing-off contexts all fall through to today's PERSONAL
        path below, unmodified — byte-identical regression.
        """
        if not credits_enabled():
            return {"enabled": False}

        ctx = self._resolve_context(user_id)
        if ctx is not None and not ctx.get("pending"):
            return self._get_credit_usage_org(ctx)

        sub = self._read_or_create_subscription(user_id)
        tier = sub.get("tier", "free")
        tier_row = self._read_tier_entitlements(tier)
        if tier_row is None:
            raise RuntimeError(f"Missing tier_entitlements row for tier={tier!r}")
        override = self._read_override(user_id)
        caps, _, _ = self._merge(tier_row, override)
        # Precedence (Task 1, spec §1): explicit override > grandfathered >
        # tier value — must match get_for_user, check_credits, and the sweep.
        monthly_grant = self._resolve_monthly_grant(sub, override, caps.monthly_credits)

        wallet = self._read_or_create_wallet(user_id)
        wallet = self._maybe_rollover_wallet(wallet, monthly_grant)
        period_start = wallet.get("period_start")
        prices = self._get_credit_prices()

        tools = self._aggregate_tool_usage(
            wallet.get("id") if period_start is not None else None, prices, since=period_start
        )

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

    def _member_cap(self, org_id: str, org_member_id: str) -> tuple[int | None, int]:
        """(effective cap, credits used this cap period) for one member.

        The cap falls through org_members.monthly_cap -> organizations.
        default_member_cap -> None (uncapped, pool is the only limit). Every org
        carries a default (2,000, seeded by 20260814000001), so in practice a
        new member IS capped and the terminal None is reached only when an admin
        deliberately clears the org default.

        A NEGATIVE stored cap is the "explicitly unlimited" sentinel (-1, the
        repo's tier_entitlements idiom) and normalizes to None HERE, so nothing
        above this function ever sees it: "None means no ceiling" stays the only
        rule in the Python layer, and the sentinel is purely how the DB spells
        it. debit_credits does the same normalization under the pool lock.

        `cap_used` is the counter debit_credits maintains; it reads as 0 once its
        period has lapsed, because the RPC rolls it lazily on the next debit
        rather than needing a sweep to reset every member.

        Never raises: an unreadable membership row degrades to uncapped, matching
        the fail-open posture of every other read on this path.
        """
        try:
            member = self._first_row(
                self.supabase.table("org_members")
                .select("monthly_cap, cap_used, cap_period_end")
                .eq("id", org_member_id)
                .execute()
            )
            if member is None:
                return None, 0
            cap = member.get("monthly_cap")
            if cap is None:
                org = self._first_row(
                    self.supabase.table("organizations").select("default_member_cap").eq("id", org_id).execute()
                )
                cap = (org or {}).get("default_member_cap")
            if cap is not None and cap < 0:
                cap = None  # -1 sentinel: explicitly unlimited
            period_end = _parse_iso(member.get("cap_period_end"))
            lapsed = period_end is None or period_end <= datetime.now(UTC)
            return cap, 0 if lapsed else (member.get("cap_used") or 0)
        except Exception:
            logging.exception("_member_cap read failed org=%s member=%s", org_id, org_member_id)
            return None, 0

    def _get_credit_usage_org(self, ctx: dict) -> dict:
        """Org-context credit usage for the CALLER: their own per-tool spend out
        of the pool, plus their cap and what they've spent against it.

        The pool's period is the dispersal period, so flooring the ledger scan on
        period_start gives "this month's spend" — the same window the member's cap
        counter uses. `monthlyGrant` carries the member's CAP: nothing is
        allocated to them, so the number that means something is their ceiling.

        SCOPED TO THE CALLER, admins included. This backs `/me/credits/usage`,
        which is a view of MY usage — it used to aggregate the pool's whole
        ledger and hand every member the org's total spend, which both leaked
        other members' activity and mislabelled it ("Your usage"). Org-wide
        rollups live on the admin-only console (`GET /orgs/{id}/usage`).

        Pool balances follow `_pool_visible_to`: admins see them, members get
        None (see CreditsInfo for why None and not 0).
        """
        from orgs.wallets import read_or_create_org_wallet

        pool = read_or_create_org_wallet(self.supabase, ctx["org_id"])
        prices = self._get_credit_prices()
        period_start = pool.get("period_start")
        tools = self._aggregate_tool_usage(
            pool.get("id"), prices, since=period_start, org_member_id=ctx["org_member_id"]
        )
        cap, cap_used = self._member_cap(ctx["org_id"], ctx["org_member_id"])

        show_pool = _pool_visible_to(ctx.get("role"))
        bundle = pool.get("bundle_balance", 0) if show_pool else None
        reserve = pool.get("reserve_balance", 0) if show_pool else None
        return {
            "enabled": True,
            "managedByOrg": {
                "orgId": ctx["org_id"],
                "orgName": ctx["org_name"],
                "role": ctx["role"],
                "kind": ctx.get("kind"),
            },
            "periodStart": period_start,
            "periodEnd": pool.get("period_end"),
            "monthlyGrant": cap or 0,
            "memberCap": cap,
            "memberCapUsed": cap_used,
            "bundleBalance": bundle,
            "reserveBalance": reserve,
            "balance": (bundle + reserve) if show_pool else None,
            "overageThisPeriod": 0,
            "tools": tools,
        }

    def get_credit_usage_safe(self, user_id: str) -> dict:
        """Never-raises wrapper for the endpoint."""
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

        KNOWN DIVERGENCE — PERSONAL-ONLY BY DESIGN (Licensing Phase B, spec §5):
        unlike `get_for_user`, this path is NOT billing-context-aware. It feeds
        host-wins CAP checks across many owners, not a billing decision, so it
        always resolves each user's PERSONAL tier — a cap checked through the bulk
        path for a user who happens to be in ACTIVE org context gets the PERSONAL
        answer, not the enterprise one. Accepted for v1 (the caps/context mismatch
        the spec calls out); folded into the resource-derivation follow-up. Do NOT
        teach this method to read billing context — it would add a per-user
        profiles+org read to every list endpoint for a billing concern the caps
        path doesn't own.
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
                    integrations_allowed=["google_drive", "dropbox"],
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
