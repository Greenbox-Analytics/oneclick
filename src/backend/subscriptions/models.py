"""Typed contracts for the subscriptions / entitlements module.

Used by:
- subscriptions.service.EntitlementsService — produces Entitlements
- subscriptions.router — serializes Entitlements to JSON for the frontend
- All future gated endpoints (Sub-project 3) — call EntitlementsService.can()
  with an Action and receive a CheckResult.

Convention: any `max_*` field set to -1 means "unlimited".
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from subscriptions.ai_pricing import overage_usd_per_credit


@dataclass
class Caps:
    max_artists: int
    max_projects: int
    max_tasks: int
    max_storage_bytes: int
    max_split_sheets_per_month: int
    max_oneclick_runs_per_month: int = 1
    monthly_credits: int = 0
    max_works: int = -1
    included_storage_bytes: int = -1
    # Self-serve teams (spec 2026-08-15 §1): slots this tier may COVER, seats
    # per team EXCLUDING the covering owner, and the per-OWNER storage pool
    # across all owned teams. 0 on free; never -1 (these are not count caps).
    max_teams: int = 0
    max_team_members: int = 0
    team_storage_bytes: int = 0


@dataclass
class Features:
    zoe_enabled: bool
    oneclick_enabled: bool
    registry_enabled: bool
    integrations_allowed: list[str]


@dataclass
class Usage:
    total_storage_bytes: int
    split_sheets_this_period: int
    zoe_queries_this_period: int
    oneclick_runs_this_period: int
    period_end: datetime


@dataclass
class ManagedByOrg:
    """Marker on an Entitlements resolved in ORG billing context (Licensing
    Phase B, spec §5). Its mere presence means this entitlements object came
    from an active org seat — used by can()'s UPLOAD_BYTES branch to pick the
    support-pointing storage-wall copy (rule 13), and surfaced in to_dict()
    under credits.managedByOrg (when the credits block exists) AND under the
    top-level billingContext field (Task 3 follow-up — present regardless of
    CREDITS_ENABLED, so the frontend can identify org context even when there
    is no credits block to read managedByOrg off of)."""

    org_id: str
    org_name: str
    role: str
    # Org's kind ("self_serve" | "enterprise"); may be None for a pre-migration
    # row. Drives the frontend's team-vs-organization noun (orgNoun in tiers.ts).
    kind: str | None = None


@dataclass
class Entitlements:
    user_id: str
    tier: Literal["free", "basic", "pro"]
    status: Literal["active", "canceled", "past_due", "trialing"]
    caps: Caps
    features: Features
    usage: Usage
    has_overrides: bool
    degraded: bool = False
    credits: "CreditsInfo | None" = None
    # Stripe billing fields — None for free / admin-override-only users
    stripe_subscription_id: str | None = None
    stripe_price_id: str | None = None
    current_period_end: datetime | None = None
    cancel_at_period_end: bool = False
    # Licensing Phase B (spec §5) — both default None so the personal / licensing-off
    # to_dict() payload is byte-identical to pre-licensing. `managed_by_org` is set
    # ONLY in active-org context — surfaced as credits.managedByOrg (when the
    # credits block exists) AND as the top-level billingContext.type=="org" field
    # (Task 3 follow-up, present regardless of CREDITS_ENABLED); `available_contexts`
    # is a list ONLY when LICENSING_ENABLED is on (surfaced as the top-level
    # availableContexts key, and gates billingContext's presence too — same flag),
    # None otherwise so both keys are omitted entirely.
    managed_by_org: "ManagedByOrg | None" = None
    available_contexts: list[dict] | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict using camelCase keys for the frontend."""
        # Derive plan_period from price_id suffix: prices ending in "_annual" → "annual"
        plan_period: str | None = None
        if self.stripe_price_id:
            plan_period = "annual" if "annual" in self.stripe_price_id else "monthly"

        credits_dict = None
        if self.credits:
            credits_dict = {
                "balance": self.credits.balance,
                "bundleBalance": self.credits.bundle_balance,
                "reserveBalance": self.credits.reserve_balance,
                "monthlyGrant": self.credits.monthly_grant,
                "overageThisPeriod": self.credits.overage_this_period,
                "overageEnabled": self.credits.overage_enabled,
                # Shipped so the UI never hardcodes the rate it quotes (it must
                # match what overage_billing.py actually charges).
                "overageUsdPerCredit": overage_usd_per_credit(),
                "overageCapCredits": self.credits.overage_cap_credits,
                # Org context: the member's ceiling on the shared pool and their
                # spend against it. Absent (None/0) in personal context.
                "memberCap": self.credits.member_cap,
                "memberCapUsed": self.credits.member_cap_used,
                "periodEnd": self.credits.period_end.isoformat() if self.credits.period_end else None,
                # Hand-built, so a new credit_prices row does NOT reach the
                # client until it is added here. Keep in step with CreditAction
                # and with CreditPrices in src/hooks/useEntitlements.ts.
                "prices": {
                    "zoeMessage": self.credits.prices.get("zoe_message", 0),
                    "oneclickRun": self.credits.prices.get("oneclick_run", 0),
                    "registryParse": self.credits.prices.get("registry_parse", 0),
                    "splitSheet": self.credits.prices.get("split_sheet", 0),
                },
            }
            # Org context only — the KEY itself is absent in personal context, so
            # the personal payload stays byte-identical to pre-licensing.
            if self.managed_by_org is not None:
                credits_dict["managedByOrg"] = {
                    "orgId": self.managed_by_org.org_id,
                    "orgName": self.managed_by_org.org_name,
                    "role": self.managed_by_org.role,
                    "kind": self.managed_by_org.kind,
                }

        result = {
            "tier": self.tier,
            "status": self.status,
            "caps": {
                "maxArtists": self.caps.max_artists,
                "maxProjects": self.caps.max_projects,
                "maxTasks": self.caps.max_tasks,
                "maxStorageBytes": self.caps.max_storage_bytes,
                "maxSplitSheetsPerMonth": self.caps.max_split_sheets_per_month,
                "maxOneclickRunsPerMonth": self.caps.max_oneclick_runs_per_month,
                "maxWorks": self.caps.max_works,
                "includedStorageBytes": self.caps.included_storage_bytes,
                "monthlyCredits": self.caps.monthly_credits,
                "maxTeams": self.caps.max_teams,
                "maxTeamMembers": self.caps.max_team_members,
                "teamStorageBytes": self.caps.team_storage_bytes,
            },
            "features": {
                "zoeEnabled": self.features.zoe_enabled,
                "oneclickEnabled": self.features.oneclick_enabled,
                "registryEnabled": self.features.registry_enabled,
                "integrationsAllowed": list(self.features.integrations_allowed),
            },
            "usage": {
                "totalStorageBytes": self.usage.total_storage_bytes,
                "splitSheetsThisPeriod": self.usage.split_sheets_this_period,
                "zoeQueriesThisPeriod": self.usage.zoe_queries_this_period,
                "oneclickRunsThisPeriod": self.usage.oneclick_runs_this_period,
                "periodEnd": self.usage.period_end.isoformat(),
            },
            "credits": credits_dict,
            "hasOverrides": self.has_overrides,
            "degraded": self.degraded,
            "subscription": {
                "stripeSubscriptionId": self.stripe_subscription_id,
                "stripePriceId": self.stripe_price_id,
                "currentPeriodEnd": self.current_period_end.isoformat() if self.current_period_end else None,
                "cancelAtPeriodEnd": self.cancel_at_period_end,
                "planPeriod": plan_period,
            },
        }
        # LICENSING_ENABLED gate lives in the service (get_for_user sets
        # available_contexts to a list only when licensing is on); here we simply
        # omit the key entirely when it's None so the pre-licensing payload is
        # byte-identical.
        if self.available_contexts is not None:
            result["availableContexts"] = self.available_contexts
            # billingContext (Task 3 follow-up): a stable identity signal that is
            # present whenever licensing is on, REGARDLESS of CREDITS_ENABLED —
            # unlike credits.managedByOrg, which only exists when the credits
            # block itself is built. Frontend org/personal rendering should key
            # off this field (falling back to credits.managedByOrg for callers
            # written before this field existed).
            if self.managed_by_org is not None:
                result["billingContext"] = {
                    "type": "org",
                    "orgId": self.managed_by_org.org_id,
                    "orgName": self.managed_by_org.org_name,
                    "role": self.managed_by_org.role,
                    "kind": self.managed_by_org.kind,
                }
            else:
                result["billingContext"] = {"type": "personal"}
        return result


class Action(StrEnum):
    """Every action the entitlements layer can gate.

    Per-action context kwargs (passed to EntitlementsService.can(...)):
      CREATE_ARTIST / CREATE_PROJECT / CREATE_TASK:
          current_count: int — count of resources YOU own (not shared with you).
                              See spec §2 'Resource counting'.
      UPLOAD_BYTES:
          size: int — bytes about to be uploaded.
          host_user_id: if uploading into a project owned by someone else,
                        pass that owner's user_id. Storage is then checked
                        against the OWNER's cap, not yours (storage is owner-scoped).
      USE_INTEGRATION:
          name: str — integration identifier (e.g. 'google_drive').
          host_user_id: if acting on a host's project, pass the host's user_id.
                        Allowed if either you or the host has the integration.
      USE_ZOE / USE_ONECLICK / USE_REGISTRY:
          host_user_id: if acting on a host's project, pass the host's user_id
                        (host-wins: allowed if either has the feature).
      GENERATE_SPLIT_SHEET:
          no context — uses your own internal counter.
    """

    CREATE_ARTIST = "create_artist"
    CREATE_PROJECT = "create_project"
    CREATE_TASK = "create_task"
    CREATE_WORK = "create_work"
    UPLOAD_BYTES = "upload_bytes"
    GENERATE_SPLIT_SHEET = "generate_split_sheet"
    USE_ZOE = "use_zoe"
    USE_ONECLICK = "use_oneclick"
    USE_REGISTRY = "use_registry"
    USE_INTEGRATION = "use_integration"


@dataclass
class CheckResult:
    allowed: bool
    reason: str | None
    upgrade_required: bool


class CreditAction(StrEnum):
    """Metered actions that draw down credits. Values ARE the credit_prices keys."""

    ZOE_MESSAGE = "zoe_message"
    ONECLICK_RUN = "oneclick_run"
    REGISTRY_PARSE = "registry_parse"
    # Non-AI: the generator makes no LLM call, so credits_for_llm_usage() returns
    # None and the charge is always the base. No set_llm_context needed.
    SPLIT_SHEET = "split_sheet"


@dataclass
class CreditsInfo:
    """Wallet state surfaced in Entitlements (spec §6/§7).

    The balances are None when the caller is NOT allowed to see them: a
    non-admin member of an org may not see the shared pool (it is a commercial
    fact about the org, not about their own access). They are REDACTED rather
    than zeroed — a 0 reads as "the org is out of credits", which is a lie a
    member would act on. None is unambiguous, and since this block only exists
    at all when credits are enabled, the frontend can read None as exactly
    "pool hidden" and fall back to the member's own limit.
    """

    bundle_balance: int | None
    reserve_balance: int | None
    monthly_grant: int
    overage_this_period: int
    overage_enabled: bool
    overage_cap_credits: int | None
    period_end: datetime | None
    prices: dict[str, int]
    # Org context only: the member's monthly ceiling on the shared pool and what
    # they have spent against it this period. None/0 in personal context, where
    # the wallet balance IS the limit.
    member_cap: int | None = None
    member_cap_used: int = 0

    @property
    def balance(self) -> int | None:
        if self.bundle_balance is None or self.reserve_balance is None:
            return None
        return self.bundle_balance + self.reserve_balance


@dataclass
class CreditCheckResult:
    """Outcome of EntitlementsService.check_credits()."""

    allowed: bool
    price: int
    reason: str | None = None
    use_overage: bool = False  # allowed only via opt-in pay-per-use
    overage_available: bool = False  # paid tier could enable overage to proceed
    upgrade_required: bool = False  # free tier: upgrade is the unlock
    reset_date: datetime | None = None
    degraded: bool = False
    # Licensing Phase B (spec §5, rules 8/9). `wallet_id` is the id of the wallet
    # the check passed against (personal wallet in personal context, the ORG POOL
    # in org context); gated_credits copies it into the CreditGrant so the debit
    # targets the SAME wallet the check cleared — a context switch mid-action can
    # never move the charge. None on a zero-price or degraded (uncharged) result.
    # `managed_by_org` marks an org-context outcome: on denial the enforcement 402
    # gains managedByOrg/requestUrl and carries NO overage/upgrade path (the member
    # asks their admin instead).
    #
    # `org_member_id` rides along so the debit can move that member's cap counter
    # atomically; `cap_reached` distinguishes the two org walls, which have
    # DIFFERENT remedies — the member's own ceiling (ask for a raise) versus a dry
    # pool (the admin must buy credits). The wall copy and the CTA both key off it.
    wallet_id: str | None = None
    org_member_id: str | None = None
    managed_by_org: bool = False
    cap_reached: bool = False
    # Licensing Phase C (spec §6/§11, rule 11) — owner-aware dry-seat wall. Set
    # ONLY on a DERIVED-resource org DENY where the caller OWNS the linked
    # project (a lazy, deny-path-only ownership check in `_check_credits_org`).
    # `owner_can_unlink` tells the enforcement 402 to render a second CTA —
    # unlink this project to fall back to the owner's personal plan — which
    # CO-OCCURS with managedByOrg/requestUrl (the owner-who-is-also-admin persona
    # is the common one; the two are never mutually exclusive). `project_id` is
    # the linked project to unlink, REQUIRED alongside the flag (contract-derived
    # surfaces like Zoe hold no project locally, so the hint would be dead text
    # without it).
    owner_can_unlink: bool = False
    project_id: str | None = None


@dataclass
class CreditGrant:
    """Handed from gated_credits() to debit_for_action() at a call site.

    enabled=False → debit_for_action is a no-op (credits disabled / admin bypass).
    """

    request_id: str
    action: str
    price: int
    kind: Literal["debit", "overage_debit"]
    enabled: bool
    # Licensing Phase B (rule 9): the wallet the debit MUST target — the exact
    # wallet the credit check passed against (the ORG POOL in org context, the
    # personal wallet otherwise). When set, debit_for_action charges it DIRECTLY,
    # resolving nothing, so a billing-context switch between check and debit cannot
    # relocate the charge. None for legacy / free / zero-price grants →
    # debit_for_action falls back to today's personal-wallet resolve (a no-op when
    # disabled anyway).
    wallet_id: str | None = None
    # Org-pool spend only: whose cap counter the debit moves. None on personal
    # wallets, so the RPC's org_members lock never fires for them.
    org_member_id: str | None = None


class OverridePayload(BaseModel):
    """Sparse override payload for POST /admin/users/{id}/override.

    Only fields the admin explicitly sets are written to tier_overrides;
    unset fields leave the row column NULL so the entitlements merge falls
    back to the tier default.
    """

    max_artists: int | None = Field(None, ge=-1)
    max_projects: int | None = Field(None, ge=-1)
    max_tasks: int | None = Field(None, ge=-1)
    max_storage_bytes: int | None = Field(None, ge=-1)
    max_split_sheets_per_month: int | None = Field(None, ge=-1)
    monthly_credits: int | None = Field(None, ge=0)
    max_works: int | None = Field(None, ge=-1)
    zoe_enabled: bool | None = None
    oneclick_enabled: bool | None = None
    registry_enabled: bool | None = None
    integrations_allowed: list[str] | None = None
    reason: str | None = None
    expires_days: int | None = Field(None, ge=1)
