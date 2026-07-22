"""Thin wrappers around EntitlementsService.can() that raise HTTPException(402)
on denial. One-line gates at endpoint entry; consistent error shape."""

import logging
import uuid

from fastapi import HTTPException

from analytics import capture as analytics_capture
from subscriptions.deps import _get_entitlements_service
from subscriptions.models import Action, CreditAction, CreditGrant
from subscriptions.service import EntitlementsService, credits_enabled

logger = logging.getLogger(__name__)

_RESOURCE_TO_ACTION = {
    "artist": Action.CREATE_ARTIST,
    "project": Action.CREATE_PROJECT,
    "task": Action.CREATE_TASK,
    "work": Action.CREATE_WORK,
    # "board" intentionally absent — cap dropped in SP3
}


def _service() -> EntitlementsService:
    """Indirection so tests can monkeypatch this name to inject a fake service."""
    return _get_entitlements_service()


def gated_create(
    user_id: str,
    resource: str,
    current_count: int,
    resource_project_id: str | None = None,
) -> None:
    """402 if user is at the create cap for this resource.

    `resource_project_id` (Licensing Phase C) is the project the resource will
    live in; when it is linked to an org where the caller holds an active seat,
    the org's (unlimited) count caps apply — threaded into can() (rule 9). Only
    CREATE_WORK derives; other resources ignore it. Default None → today's
    personal cap check, byte-identical.
    """
    action = _RESOURCE_TO_ACTION.get(resource)
    if action is None:
        # Unknown resource — silently no-op so a typo doesn't block requests,
        # but log a warning so the typo is visible in observability.
        logger.warning(
            "gated_create: unknown resource %r — treating as ungated",
            resource,
        )
        return
    can_kwargs = {"current_count": current_count}
    # Thread the resource id ONLY when present so the None case is a byte-identical
    # call to pre-Phase-C (can() defaults it to None either way) — existing exact-arg
    # tests stay unmodified, per the plan's regression discipline.
    if resource_project_id is not None:
        can_kwargs["resource_project_id"] = resource_project_id
    result = _service().can(user_id, action, **can_kwargs)
    if not result.allowed:
        analytics_capture(
            user_id,
            "paywall_blocked",
            {
                "kind": "create_cap",
                "resource": resource,
                "reason": result.reason,
            },
        )
        raise HTTPException(status_code=402, detail=result.reason or "Limit reached")


def gated_feature(
    user_id: str,
    action: Action,
    host_user_id: str | None = None,
    **ctx,
) -> None:
    """402 if user can't use this feature (with host-wins resolution)."""
    result = _service().can(user_id, action, host_user_id=host_user_id, **ctx)
    if not result.allowed:
        analytics_capture(
            user_id,
            "paywall_blocked",
            {
                "kind": "feature",
                "action": str(action),
                "reason": result.reason,
            },
        )
        raise HTTPException(status_code=402, detail=result.reason or "Pro feature")


def gated_upload(
    user_id: str,
    size: int,
    host_user_id: str | None = None,
    resource_project_id: str | None = None,
) -> None:
    """402 if upload would exceed the project owner's storage cap.

    `resource_project_id` (Licensing Phase C) is the project being uploaded to;
    when the caller IS the storage-counter owner AND that project is linked to
    an org where they hold an active seat, the org's larger seat storage applies
    (rule 9 — derivation NEVER fires when host_user_id is a different owner).
    Default None → today's owner-scoped check, byte-identical.
    """
    can_kwargs = {"size": size, "host_user_id": host_user_id}
    # Thread the resource id ONLY when present so the None case is a byte-identical
    # call to pre-Phase-C (can() defaults it to None either way) — existing exact-arg
    # tests stay unmodified, per the plan's regression discipline.
    if resource_project_id is not None:
        can_kwargs["resource_project_id"] = resource_project_id
    result = _service().can(user_id, Action.UPLOAD_BYTES, **can_kwargs)
    if not result.allowed:
        analytics_capture(
            user_id,
            "paywall_blocked",
            {
                "kind": "upload_cap",
                "size": size,
                "reason": result.reason,
            },
        )
        raise HTTPException(status_code=402, detail=result.reason or "Storage limit reached")


def gated_split_sheet(user_id: str) -> None:
    """402 if user is over their per-period split-sheet cap."""
    result = _service().can(user_id, Action.GENERATE_SPLIT_SHEET)
    if not result.allowed:
        analytics_capture(
            user_id,
            "paywall_blocked",
            {
                "kind": "splitsheet_cap",
                "reason": result.reason,
            },
        )
        raise HTTPException(status_code=402, detail=result.reason or "Split sheet limit reached")


# Legacy feature-flag equivalent for each credit action (fallback when
# CREDITS_ENABLED is off — preserves today's behavior exactly).
_CREDIT_TO_LEGACY = {
    CreditAction.ZOE_MESSAGE: Action.USE_ZOE,
    CreditAction.ONECLICK_RUN: Action.USE_ONECLICK,
    CreditAction.REGISTRY_PARSE: Action.USE_REGISTRY,
}


def gated_credits(
    user_id: str,
    action: CreditAction,
    host_user_id: str | None = None,
    *,
    is_admin: bool = False,
    resource_project_id: str | None = None,
    resource_contract_ids: list[str] | None = None,
) -> CreditGrant:
    """Credit gate for metered AI actions. Returns a CreditGrant to hand to
    debit_for_action() after the action SUCCEEDS (charge-on-success, spec §3).

    request_id is a fresh uuid4 per invocation: internal RPC retries dedupe on
    it; an intentional user re-run is a new invocation and charges again.
    The ACTING user pays — host_user_id only feeds the legacy fallback.

    `resource_project_id` / `resource_contract_ids` (Licensing Phase C) are the
    resource the action operates on; they thread straight into check_credits so
    an action on an org-linked project bills the linked org's seat (rule 5).
    Both default None → the pre-Phase-C ambient/personal decision, unchanged.
    """
    if not credits_enabled():
        legacy = _CREDIT_TO_LEGACY.get(action)
        if legacy is None:
            # Unmapped CreditAction — fail LOUD (this is a gating path).
            raise RuntimeError(f"gated_credits: no legacy mapping for {action!r}")
        gated_feature(user_id, legacy, host_user_id=host_user_id)
        return CreditGrant(request_id="", action=str(action), price=0, kind="debit", enabled=False)

    result = _service().check_credits(
        user_id,
        str(action),
        is_admin=is_admin,
        resource_project_id=resource_project_id,
        resource_contract_ids=resource_contract_ids,
    )
    if not result.allowed:
        analytics_capture(
            user_id,
            "paywall_blocked",
            {
                "kind": "credits",
                "gate": "credits",
                "action": str(action),
                "reason": result.reason,
                "degraded": result.degraded,
                "managed_by_org": result.managed_by_org,
            },
        )
        detail = {
            "reason": result.reason or "Not enough credits.",
            "price": result.price,
            "resetDate": result.reset_date.isoformat() if result.reset_date else None,
            "upgradeRequired": result.upgrade_required,
            "overageAvailable": result.overage_available,
        }
        if result.managed_by_org:
            # Seat wall (Licensing Phase B, rule 8): there is no upgrade / overage
            # path — the member asks their org admin. `requestUrl` points the wall
            # at the org member view where the credit-request form lives (Task 9).
            detail["managedByOrg"] = True
            detail["requestUrl"] = "/organization"
        if result.owner_can_unlink:
            # Owner-aware dry-seat wall (Licensing Phase C, rule 11): the caller
            # OWNS the linked project, so offer a second CTA — unlink it to fall
            # back to their personal plan. `projectId` is REQUIRED alongside the
            # flag (round 5: contract-derived surfaces like Zoe hold no project
            # locally, so the hint is dead text without it). These CO-OCCUR with
            # managedByOrg/requestUrl above — an owner who is also an org admin
            # sees BOTH the buy/request path and the unlink path; they are never
            # rendered mutually exclusive (round 4).
            detail["ownerCanUnlink"] = True
            detail["projectId"] = result.project_id
        raise HTTPException(status_code=402, detail=detail)
    return CreditGrant(
        request_id=str(uuid.uuid4()),
        action=str(action),
        price=result.price,
        kind="overage_debit" if result.use_overage else "debit",
        enabled=result.price > 0,
        # Rule 9: the debit targets the exact wallet the check cleared (seat wallet
        # in org context, personal wallet otherwise). None on legacy / zero-price.
        wallet_id=result.wallet_id,
    )


def free_credit_grant(action: CreditAction) -> CreditGrant:
    """Grant for a PRE-VERIFIED zero-cost invocation (result-cache hit,
    conversational fast-path): skips the wall AND the debit. Only correct when
    the caller has deterministically established no LLM cost will be incurred
    (spec §3: cached/canned actions are free and must stay reachable at zero
    balance). Callers gate usage on credits_enabled() so legacy-mode feature
    gating is preserved when the flag is off."""
    return CreditGrant(request_id="", action=str(action), price=0, kind="debit", enabled=False)
