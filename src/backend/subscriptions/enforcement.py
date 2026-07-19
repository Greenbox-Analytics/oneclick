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


def gated_create(user_id: str, resource: str, current_count: int) -> None:
    """402 if user is at the create cap for this resource."""
    action = _RESOURCE_TO_ACTION.get(resource)
    if action is None:
        # Unknown resource — silently no-op so a typo doesn't block requests,
        # but log a warning so the typo is visible in observability.
        logger.warning(
            "gated_create: unknown resource %r — treating as ungated",
            resource,
        )
        return
    result = _service().can(user_id, action, current_count=current_count)
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
) -> None:
    """402 if upload would exceed the project owner's storage cap."""
    result = _service().can(
        user_id,
        Action.UPLOAD_BYTES,
        size=size,
        host_user_id=host_user_id,
    )
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
) -> CreditGrant:
    """Credit gate for metered AI actions. Returns a CreditGrant to hand to
    debit_for_action() after the action SUCCEEDS (charge-on-success, spec §3).

    request_id is a fresh uuid4 per invocation: internal RPC retries dedupe on
    it; an intentional user re-run is a new invocation and charges again.
    The ACTING user pays — host_user_id only feeds the legacy fallback.
    """
    if not credits_enabled():
        legacy = _CREDIT_TO_LEGACY.get(action)
        if legacy is None:
            # Unmapped CreditAction — fail LOUD (this is a gating path).
            raise RuntimeError(f"gated_credits: no legacy mapping for {action!r}")
        gated_feature(user_id, legacy, host_user_id=host_user_id)
        return CreditGrant(request_id="", action=str(action), price=0, kind="debit", enabled=False)

    result = _service().check_credits(user_id, str(action), is_admin=is_admin)
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
            },
        )
        raise HTTPException(
            status_code=402,
            detail={
                "reason": result.reason or "Not enough credits.",
                "price": result.price,
                "resetDate": result.reset_date.isoformat() if result.reset_date else None,
                "upgradeRequired": result.upgrade_required,
                "overageAvailable": result.overage_available,
            },
        )
    return CreditGrant(
        request_id=str(uuid.uuid4()),
        action=str(action),
        price=result.price,
        kind="overage_debit" if result.use_overage else "debit",
        enabled=result.price > 0,
    )
