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
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class Caps:
    max_artists: int
    max_projects: int
    max_tasks: int
    max_storage_bytes: int
    max_split_sheets_per_month: int


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
class Entitlements:
    user_id: str
    tier: Literal["free", "pro"]
    status: Literal["active", "canceled", "past_due", "trialing"]
    caps: Caps
    features: Features
    usage: Usage
    has_overrides: bool
    degraded: bool = False

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict using camelCase keys for the frontend."""
        return {
            "tier": self.tier,
            "status": self.status,
            "caps": {
                "maxArtists": self.caps.max_artists,
                "maxProjects": self.caps.max_projects,
                "maxTasks": self.caps.max_tasks,
                "maxStorageBytes": self.caps.max_storage_bytes,
                "maxSplitSheetsPerMonth": self.caps.max_split_sheets_per_month,
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
            "hasOverrides": self.has_overrides,
            "degraded": self.degraded,
        }


class Action(str, Enum):
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
          name: str — integration identifier (e.g. 'slack').
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
    zoe_enabled: bool | None = None
    oneclick_enabled: bool | None = None
    registry_enabled: bool | None = None
    integrations_allowed: list[str] | None = None
    reason: str | None = None
    expires_days: int | None = Field(None, ge=1)
