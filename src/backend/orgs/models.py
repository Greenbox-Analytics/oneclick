from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class OrgCreate(BaseModel):
    name: str = Field(min_length=1)


class OrgUpdate(BaseModel):
    """PUT /orgs/{org_id} body. Fields not present in the request are left
    untouched by the service (router calls `model_dump(exclude_unset=True)`).

    default_member_cap is what a NEW member inherits (2,000 out of the box —
    20260814000001). `-1` sets the org to no limit; an explicit `null` also
    leaves members uncapped, since the fallback chain then ends at nothing.
    Prefer -1: it states the intent rather than relying on a NULL terminal."""

    name: str | None = Field(default=None, min_length=1)
    default_member_cap: int | None = Field(default=None, ge=-1)


class DissolveBody(BaseModel):
    """POST /orgs/{org_id}/dissolve body. `confirm_name` is the team's name
    typed back — the service compares it (trimmed) to the stored name and
    refuses with a 400 before writing anything, which is the only thing
    standing between a mis-click and a terminal, one-way operation."""

    confirm_name: str = Field(min_length=1)


class InviteCreate(BaseModel):
    """POST /orgs/{org_id}/invites body. Mirrors teams.models.InviteCreate."""

    email: EmailStr
    role: str = "member"  # 'admin' | 'member'


class MemberRoleUpdate(BaseModel):
    """PUT /orgs/{org_id}/members/{member_id}/role body."""

    role: str  # 'admin' | 'member'


class MemberCapUpdate(BaseModel):
    """PUT /orgs/{org_id}/members/{member_id}/cap body (admin-only).

    Three settings, and null is NOT "no limit":
      cap=N     this member's own ceiling.
      cap=null  INHERIT the org's `default_member_cap` (2,000 by default).
      cap=-1    NO LIMIT for this member, whatever the org default is.

    No idempotency key: writing a ceiling is idempotent by nature — there is no
    money to move twice.
    """

    cap: int | None = Field(default=None, ge=-1, le=10_000_000)


class OrgDispersalUpdate(BaseModel):
    """PUT /admin/orgs/{org_id}/dispersal body — MSANII ADMIN ONLY (see
    orgs.service.set_org_dispersal for why this can't live on /orgs/*).

    `monthly_dispersal_credits` is what the sweep adds to the org's pool each
    period: the contract volume. `default_member_cap` is NOT here — dividing up
    what they've paid for is the customer's own business, and it rides
    PUT /orgs/{org_id} like the org name.
    """

    monthly_dispersal_credits: int = Field(ge=0, le=100_000_000)


class CreditRequestCreate(BaseModel):
    """POST /orgs/{org_id}/credit-requests body (any ACTIVE member) — a request
    to RAISE this member's monthly cap. `requested_cap=None` means "raise it,
    admin decides" (matches the nullable CHECK on credit_requests.requested_cap
    — the column allows NULL, but any provided value must be > 0)."""

    requested_cap: int | None = Field(default=None, gt=0, le=10_000_000)
    note: str | None = None


class CreditRequestApprove(BaseModel):
    """POST /orgs/{org_id}/credit-requests/{request_id}/approve body
    (admin-only). The admin decides the new cap — it may differ from what was
    requested."""

    cap: int = Field(ge=0, le=10_000_000)


class CreditRequestDeny(BaseModel):
    """POST /orgs/{org_id}/credit-requests/{request_id}/deny body
    (admin-only)."""

    note: str | None = None


class TransferCreditsBody(BaseModel):
    """POST /orgs/{org_id}/transfer-credits body (active admin only, spec
    §4.1) — moves `amount` credits from the caller's OWN personal reserve
    into this org's pool. `le=1_000_000` is a sanity ceiling on a single
    transfer, not a business limit; the real constraint is the caller's
    actual reserve balance, enforced by the `transfer_credits` RPC (409 on
    insufficient reserve)."""

    amount: int = Field(gt=0, le=1_000_000)


class ProjectMemberRoleUpdate(BaseModel):
    """PUT /orgs/{org_id}/projects/{project_id}/members/{member_id} body
    (Licensing Phase C, Task 3 — org-admin-driven project membership). The
    Literal type is the ONLY "never set to owner" validation this endpoint
    needs: 'owner' isn't even a representable value, so
    projects.service.update_member_role's equivalent runtime ValueError
    guard has nothing left to add here."""

    role: Literal["viewer", "editor", "admin"]
