from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class OrgCreate(BaseModel):
    name: str = Field(min_length=1)


class OrgUpdate(BaseModel):
    """PUT /orgs/{org_id} body. Fields not present in the request are left
    untouched by the service (router calls `model_dump(exclude_unset=True)`);
    an explicit `null` for default_member_cap clears it back to
    manual-only allocation (NULL/0 = manual-only, spec §4)."""

    name: str | None = Field(default=None, min_length=1)
    default_member_cap: int | None = Field(default=None, ge=0)


class InviteCreate(BaseModel):
    """POST /orgs/{org_id}/invites body. Mirrors teams.models.InviteCreate."""

    email: EmailStr
    role: str = "member"  # 'admin' | 'member'


class MemberRoleUpdate(BaseModel):
    """PUT /orgs/{org_id}/members/{member_id}/role body."""

    role: str  # 'admin' | 'member'


class MemberCapUpdate(BaseModel):
    """PUT /orgs/{org_id}/members/{member_id}/cap body (admin-only).

    `cap=null` clears this member's own cap so they inherit the org's
    `default_member_cap` (uncapped if that is NULL too). No idempotency key:
    writing a ceiling is idempotent by nature — there is no money to move twice.
    """

    cap: int | None = Field(default=None, ge=0, le=10_000_000)


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


class ProjectMemberRoleUpdate(BaseModel):
    """PUT /orgs/{org_id}/projects/{project_id}/members/{member_id} body
    (Licensing Phase C, Task 3 — org-admin-driven project membership). The
    Literal type is the ONLY "never set to owner" validation this endpoint
    needs: 'owner' isn't even a representable value, so
    projects.service.update_member_role's equivalent runtime ValueError
    guard has nothing left to add here."""

    role: Literal["viewer", "editor", "admin"]
