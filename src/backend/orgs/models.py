from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class OrgCreate(BaseModel):
    name: str = Field(min_length=1)


class OrgUpdate(BaseModel):
    """PUT /orgs/{org_id} body. Fields not present in the request are left
    untouched by the service (router calls `model_dump(exclude_unset=True)`);
    an explicit `null` for default_seat_allowance clears it back to
    manual-only allocation (NULL/0 = manual-only, spec §4)."""

    name: str | None = Field(default=None, min_length=1)
    default_seat_allowance: int | None = Field(default=None, ge=0)


class InviteCreate(BaseModel):
    """POST /orgs/{org_id}/invites body. Mirrors teams.models.InviteCreate."""

    email: EmailStr
    role: str = "member"  # 'admin' | 'member'


class MemberRoleUpdate(BaseModel):
    """PUT /orgs/{org_id}/members/{member_id}/role body."""

    role: str  # 'admin' | 'member'


class AllocateCredits(BaseModel):
    """POST /orgs/{org_id}/members/{member_id}/allocate body (admin-only
    pool -> seat transfer). `idempotency_key` becomes the RPC's BASE
    request_id (`alloc:{idempotency_key}`) — transfer_credits appends its own
    :from/:to suffixes."""

    amount: int = Field(gt=0, le=1_000_000)
    idempotency_key: str = Field(min_length=1)


class ReclaimCredits(BaseModel):
    """POST /orgs/{org_id}/members/{member_id}/reclaim body (admin-only
    seat -> pool transfer). `amount=null` means reclaim-all: the service
    reads the seat's current balance and uses that as the transfer amount (or
    no-ops at `{"removed": 0}` if the balance isn't positive — see
    orgs.service.reclaim_credits)."""

    amount: int | None = Field(default=None, gt=0, le=1_000_000)
    idempotency_key: str = Field(min_length=1)


class CreditRequestCreate(BaseModel):
    """POST /orgs/{org_id}/credit-requests body (any ACTIVE member).
    `requested_credits=None` means "more, admin decides" (matches the
    nullable CHECK on credit_requests.requested_credits — the column allows
    NULL, but any provided value must be > 0)."""

    requested_credits: int | None = Field(default=None, gt=0, le=1_000_000)
    note: str | None = None


class CreditRequestApprove(BaseModel):
    """POST /orgs/{org_id}/credit-requests/{request_id}/approve body
    (admin-only). The admin decides the amount — it may differ from
    requested_credits."""

    credits: int = Field(gt=0, le=1_000_000)


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
