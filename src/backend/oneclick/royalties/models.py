"""Pydantic models for the OneClick royalties aggregation and payout endpoints."""

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CreatePayoutRequest(BaseModel):
    payee_ids: list[str]
    idempotency_key: str | None = None
    note: str | None = None


class PatchPayeeRequest(BaseModel):
    payout_currency: str | None = None
    registry_user_id: str | None = None
    email: str | None = None


class SplitPayeeRequest(BaseModel):
    line_ids: list[str]
    new_display_name: str


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PayeeSummary(BaseModel):
    id: str
    display_name: str
    payout_currency: str
    registry_user_id: str | None = None
    email: str | None = None
    collision: bool = False
    project_count: int = 0
    status: str  # "owed" | "scheduled" | "settled"
    # reporting-currency totals
    earned: float
    paid: float
    drafted: float
    owed: float
    # payee payout-currency totals
    earned_native: float
    paid_native: float
    drafted_native: float
    owed_native: float


class PayeeLine(BaseModel):
    line_id: str
    song_title: str
    role: str | None = None
    royalty_type: str | None = None
    percentage: float | None = None
    amount_owed: float  # statement currency
    statement_currency: str


class PayeeStatement(BaseModel):
    royalty_statement_id: str
    period_start: str | None = None
    period_end: str | None = None
    statement_currency: str
    statement_total: float | None = None  # true per-project statement net (may be null for pre-feature calcs)
    earned: float
    paid: float
    drafted: float
    owed: float
    state: str  # "owed" | "scheduled" | "settled"
    lines: list[PayeeLine]


class PayeeProject(BaseModel):
    project_id: str
    name: str
    statements: list[PayeeStatement]


class PayoutOut(BaseModel):
    id: str
    payee_id: str
    status: str  # "draft" | "paid"
    pay_currency: str
    fx_rate_date: str
    total_amount: float
    note: str | None = None
    created_at: str
    paid_at: str | None = None
    breakdown_snapshot: dict
    orphan_state: str = "none"  # "none" | "partial" | "orphaned"  (derived)


class PayeeDetail(BaseModel):
    summary: PayeeSummary
    projects: list[PayeeProject]
    payouts: list[PayoutOut]


class PeriodCell(BaseModel):
    royalty_statement_id: str
    period_start: str | None = None
    period_end: str | None = None
    earned: float
    state: str


class PeriodLedgerRow(BaseModel):
    payee_id: str
    display_name: str
    cells: list[PeriodCell]
    total: float


class PeriodLedger(BaseModel):
    base: str
    rows: list[PeriodLedgerRow]
