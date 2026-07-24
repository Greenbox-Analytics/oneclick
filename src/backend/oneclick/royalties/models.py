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


class SaveReceiptRequest(BaseModel):
    artist_id: str
    project_id: str


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
    status: str  # "owed" | "overpaid" | "scheduled" | "settled"
    # reporting-currency totals
    earned: float
    paid: float
    drafted: float
    owed: float  # earned − paid − drafted (available to draft; drives status/eligibility)
    unpaid: float  # earned − paid (outstanding until actually paid; drives "Outstanding" displays)
    # payee payout-currency totals
    earned_native: float
    paid_native: float
    drafted_native: float
    owed_native: float
    unpaid_native: float
    # buckets whose statement currency could not be converted to the reporting base
    unconvertible_count: int = 0
    # overpayment credit, derived from PAID coverage only, per statement currency —
    # never FX-netted across currencies (mirrors the owed/unpaid per-bucket clamp invariant)
    credit_by_ccy: dict[str, float] = {}


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
    owed: float  # earned − paid − drafted
    unpaid: float  # earned − paid (outstanding until actually paid)
    state: str  # "owed" | "overpaid" | "scheduled" | "settled"
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
    payment_method: str = "manual"  # "manual" | "paypal"
    paypal_capture_id: str | None = None


class PaypalOrderOut(BaseModel):
    paypal_order_id: str


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
    unconvertible_count: int = 0


class PeriodLedger(BaseModel):
    base: str
    rows: list[PeriodLedgerRow]


# ---------------------------------------------------------------------------
# Analytics models
# ---------------------------------------------------------------------------


class MonthPoint(BaseModel):
    """A single month bucket for time-series analytics.

    Used in overview paid_by_month (amount only) and per-artist / per-payee
    by_month (earned + paid).  Fields not applicable to a given context
    default to None so one model covers both shapes.
    """

    month: str  # "YYYY-MM"
    amount: float = 0.0  # used in overview paid_by_month
    earned: float = 0.0  # used in artist/payee by_month
    paid: float = 0.0  # used in artist/payee by_month


class TopOwed(BaseModel):
    payee_id: str
    display_name: str
    owed: float


class OverviewOut(BaseModel):
    base: str
    outstanding_total: float
    payees_owed_count: int
    drafted_total: float
    draft_count: int
    paid_total: float
    paid_last_30d: float
    paid_by_month: list[MonthPoint]
    top_owed: list[TopOwed]
    unconvertible_count: int


class ArtistAnalyticsOut(BaseModel):
    artist_id: str
    base: str
    summary: dict  # {earned_total, owed_now, paid_total}
    by_month: list[MonthPoint]
    unconvertible_count: int


class PayeeAnalyticsOut(BaseModel):
    payee_id: str
    display_name: str
    base: str
    summary: dict  # {earned_total, paid_total, owed}
    by_month: list[MonthPoint]
    unconvertible_count: int
