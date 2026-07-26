from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class MemberAdd(BaseModel):
    email: EmailStr
    role: str  # admin, editor, viewer (not owner)


class MemberUpdate(BaseModel):
    role: str  # admin, editor, viewer (not owner)


EXPENSE_CATEGORIES = (
    "studio",
    "mixing_mastering",
    "marketing",
    "travel",
    "equipment",
    "distribution",
    "other",
)


# Keep in sync with EXPENSE_CURRENCIES in src/hooks/useProjectExpenses.ts and the
# CHECK constraint on project_expenses.currency.
EXPENSE_CURRENCIES = ("USD", "EUR", "CAD", "AUD")
ExpenseCurrency = Literal["USD", "EUR", "CAD", "AUD"]


class ExpenseCreate(BaseModel):
    description: str = ""  # optional; "" convention (column is NOT NULL)
    amount: float = Field(ge=0)
    currency: ExpenseCurrency = "USD"
    category: str | None = None
    incurred_on: str | None = None  # ISO date (YYYY-MM-DD)
    work_ids: list[str] = Field(default_factory=list)


class ExpenseUpdate(BaseModel):
    description: str | None = None
    amount: float | None = Field(default=None, ge=0)
    currency: ExpenseCurrency | None = None
    category: str | None = None
    incurred_on: str | None = None
    work_ids: list[str] | None = None  # None = leave links unchanged; [] = clear all links
