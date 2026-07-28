"""Shared helpers for expense-report exports (PDF + XLSX).

Mirrors the category labels and filter semantics used on the frontend Expense
Tracker (``EXPENSE_CATEGORY_LABELS`` and ``filterExpenseRows`` in
``src/pages/ExpenseTracker.tsx``) so the exported report matches exactly what
the user sees on screen. Line items are shown in their original currency;
totals sum the USD-converted ``amount_usd`` (falling back to ``amount`` for
pre-currency rows), matching the page's USD totals.
"""

# Keep in sync with EXPENSE_CATEGORY_LABELS in src/hooks/useProjectExpenses.ts
CATEGORY_LABELS: dict[str, str] = {
    "studio": "Studio",
    "mixing_mastering": "Mixing / Mastering",
    "marketing": "Marketing",
    "travel": "Travel",
    "equipment": "Equipment",
    "distribution": "Distribution",
    "other": "Other",
}


def category_label(cat: str | None) -> str:
    """Human label for a category code; falls back to a title-cased value."""
    if not cat:
        return "Other"
    return CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())


# Keep in sync with EXPENSE_CURRENCIES in projects/models.py and
# CURRENCY_SYMBOLS in src/lib/currency.ts
CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "US$",
    "EUR": "€",
    "CAD": "CA$",
    "AUD": "A$",
}


def fmt_money(n, currency: str = "USD") -> str:
    """Format a number in the given currency, matching the page's formatting."""
    symbol = CURRENCY_SYMBOLS.get((currency or "USD").upper())
    if symbol is None:
        return f"{float(n or 0):,.2f} {currency}"
    return f"{symbol}{float(n or 0):,.2f}"


def usd_amount(r: dict) -> float:
    """USD value of an expense row (``amount_usd``; ``amount`` for legacy rows)."""
    return float(r.get("amount_usd", r.get("amount")) or 0)


def filter_expense_rows(
    rows: list[dict],
    project_id: str | None,
    category: str | None,
    artist_id: str | None = None,
) -> list[dict]:
    """Apply the same scoping the page applies via ``filterExpenseRows``.

    Uncategorized expenses count as ``other`` so a category filter behaves the
    same way the on-screen ``byCategory`` bucketing does. Rows without an
    artist are excluded whenever an artist filter is set, matching the page.
    """
    out = []
    for r in rows:
        if project_id and r.get("project_id") != project_id:
            continue
        if category and (r.get("category") or "other") != category:
            continue
        if artist_id and r.get("artist_id") != artist_id:
            continue
        out.append(r)
    return out


def sorted_rows(rows: list[dict]) -> list[dict]:
    """Sort itemized rows by date ascending; undated rows sink to the bottom."""
    return sorted(rows, key=lambda r: (r.get("incurred_on") is None, r.get("incurred_on") or ""))


def category_totals(rows: list[dict]) -> list[tuple[str, float]]:
    """(category_code, USD total) pairs sorted by total descending."""
    acc: dict[str, float] = {}
    for r in rows:
        key = r.get("category") or "other"
        acc[key] = acc.get(key, 0.0) + usd_amount(r)
    return sorted(acc.items(), key=lambda kv: kv[1], reverse=True)


def grand_total(rows: list[dict]) -> float:
    """USD grand total across rows."""
    return sum(usd_amount(r) for r in rows)
