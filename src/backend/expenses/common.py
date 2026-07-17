"""Shared helpers for expense-report exports (PDF + XLSX).

Mirrors the category labels and filter semantics used on the frontend Expense
Tracker (``EXPENSE_CATEGORY_LABELS`` and ``filterExpenseRows`` in
``src/pages/ExpenseTracker.tsx``) so the exported report matches exactly what
the user sees on screen. Amounts are USD, matching the page's hardcoded
formatting — currency selection is intentionally out of scope.
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


def fmt_money(n) -> str:
    """Format a number as USD, matching the page's ``$`` formatting."""
    return f"${float(n or 0):,.2f}"


def filter_expense_rows(rows: list[dict], project_id: str | None, category: str | None) -> list[dict]:
    """Apply the same scoping the page applies via ``filterExpenseRows``.

    Uncategorized expenses count as ``other`` so a category filter behaves the
    same way the on-screen ``byCategory`` bucketing does.
    """
    out = []
    for r in rows:
        if project_id and r.get("project_id") != project_id:
            continue
        if category and (r.get("category") or "other") != category:
            continue
        out.append(r)
    return out


def sorted_rows(rows: list[dict]) -> list[dict]:
    """Sort itemized rows by date ascending; undated rows sink to the bottom."""
    return sorted(rows, key=lambda r: (r.get("incurred_on") is None, r.get("incurred_on") or ""))


def category_totals(rows: list[dict]) -> list[tuple[str, float]]:
    """(category_code, total) pairs sorted by total descending."""
    acc: dict[str, float] = {}
    for r in rows:
        key = r.get("category") or "other"
        acc[key] = acc.get(key, 0.0) + float(r.get("amount") or 0)
    return sorted(acc.items(), key=lambda kv: kv[1], reverse=True)


def grand_total(rows: list[dict]) -> float:
    return sum(float(r.get("amount") or 0) for r in rows)
