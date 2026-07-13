import { describe, it, expect } from "vitest";
import { filterExpenseRows } from "../ExpenseTracker";
import type { ExpenseSummaryRow } from "@/hooks/useProjectExpenses";

const row = (overrides: Partial<ExpenseSummaryRow>): ExpenseSummaryRow => ({
  id: "e1",
  project_id: "p1",
  project_name: "Project One",
  artist_id: "a1",
  artist_name: "Artist One",
  description: "Studio session",
  amount: 100,
  category: "studio",
  incurred_on: "2026-01-15",
  is_tagged: false,
  ...overrides,
});

const rows: ExpenseSummaryRow[] = [
  row({ id: "e1", project_id: "p1", category: "studio" }),
  row({ id: "e2", project_id: "p1", category: "other" }),
  row({ id: "e3", project_id: "p2", category: null }),
  row({ id: "e4", project_id: "p2", category: "marketing" }),
];

describe("filterExpenseRows", () => {
  it("returns all rows when both filters are 'all'", () => {
    expect(filterExpenseRows(rows, "all", "all")).toEqual(rows);
  });

  it("filters by project id", () => {
    const result = filterExpenseRows(rows, "p1", "all");
    expect(result.map((r) => r.id)).toEqual(["e1", "e2"]);
  });

  it("filters by category", () => {
    const result = filterExpenseRows(rows, "all", "marketing");
    expect(result.map((r) => r.id)).toEqual(["e4"]);
  });

  it("treats uncategorized rows as 'other'", () => {
    const result = filterExpenseRows(rows, "all", "other");
    expect(result.map((r) => r.id)).toEqual(["e2", "e3"]);
  });

  it("intersects project and category filters", () => {
    expect(filterExpenseRows(rows, "p2", "other").map((r) => r.id)).toEqual(["e3"]);
    expect(filterExpenseRows(rows, "p1", "marketing")).toEqual([]);
  });
});
