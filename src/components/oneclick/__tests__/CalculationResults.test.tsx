import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import {
  filterRows,
  findingForRow,
  ReviewBanner,
  sortRows,
  type RoyaltyPayment,
  type SplitFinding,
  type SplitReview,
} from "@/components/oneclick/CalculationResults";

afterEach(() => cleanup());

// Minimal fixture builder — every field defaults to something plausible so
// each test only spells out what it cares about.
const makePayment = (overrides: Partial<RoyaltyPayment> = {}): RoyaltyPayment => ({
  song_title: "Song A",
  party_name: "Alice",
  role: "writer",
  royalty_type: "master",
  percentage: 50,
  total_royalty: 1000,
  amount_to_pay: 500,
  ...overrides,
});

describe("findingForRow", () => {
  const findings: SplitFinding[] = [
    { party_name: "Alice", royalty_type: "Master", extracted_percentage: 50.005, verdict: "verified" },
    {
      party_name: "(unreadable contract)",
      royalty_type: "unknown",
      extracted_percentage: 0,
      verdict: "unverified",
      note: "Contract could not be parsed.",
    },
  ];
  const review: SplitReview = { overall: "needs_review", checked: 2, flagged: 1, findings };

  it("matches on normalized party_name + royalty_type and percentage within 0.01 tolerance", () => {
    const row = makePayment({ party_name: "  alice ", royalty_type: "MASTER", percentage: 50 });
    expect(findingForRow(review, row)).toBe(findings[0]);
  });

  it("does not match when percentage differs by more than the 0.01 tolerance", () => {
    const row = makePayment({ party_name: "Alice", royalty_type: "Master", percentage: 50.02 });
    expect(findingForRow(review, row)).toBeUndefined();
  });

  it("returns undefined when no finding matches party/type/percentage", () => {
    const row = makePayment({ party_name: "Bob", royalty_type: "master", percentage: 50 });
    expect(findingForRow(review, row)).toBeUndefined();
  });

  it("never joins the synthesized '(unreadable contract)' finding to a real payment row", () => {
    const row = makePayment({ party_name: "Charlie", royalty_type: "unknown", percentage: 0 });
    expect(findingForRow(review, row)).toBeUndefined();
  });

  it("returns undefined for a null or undefined review", () => {
    const row = makePayment();
    expect(findingForRow(null, row)).toBeUndefined();
    expect(findingForRow(undefined, row)).toBeUndefined();
  });
});

describe("ReviewBanner", () => {
  it("renders the unavailable message when review is null", () => {
    render(<ReviewBanner review={null} />);
    expect(screen.getByText("Split verification wasn't available for this result.")).toBeTruthy();
  });

  it("renders the unavailable message when overall is 'unavailable'", () => {
    render(<ReviewBanner review={{ overall: "unavailable", checked: 0, flagged: 0, findings: [] }} />);
    expect(screen.getByText("Split verification wasn't available for this result.")).toBeTruthy();
  });

  it("renders the verified banner with singular wording for a single check", () => {
    render(<ReviewBanner review={{ overall: "verified", checked: 1, flagged: 0, findings: [] }} />);
    expect(screen.getByText("All 1 extracted split match the contract.")).toBeTruthy();
  });

  it("renders the verified banner with plural wording for multiple checks", () => {
    render(<ReviewBanner review={{ overall: "verified", checked: 3, flagged: 0, findings: [] }} />);
    expect(screen.getByText("All 3 extracted splits match the contract.")).toBeTruthy();
  });

  it("renders the needs_review banner with the flagged/checked counts", () => {
    render(<ReviewBanner review={{ overall: "needs_review", checked: 3, flagged: 2, findings: [] }} />);
    expect(
      screen.getByText("2 of 3 splits couldn't be confirmed against the contract — double-check the percentages below against your contract before paying."),
    ).toBeTruthy();
  });
});

describe("filterRows / sortRows", () => {
  const rows: RoyaltyPayment[] = [
    makePayment({ song_title: "Song A", party_name: "Alice", basis: "net", expenses_applied: 100, amount_to_pay: 500 }),
    makePayment({ song_title: "Song B", party_name: "Bob", basis: "gross", amount_to_pay: 900 }),
    makePayment({ song_title: "Song B", party_name: "Cara", amount_to_pay: 100 }),
  ];
  const noFilters = { search: "", song: "all", basis: "all", status: "any", segment: null } as const;

  it("filters by basis without touching any row amounts", () => {
    const out = filterRows(rows, { ...noFilters, basis: "net" }, null);
    expect(out).toEqual([rows[0]]);
    expect(out[0].amount_to_pay).toBe(500);
    expect(out[0].expenses_applied).toBe(100);
  });

  it("filters by search text and by chart segment selection", () => {
    expect(filterRows(rows, { ...noFilters, search: "bob" }, null)).toEqual([rows[1]]);
    expect(
      filterRows(rows, { ...noFilters, segment: { kind: "song", name: "Song B" } }, null),
    ).toEqual([rows[1], rows[2]]);
    expect(
      filterRows(rows, { ...noFilters, segment: { kind: "payee", name: "Alice" } }, null),
    ).toEqual([rows[0]]);
  });

  it("sorts by amount desc and payee asc without mutating the input", () => {
    expect(sortRows(rows, "amount", "desc").map((r) => r.amount_to_pay)).toEqual([900, 500, 100]);
    expect(sortRows(rows, "payee", "asc").map((r) => r.party_name)).toEqual(["Alice", "Bob", "Cara"]);
    // Original order untouched.
    expect(rows.map((r) => r.amount_to_pay)).toEqual([500, 900, 100]);
  });
});
