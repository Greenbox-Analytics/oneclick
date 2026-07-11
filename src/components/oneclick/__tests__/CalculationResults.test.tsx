import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import {
  findingForRow,
  ReviewBanner,
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
