import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
// Registers jest-dom matchers on vitest's `expect` for this file — no shared
// vitest.config.ts setupFiles entry exists yet (see gate-dialogs.test.tsx).
import "@testing-library/jest-dom/vitest";
import { PartiesTable } from "../PartiesTable";
import type { PayeeSummary } from "@/hooks/useRoyalties";

// No global `afterEach`/auto-cleanup registered (vitest.config.ts doesn't set
// test.globals: true), and unlike gate-dialogs.test.tsx this file's two cases
// render overlapping content — clean up explicitly so test 2 doesn't see
// test 1's still-mounted chip.
afterEach(() => cleanup());

const overpaidPayee: PayeeSummary = {
  id: "payee-1",
  display_name: "Romes",
  payout_currency: "USD",
  collision: false,
  project_count: 1,
  status: "overpaid",
  earned: 100,
  paid: 100.5,
  drafted: 0,
  owed: 0,
  unpaid: 0,
  earned_native: 100,
  paid_native: 100.5,
  drafted_native: 0,
  owed_native: 0,
  unpaid_native: 0,
  credit_by_ccy: { USD: 0.5 },
};

const noop = () => {};

describe("overpaid party state", () => {
  it("renders the Overpaid status badge and a credit chip", () => {
    render(
      <PartiesTable
        parties={[overpaidPayee]}
        base="USD"
        selection={[]}
        toggleSel={noop}
        onOpenParty={noop}
        onPaySelected={noop}
        onClear={noop}
      />,
    );

    expect(screen.getByText("Overpaid")).toBeInTheDocument();
    expect(screen.getByText(/Overpaid \$0\.50 USD/)).toBeInTheDocument();
  });

  it("renders no credit chip when credit_by_ccy is empty", () => {
    render(
      <PartiesTable
        parties={[{ ...overpaidPayee, status: "owed", owed: 10, unpaid: 10, credit_by_ccy: {} }]}
        base="USD"
        selection={[]}
        toggleSel={noop}
        onOpenParty={noop}
        onPaySelected={noop}
        onClear={noop}
      />,
    );

    expect(screen.queryByText(/Overpaid \$/)).not.toBeInTheDocument();
    expect(screen.getByText("Unpaid")).toBeInTheDocument();
  });
});
