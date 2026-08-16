import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
// Registers jest-dom matchers (toBeDisabled, etc.) on vitest's `expect` for
// this file — mirrors ../dissolve-dialog.test.tsx / ../transfer-dialog.test.tsx.
import "@testing-library/jest-dom/vitest";
import { OrgBillingPanel } from "../OrgBillingPanel";
import type { OrgDetail, OrgLedgerEntry, OrgUsage } from "@/hooks/useOrgs";
import type { CreditPack } from "@/hooks/useCreditPacks";

// Render-level coverage for the queued "Important" item: storage meter math
// (under/over pool), the required "team pool" copy, and the top-up Start
// button wiring. Pure-function label/fallback logic already lives in
// ../org-billing-panel.test.ts — this file only covers what needs a DOM.

const GB = 1024 ** 3;

const startTopupMutate = vi.fn();
const cancelTopupMutate = vi.fn();

let ledgerData: OrgLedgerEntry[] | undefined = [];
let usageData: OrgUsage | undefined;
let packsData: { packs: CreditPack[] } | undefined = { packs: [] };
let packsLoading = false;

vi.mock("@/hooks/useOrgs", () => ({
  useOrgLedger: () => ({ data: ledgerData, isLoading: false }),
  useOrgUsage: () => ({ data: usageData }),
  useStartOrgTopup: () => ({ mutate: startTopupMutate, isPending: false, error: null, variables: undefined }),
  useCancelOrgTopup: () => ({ mutate: cancelTopupMutate, isPending: false }),
  useTransferCredits: () => ({ mutate: vi.fn(), isPending: false, error: null, reset: vi.fn() }),
}));

vi.mock("@/hooks/useCreditPacks", () => ({
  useCreditPacks: () => ({ data: packsData, isLoading: packsLoading }),
}));

vi.mock("@/hooks/useBilling", () => ({
  useCreatePortalSession: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

vi.mock("@/hooks/useEntitlements", () => ({
  useEntitlements: () => ({ data: { credits: { reserveBalance: 0 } } }),
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  ledgerData = [];
  usageData = undefined;
  packsData = { packs: [] };
  packsLoading = false;
});

const baseOrg: OrgDetail = {
  id: "org-1",
  name: "Greenbox Analytics",
  status: "active",
  member_count: 3,
  admins: [],
};

describe("OrgBillingPanel", () => {
  it("renders the required 'team pool' copy", () => {
    render(<OrgBillingPanel org={baseOrg} />);
    expect(screen.getByText(/team ai work uses the team pool/i)).toBeInTheDocument();
  });

  it("renders the storage meter under the pool with no overage line", () => {
    const org: OrgDetail = {
      ...baseOrg,
      teamStorage: { usedBytes: 5 * GB, poolBytes: 10 * GB, overageGb: 0, ratePerGb: 0.5 },
    };
    render(<OrgBillingPanel org={org} />);

    expect(screen.getByText(/5\.0 GB of 10 GB used/)).toBeInTheDocument();
    expect(screen.queryByText(/GB over/)).toBeNull();
  });

  it("renders the amber overage line and monthly cost when over the pool", () => {
    const org: OrgDetail = {
      ...baseOrg,
      teamStorage: { usedBytes: 12 * GB, poolBytes: 10 * GB, overageGb: 2, ratePerGb: 0.5 },
    };
    render(<OrgBillingPanel org={org} />);

    expect(screen.getByText(/12 GB of 10 GB used/)).toBeInTheDocument();
    const overLine = screen.getByText(/GB over/);
    expect(overLine.textContent).toContain("2 GB over");
    expect(overLine.textContent).toContain("$1.00/mo");
  });

  it("starts the monthly top-up with the selected pack's key", () => {
    packsData = {
      packs: [{ key: "pack_500", credits: 500, price_cents: 999, sort_order: 1, recurringPriceId: "price_123" }],
    };
    render(<OrgBillingPanel org={baseOrg} />);

    fireEvent.click(screen.getByRole("button", { name: /^start$/i }));
    expect(startTopupMutate).toHaveBeenCalledWith({ orgId: baseOrg.id, key: "pack_500" });
  });
});
