import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
// Registers jest-dom matchers on vitest's `expect` for this file — mirrors
// ../../orgs/__tests__/billing-panel.test.tsx.
import "@testing-library/jest-dom/vitest";
import { CreditsUsageCard } from "../CreditsUsageCard";
import type { CreditUsage } from "@/hooks/useCreditUsage";

// Base rates (spec 2026-08-17): split sheets are a metered action, so the card
// must render exactly ONE "Split sheet" row — the one that arrives on
// usage.tools — and quote its base rate rather than calling it free.

let usageData: CreditUsage | undefined;

vi.mock("react-router-dom", () => ({ useNavigate: () => vi.fn() }));

vi.mock("@/hooks/useCreditUsage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/hooks/useCreditUsage")>()),
  useCreditUsage: () => ({ data: usageData, isLoading: false }),
}));

vi.mock("@/hooks/useEntitlements", () => ({
  useEntitlements: () => ({ data: { tier: "pro", credits: {}, caps: {}, usage: {} } }),
}));

vi.mock("@/hooks/useCreditPacks", () => ({
  useCreditPacks: () => ({ data: { packs: [] }, isLoading: false }),
  useCreateTopupSession: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("@/hooks/useBilling", () => ({
  useSetBillingPrefs: () => ({ mutate: vi.fn(), isPending: false }),
}));

afterEach(() => {
  cleanup();
  usageData = undefined;
});

const renderCard = (opts: { usage: Partial<CreditUsage> }) => {
  usageData = { enabled: true, ...opts.usage } as CreditUsage;
  return render(<CreditsUsageCard />);
};

describe("CreditsUsageCard", () => {
  it("renders exactly one Split sheet row", async () => {
    renderCard({
      usage: {
        enabled: true,
        tools: [
          { action: "split_sheet", price: 20, count: 2, spent: 40 },
          { action: "oneclick_run", price: 30, count: 1, spent: 30 },
        ],
      },
    });
    const rows = await screen.findAllByText("Split sheet");
    expect(rows).toHaveLength(1);
  });

  it("prices the split sheet row instead of calling it free", async () => {
    renderCard({
      usage: { enabled: true, tools: [{ action: "split_sheet", price: 20, count: 0, spent: 0 }] },
    });
    expect(await screen.findByText(/20 cr/)).toBeInTheDocument();
    expect(screen.queryByText(/not metered/)).not.toBeInTheDocument();
  });

  it("no longer claims registry cache hits are free", async () => {
    renderCard({
      usage: { enabled: true, tools: [{ action: "registry_parse", price: 30, count: 1, spent: 30 }] },
    });
    expect(await screen.findByText("Registry parse")).toBeInTheDocument();
    expect(screen.queryByText(/cache hits free/)).not.toBeInTheDocument();
  });
});
