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
const PERSONAL_ENT = { tier: "pro", credits: {}, caps: {}, usage: {} };
let entData: Record<string, unknown> = PERSONAL_ENT;

vi.mock("react-router-dom", () => ({ useNavigate: () => vi.fn() }));

vi.mock("@/hooks/useCreditUsage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/hooks/useCreditUsage")>()),
  useCreditUsage: () => ({ data: usageData, isLoading: false }),
}));

vi.mock("@/hooks/useEntitlements", () => ({
  useEntitlements: () => ({ data: entData }),
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
  entData = PERSONAL_ENT;
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

  describe("org billing context", () => {
    const asOrgMember = (credits: Record<string, unknown>) => {
      entData = {
        tier: "free",
        caps: {},
        usage: {},
        billingContext: { type: "org", orgId: "o1", orgName: "Acme Records", role: "member" },
        credits,
      };
    };

    it("shows a member their own per-tool spend", async () => {
      // The pool is shared, so where a member's OWN credits went is the only
      // spend they can act on. The backend already scopes /me/credits/usage to
      // the caller (metadata.org_member_id); this card used to drop it.
      asOrgMember({ memberCap: 2000, memberCapUsed: 50, balance: null });
      renderCard({
        usage: {
          enabled: true,
          memberCap: 2000,
          memberCapUsed: 50,
          tools: [
            { action: "oneclick_run", price: 30, count: 1, spent: 30 },
            { action: "zoe_message", price: 5, count: 4, spent: 20 },
          ],
        },
      });

      expect(await screen.findByText("OneClick run")).toBeInTheDocument();
      expect(screen.getByText("Zoe message")).toBeInTheDocument();
      expect(screen.getByText("30 cr")).toBeInTheDocument();
      expect(screen.getByText("20 cr")).toBeInTheDocument();
    });

    it("still hides the pool balance from a plain member", async () => {
      // Redaction is None/absent, never 0 — a 0 reads as "the org is out of
      // credits", which a member would act on. Adding the breakdown must not
      // smuggle the pool in alongside it.
      asOrgMember({ memberCap: 2000, memberCapUsed: 50, balance: null });
      renderCard({
        usage: {
          enabled: true,
          memberCap: 2000,
          memberCapUsed: 50,
          balance: null,
          bundleBalance: null,
          reserveBalance: null,
          tools: [{ action: "oneclick_run", price: 30, count: 1, spent: 30 }],
        },
      });

      expect(await screen.findByText("OneClick run")).toBeInTheDocument();
      expect(screen.queryByText(/in the shared pool/)).not.toBeInTheDocument();
      expect(screen.queryByText(/0 credits/)).not.toBeInTheDocument();
    });
  });
});
