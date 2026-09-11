// Billing's plan card: the "Finish upgrading" resume button funnels the
// endpoint's 409 (one live personal subscription per user) through the shared
// conflict handler instead of the generic error toast.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { toast } from "sonner";
import type { Entitlements } from "@/hooks/useEntitlements";
import { readPendingPlan, stashPendingPlan } from "@/lib/pendingPlan";

const h = vi.hoisted(() => ({
  ent: { data: undefined as Partial<Entitlements> | undefined },
  mutateAsync: vi.fn(),
  openPortal: vi.fn(),
  openCancelFlow: vi.fn(),
  handleConflict: vi.fn(),
  navigate: vi.fn(),
}));

vi.mock("react-router-dom", () => ({ useNavigate: () => h.navigate }));
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({ user: { id: "u1", email: "u1@example.test" } }),
}));
vi.mock("@/hooks/useEntitlements", () => ({ useEntitlements: () => ({ data: h.ent.data }) }));
vi.mock("@/hooks/useBilling", () => ({
  useCreateCheckoutSession: () => ({ mutateAsync: h.mutateAsync, isPending: false }),
  useOpenBillingPortal: () => ({ openPortal: h.openPortal, isPending: false }),
  useOpenCancelFlow: () => ({ openCancelFlow: h.openCancelFlow, isPending: false }),
  useSubscriptionConflict: () => h.handleConflict,
}));
vi.mock("@/hooks/useAdmin", () => ({ useIsAdmin: () => ({ isAdmin: false }) }));
vi.mock("@/components/admin/AdminBadge", () => ({ AdminBadge: () => null }));
vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { success: vi.fn(), info: vi.fn(), error: vi.fn() }),
}));

const { PlanCard } = await import("@/components/billing/PlanCard");

const FREE: Partial<Entitlements> = {
  tier: "free",
  status: "active",
  degraded: false,
  subscription: {
    stripeSubscriptionId: null,
    stripePriceId: null,
    currentPeriodEnd: null,
    cancelAtPeriodEnd: false,
    planPeriod: null,
  },
};

beforeEach(() => {
  localStorage.clear();
  h.ent.data = FREE;
  h.mutateAsync.mockResolvedValue("https://checkout.stripe.test/cs_1");
  h.handleConflict.mockResolvedValue(false);
});

afterEach(() => {
  cleanup();
});

describe("PlanCard — finishing an abandoned checkout", () => {
  it("resumes with the remembered plan", async () => {
    stashPendingPlan("u1", "basic_monthly");
    render(<PlanCard />);

    fireEvent.click(screen.getByRole("button", { name: /Finish upgrading to Basic/ }));

    await waitFor(() => expect(h.mutateAsync).toHaveBeenCalledWith("basic_monthly"));
  });

  it("hands a subscription conflict to the shared handler, not the error toast", async () => {
    stashPendingPlan("u1", "basic_monthly");
    const err = new Error("You already have an active subscription.");
    h.mutateAsync.mockRejectedValue(err);
    h.handleConflict.mockResolvedValue(true);
    render(<PlanCard />);

    fireEvent.click(screen.getByRole("button", { name: /Finish upgrading to Basic/ }));

    await waitFor(() => expect(h.handleConflict).toHaveBeenCalledWith(err));
    expect(vi.mocked(toast.error)).not.toHaveBeenCalled();
  });

  it("still toasts on any other failure and keeps the memory", async () => {
    stashPendingPlan("u1", "basic_monthly");
    h.mutateAsync.mockRejectedValue(new Error("boom"));
    render(<PlanCard />);

    fireEvent.click(screen.getByRole("button", { name: /Finish upgrading to Basic/ }));

    await waitFor(() => expect(vi.mocked(toast.error)).toHaveBeenCalled());
    expect(readPendingPlan("u1")?.plan).toBe("basic_monthly");
  });
});

describe("PlanCard — the period row and Cancel plan", () => {
  const BASIC_LIVE: Partial<Entitlements> = {
    tier: "basic",
    status: "active",
    degraded: false,
    subscription: {
      stripeSubscriptionId: "sub_1",
      stripePriceId: "price_1",
      currentPeriodEnd: "2026-10-10T12:00:00.000Z",
      cancelAtPeriodEnd: false,
      planPeriod: "monthly",
    },
  };
  const BASIC_ENDING: Partial<Entitlements> = {
    ...BASIC_LIVE,
    subscription: { ...BASIC_LIVE.subscription!, cancelAtPeriodEnd: true },
  };

  afterEach(() => {
    vi.useRealTimers();
  });

  it("reads Renews and offers Cancel plan on a live subscription", () => {
    h.ent.data = BASIC_LIVE;
    render(<PlanCard />);

    expect(screen.getByText("Renews")).toBeTruthy();
    expect(screen.getByText("Oct 10, 2026")).toBeTruthy();
    expect(screen.getByRole("button", { name: /Manage subscription/ })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /Cancel plan/ }));
    expect(h.openCancelFlow).toHaveBeenCalledTimes(1);
  });

  it("reads Ends with the days left once the cancel is scheduled, and drops Cancel plan", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 8, 28, 12, 0, 0)); // Sep 28 — 12 calendar days before Oct 10
    h.ent.data = BASIC_ENDING;
    render(<PlanCard />);

    expect(screen.getByText("Ends")).toBeTruthy();
    expect(screen.queryByText("Renews")).toBeNull();
    expect(screen.getByText("Oct 10, 2026")).toBeTruthy();
    expect(screen.getByText(/12 days left/)).toBeTruthy();
    // Reactivating is "Renew plan" on the Portal home, so that button stays.
    expect(screen.getByRole("button", { name: /Manage subscription/ })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Cancel plan/ })).toBeNull();
  });

  it("offers no Cancel plan to an admin-granted tier or to Free", () => {
    h.ent.data = { ...BASIC_LIVE, tier: "pro", subscription: FREE.subscription }; // paid tier, no Stripe = admin grant
    const { unmount } = render(<PlanCard />);
    expect(screen.queryByRole("button", { name: /Cancel plan/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /Manage subscription/ })).toBeNull();
    unmount();

    h.ent.data = FREE;
    render(<PlanCard />);
    expect(screen.queryByRole("button", { name: /Cancel plan/ })).toBeNull();
  });
});
