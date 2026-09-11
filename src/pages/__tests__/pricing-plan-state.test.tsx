// /pricing for someone who already pays: the current tier's card says so, and
// the other paid card switches plans through the Customer Portal — never a
// second Checkout, which the endpoint would 409 anyway (one live personal
// subscription per user). Everyone else keeps the Upgrade buttons.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { MemoryRouter } from "react-router-dom";
import { toast } from "sonner";
import type { Entitlements, EntitlementSubscription } from "@/hooks/useEntitlements";
import { readPendingPlan } from "@/lib/pendingPlan";

const h = vi.hoisted(() => ({
  user: { id: "u1", email: "u1@example.test" } as { id: string; email: string } | null,
  ent: { data: undefined as Partial<Entitlements> | undefined },
  mutateAsync: vi.fn(),
  openPortal: vi.fn(),
  handleConflict: vi.fn(),
}));

vi.mock("@/contexts/AuthContext", () => ({ useAuth: () => ({ user: h.user }) }));
vi.mock("@/hooks/useEntitlements", () => ({ useEntitlements: () => ({ data: h.ent.data }) }));
vi.mock("@/hooks/useBilling", () => ({
  useCreateCheckoutSession: () => ({ mutateAsync: h.mutateAsync, isPending: false }),
  useOpenBillingPortal: () => ({ openPortal: h.openPortal, isPending: false }),
  useSubscriptionConflict: () => h.handleConflict,
}));
vi.mock("@/hooks/useAnalytics", () => ({
  useAnalytics: () => ({ captureCheckoutStarted: vi.fn() }),
}));
vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { success: vi.fn(), info: vi.fn(), error: vi.fn() }),
}));

const { default: Pricing } = await import("@/pages/Pricing");

const NO_STRIPE: EntitlementSubscription = {
  stripeSubscriptionId: null,
  stripePriceId: null,
  currentPeriodEnd: null,
  cancelAtPeriodEnd: false,
  planPeriod: null,
};
const STRIPE: EntitlementSubscription = { ...NO_STRIPE, stripeSubscriptionId: "sub_1", stripePriceId: "price_1", planPeriod: "monthly" };

const ent = (tier: "free" | "basic" | "pro", extra: Partial<Entitlements> = {}): Partial<Entitlements> => ({
  tier,
  status: "active",
  degraded: false,
  subscription: STRIPE,
  ...extra,
});

function renderPricing() {
  return render(
    <MemoryRouter initialEntries={["/pricing"]}>
      <Pricing />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  localStorage.clear();
  h.user = { id: "u1", email: "u1@example.test" };
  h.ent.data = undefined;
  h.mutateAsync.mockResolvedValue("https://checkout.stripe.test/cs_1");
  h.openPortal.mockResolvedValue(undefined);
  h.handleConflict.mockResolvedValue(false);
});

afterEach(() => {
  cleanup();
});

describe("Pricing — a live subscriber", () => {
  it("marks the current tier and switches to the other one through the portal", async () => {
    h.ent.data = ent("basic");
    renderPricing();

    expect(screen.getByRole("button", { name: "Current plan" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: /Upgrade to/ })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Switch to Pro" }));
    await waitFor(() => expect(h.openPortal).toHaveBeenCalledTimes(1));
    expect(h.mutateAsync).not.toHaveBeenCalled();
    expect(readPendingPlan("u1")).toBeNull();
  });

  it("mirrors for a Pro subscriber", () => {
    h.ent.data = ent("pro");
    renderPricing();
    expect(screen.getByRole("button", { name: "Current plan" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Switch to Basic" })).toBeEnabled();
  });
});

describe("Pricing — everyone else keeps the Upgrade buttons", () => {
  it.each<[string, Partial<Entitlements> | undefined]>([
    ["an admin-granted Pro (no Stripe subscription)", ent("pro", { subscription: NO_STRIPE })],
    ["a canceled subscription whose id is still on the row", ent("free", { status: "canceled" })],
    ["a degraded entitlements read", ent("basic", { degraded: true })],
    ["a free user", ent("free", { subscription: NO_STRIPE })],
    ["entitlements still loading", undefined],
  ])("%s", (_label, data) => {
    h.ent.data = data;
    renderPricing();
    expect(screen.getByRole("button", { name: "Upgrade to Basic" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Upgrade to Pro" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Current plan" })).not.toBeInTheDocument();
  });

  it("a signed-out visitor", () => {
    h.user = null;
    renderPricing();
    expect(screen.getByRole("button", { name: "Upgrade to Basic" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Upgrade to Pro" })).toBeEnabled();
  });
});

describe("Pricing — the endpoint refuses a second subscription", () => {
  it("hands the 409 to the shared conflict handler instead of the generic error toast", async () => {
    // Stale entitlements: the page still shows Upgrade, the server knows better.
    h.ent.data = ent("free", { subscription: NO_STRIPE });
    const err = new Error("You already have an active subscription.");
    h.mutateAsync.mockRejectedValue(err);
    h.handleConflict.mockResolvedValue(true);
    renderPricing();

    fireEvent.click(screen.getByRole("button", { name: "Upgrade to Pro" }));

    await waitFor(() => expect(h.handleConflict).toHaveBeenCalledWith(err));
    expect(vi.mocked(toast.error)).not.toHaveBeenCalled();
    expect(readPendingPlan("u1")).toBeNull();
  });
});
