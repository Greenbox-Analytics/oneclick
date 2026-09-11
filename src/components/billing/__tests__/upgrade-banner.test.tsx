// The dashboard strip is where a user who abandoned Stripe Checkout lands.
// With a remembered plan it offers to finish that checkout; without one it is
// the pre-existing free-tier nudge. The pending variant is gated on
// entitlements (server truth) — never on the remembered intent alone — so it
// can never tell a paying user they "didn't finish".
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { toast } from "sonner";
import type { Entitlements } from "@/hooks/useEntitlements";
import { readPendingPlan, stashPendingPlan } from "@/lib/pendingPlan";

const h = vi.hoisted(() => ({
  ent: { data: undefined as Partial<Entitlements> | undefined },
  ctx: null as { plan: string } | null,
  navigate: vi.fn(),
  mutateAsync: vi.fn(),
  handleConflict: vi.fn(),
}));

vi.mock("react-router-dom", () => ({ useNavigate: () => h.navigate }));
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({ user: { id: "u1", email: "u1@example.test" } }),
}));
vi.mock("@/hooks/useEntitlements", () => ({ useEntitlements: () => ({ data: h.ent.data }) }));
vi.mock("@/hooks/useBilling", () => ({
  useCreateCheckoutSession: () => ({ mutateAsync: h.mutateAsync, isPending: false }),
  useSubscriptionConflict: () => h.handleConflict,
}));
vi.mock("@/hooks/useAnalyticsContext", () => ({
  peekCachedAnalyticsContext: () => h.ctx,
}));
vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { success: vi.fn(), info: vi.fn(), error: vi.fn() }),
}));

const { UpgradeBanner } = await import("@/components/billing/UpgradeBanner");

const DISMISS_KEY = "msanii.upgrade_banner.dismissed.u1";
const PENDING_TEXT = /You didn't finish upgrading to Basic \(monthly\)/;
const GENERIC_TEXT = /You're on the Free plan/;

beforeEach(() => {
  localStorage.clear();
  h.ent.data = { tier: "free", degraded: false };
  h.ctx = { plan: "free" };
  h.mutateAsync.mockResolvedValue("https://checkout.stripe.test/cs_1");
  h.handleConflict.mockResolvedValue(false);
});

afterEach(() => {
  cleanup();
});

describe("UpgradeBanner — remembered checkout", () => {
  it("offers to finish the abandoned checkout and resumes it with the same plan", async () => {
    stashPendingPlan("u1", "basic_monthly");
    render(<UpgradeBanner />);

    expect(screen.getByText(PENDING_TEXT)).toBeInTheDocument();
    expect(screen.queryByText(GENERIC_TEXT)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Finish upgrading" }));
    await waitFor(() => expect(h.mutateAsync).toHaveBeenCalledWith("basic_monthly"));
    // Still remembered: if they abandon again, the nudge comes back.
    expect(readPendingPlan("u1")?.plan).toBe("basic_monthly");
  });

  it("hands a subscription conflict on resume to the shared handler, not the error toast", async () => {
    // Entitlements said free, the server knows better (a second tab, the 60s cache).
    stashPendingPlan("u1", "basic_monthly");
    const err = new Error("You already have an active subscription.");
    h.mutateAsync.mockRejectedValue(err);
    h.handleConflict.mockResolvedValue(true);
    render(<UpgradeBanner />);

    fireEvent.click(screen.getByRole("button", { name: "Finish upgrading" }));

    await waitFor(() => expect(h.handleConflict).toHaveBeenCalledWith(err));
    expect(vi.mocked(toast.error)).not.toHaveBeenCalled();
  });

  it("dismiss forgets the plan without setting the permanent free-plan dismiss flag", () => {
    stashPendingPlan("u1", "basic_monthly");
    render(<UpgradeBanner />);

    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));

    expect(screen.queryByText(PENDING_TEXT)).not.toBeInTheDocument();
    // Dismissing one strip must not surface another underneath it.
    expect(screen.queryByText(GENERIC_TEXT)).not.toBeInTheDocument();
    expect(readPendingPlan("u1")).toBeNull();
    expect(localStorage.getItem(DISMISS_KEY)).toBeNull();
  });

  it("shows even when the generic strip was dismissed long ago", () => {
    localStorage.setItem(DISMISS_KEY, "1");
    stashPendingPlan("u1", "basic_monthly");
    render(<UpgradeBanner />);
    expect(screen.getByText(PENDING_TEXT)).toBeInTheDocument();
  });

  it("never nags a paying user — clears the memory when entitlements say paid", async () => {
    stashPendingPlan("u1", "basic_monthly");
    h.ent.data = { tier: "basic", degraded: false };
    const { container } = render(<UpgradeBanner />);
    expect(container).toBeEmptyDOMElement();
    await waitFor(() => expect(readPendingPlan("u1")).toBeNull());
  });

  it("stays quiet, and keeps the memory, while entitlements are still loading", () => {
    stashPendingPlan("u1", "basic_monthly");
    h.ent.data = undefined;
    const { container } = render(<UpgradeBanner />);
    expect(container).toBeEmptyDOMElement();
    expect(readPendingPlan("u1")?.plan).toBe("basic_monthly");
  });

  it("stays quiet, and keeps the memory, on a degraded entitlements read", () => {
    stashPendingPlan("u1", "basic_monthly");
    h.ent.data = { tier: "free", degraded: true };
    const { container } = render(<UpgradeBanner />);
    expect(container).toBeEmptyDOMElement();
    expect(readPendingPlan("u1")?.plan).toBe("basic_monthly");
  });
});

describe("UpgradeBanner — no remembered checkout (pre-existing behaviour)", () => {
  it("shows the free-plan nudge from the analytics cache", () => {
    render(<UpgradeBanner />);
    expect(screen.getByText(GENERIC_TEXT)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /See what Pro unlocks/ }));
    expect(h.navigate).toHaveBeenCalledWith("/pricing");
  });

  it("honours the permanent dismiss flag", () => {
    localStorage.setItem(DISMISS_KEY, "1");
    const { container } = render(<UpgradeBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing before the analytics cache is populated", () => {
    h.ctx = null;
    const { container } = render(<UpgradeBanner />);
    expect(container).toBeEmptyDOMElement();
  });
});
