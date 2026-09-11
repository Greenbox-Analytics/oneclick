// /pricing is the backend's default cancel_url (?canceled=true). Nothing used
// to read it, so backing out of Stripe was silent and left the param in the
// URL. It is also a checkout entry point, so it remembers the chosen plan for
// the dashboard's "finish upgrading" nudge.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { MemoryRouter, useLocation } from "react-router-dom";
import { toast } from "sonner";
import { readPendingPlan, stashPendingPlan } from "@/lib/pendingPlan";

const h = vi.hoisted(() => ({
  user: { id: "u1", email: "u1@example.test" } as { id: string; email: string } | null,
  mutateAsync: vi.fn(),
}));

vi.mock("@/contexts/AuthContext", () => ({ useAuth: () => ({ user: h.user }) }));
// Plan state (Current plan / Switch) has its own suite: pricing-plan-state.test.tsx.
vi.mock("@/hooks/useEntitlements", () => ({ useEntitlements: () => ({ data: undefined }) }));
vi.mock("@/hooks/useBilling", () => ({
  useCreateCheckoutSession: () => ({ mutateAsync: h.mutateAsync, isPending: false }),
  useOpenBillingPortal: () => ({ openPortal: vi.fn(), isPending: false }),
  useSubscriptionConflict: () => async () => false,
}));
vi.mock("@/hooks/useAnalytics", () => ({
  useAnalytics: () => ({ captureCheckoutStarted: vi.fn() }),
}));
vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { success: vi.fn(), info: vi.fn(), error: vi.fn() }),
}));

const { default: Pricing } = await import("@/pages/Pricing");

const Probe = () => {
  const { pathname, search } = useLocation();
  return <div data-testid="loc">{pathname + search}</div>;
};

function renderAt(url: string) {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <Probe />
      <Pricing />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  localStorage.clear();
  h.user = { id: "u1", email: "u1@example.test" };
  h.mutateAsync.mockResolvedValue("https://checkout.stripe.test/cs_1");
});

afterEach(() => {
  cleanup();
});

describe("Pricing — cancelled checkout return", () => {
  it("acknowledges the cancel exactly once and strips the param", () => {
    renderAt("/pricing?canceled=true");
    // Two paid cards share the page: the handler must live in the page, not
    // the card, or the toast would fire twice.
    expect(vi.mocked(toast.info)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(toast.info).mock.calls[0][0]).toMatch(/cancelled/i);
    expect(vi.mocked(toast.info).mock.calls[0][0]).toMatch(/haven't been charged/i);
    expect(screen.getByTestId("loc")).toHaveTextContent("/pricing");
    expect(screen.getByTestId("loc")).not.toHaveTextContent("canceled");
  });

  it("says nothing on a plain visit", () => {
    renderAt("/pricing");
    expect(vi.mocked(toast.info)).not.toHaveBeenCalled();
  });
});

describe("Pricing — remembering the chosen plan", () => {
  it("stashes the plan once the checkout session exists", async () => {
    renderAt("/pricing");
    fireEvent.click(screen.getByRole("button", { name: "Upgrade to Pro" }));
    await waitFor(() => expect(h.mutateAsync).toHaveBeenCalledWith("pro_monthly"));
    await waitFor(() => expect(readPendingPlan("u1")?.plan).toBe("pro_monthly"));
  });

  it("stashes nothing when the session could not be created", async () => {
    h.mutateAsync.mockRejectedValue(new Error("boom"));
    renderAt("/pricing");
    fireEvent.click(screen.getByRole("button", { name: "Upgrade to Basic" }));
    await waitFor(() => expect(vi.mocked(toast.error)).toHaveBeenCalled());
    expect(readPendingPlan("u1")).toBeNull();
  });

  it("forgets the plan when a signed-in user chooses Free", () => {
    stashPendingPlan("u1", "basic_monthly");
    renderAt("/pricing");
    fireEvent.click(screen.getByRole("button", { name: "Continue with Free" }));
    expect(readPendingPlan("u1")).toBeNull();
    expect(screen.getByTestId("loc")).toHaveTextContent("/dashboard");
  });

  it("stashes nothing for a signed-out visitor (they are sent to sign in first)", () => {
    h.user = null;
    renderAt("/pricing");
    fireEvent.click(screen.getByRole("button", { name: "Upgrade to Basic" }));
    expect(h.mutateAsync).not.toHaveBeenCalled();
    expect(screen.getByTestId("loc")).toHaveTextContent("/auth?redirect=/pricing&plan=basic_monthly");
  });
});
