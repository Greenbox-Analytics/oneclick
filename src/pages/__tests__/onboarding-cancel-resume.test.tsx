// A user who picks Basic on the onboarding plan step and then backs out of
// Stripe Checkout returns to /onboarding?upgrade=cancelled. Onboarding saves
// the profile (onboarding_completed = true) BEFORE the redirect on purpose,
// so the mount-time "already onboarded → /dashboard" bounce used to win over
// the cancel handler: the user landed on the dashboard as Free with the plan
// step unreachable. These tests pin the resume.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { toast } from "sonner";
import { readPendingPlan, stashPendingPlan } from "@/lib/pendingPlan";

const h = vi.hoisted(() => ({
  profileRow: {} as Record<string, unknown>,
  upsert: vi.fn(),
  mutateAsync: vi.fn(),
}));

vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({ user: { id: "u1", email: "u1@example.test" } }),
}));
vi.mock("@/integrations/supabase/client", () => ({
  supabase: {
    from: () => ({
      select: () => ({ eq: () => ({ single: () => Promise.resolve({ data: h.profileRow }) }) }),
      upsert: (row: unknown) => h.upsert(row),
    }),
  },
}));
vi.mock("@/hooks/useAnalytics", () => ({
  useAnalytics: () => ({ captureOnboardingStepCompleted: vi.fn(), captureOnboardingFinished: vi.fn() }),
}));
vi.mock("@/hooks/useBilling", () => ({
  useCreateCheckoutSession: () => ({ mutateAsync: h.mutateAsync }),
  useSubscriptionConflict: () => async () => false,
}));
vi.mock("@/hooks/useOrgs", () => ({
  useOrgInvitePreview: () => ({ isSuccess: false, isPending: false, data: undefined }),
}));
vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { success: vi.fn(), info: vi.fn(), error: vi.fn() }),
}));

const { default: Onboarding } = await import("@/pages/Onboarding");

const Probe = () => {
  const { pathname, search } = useLocation();
  return <div data-testid="loc">{pathname + search}</div>;
};

function renderAt(url: string) {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <Probe />
      <Routes>
        <Route path="/onboarding" element={<Onboarding />} />
        <Route path="/dashboard" element={<div>DASHBOARD</div>} />
      </Routes>
    </MemoryRouter>,
  );
}

const COMPLETED_PROFILE = {
  first_name: "Ada",
  last_name: "Lovelace",
  given_name: null,
  full_name: "Ada Lovelace",
  company: "Analytical Engines",
  role: "manager",
  onboarding_completed: true,
};

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  h.profileRow = { ...COMPLETED_PROFILE };
  h.upsert.mockResolvedValue({ error: null });
  h.mutateAsync.mockResolvedValue("https://checkout.stripe.test/cs_1");
});

afterEach(() => {
  cleanup();
});

describe("Onboarding — returning from a cancelled checkout", () => {
  it("resumes on the plan step instead of bouncing to the dashboard, and cleans the URL", async () => {
    renderAt("/onboarding?upgrade=cancelled");

    expect(await screen.findByText("Pick your plan")).toBeInTheDocument();
    expect(screen.queryByText("DASHBOARD")).not.toBeInTheDocument();
    expect(screen.getByTestId("loc")).toHaveTextContent("/onboarding");
    expect(screen.getByTestId("loc")).not.toHaveTextContent("upgrade=");

    expect(vi.mocked(toast.info)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(toast.info).mock.calls[0][0]).toMatch(/cancelled/i);
    expect(vi.mocked(toast.info).mock.calls[0][0]).toMatch(/haven't been charged/i);
  });

  it("keeps the saved role when the user then continues with Free (no data loss on re-save)", async () => {
    stashPendingPlan("u1", "basic_monthly");
    renderAt("/onboarding?upgrade=cancelled");
    await screen.findByText("Pick your plan");

    fireEvent.click(screen.getByRole("button", { name: "Continue with Free" }));

    await waitFor(() => expect(h.upsert).toHaveBeenCalledTimes(1));
    expect(h.upsert.mock.calls[0][0]).toEqual(
      expect.objectContaining({
        id: "u1",
        role: "manager",
        company: "Analytical Engines",
        first_name: "Ada",
        onboarding_completed: true,
      }),
    );
    // An explicit Free choice ends the "finish upgrading" nudge.
    expect(readPendingPlan("u1")).toBeNull();
  });

  it("remembers the plan when the user heads back to Stripe", async () => {
    renderAt("/onboarding?upgrade=cancelled");
    await screen.findByText("Pick your plan");

    fireEvent.click(screen.getByRole("button", { name: "Annual save ~20%" }));
    fireEvent.click(screen.getByRole("button", { name: /Upgrade to Basic/ }));

    await waitFor(() =>
      expect(h.mutateAsync).toHaveBeenCalledWith(
        expect.objectContaining({ plan: "basic_annual", cancel_path: "/onboarding?upgrade=cancelled" }),
      ),
    );
    await waitFor(() => expect(readPendingPlan("u1")?.plan).toBe("basic_annual"));
  });

  it("does not remember a plan when checkout could not be started", async () => {
    h.mutateAsync.mockRejectedValue(new Error("boom"));
    renderAt("/onboarding?upgrade=cancelled");
    await screen.findByText("Pick your plan");

    fireEvent.click(screen.getByRole("button", { name: /Upgrade to Basic/ }));

    await waitFor(() => expect(vi.mocked(toast.error)).toHaveBeenCalled());
    expect(readPendingPlan("u1")).toBeNull();
  });

  it("still sends an already-onboarded user to the dashboard when there is no cancel param", async () => {
    renderAt("/onboarding");
    expect(await screen.findByText("DASHBOARD")).toBeInTheDocument();
    expect(vi.mocked(toast.info)).not.toHaveBeenCalled();
  });
});
