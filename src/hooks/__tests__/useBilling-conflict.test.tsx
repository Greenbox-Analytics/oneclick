// The checkout endpoint refuses a second subscription with a 409 — one live
// personal subscription per user. Every checkout caller funnels that error
// through useSubscriptionConflict: forget the remembered plan, refetch
// entitlements, explain, and open the Customer Portal, where plan changes for
// a subscriber actually happen.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { toast } from "sonner";
import { ApiError } from "@/lib/apiFetch";
import { readPendingPlan, stashPendingPlan } from "@/lib/pendingPlan";

const h = vi.hoisted(() => ({
  apiFetch: vi.fn(),
}));

vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({ user: { id: "u1", email: "u1@example.test" } }),
}));
vi.mock("@/lib/apiFetch", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/apiFetch")>();
  return { ...actual, apiFetch: h.apiFetch };
});
vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { success: vi.fn(), info: vi.fn(), error: vi.fn() }),
}));

const { isSubscriptionConflict, useSubscriptionConflict } = await import("@/hooks/useBilling");

const conflict = () => {
  const err = new ApiError("You already have an active subscription.", 409);
  err.detail = { code: "subscription_exists", reason: "You already have an active subscription.", tier: "basic" };
  return err;
};

const wrapperWith =
  (qc: QueryClient) =>
  ({ children }: { children: ReactNode }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider>;

beforeEach(() => {
  localStorage.clear();
  h.apiFetch.mockResolvedValue({ url: "https://billing.stripe.test/p/session_1" });
});

afterEach(() => {
  cleanup();
});

describe("isSubscriptionConflict", () => {
  it("recognises the endpoint's 409 by its code", () => {
    expect(isSubscriptionConflict(conflict())).toBe(true);
  });

  it("ignores every other error", () => {
    const plain409 = new ApiError("Conflict", 409);
    const wrongStatus = new ApiError("Out of credits", 402);
    wrongStatus.detail = { code: "subscription_exists" };
    expect(isSubscriptionConflict(plain409)).toBe(false);
    expect(isSubscriptionConflict(wrongStatus)).toBe(false);
    expect(isSubscriptionConflict(new Error("boom"))).toBe(false);
    expect(isSubscriptionConflict(undefined)).toBe(false);
  });
});

describe("useSubscriptionConflict", () => {
  it("forgets the plan, refetches entitlements, explains, and opens the portal", async () => {
    stashPendingPlan("u1", "pro_monthly");
    const qc = new QueryClient();
    const invalidate = vi.spyOn(qc, "invalidateQueries");
    const { result } = renderHook(() => useSubscriptionConflict(), { wrapper: wrapperWith(qc) });

    await expect(result.current(conflict())).resolves.toBe(true);

    expect(readPendingPlan("u1")).toBeNull();
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["entitlements", "u1"] });
    expect(vi.mocked(toast.info)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(toast.info).mock.calls[0][0]).toMatch(/already have an active subscription/i);
    await waitFor(() =>
      expect(h.apiFetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/billing\/create-portal-session$/),
        expect.objectContaining({ method: "POST" }),
      ),
    );
  });

  it("leaves anything else to the caller", async () => {
    stashPendingPlan("u1", "pro_monthly");
    const qc = new QueryClient();
    const { result } = renderHook(() => useSubscriptionConflict(), { wrapper: wrapperWith(qc) });

    await expect(result.current(new Error("boom"))).resolves.toBe(false);

    expect(readPendingPlan("u1")?.plan).toBe("pro_monthly");
    expect(vi.mocked(toast.info)).not.toHaveBeenCalled();
    expect(h.apiFetch).not.toHaveBeenCalled();
  });
});
