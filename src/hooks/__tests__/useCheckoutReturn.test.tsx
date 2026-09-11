// The subscription-checkout return: Stripe sends the browser back to
// /profile?stripe_session_id=...&welcome=true and the webhook that writes the
// subscription row races it. The hook must (1) strip the URL immediately,
// (2) keep polling AFTER the strip — the original bug was an effect keyed on
// the URL that killed its own timers the moment it cleaned the URL, leaving
// "Activating your subscription…" up forever — and (3) always take the
// overlay down, on success and on timeout alike.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StrictMode, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, useLocation } from "react-router-dom";
import { act, cleanup, renderHook } from "@testing-library/react";
import { toast } from "sonner";
import type { Entitlements } from "@/hooks/useEntitlements";
import { readPendingPlan, stashPendingPlan } from "@/lib/pendingPlan";

const h = vi.hoisted(() => ({
  ent: { data: undefined as Partial<Entitlements> | undefined },
  captureCheckoutCompleted: vi.fn(),
  refreshAnalyticsContext: vi.fn(),
}));

vi.mock("@/hooks/useEntitlements", () => ({ useEntitlements: () => ({ data: h.ent.data }) }));
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({ user: { id: "u1", email: "u1@example.test" } }),
}));
vi.mock("@/hooks/useAnalytics", () => ({
  useAnalytics: () => ({ captureCheckoutCompleted: h.captureCheckoutCompleted }),
}));
vi.mock("@/hooks/useAnalyticsContext", () => ({
  refreshAnalyticsContext: h.refreshAnalyticsContext,
}));
vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { success: vi.fn(), info: vi.fn(), error: vi.fn() }),
}));

const { useCheckoutReturn } = await import("@/hooks/useCheckoutReturn");

const RETURN_URL = "/profile?stripe_session_id=cs_test_1&welcome=true";

function setup(url: string, { strict = false } = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const invalidate = vi.spyOn(qc, "invalidateQueries");
  const seen: boolean[] = [];
  const Wrapper = ({ children }: { children: ReactNode }) => {
    const tree = (
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={[url]}>{children}</MemoryRouter>
      </QueryClientProvider>
    );
    return strict ? <StrictMode>{tree}</StrictMode> : tree;
  };
  const hook = renderHook(
    () => {
      const ret = useCheckoutReturn();
      seen.push(ret.activating);
      return { ret, search: useLocation().search };
    },
    { wrapper: Wrapper },
  );
  return { ...hook, invalidate, seen };
}

const success = vi.mocked(toast.success);
const info = vi.mocked(toast.info);

beforeEach(() => {
  vi.useFakeTimers();
  localStorage.clear();
  h.ent.data = { tier: "free" };
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("useCheckoutReturn", () => {
  it("strips the return params and forgets the pending plan the moment it lands", () => {
    stashPendingPlan("u1", "basic_monthly");
    const { result } = setup(RETURN_URL);
    expect(result.current.search).toBe("");
    // Reaching the success URL means Checkout completed. Even if the webhook
    // is late, no surface may offer a "Finish upgrading" that starts a SECOND
    // paid checkout — so the intent is dropped on arrival, not on "paid".
    expect(readPendingPlan("u1")).toBeNull();
  });

  it("keeps polling after it has cleaned the URL (the stuck-overlay regression)", () => {
    const { result, invalidate } = setup(RETURN_URL);
    expect(result.current.search).toBe("");
    expect(result.current.ret.activating).toBe(true);
    invalidate.mockClear();
    act(() => vi.advanceTimersByTime(3_000));
    expect(invalidate).toHaveBeenCalledTimes(3);
    // Never cancel the fetch a tick is waiting for (the invalidate default).
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["entitlements"] }, { cancelRefetch: false });
    expect(result.current.ret.activating).toBe(true);
  });

  it("takes the overlay down and welcomes the user once the webhook lands", () => {
    const { result, rerender } = setup(RETURN_URL);
    act(() => vi.advanceTimersByTime(2_000));
    expect(result.current.ret.activating).toBe(true);

    h.ent.data = {
      tier: "basic",
      subscription: { planPeriod: "annual" } as Entitlements["subscription"],
    };
    rerender();
    act(() => vi.advanceTimersByTime(1_000));

    expect(result.current.ret.activating).toBe(false);
    expect(success).toHaveBeenCalledTimes(1);
    expect(success.mock.calls[0][0]).toMatch(/^Welcome to Basic!/);
    expect(h.captureCheckoutCompleted).toHaveBeenCalledTimes(1);
    expect(h.captureCheckoutCompleted).toHaveBeenCalledWith("annual");
    expect(h.refreshAnalyticsContext).toHaveBeenCalledWith("u1", "u1@example.test");

    // Settled means settled: the timeout must not fire a second message later.
    act(() => vi.advanceTimersByTime(20_000));
    expect(success).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
  });

  it("gives up after 10s without implying the payment failed", () => {
    const { result, invalidate } = setup(RETURN_URL);
    act(() => vi.advanceTimersByTime(10_000));
    expect(result.current.ret.activating).toBe(false);
    expect(info).toHaveBeenCalledTimes(1);
    expect(info.mock.calls[0][0]).toMatch(/still activating/i);
    expect(info.mock.calls[0][0]).not.toMatch(/fail|declin|error/i);
    expect(success).not.toHaveBeenCalled();
    expect(h.captureCheckoutCompleted).not.toHaveBeenCalled();
    expect(invalidate.mock.calls.length).toBeGreaterThanOrEqual(10);

    act(() => vi.advanceTimersByTime(20_000));
    expect(info).toHaveBeenCalledTimes(1);
  });

  it("never shows the overlay when the tier is already paid on arrival", () => {
    h.ent.data = { tier: "pro" };
    const { result, seen } = setup(RETURN_URL);
    expect(seen).not.toContain(true);
    expect(result.current.ret.activating).toBe(false);
    expect(result.current.search).toBe("");
    expect(success).toHaveBeenCalledTimes(1);
    expect(success.mock.calls[0][0]).toMatch(/^Welcome to Pro!/);
    expect(h.captureCheckoutCompleted).toHaveBeenCalledWith("monthly");

    act(() => vi.advanceTimersByTime(20_000));
    expect(success).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
  });

  it("does nothing without the return params", () => {
    stashPendingPlan("u1", "basic_monthly");
    const { result, invalidate } = setup("/profile?tab=usage");
    act(() => vi.advanceTimersByTime(15_000));
    expect(result.current.ret.activating).toBe(false);
    expect(result.current.search).toBe("?tab=usage");
    expect(invalidate).not.toHaveBeenCalled();
    expect(success).not.toHaveBeenCalled();
    expect(info).not.toHaveBeenCalled();
    expect(readPendingPlan("u1")?.plan).toBe("basic_monthly");
  });

  it("ignores welcome=true without a session id, and a session id without welcome", () => {
    setup("/profile?welcome=true");
    setup("/profile?stripe_session_id=cs_1");
    act(() => vi.advanceTimersByTime(15_000));
    expect(success).not.toHaveBeenCalled();
    expect(info).not.toHaveBeenCalled();
  });

  it("settles exactly once under StrictMode's double effect run", () => {
    h.ent.data = { tier: "basic" };
    setup(RETURN_URL, { strict: true });
    act(() => vi.advanceTimersByTime(15_000));
    expect(success).toHaveBeenCalledTimes(1);
    expect(h.captureCheckoutCompleted).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
  });
});
