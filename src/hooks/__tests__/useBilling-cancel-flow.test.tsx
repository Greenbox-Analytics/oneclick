// "Cancel plan" opens the Customer Portal's cancel FLOW — the only Stripe
// surface that redirects on its own after a cancel (the Portal home always
// needs a "Return" click). The hook treats the endpoint's 409 as "the card was
// stale" (refetch, explain, no portal) and degrades to the Portal home when
// Stripe refuses the flow for any other reason.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { toast } from "sonner";
import { ApiError } from "@/lib/apiFetch";

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

const { useOpenCancelFlow } = await import("@/hooks/useBilling");

const apiError = (status: number, detail: unknown) => {
  const message = typeof detail === "string" ? detail : (detail as { reason: string }).reason;
  const err = new ApiError(message, status);
  err.detail = detail;
  return err;
};

const wrapperWith =
  (qc: QueryClient) =>
  ({ children }: { children: ReactNode }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider>;

const PORTAL_ENDPOINT = /\/billing\/create-portal-session$/;

beforeEach(() => {
  h.apiFetch.mockResolvedValue({ url: "https://billing.stripe.test/p/cancel_1" });
});

afterEach(() => {
  cleanup();
});

describe("useOpenCancelFlow", () => {
  it("asks for the cancel flow, not the portal home", async () => {
    const { result } = renderHook(() => useOpenCancelFlow(), { wrapper: wrapperWith(new QueryClient()) });

    await result.current.openCancelFlow();

    expect(h.apiFetch).toHaveBeenCalledTimes(1);
    expect(h.apiFetch).toHaveBeenCalledWith(
      expect.stringMatching(PORTAL_ENDPOINT),
      expect.objectContaining({ method: "POST", body: JSON.stringify({ flow: "subscription_cancel" }) }),
    );
    expect(vi.mocked(toast.info)).not.toHaveBeenCalled();
    expect(vi.mocked(toast.error)).not.toHaveBeenCalled();
  });

  it("treats a 409 as a stale card: refetch, explain, no portal", async () => {
    const reason = "Your subscription is already set to end at the close of this billing period.";
    h.apiFetch.mockRejectedValueOnce(apiError(409, { code: "already_canceling", reason }));
    const qc = new QueryClient();
    const invalidate = vi.spyOn(qc, "invalidateQueries");
    const { result } = renderHook(() => useOpenCancelFlow(), { wrapper: wrapperWith(qc) });

    await result.current.openCancelFlow();

    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["entitlements", "u1"] });
    expect(vi.mocked(toast.info)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(toast.info).mock.calls[0][0]).toBe(reason);
    expect(h.apiFetch).toHaveBeenCalledTimes(1); // no fallback to the portal home
  });

  it("says so when there is no billing portal at all", async () => {
    h.apiFetch.mockRejectedValueOnce(apiError(404, "No Stripe subscription on file."));
    const { result } = renderHook(() => useOpenCancelFlow(), { wrapper: wrapperWith(new QueryClient()) });

    await result.current.openCancelFlow();

    expect(vi.mocked(toast.error)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(toast.info)).not.toHaveBeenCalled();
    expect(h.apiFetch).toHaveBeenCalledTimes(1);
  });

  it("falls back to the portal home when Stripe refuses the flow", async () => {
    h.apiFetch.mockRejectedValueOnce(
      apiError(502, { code: "portal_flow_unavailable", reason: "Couldn't open the cancel page." }),
    );
    const { result } = renderHook(() => useOpenCancelFlow(), { wrapper: wrapperWith(new QueryClient()) });

    await result.current.openCancelFlow();

    await waitFor(() => expect(h.apiFetch).toHaveBeenCalledTimes(2));
    expect(vi.mocked(toast.info)).toHaveBeenCalledTimes(1);
    expect(vi.mocked(toast.info).mock.calls[0][0]).toMatch(/billing portal instead/i);
    const [url, init] = h.apiFetch.mock.calls[1];
    expect(url).toMatch(PORTAL_ENDPOINT);
    expect(init).toEqual(expect.objectContaining({ method: "POST" }));
    expect(init).not.toHaveProperty("body"); // the Portal home: body-less, as always
  });
});
