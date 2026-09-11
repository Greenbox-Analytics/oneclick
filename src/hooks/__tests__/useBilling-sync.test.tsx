// POST /billing/sync-subscription: mirror the user's live subscription from
// Stripe onto their row, so a Portal return doesn't wait on the webhook.
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
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

const { useSyncSubscription } = await import("@/hooks/useBilling");

const wrapper = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
);

afterEach(() => {
  cleanup();
  h.apiFetch.mockReset();
});

describe("useSyncSubscription", () => {
  it("posts to the sync endpoint with no body and resolves with the outcome", async () => {
    h.apiFetch.mockResolvedValue({ synced: true });
    const { result } = renderHook(() => useSyncSubscription(), { wrapper });

    await expect(result.current.mutateAsync()).resolves.toEqual({ synced: true });

    expect(h.apiFetch).toHaveBeenCalledTimes(1);
    const [url, init] = h.apiFetch.mock.calls[0];
    expect(url).toMatch(/\/billing\/sync-subscription$/);
    expect(init).toEqual({ method: "POST" });
  });

  it("rejects when Stripe can't be read, so the caller falls back to the webhook", async () => {
    const err = new ApiError("Couldn't reach Stripe.", 502);
    h.apiFetch.mockRejectedValue(err);
    const { result } = renderHook(() => useSyncSubscription(), { wrapper });

    await expect(result.current.mutateAsync()).rejects.toBe(err);
  });
});
