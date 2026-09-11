// The return from the Stripe Customer Portal, on /profile. ?portal=canceled is
// where the Portal's cancel FLOW redirects the moment the user confirms — the
// cancel already happened, only the row may be behind — so the hook must strip
// the URL, mirror the subscription from Stripe (POST /billing/sync-subscription)
// and refetch, keep polling as the fallback, and always take the overlay down.
// ?portal=return is the Portal home's "Return to Msanii" link, after which
// nothing is known: the same sync, then a short silent refetch burst.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StrictMode, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, useLocation } from "react-router-dom";
import { act, cleanup, renderHook } from "@testing-library/react";
import { toast } from "sonner";
import type { Entitlements } from "@/hooks/useEntitlements";

const h = vi.hoisted(() => ({
  ent: { data: undefined as Partial<Entitlements> | undefined },
  sync: vi.fn(),
}));

vi.mock("@/hooks/useEntitlements", () => ({ useEntitlements: () => ({ data: h.ent.data }) }));
vi.mock("@/hooks/useBilling", () => ({ useSyncSubscription: () => ({ mutateAsync: h.sync }) }));
vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { success: vi.fn(), info: vi.fn(), error: vi.fn() }),
}));

const { usePortalReturn } = await import("@/hooks/usePortalReturn");

const CANCELED_URL = "/profile?portal=canceled";
const RETURN_URL = "/profile?portal=return";
const ENTITLEMENTS = { queryKey: ["entitlements"] };
// A poll tick must never cancel the fetch it is waiting for.
const POLL_OPTS = { cancelRefetch: false };

type Sub = Entitlements["subscription"];
const sub = (over: Partial<Sub> = {}): Sub => ({
  stripeSubscriptionId: "sub_1",
  stripePriceId: "price_1",
  currentPeriodEnd: "2026-10-10T12:00:00.000Z",
  cancelAtPeriodEnd: false,
  planPeriod: "monthly",
  ...over,
});
const LIVE: Partial<Entitlements> = { tier: "basic", status: "active", degraded: false, subscription: sub() };
const ENDING: Partial<Entitlements> = { ...LIVE, subscription: sub({ cancelAtPeriodEnd: true }) };
const FREE_AGAIN: Partial<Entitlements> = {
  tier: "free",
  status: "canceled",
  degraded: false,
  subscription: sub({ stripeSubscriptionId: null, stripePriceId: null, currentPeriodEnd: null, planPeriod: null }),
};

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
      const ret = usePortalReturn();
      seen.push(ret.syncing);
      return { ret, search: useLocation().search };
    },
    { wrapper: Wrapper },
  );
  return { ...hook, invalidate, seen };
}

// Let the sync's promise settle (microtasks only; the fake clock stays put).
const flushSync = () => act(async () => {});

const success = vi.mocked(toast.success);
const info = vi.mocked(toast.info);

beforeEach(() => {
  vi.useFakeTimers();
  h.ent.data = LIVE;
  h.sync.mockReset().mockResolvedValue({ synced: true });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("usePortalReturn — ?portal=canceled", () => {
  it("strips the signal and keeps polling after the strip", () => {
    const { result, invalidate } = setup(CANCELED_URL);
    expect(result.current.search).toBe("");
    expect(result.current.ret.syncing).toBe(true);
    invalidate.mockClear();
    act(() => vi.advanceTimersByTime(3_000));
    expect(invalidate).toHaveBeenCalledTimes(3);
    expect(invalidate).toHaveBeenCalledWith(ENTITLEMENTS, POLL_OPTS);
    expect(result.current.ret.syncing).toBe(true);
  });

  it("mirrors the subscription from Stripe first, then refetches without waiting for a tick", async () => {
    const { invalidate } = setup(CANCELED_URL);
    expect(h.sync).toHaveBeenCalledTimes(1);
    expect(invalidate).not.toHaveBeenCalled();

    await flushSync();

    // The post-sync refetch is the "fetch it now" kind: it MAY cancel an
    // in-flight fetch, which by then predates the row write.
    expect(invalidate).toHaveBeenCalledTimes(1);
    expect(invalidate).toHaveBeenLastCalledWith(ENTITLEMENTS);
  });

  it("keeps polling — the webhook fallback — when the sync fails", async () => {
    h.sync.mockRejectedValueOnce(new Error("502"));
    const { result, invalidate } = setup(CANCELED_URL);
    await flushSync();
    expect(invalidate).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(3_000));
    expect(invalidate).toHaveBeenCalledTimes(3);
    expect(invalidate).toHaveBeenCalledWith(ENTITLEMENTS, POLL_OPTS);
    expect(result.current.ret.syncing).toBe(true);
    expect(vi.mocked(toast.error)).not.toHaveBeenCalled();
  });

  it("takes the overlay down and names the end date once the row says so", () => {
    const { result, rerender } = setup(CANCELED_URL);
    act(() => vi.advanceTimersByTime(2_000));
    expect(result.current.ret.syncing).toBe(true);

    h.ent.data = ENDING;
    rerender();
    act(() => vi.advanceTimersByTime(1_000));

    expect(result.current.ret.syncing).toBe(false);
    expect(success).toHaveBeenCalledTimes(1);
    expect(success.mock.calls[0][0]).toMatch(/^Your Basic plan is set to end on Oct 10, 2026\./);
    expect(success.mock.calls[0][0]).toMatch(/full access until then/);

    // Settled means settled: the timeout must not fire a second message later.
    act(() => vi.advanceTimersByTime(20_000));
    expect(success).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
  });

  it("recognises an immediate cancel (a portal configured that way) as done", () => {
    const { result, rerender } = setup(CANCELED_URL);
    h.ent.data = FREE_AGAIN;
    rerender();
    act(() => vi.advanceTimersByTime(1_000));

    expect(result.current.ret.syncing).toBe(false);
    expect(info).toHaveBeenCalledTimes(1);
    expect(info.mock.calls[0][0]).toMatch(/has been canceled/i);
    expect(success).not.toHaveBeenCalled();
  });

  it("does not settle on a missing or degraded read — that would look like an immediate cancel", () => {
    h.ent.data = undefined;
    const { result, rerender } = setup(CANCELED_URL);
    act(() => vi.advanceTimersByTime(2_000));
    expect(result.current.ret.syncing).toBe(true);

    h.ent.data = { ...FREE_AGAIN, degraded: true };
    rerender();
    act(() => vi.advanceTimersByTime(2_000));
    expect(result.current.ret.syncing).toBe(true);
    expect(info).not.toHaveBeenCalled();
    expect(success).not.toHaveBeenCalled();
  });

  it("gives up after 10s without implying the cancel failed", () => {
    const { result, invalidate } = setup(CANCELED_URL);
    act(() => vi.advanceTimersByTime(10_000));
    expect(result.current.ret.syncing).toBe(false);
    expect(info).toHaveBeenCalledTimes(1);
    expect(info.mock.calls[0][0]).toMatch(/cancellation went through/i);
    expect(info.mock.calls[0][0]).not.toMatch(/fail|error|try again/i);
    expect(success).not.toHaveBeenCalled();
    expect(invalidate.mock.calls.length).toBeGreaterThanOrEqual(10);

    act(() => vi.advanceTimersByTime(20_000));
    expect(info).toHaveBeenCalledTimes(1);
  });

  it("never shows the overlay, nor syncs, when the read already says the plan is ending", () => {
    h.ent.data = ENDING;
    const { result, seen } = setup(CANCELED_URL);
    expect(seen).not.toContain(true);
    expect(result.current.ret.syncing).toBe(false);
    expect(result.current.search).toBe("");
    expect(success).toHaveBeenCalledTimes(1);
    expect(h.sync).not.toHaveBeenCalled();

    act(() => vi.advanceTimersByTime(20_000));
    expect(success).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
  });

  it("settles exactly once under StrictMode's double effect run", () => {
    h.ent.data = ENDING;
    setup(CANCELED_URL, { strict: true });
    act(() => vi.advanceTimersByTime(15_000));
    expect(success).toHaveBeenCalledTimes(1);
    expect(info).not.toHaveBeenCalled();
  });

  it("syncs exactly once under StrictMode's double effect run", () => {
    setup(CANCELED_URL, { strict: true });
    expect(h.sync).toHaveBeenCalledTimes(1);
  });
});

describe("usePortalReturn — ?portal=return", () => {
  it("mirrors the subscription from Stripe, then refetches quietly for a few seconds: no overlay, no toast", async () => {
    const { result, invalidate, seen } = setup(RETURN_URL);
    expect(result.current.search).toBe("");
    expect(h.sync).toHaveBeenCalledTimes(1);

    await flushSync();
    expect(invalidate).toHaveBeenCalledTimes(1);
    expect(invalidate).toHaveBeenLastCalledWith(ENTITLEMENTS);

    act(() => vi.advanceTimersByTime(30_000));
    expect(seen).not.toContain(true);
    expect(invalidate).toHaveBeenCalledTimes(6);
    expect(invalidate).toHaveBeenLastCalledWith(ENTITLEMENTS, POLL_OPTS);
    expect(success).not.toHaveBeenCalled();
    expect(info).not.toHaveBeenCalled();
  });

  it("still bursts when the sync fails", async () => {
    h.sync.mockRejectedValueOnce(new Error("502"));
    const { invalidate } = setup(RETURN_URL);
    await flushSync();
    act(() => vi.advanceTimersByTime(30_000));
    expect(invalidate).toHaveBeenCalledTimes(5);
    expect(vi.mocked(toast.error)).not.toHaveBeenCalled();
  });
});

describe("usePortalReturn — anything else", () => {
  it("does nothing without the signal", () => {
    const { result, invalidate } = setup("/profile?tab=usage");
    act(() => vi.advanceTimersByTime(15_000));
    expect(result.current.ret.syncing).toBe(false);
    expect(result.current.search).toBe("?tab=usage");
    expect(invalidate).not.toHaveBeenCalled();
    expect(h.sync).not.toHaveBeenCalled();
    expect(success).not.toHaveBeenCalled();
    expect(info).not.toHaveBeenCalled();
  });

  it("ignores a value it doesn't know", () => {
    const { result, invalidate } = setup("/profile?portal=bogus");
    act(() => vi.advanceTimersByTime(15_000));
    expect(result.current.ret.syncing).toBe(false);
    expect(invalidate).not.toHaveBeenCalled();
    expect(h.sync).not.toHaveBeenCalled();
    expect(success).not.toHaveBeenCalled();
    expect(info).not.toHaveBeenCalled();
  });
});
