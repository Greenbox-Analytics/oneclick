// The credit-purchase return (?topup=success|canceled). Same shape of bug as
// useCheckoutReturn: the hook strips its own param first, and an effect keyed
// on that param would tear down the poll it just armed — so the success toast
// could never fire. These tests pin the poll surviving the strip.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, useLocation } from "react-router-dom";
import { act, cleanup, renderHook } from "@testing-library/react";
import { toast } from "sonner";
import type { Entitlements } from "@/hooks/useEntitlements";

const h = vi.hoisted(() => ({
  ent: { data: undefined as Partial<Entitlements> | undefined },
  invalidateCreditSurfaces: vi.fn(),
}));

vi.mock("@/hooks/useEntitlements", () => ({ useEntitlements: () => ({ data: h.ent.data }) }));
vi.mock("@/hooks/useCreditUsage", () => ({ invalidateCreditSurfaces: h.invalidateCreditSurfaces }));
vi.mock("sonner", () => ({
  toast: Object.assign(vi.fn(), { success: vi.fn(), info: vi.fn(), error: vi.fn() }),
}));

const { useTopupReturn } = await import("@/hooks/useTopupReturn");

const reserve = (reserveBalance: number): Partial<Entitlements> =>
  ({ credits: { reserveBalance } }) as unknown as Partial<Entitlements>;

function setup(url: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[url]}>{children}</MemoryRouter>
    </QueryClientProvider>
  );
  return renderHook(
    () => {
      useTopupReturn();
      return { search: useLocation().search };
    },
    { wrapper: Wrapper },
  );
}

const success = vi.mocked(toast.success);
const info = vi.mocked(toast.info);

beforeEach(() => {
  vi.useFakeTimers();
  h.ent.data = reserve(100);
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("useTopupReturn", () => {
  it("strips the param, then still notices the grant landing after the strip", () => {
    const { result, rerender } = setup("/profile?topup=success");
    expect(result.current.search).toBe("");

    h.ent.data = reserve(400);
    rerender();
    act(() => vi.advanceTimersByTime(1_000));

    expect(success).toHaveBeenCalledTimes(1);
    expect(success.mock.calls[0][0]).toMatch(/Credits added — 300 credits/);

    act(() => vi.advanceTimersByTime(20_000));
    expect(success).toHaveBeenCalledTimes(1);
  });

  it("keeps refetching until the balance moves", () => {
    setup("/profile?topup=success");
    h.invalidateCreditSurfaces.mockClear();
    act(() => vi.advanceTimersByTime(3_000));
    expect(h.invalidateCreditSurfaces).toHaveBeenCalledTimes(3);
  });

  it("acknowledges the payment on timeout without claiming failure", () => {
    setup("/profile?topup=success");
    act(() => vi.advanceTimersByTime(10_000));
    expect(success).toHaveBeenCalledTimes(1);
    expect(success.mock.calls[0][0]).toMatch(/Payment received/);
  });

  it("toasts a cancel and strips the param", () => {
    const { result } = setup("/teams?topup=canceled");
    expect(result.current.search).toBe("");
    expect(info).toHaveBeenCalledTimes(1);
    expect(info.mock.calls[0][0]).toMatch(/cancelled/i);
    act(() => vi.advanceTimersByTime(15_000));
    expect(success).not.toHaveBeenCalled();
  });

  it("does nothing without the param", () => {
    const { result } = setup("/profile?tab=usage");
    act(() => vi.advanceTimersByTime(15_000));
    expect(result.current.search).toBe("?tab=usage");
    expect(success).not.toHaveBeenCalled();
    expect(info).not.toHaveBeenCalled();
    expect(h.invalidateCreditSurfaces).not.toHaveBeenCalled();
  });
});
