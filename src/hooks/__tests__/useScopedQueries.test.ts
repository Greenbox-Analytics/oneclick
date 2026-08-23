// The scope-threading contract every listing hook follows: scopeKey in the
// query key, withScope on the URL, ready gating the fetch. Exercised through
// two representative hooks (useArtistsList — the shared artist roster — and
// useExpenseSummary — the Expense Tracker), plus the switch-invalidation that
// makes changing workspace actually refresh what's on screen.
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor, cleanup } from "@testing-library/react";
import type { Entitlements } from "@/hooks/useEntitlements";

const mockEntitlements = vi.fn();
vi.mock("@/hooks/useEntitlements", () => ({
  useEntitlements: () => mockEntitlements(),
  clearRememberedBillingContext: vi.fn(),
}));
vi.mock("@/contexts/AuthContext", () => ({ useAuth: () => ({ user: { id: "u1" } }) }));
vi.mock("sonner", () => ({ toast: Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }) }));

const apiFetch = vi.fn();
vi.mock("@/lib/apiFetch", () => ({
  API_URL: "http://test",
  apiFetch: (...args: unknown[]) => apiFetch(...args),
  getAuthHeaders: vi.fn().mockResolvedValue({}),
}));

const { useArtistsList } = await import("@/hooks/useArtistsList");
const { useExpenseSummary } = await import("@/hooks/useProjectExpenses");
const { useSetBillingContext } = await import("@/hooks/useBillingContext");

const ORG = "org-a";

function scoped(payload: Partial<Entitlements> | undefined, isLoading = false) {
  mockEntitlements.mockReturnValue({ data: payload, isLoading });
}

function render<T>(hook: () => T) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client: qc }, children);
  return { qc, ...renderHook(hook, { wrapper }) };
}

beforeEach(() => {
  mockEntitlements.mockReset();
  apiFetch.mockReset();
  apiFetch.mockResolvedValue([]);
});
afterEach(() => cleanup());

describe("scope threading in listing hooks", () => {
  it("appends the org scope to the artists request", async () => {
    scoped({ workspaceScope: { type: "org", orgId: ORG, orgName: "Acme" } });
    render(() => useArtistsList());

    await waitFor(() => expect(apiFetch).toHaveBeenCalledWith(`http://test/artists?scope=${ORG}`));
  });

  it("appends scope=personal in the personal workspace", async () => {
    scoped({ workspaceScope: { type: "personal" } });
    apiFetch.mockResolvedValue({ expenses: [] });
    render(() => useExpenseSummary());

    await waitFor(() =>
      expect(apiFetch).toHaveBeenCalledWith("http://test/expenses/summary?scope=personal"),
    );
  });

  it("sends no scope param when scoping is off — the rollback path", async () => {
    scoped({});
    render(() => useArtistsList());

    await waitFor(() => expect(apiFetch).toHaveBeenCalledWith("http://test/artists"));
  });

  it("does not fetch until the scope is known", async () => {
    // Fetching early would cache a param-less response under the wrong key,
    // then refetch — a double fetch and a flash of the wrong workspace.
    scoped(undefined, true);
    render(() => useArtistsList());

    await new Promise((r) => setTimeout(r, 30));
    expect(apiFetch).not.toHaveBeenCalled();
  });

  it("refetches under the new scope when the workspace changes", async () => {
    scoped({ workspaceScope: { type: "org", orgId: ORG, orgName: "Acme" } });
    const { rerender } = render(() => useArtistsList());
    await waitFor(() => expect(apiFetch).toHaveBeenCalledWith(`http://test/artists?scope=${ORG}`));

    scoped({ workspaceScope: { type: "personal" } });
    rerender();

    // A new scope = a new cache key = a fresh request — never org A's roster
    // served from cache under the Personal label.
    await waitFor(() => expect(apiFetch).toHaveBeenCalledWith("http://test/artists?scope=personal"));
  });
});

describe("workspace switch invalidation", () => {
  it("invalidates every query, not a hand-kept list of keys", async () => {
    scoped({ workspaceScope: { type: "personal" } });
    apiFetch.mockResolvedValue({});
    const { qc, result } = render(() => useSetBillingContext());
    const invalidate = vi.spyOn(qc, "invalidateQueries");

    result.current.mutate({ orgId: ORG });

    // The context now decides what every list SHOWS — a partial invalidation
    // list would silently rot as scoped surfaces are added.
    await waitFor(() => expect(invalidate).toHaveBeenCalledWith());
  });
});
