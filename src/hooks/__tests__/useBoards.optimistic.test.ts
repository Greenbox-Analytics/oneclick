import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor, cleanup } from "@testing-library/react";

// useBoards pulls the user from AuthContext and talks to the backend via apiFetch —
// stub both so the hook mounts and its background queries resolve to empty lists.
vi.mock("@/contexts/AuthContext", () => ({ useAuth: () => ({ user: { id: "u1" } }) }));
vi.mock("@/lib/apiFetch", () => ({
  API_URL: "http://test",
  apiFetch: vi.fn().mockResolvedValue({ columns: [], tasks: [] }),
  getAuthHeaders: vi.fn().mockResolvedValue({}),
}));

import { useBoards } from "@/hooks/useBoards";

// The board renders from a PERIOD-scoped key; the panel's useBoards({ boardId }) uses a
// plain key. Seeding only the period key proves optimistic writes reach the key the
// board actually renders — the bug that made create/delete/update feel non-instant.
const PERIOD_KEY = ["board-tasks", "u1", "2026-07-01", "2026-07-31", true, "b1"];

type CachedTask = { id: string; title?: string; priority?: string };

function setup() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  qc.setQueryData(PERIOD_KEY, [{ id: "t1", title: "Existing" }]);
  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client: qc }, children);
  const { result } = renderHook(() => useBoards({ boardId: "b1" }), { wrapper });
  return { qc, result };
}

const period = (qc: QueryClient) => qc.getQueryData(PERIOD_KEY) as CachedTask[];

beforeEach(() => vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true })));
afterEach(() => cleanup());

describe("useBoards optimistic mutations reach the board's period-scoped cache", () => {
  it("create inserts a temp task immediately, and rolls back on demand", async () => {
    const { qc, result } = setup();

    const ctx = await result.current.applyOptimisticTaskCreate({
      title: "New task",
      column_id: "c1",
      board_id: "b1",
    });
    expect(period(qc).map((t) => t.title)).toContain("New task");

    result.current.rollbackTaskCaches(ctx);
    expect(period(qc).map((t) => t.title)).toEqual(["Existing"]);
  });

  it("update patches the matching task", async () => {
    const { qc, result } = setup();
    result.current.updateTask({ id: "t1", priority: "high" });
    await waitFor(() =>
      expect(period(qc).find((t) => t.id === "t1")?.priority).toBe("high"),
    );
  });

  it("delete removes the task", async () => {
    const { qc, result } = setup();
    result.current.deleteTask("t1");
    await waitFor(() => expect(period(qc).find((t) => t.id === "t1")).toBeUndefined());
  });
});
