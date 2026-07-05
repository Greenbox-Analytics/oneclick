import { describe, it, expect } from "vitest";
import { QueryClient } from "@tanstack/react-query";
import {
  applyAssigneePatch,
  snapshotAssigneeCaches,
  rollbackAssigneeCaches,
  type Assignee,
} from "@/hooks/useTaskAssignees";

// The panel's assignee picker reads from board-task-detail; the kanban card reads
// from board-tasks. The board renders a PERIOD-scoped board-tasks key, distinct from
// the panel hook's plain key — so an optimistic assignee change must span both.
const DETAIL_KEY = ["board-task-detail", "t1"];
const PLAIN_KEY = ["board-tasks", "u1", "b1"];
const PERIOD_KEY = ["board-tasks", "u1", "2026-07-01", "2026-07-31", true, "b1"];

const addTo = (userId: string, entry: Assignee) => (list: Assignee[]) =>
  list.some((a) => a.user_id === userId) ? list : [...list, entry];
const removeFrom = (userId: string) => (list: Assignee[]) =>
  list.filter((a) => a.user_id !== userId);

function seed() {
  const qc = new QueryClient();
  qc.setQueryData(DETAIL_KEY, { id: "t1", assignees: [] });
  qc.setQueryData(PLAIN_KEY, [{ id: "t1", assignees: [] }]);
  qc.setQueryData(PERIOD_KEY, [{ id: "t1", assignees: [] }]);
  return qc;
}

const detailAssignees = (qc: QueryClient) =>
  (qc.getQueryData(DETAIL_KEY) as { assignees: Assignee[] }).assignees;
const cardAssignees = (qc: QueryClient, key: readonly unknown[]) =>
  (qc.getQueryData(key) as { id: string; assignees: Assignee[] }[]).find((t) => t.id === "t1")!
    .assignees;

describe("assignee optimistic cache helpers", () => {
  it("adds an assignee to the detail cache AND every board-tasks variant", () => {
    const qc = seed();
    applyAssigneePatch(qc, "t1", addTo("u2", { user_id: "u2", full_name: "Bob" }));

    expect(detailAssignees(qc)).toEqual([{ user_id: "u2", full_name: "Bob" }]);
    expect(cardAssignees(qc, PLAIN_KEY)).toEqual([{ user_id: "u2", full_name: "Bob" }]);
    expect(cardAssignees(qc, PERIOD_KEY)).toEqual([{ user_id: "u2", full_name: "Bob" }]);
  });

  it("does not duplicate an assignee already present", () => {
    const qc = seed();
    applyAssigneePatch(qc, "t1", addTo("u2", { user_id: "u2" }));
    applyAssigneePatch(qc, "t1", addTo("u2", { user_id: "u2" }));
    expect(detailAssignees(qc)).toHaveLength(1);
  });

  it("removes an assignee from all caches", () => {
    const qc = seed();
    applyAssigneePatch(qc, "t1", addTo("u2", { user_id: "u2" }));
    applyAssigneePatch(qc, "t1", removeFrom("u2"));

    expect(detailAssignees(qc)).toEqual([]);
    expect(cardAssignees(qc, PERIOD_KEY)).toEqual([]);
  });

  it("rolls back to the snapshot on error", () => {
    const qc = seed();
    applyAssigneePatch(qc, "t1", addTo("u9", { user_id: "u9" })); // pre-existing state
    const ctx = snapshotAssigneeCaches(qc, "t1");

    applyAssigneePatch(qc, "t1", addTo("u2", { user_id: "u2" })); // optimistic write
    expect(detailAssignees(qc)).toHaveLength(2);

    rollbackAssigneeCaches(qc, "t1", ctx);
    expect(detailAssignees(qc)).toEqual([{ user_id: "u9" }]);
    expect(cardAssignees(qc, PERIOD_KEY)).toEqual([{ user_id: "u9" }]);
  });
});
