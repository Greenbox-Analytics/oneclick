import { useMutation, useQueryClient, type QueryClient, type QueryKey } from "@tanstack/react-query";
import { toast } from "sonner";

import { API_URL, apiFetch } from "@/lib/apiFetch";
import type { BoardTask, BoardTaskDetail } from "@/types/integrations";

export type Assignee = { user_id: string; full_name?: string | null; avatar_url?: string | null };

export type AssigneeContext = {
  prevDetail: BoardTaskDetail | undefined;
  prevTasksQueries: [QueryKey, BoardTask[] | undefined][];
};

// --- Pure cache helpers (exported for unit tests) -------------------------------
// The panel's assignee picker derives its selection from `task.assignees` in the
// board-task-detail cache, and the kanban card shows assignee avatars from the
// board-tasks cache. Patch BOTH optimistically so a click registers instantly.

/** Snapshot the caches an assignee change touches, for rollback on error. */
export function snapshotAssigneeCaches(qc: QueryClient, taskId: string): AssigneeContext {
  return {
    prevDetail: qc.getQueryData<BoardTaskDetail>(["board-task-detail", taskId]),
    prevTasksQueries: qc.getQueriesData<BoardTask[]>({ queryKey: ["board-tasks"] }),
  };
}

/** Apply `update` to the task's assignee list in every cache that renders it. */
export function applyAssigneePatch(
  qc: QueryClient,
  taskId: string,
  update: (list: Assignee[]) => Assignee[],
) {
  qc.setQueryData<BoardTaskDetail>(["board-task-detail", taskId], (old) =>
    old ? { ...old, assignees: update(old.assignees ?? []) } : old,
  );
  qc.setQueriesData<BoardTask[]>({ queryKey: ["board-tasks"] }, (old) =>
    old ? old.map((t) => (t.id === taskId ? { ...t, assignees: update(t.assignees ?? []) } : t)) : old,
  );
}

/** Restore the caches captured by snapshotAssigneeCaches. */
export function rollbackAssigneeCaches(qc: QueryClient, taskId: string, ctx?: AssigneeContext) {
  if (ctx?.prevDetail !== undefined) qc.setQueryData(["board-task-detail", taskId], ctx.prevDetail);
  if (ctx?.prevTasksQueries) {
    for (const [key, value] of ctx.prevTasksQueries) qc.setQueryData(key, value);
  }
}

function reconcileAssigneeCaches(qc: QueryClient, taskId: string) {
  qc.invalidateQueries({ queryKey: ["board-task-detail", taskId] });
  qc.invalidateQueries({ queryKey: ["board-tasks"] });
  qc.invalidateQueries({ queryKey: ["board-tasks-calendar"] });
  qc.invalidateQueries({ queryKey: ["parent-tasks"] });
}

const addTo = (userId: string, entry: Assignee) => (list: Assignee[]) =>
  list.some((a) => a.user_id === userId) ? list : [...list, entry];
const removeFrom = (userId: string) => (list: Assignee[]) => list.filter((a) => a.user_id !== userId);

// --- Hooks ----------------------------------------------------------------------

export function useAddAssignee() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ taskId, userId }: { taskId: string; userId: string; assignee?: Assignee }) =>
      apiFetch(`${API_URL}/boards/tasks/${taskId}/assignees`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      }),
    onMutate: ({ taskId, userId, assignee }): AssigneeContext => {
      const ctx = snapshotAssigneeCaches(qc, taskId);
      applyAssigneePatch(qc, taskId, addTo(userId, assignee ?? { user_id: userId }));
      return ctx;
    },
    onError: (e: Error, { taskId }, ctx) => {
      rollbackAssigneeCaches(qc, taskId, ctx);
      toast.error(e.message);
    },
    onSettled: (_d, _e, { taskId }) => reconcileAssigneeCaches(qc, taskId),
  });
}

export function useRemoveAssignee() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ taskId, userId }: { taskId: string; userId: string }) =>
      apiFetch(`${API_URL}/boards/tasks/${taskId}/assignees/${userId}`, { method: "DELETE" }),
    onMutate: ({ taskId, userId }): AssigneeContext => {
      const ctx = snapshotAssigneeCaches(qc, taskId);
      applyAssigneePatch(qc, taskId, removeFrom(userId));
      return ctx;
    },
    onError: (e: Error, { taskId }, ctx) => {
      rollbackAssigneeCaches(qc, taskId, ctx);
      toast.error(e.message);
    },
    onSettled: (_d, _e, { taskId }) => reconcileAssigneeCaches(qc, taskId),
  });
}
