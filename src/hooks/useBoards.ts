import { useQuery, useMutation, useQueryClient, keepPreviousData, type QueryKey } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { BoardColumn, BoardTask, ParentTaskWithChildren } from "@/types/integrations";
import { API_URL, apiFetch, getAuthHeaders } from "@/lib/apiFetch";

interface UseBoardsOptions {
  artistId?: string;
  boardId?: string;
  periodStart?: string;
  periodEnd?: string;
  isCurrentPeriod?: boolean;
}

type ParentTaskQueryData = { parents: ParentTaskWithChildren[]; ungrouped: BoardTask[] };

/** Fields needed to build an optimistic temp task. Superset of the create payload:
 *  the enriched `artists/projects/documents` arrays let a caller (e.g. TaskDetailPanel)
 *  render linked NAMES on the card immediately, before the server round-trip. */
export type OptimisticTaskData = {
  column_id?: string;
  title: string;
  description?: string;
  priority?: string;
  start_date?: string;
  due_date?: string;
  color?: string;
  parent_task_id?: string;
  is_parent?: boolean;
  board_id?: string;
  artist_ids?: string[];
  project_ids?: string[];
  contract_ids?: string[];
  labels?: string[];
  artists?: { id: string; name: string; can_open: boolean }[];
  projects?: { id: string; name: string; can_open: boolean }[];
  documents?: { id: string; name: string; can_open: boolean }[];
};

/** Rollback snapshot returned by applyOptimisticTaskCreate. Captures every
 *  board-tasks cache variant (period-scoped + plain) so rollback is exact. */
export type OptimisticTaskContext = {
  prevTasksQueries: [QueryKey, BoardTask[] | undefined][];
  prevParentQueries: [QueryKey, ParentTaskQueryData | undefined][];
};

export function useBoards(artistIdOrOptions?: string | UseBoardsOptions) {
  const options: UseBoardsOptions = typeof artistIdOrOptions === "string"
    ? { artistId: artistIdOrOptions }
    : artistIdOrOptions || {};
  const { artistId, boardId, periodStart, periodEnd, isCurrentPeriod } = options;

  const { user } = useAuth();
  const queryClient = useQueryClient();

  const columnsQuery = useQuery<BoardColumn[]>({
    queryKey: ["board-columns", user?.id, artistId, boardId],
    queryFn: async () => {
      if (!user?.id) return [];
      const params = new URLSearchParams();
      if (artistId) params.set("artist_id", artistId);
      if (boardId) params.set("board_id", boardId);
      const qs = params.toString();
      const data = await apiFetch<{ columns: BoardColumn[] }>(`${API_URL}/boards/columns${qs ? `?${qs}` : ""}`);
      return data.columns;
    },
    enabled: !!user?.id,
  });

  const hasPeriod = !!(periodStart && periodEnd);

  const tasksQueryKey = hasPeriod
    ? ["board-tasks", user?.id, periodStart, periodEnd, isCurrentPeriod, boardId]
    : ["board-tasks", user?.id, boardId];

  const tasksQuery = useQuery<BoardTask[]>({
    queryKey: tasksQueryKey,
    queryFn: async () => {
      if (!user?.id) return [];
      if (hasPeriod) {
        const params = new URLSearchParams({
          period_start: periodStart!,
          period_end: periodEnd!,
          is_current: String(isCurrentPeriod ?? true),
        });
        if (boardId) params.set("board_id", boardId);
        const data = await apiFetch<{ tasks: BoardTask[] }>(`${API_URL}/boards/tasks/period?${params}`);
        return data.tasks;
      }
      const data = await apiFetch<{ tasks: BoardTask[] }>(
        `${API_URL}/boards/tasks${boardId ? `?board_id=${boardId}` : ""}`,
      );
      return data.tasks;
    },
    enabled: !!user?.id,
    placeholderData: keepPreviousData,
  });

  const createColumnMutation = useMutation({
    mutationFn: async (data: { title: string; color?: string; artist_id?: string; board_id?: string; position?: number }) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch(`${API_URL}/boards/columns`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...data, board_id: data.board_id ?? boardId }),
      });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["board-columns"] }),
  });

  const updateColumnMutation = useMutation({
    mutationFn: async ({ id, ...data }: { id: string; title?: string; color?: string; position?: number }) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch(`${API_URL}/boards/columns/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["board-columns"] }),
  });

  const deleteColumnMutation = useMutation({
    mutationFn: async (columnId: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      const authHeaders = await getAuthHeaders();
      const res = await fetch(`${API_URL}/boards/columns/${columnId}`, {
        method: "DELETE",
        headers: authHeaders,
      });
      if (!res.ok) throw new Error("Failed to delete column");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["board-columns"] });
      queryClient.invalidateQueries({ queryKey: ["board-tasks"] });
    },
  });

  // Optimistic-create primitives — shared by the internal create mutation (subtasks)
  // and the gated top-level create in TaskDetailPanel. They write to EVERY board-tasks
  // cache (period-scoped + plain) via setQueriesData, so the optimistic card lands no
  // matter which key the renderer uses — the panel's key differs from the board's.
  const applyOptimisticTaskCreate = async (data: OptimisticTaskData): Promise<OptimisticTaskContext> => {
    // Snapshot ALL board-tasks variants (period-scoped + plain) + parent caches so we can
    // roll back, and so the optimistic write lands on whichever key the board renders from.
    const prevTasksQueries = queryClient.getQueriesData<BoardTask[]>({ queryKey: ["board-tasks"] });
    const prevParentQueries = queryClient.getQueriesData<ParentTaskQueryData>({ queryKey: ["parent-tasks"] });

    // Create temporary optimistic task
    const tempTask: BoardTask = {
      id: `temp-${Date.now()}`,
      user_id: user?.id || "",
      title: data.title,
      position: 0,
      column_id: data.column_id,
      parent_task_id: data.parent_task_id,
      is_parent: data.is_parent,
      priority: data.priority as BoardTask["priority"],
      color: data.color,
      board_id: data.board_id ?? boardId,
      start_date: data.start_date,
      due_date: data.due_date,
      description: data.description,
      labels: data.labels,
      artist_ids: data.artist_ids || [],
      project_ids: data.project_ids || [],
      contract_ids: data.contract_ids || [],
      artists: data.artists || [],
      projects: data.projects || [],
      documents: data.documents || [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    // Optimistically add to EVERY board-tasks cache — synchronously (before any await)
    // so the card renders this frame; the board reads a period-scoped key, not ours.
    queryClient.setQueriesData<BoardTask[]>(
      { queryKey: ["board-tasks"] },
      (old) => (old ? [...old, tempTask] : old)
    );

    // Optimistically add subtask to parent's children in all parent-tasks caches
    if (data.parent_task_id) {
      for (const [key, value] of prevParentQueries) {
        if (!value) continue;
        queryClient.setQueryData(key, {
          ...value,
          parents: value.parents.map((p: ParentTaskWithChildren) =>
            p.id === data.parent_task_id
              ? { ...p, children: [...(p.children || []), tempTask], child_count: (p.child_count || 0) + 1 }
              : p
          ),
        });
      }

      // Optimistically add to task detail cache if viewing the parent
      const detailData = queryClient.getQueryData<{ children?: BoardTask[] }>(["board-task-detail", data.parent_task_id]);
      if (detailData) {
        queryClient.setQueryData(["board-task-detail", data.parent_task_id], {
          ...detailData,
          children: [...(detailData.children || []), tempTask],
        });
      }
    }

    // Cancel any in-flight refetch AFTER writing so it can't clobber the optimistic card.
    queryClient.cancelQueries({ queryKey: ["board-tasks"] });
    queryClient.cancelQueries({ queryKey: ["parent-tasks"] });
    queryClient.cancelQueries({ queryKey: ["board-task-detail"] });

    return { prevTasksQueries, prevParentQueries };
  };

  const rollbackTaskCaches = (context?: OptimisticTaskContext) => {
    if (context?.prevTasksQueries) {
      for (const [key, value] of context.prevTasksQueries) {
        queryClient.setQueryData(key, value);
      }
    }
    if (context?.prevParentQueries) {
      for (const [key, value] of context.prevParentQueries) {
        queryClient.setQueryData(key, value);
      }
    }
  };

  const reconcileTaskCaches = () => {
    queryClient.invalidateQueries({ queryKey: ["board-tasks"] });
    queryClient.invalidateQueries({ queryKey: ["board-tasks-calendar"] });
    queryClient.invalidateQueries({ queryKey: ["parent-tasks"] });
    queryClient.invalidateQueries({ queryKey: ["board-task-detail"] });
  };

  const createTaskMutation = useMutation({
    mutationFn: async (data: {
      column_id?: string;
      title: string;
      description?: string;
      priority?: string;
      start_date?: string;
      due_date?: string;
      color?: string;
      parent_task_id?: string;
      is_parent?: boolean;
      board_id?: string;
      artist_ids?: string[];
      project_ids?: string[];
      contract_ids?: string[];
      labels?: string[];
    }) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch(`${API_URL}/boards/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...data, board_id: data.board_id ?? boardId }),
      });
    },
    onMutate: (data) => applyOptimisticTaskCreate(data),
    onError: (_err, _vars, context) => rollbackTaskCaches(context),
    onSettled: () => reconcileTaskCaches(),
  });

  const updateTaskMutation = useMutation({
    mutationFn: async ({ id, ...data }: {
      id: string; column_id?: string | null; title?: string; description?: string;
      priority?: string; position?: number; start_date?: string; due_date?: string;
      color?: string; artist_ids?: string[]; project_ids?: string[];
      contract_ids?: string[]; assignee_name?: string; labels?: string[];
      parent_task_id?: string | null;
    }) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch(`${API_URL}/boards/tasks/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
    },
    onMutate: async ({ id, ...data }) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: ["board-tasks"] });
      await queryClient.cancelQueries({ queryKey: ["parent-tasks"] });
      // Snapshot previous using the active query key (period-aware)
      const prevTasksQueries = queryClient.getQueriesData<BoardTask[]>({ queryKey: ["board-tasks"] });
      // Optimistically update every board-tasks cache
      queryClient.setQueriesData<BoardTask[]>(
        { queryKey: ["board-tasks"] },
        (old) => (old ? old.map((t) => (t.id === id ? { ...t, ...data } : t)) : old)
      );
      // Optimistically update all parent-tasks caches (for epic board)
      const prevParentQueries = queryClient.getQueriesData<{ parents: ParentTaskWithChildren[]; ungrouped: BoardTask[] }>({ queryKey: ["parent-tasks"] });
      for (const [key, value] of prevParentQueries) {
        if (!value) continue;
        queryClient.setQueryData(key, {
          ...value,
          parents: value.parents.map((p: ParentTaskWithChildren) =>
            p.id === id ? { ...p, ...data } : p
          ),
          ungrouped: value.ungrouped.map((t: BoardTask) =>
            t.id === id ? { ...t, ...data } : t
          ),
        });
      }
      return { prevTasksQueries, prevParentQueries };
    },
    onError: (_err, _vars, context) => {
      // Rollback on error
      if (context?.prevTasksQueries) {
        for (const [key, value] of context.prevTasksQueries) {
          queryClient.setQueryData(key, value);
        }
      }
      if (context?.prevParentQueries) {
        for (const [key, value] of context.prevParentQueries) {
          queryClient.setQueryData(key, value);
        }
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["board-tasks"] });
      queryClient.invalidateQueries({ queryKey: ["board-task-detail"] });
      queryClient.invalidateQueries({ queryKey: ["board-tasks-calendar"] });
      queryClient.invalidateQueries({ queryKey: ["parent-tasks"] });
    },
  });

  const deleteTaskMutation = useMutation({
    mutationFn: async (taskId: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      const authHeaders = await getAuthHeaders();
      const res = await fetch(`${API_URL}/boards/tasks/${taskId}`, {
        method: "DELETE",
        headers: authHeaders,
      });
      if (!res.ok) throw new Error("Failed to delete task");
    },
    onMutate: async (taskId) => {
      await queryClient.cancelQueries({ queryKey: ["board-tasks"] });
      await queryClient.cancelQueries({ queryKey: ["parent-tasks"] });

      const prevTasksQueries = queryClient.getQueriesData<BoardTask[]>({ queryKey: ["board-tasks"] });
      queryClient.setQueriesData<BoardTask[]>(
        { queryKey: ["board-tasks"] },
        (old) => (old ? old.filter((t) => t.id !== taskId) : old)
      );

      // Also remove from parent-tasks caches (if it's a child, remove from parent's children)
      const prevParentQueries = queryClient.getQueriesData<{ parents: ParentTaskWithChildren[]; ungrouped: BoardTask[] }>({ queryKey: ["parent-tasks"] });
      for (const [key, value] of prevParentQueries) {
        if (!value) continue;
        queryClient.setQueryData(key, {
          ...value,
          parents: value.parents.map((p: ParentTaskWithChildren) => ({
            ...p,
            children: p.children.filter((c: BoardTask) => c.id !== taskId),
          })),
        });
      }

      return { prevTasksQueries, prevParentQueries };
    },
    onError: (_err, _vars, context) => {
      if (context?.prevTasksQueries) {
        for (const [key, value] of context.prevTasksQueries) {
          queryClient.setQueryData(key, value);
        }
      }
      if (context?.prevParentQueries) {
        for (const [key, value] of context.prevParentQueries) {
          queryClient.setQueryData(key, value);
        }
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["board-tasks"] });
      queryClient.invalidateQueries({ queryKey: ["board-tasks-calendar"] });
      queryClient.invalidateQueries({ queryKey: ["parent-tasks"] });
      queryClient.invalidateQueries({ queryKey: ["board-task-detail"] });
    },
  });

  const reorderTasksMutation = useMutation({
    mutationFn: async (reorders: { task_id: string; target_column_id: string; position: number }[]) => {
      if (!user?.id) throw new Error("Not authenticated");
      const authHeaders = await getAuthHeaders();
      const res = await fetch(`${API_URL}/boards/tasks/reorder`, {
        method: "PUT",
        headers: { ...authHeaders, "Content-Type": "application/json" },
        body: JSON.stringify({ reorders }),
      });
      if (!res.ok) throw new Error("Failed to reorder tasks");
    },
    onMutate: async (reorders) => {
      await queryClient.cancelQueries({ queryKey: ["board-tasks"] });
      const prevTasks = queryClient.getQueryData<BoardTask[]>(tasksQueryKey);
      // Optimistically move tasks to new columns/positions
      if (prevTasks) {
        const updated = [...prevTasks];
        for (const r of reorders) {
          const idx = updated.findIndex((t) => t.id === r.task_id);
          if (idx >= 0) {
            updated[idx] = { ...updated[idx], column_id: r.target_column_id, position: r.position };
          }
        }
        queryClient.setQueryData<BoardTask[]>(tasksQueryKey, updated);
      }
      return { prevTasks };
    },
    onError: (_err, _vars, context) => {
      if (context?.prevTasks) {
        queryClient.setQueryData(tasksQueryKey, context.prevTasks);
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["board-tasks"] });
      queryClient.invalidateQueries({ queryKey: ["parent-tasks"] });
    },
  });

  const createDefaultsMutation = useMutation({
    mutationFn: async () => {
      if (!user?.id) throw new Error("Not authenticated");
      const params = new URLSearchParams();
      if (artistId) params.set("artist_id", artistId);
      if (boardId) params.set("board_id", boardId);
      const qs = params.toString();
      return apiFetch(`${API_URL}/boards/columns/defaults${qs ? `?${qs}` : ""}`, {
        method: "POST",
      });
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["board-columns"] }),
  });

  return {
    columns: columnsQuery.data || [],
    tasks: tasksQuery.data || [],
    isLoading: columnsQuery.isLoading || tasksQuery.isLoading,
    createColumn: createColumnMutation.mutate,
    updateColumn: updateColumnMutation.mutate,
    deleteColumn: deleteColumnMutation.mutate,
    createTask: createTaskMutation.mutate,
    applyOptimisticTaskCreate,
    rollbackTaskCaches,
    reconcileTaskCaches,
    updateTask: updateTaskMutation.mutate,
    deleteTask: deleteTaskMutation.mutate,
    reorderTasks: reorderTasksMutation.mutate,
    createDefaults: createDefaultsMutation.mutate,
  };
}
