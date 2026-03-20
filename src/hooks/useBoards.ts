import { useQuery, useMutation, useQueryClient, keepPreviousData } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { BoardColumn, BoardTask } from "@/types/integrations";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

interface UseBoardsOptions {
  artistId?: string;
  periodStart?: string;
  periodEnd?: string;
  isCurrentPeriod?: boolean;
}

export function useBoards(artistIdOrOptions?: string | UseBoardsOptions) {
  const options: UseBoardsOptions = typeof artistIdOrOptions === "string"
    ? { artistId: artistIdOrOptions }
    : artistIdOrOptions || {};
  const { artistId, periodStart, periodEnd, isCurrentPeriod } = options;

  const { user } = useAuth();
  const queryClient = useQueryClient();

  const columnsQuery = useQuery<BoardColumn[]>({
    queryKey: ["board-columns", user?.id, artistId],
    queryFn: async () => {
      if (!user?.id) return [];
      const params = new URLSearchParams({ user_id: user.id });
      if (artistId) params.set("artist_id", artistId);
      const res = await fetch(`${API_URL}/boards/columns?${params}`);
      if (!res.ok) throw new Error("Failed to fetch columns");
      const data = await res.json();
      return data.columns;
    },
    enabled: !!user?.id,
  });

  const hasPeriod = !!(periodStart && periodEnd);

  const tasksQueryKey = hasPeriod
    ? ["board-tasks", user?.id, periodStart, periodEnd, isCurrentPeriod]
    : ["board-tasks", user?.id];

  const tasksQuery = useQuery<BoardTask[]>({
    queryKey: tasksQueryKey,
    queryFn: async () => {
      if (!user?.id) return [];
      if (hasPeriod) {
        const params = new URLSearchParams({
          user_id: user.id,
          period_start: periodStart!,
          period_end: periodEnd!,
          is_current: String(isCurrentPeriod ?? true),
        });
        const res = await fetch(`${API_URL}/boards/tasks/period?${params}`);
        if (!res.ok) throw new Error("Failed to fetch period tasks");
        const data = await res.json();
        return data.tasks;
      }
      const res = await fetch(`${API_URL}/boards/tasks?user_id=${user.id}`);
      if (!res.ok) throw new Error("Failed to fetch tasks");
      const data = await res.json();
      return data.tasks;
    },
    enabled: !!user?.id,
    placeholderData: keepPreviousData,
  });

  const createColumnMutation = useMutation({
    mutationFn: async (data: { title: string; color?: string; artist_id?: string }) => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(`${API_URL}/boards/columns?user_id=${user.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to create column");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["board-columns"] }),
  });

  const updateColumnMutation = useMutation({
    mutationFn: async ({ id, ...data }: { id: string; title?: string; color?: string; position?: number }) => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(`${API_URL}/boards/columns/${id}?user_id=${user.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to update column");
      return res.json();
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["board-columns"] }),
  });

  const deleteColumnMutation = useMutation({
    mutationFn: async (columnId: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(`${API_URL}/boards/columns/${columnId}?user_id=${user.id}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to delete column");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["board-columns"] });
      queryClient.invalidateQueries({ queryKey: ["board-tasks"] });
    },
  });

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
      artist_ids?: string[];
      project_ids?: string[];
      contract_ids?: string[];
      labels?: string[];
    }) => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(`${API_URL}/boards/tasks?user_id=${user.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to create task");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["board-tasks"] });
      queryClient.invalidateQueries({ queryKey: ["board-tasks-calendar"] });
    },
  });

  const updateTaskMutation = useMutation({
    mutationFn: async ({ id, ...data }: {
      id: string; column_id?: string | null; title?: string; description?: string;
      priority?: string; position?: number; start_date?: string; due_date?: string;
      color?: string; artist_ids?: string[]; project_ids?: string[];
      contract_ids?: string[]; assignee_name?: string; labels?: string[];
    }) => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(`${API_URL}/boards/tasks/${id}?user_id=${user.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to update task");
      return res.json();
    },
    onMutate: async ({ id, ...data }) => {
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: ["board-tasks"] });
      await queryClient.cancelQueries({ queryKey: ["parent-tasks"] });
      // Snapshot previous using the active query key (period-aware)
      const prevTasks = queryClient.getQueryData<BoardTask[]>(tasksQueryKey);
      // Optimistically update board-tasks cache
      if (prevTasks) {
        queryClient.setQueryData<BoardTask[]>(
          tasksQueryKey,
          prevTasks.map((t) => (t.id === id ? { ...t, ...data } : t))
        );
      }
      return { prevTasks };
    },
    onError: (_err, _vars, context) => {
      // Rollback on error
      if (context?.prevTasks) {
        queryClient.setQueryData(tasksQueryKey, context.prevTasks);
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
      const res = await fetch(`${API_URL}/boards/tasks/${taskId}?user_id=${user.id}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to delete task");
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["board-tasks"] }),
  });

  const reorderTasksMutation = useMutation({
    mutationFn: async (reorders: { task_id: string; target_column_id: string; position: number }[]) => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(`${API_URL}/boards/tasks/reorder?user_id=${user.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
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
      const params = new URLSearchParams({ user_id: user.id });
      if (artistId) params.set("artist_id", artistId);
      const res = await fetch(`${API_URL}/boards/columns/defaults?${params}`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to create default columns");
      return res.json();
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
    updateTask: updateTaskMutation.mutate,
    deleteTask: deleteTaskMutation.mutate,
    reorderTasks: reorderTasksMutation.mutate,
    createDefaults: createDefaultsMutation.mutate,
  };
}
