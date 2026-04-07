import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { ParentTaskWithChildren, BoardTask } from "@/types/integrations";
import { API_URL, apiFetch, getAuthHeaders } from "@/lib/apiFetch";

interface ParentsResponse {
  parents: ParentTaskWithChildren[];
  ungrouped: BoardTask[];
}

export function useParentTasks(search?: string, artistId?: string) {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const query = useQuery<ParentsResponse>({
    queryKey: ["parent-tasks", user?.id, search, artistId],
    queryFn: async () => {
      if (!user?.id) return { parents: [], ungrouped: [] };
      const params = new URLSearchParams();
      if (search) params.set("search", search);
      if (artistId) params.set("artist_id", artistId);
      const qs = params.toString();
      return apiFetch<ParentsResponse>(`${API_URL}/boards/parents${qs ? `?${qs}` : ""}`);
    },
    enabled: !!user?.id,
  });

  const createParentMutation = useMutation({
    mutationFn: async (data: {
      title: string;
      description?: string;
      priority?: string;
      start_date?: string;
      due_date?: string;
      color?: string;
      artist_ids?: string[];
      project_ids?: string[];
      labels?: string[];
    }) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch(`${API_URL}/boards/parents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["parent-tasks"] });
      queryClient.invalidateQueries({ queryKey: ["board-tasks"] });
    },
  });

  const deleteParentMutation = useMutation({
    mutationFn: async (taskId: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      const authHeaders = await getAuthHeaders();
      const res = await fetch(
        `${API_URL}/boards/tasks/${taskId}`,
        { method: "DELETE", headers: authHeaders }
      );
      if (!res.ok) throw new Error("Failed to delete parent task");
    },
    onMutate: async (taskId) => {
      await queryClient.cancelQueries({ queryKey: ["parent-tasks"] });
      await queryClient.cancelQueries({ queryKey: ["board-tasks"] });

      // Snapshot all parent-tasks caches for rollback
      const prevParentQueries = queryClient.getQueriesData<ParentsResponse>({ queryKey: ["parent-tasks"] });

      // Optimistically remove the parent from all parent-tasks caches
      for (const [key, value] of prevParentQueries) {
        if (!value) continue;
        queryClient.setQueryData(key, {
          ...value,
          parents: value.parents.filter((p: ParentTaskWithChildren) => p.id !== taskId),
        });
      }

      return { prevParentQueries };
    },
    onError: (_err, _vars, context) => {
      if (context?.prevParentQueries) {
        for (const [key, value] of context.prevParentQueries) {
          queryClient.setQueryData(key, value);
        }
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["parent-tasks"] });
      queryClient.invalidateQueries({ queryKey: ["board-tasks"] });
      queryClient.invalidateQueries({ queryKey: ["board-task-detail"] });
    },
  });

  return {
    parents: query.data?.parents || [],
    ungrouped: query.data?.ungrouped || [],
    isLoading: query.isLoading,
    createParent: createParentMutation.mutate,
    deleteParent: deleteParentMutation.mutate,
    isCreating: createParentMutation.isPending,
  };
}
