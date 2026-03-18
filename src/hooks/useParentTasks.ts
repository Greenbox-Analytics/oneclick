import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { ParentTaskWithChildren, BoardTask } from "@/types/integrations";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

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
      const params = new URLSearchParams({ user_id: user.id });
      if (search) params.set("search", search);
      if (artistId) params.set("artist_id", artistId);
      const res = await fetch(`${API_URL}/boards/parents?${params}`);
      if (!res.ok) throw new Error("Failed to fetch parent tasks");
      return res.json();
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
      const res = await fetch(`${API_URL}/boards/parents?user_id=${user.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to create parent task");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["parent-tasks"] });
    },
  });

  const deleteParentMutation = useMutation({
    mutationFn: async (taskId: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(
        `${API_URL}/boards/tasks/${taskId}?user_id=${user.id}`,
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error("Failed to delete parent task");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["parent-tasks"] });
      queryClient.invalidateQueries({ queryKey: ["board-tasks"] });
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
