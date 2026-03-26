import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { BoardTaskDetail } from "@/types/integrations";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

export function useTaskDetail(taskId: string | null) {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const detailQuery = useQuery<BoardTaskDetail>({
    queryKey: ["board-task-detail", taskId],
    queryFn: async () => {
      if (!user?.id || !taskId) throw new Error("Missing params");
      const res = await fetch(
        `${API_URL}/boards/tasks/${taskId}/detail?user_id=${user.id}`
      );
      if (!res.ok) throw new Error("Failed to fetch task detail");
      return res.json();
    },
    enabled: !!user?.id && !!taskId,
  });

  const addCommentMutation = useMutation({
    mutationFn: async ({ taskId, content }: { taskId: string; content: string }) => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(
        `${API_URL}/boards/tasks/${taskId}/comments?user_id=${user.id}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content }),
        }
      );
      if (!res.ok) throw new Error("Failed to add comment");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["board-task-detail", taskId] });
    },
  });

  const deleteCommentMutation = useMutation({
    mutationFn: async (commentId: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(
        `${API_URL}/boards/comments/${commentId}?user_id=${user.id}`,
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error("Failed to delete comment");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["board-task-detail", taskId] });
    },
  });

  return {
    task: detailQuery.data || null,
    isLoading: detailQuery.isLoading,
    addComment: addCommentMutation.mutate,
    deleteComment: deleteCommentMutation.mutate,
  };
}
