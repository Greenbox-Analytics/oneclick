import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { BoardTaskDetail } from "@/types/integrations";
import { API_URL, apiFetch, getAuthHeaders } from "@/lib/apiFetch";

export function useTaskDetail(taskId: string | null) {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const detailQuery = useQuery<BoardTaskDetail>({
    queryKey: ["board-task-detail", taskId],
    queryFn: async () => {
      if (!user?.id || !taskId) throw new Error("Missing params");
      return apiFetch<BoardTaskDetail>(
        `${API_URL}/boards/tasks/${taskId}/detail`
      );
    },
    enabled: !!user?.id && !!taskId,
  });

  const addCommentMutation = useMutation({
    mutationFn: async ({ taskId, content }: { taskId: string; content: string }) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch(
        `${API_URL}/boards/tasks/${taskId}/comments`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content }),
        }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["board-task-detail", taskId] });
    },
  });

  const deleteCommentMutation = useMutation({
    mutationFn: async (commentId: string) => {
      if (!user?.id) throw new Error("Not authenticated");
      const authHeaders = await getAuthHeaders();
      const res = await fetch(
        `${API_URL}/boards/comments/${commentId}`,
        { method: "DELETE", headers: authHeaders }
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
