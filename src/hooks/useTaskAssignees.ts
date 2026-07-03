import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { API_URL, apiFetch } from "@/lib/apiFetch";

export function useAddAssignee() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ taskId, userId }: { taskId: string; userId: string }) =>
      apiFetch(`${API_URL}/boards/tasks/${taskId}/assignees`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      }),
    onSuccess: (_d, { taskId }) => {
      qc.invalidateQueries({ queryKey: ["board-task-detail", taskId] });
      qc.invalidateQueries({ queryKey: ["board-tasks"] });
      qc.invalidateQueries({ queryKey: ["board-tasks-calendar"] });
      qc.invalidateQueries({ queryKey: ["parent-tasks"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRemoveAssignee() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ taskId, userId }: { taskId: string; userId: string }) =>
      apiFetch(`${API_URL}/boards/tasks/${taskId}/assignees/${userId}`, { method: "DELETE" }),
    onSuccess: (_d, { taskId }) => {
      qc.invalidateQueries({ queryKey: ["board-task-detail", taskId] });
      qc.invalidateQueries({ queryKey: ["board-tasks"] });
      qc.invalidateQueries({ queryKey: ["board-tasks-calendar"] });
      qc.invalidateQueries({ queryKey: ["parent-tasks"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
}
