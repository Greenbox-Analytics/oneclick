import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { WorkspaceSettings } from "@/types/integrations";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export function useWorkspaceSettings() {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const settingsQuery = useQuery<WorkspaceSettings>({
    queryKey: ["workspace-settings", user?.id],
    queryFn: async () => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<WorkspaceSettings>(`${API_URL}/settings`);
    },
    enabled: !!user?.id,
  });

  const updateMutation = useMutation({
    mutationFn: async (data: Partial<WorkspaceSettings>) => {
      if (!user?.id) throw new Error("Not authenticated");
      return apiFetch<WorkspaceSettings>(`${API_URL}/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
    },
    onSuccess: (data) => {
      queryClient.setQueryData(["workspace-settings", user?.id], data);
    },
  });

  return {
    settings: settingsQuery.data,
    isLoading: settingsQuery.isLoading,
    updateSettings: updateMutation.mutate,
  };
}
