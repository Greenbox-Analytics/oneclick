import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { WorkspaceSettings } from "@/types/integrations";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

export function useWorkspaceSettings() {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const settingsQuery = useQuery<WorkspaceSettings>({
    queryKey: ["workspace-settings", user?.id],
    queryFn: async () => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(`${API_URL}/settings?user_id=${user.id}`);
      if (!res.ok) throw new Error("Failed to fetch settings");
      return res.json();
    },
    enabled: !!user?.id,
  });

  const updateMutation = useMutation({
    mutationFn: async (data: Partial<WorkspaceSettings>) => {
      if (!user?.id) throw new Error("Not authenticated");
      const res = await fetch(`${API_URL}/settings?user_id=${user.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Failed to update settings");
      return res.json();
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
