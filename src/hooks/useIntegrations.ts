import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { IntegrationConnection, IntegrationProvider } from "@/types/integrations";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

export function useIntegrations() {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const connectionsQuery = useQuery<IntegrationConnection[]>({
    queryKey: ["integrations", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const providers: IntegrationProvider[] = ["google_drive", "slack", "notion", "monday"];
      // Fetch all connections from each provider's status or a combined endpoint
      // For now, we query the backend for each provider's connection status
      const connections: IntegrationConnection[] = [];
      for (const provider of providers) {
        try {
          const res = await fetch(
            `${API_URL}/integrations/${provider.replace("_", "-")}/auth?user_id=${user.id}`,
            { method: "HEAD" }
          );
          // The connection data comes from the integration_connections table
          // We'll use a dedicated status endpoint
        } catch {
          // Provider not connected
        }
      }
      return connections;
    },
    enabled: !!user?.id,
  });

  const connectMutation = useMutation({
    mutationFn: async (provider: IntegrationProvider) => {
      if (!user?.id) throw new Error("Not authenticated");
      const providerPath = provider.replace("_", "-");
      const res = await fetch(
        `${API_URL}/integrations/${providerPath}/auth?user_id=${user.id}`
      );
      if (!res.ok) throw new Error("Failed to initiate OAuth");
      const data = await res.json();
      // Open OAuth URL in a new window
      window.open(data.auth_url, "_blank", "width=600,height=700");
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: async (provider: IntegrationProvider) => {
      if (!user?.id) throw new Error("Not authenticated");
      const providerPath = provider.replace("_", "-");
      const res = await fetch(
        `${API_URL}/integrations/${providerPath}/disconnect?user_id=${user.id}`,
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error("Failed to disconnect");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["integrations"] });
    },
  });

  return {
    connections: connectionsQuery.data || [],
    isLoading: connectionsQuery.isLoading,
    connect: connectMutation.mutate,
    disconnect: disconnectMutation.mutate,
    isConnecting: connectMutation.isPending,
  };
}
