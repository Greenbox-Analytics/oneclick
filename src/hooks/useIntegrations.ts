import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { toast } from "sonner";
import type { IntegrationConnection, IntegrationProvider } from "@/types/integrations";

const PROVIDER_NAMES: Record<IntegrationProvider, string> = {
  google_drive: "Google Drive",
  slack: "Slack",
  notion: "Notion",
  monday: "Monday.com",
};

// Map provider key to backend URL segment
const PROVIDER_URL_SEGMENT: Record<IntegrationProvider, string> = {
  google_drive: "google-drive",
  slack: "slack",
  notion: "notion",
  monday: "monday",
};

export function useIntegrations() {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const connectionsQuery = useQuery<IntegrationConnection[]>({
    queryKey: ["integrations", user?.id],
    queryFn: async () => {
      const data = await apiFetch<{ connections: IntegrationConnection[] }>(
        `${API_URL}/integrations/connections`
      );
      return data.connections;
    },
    enabled: !!user?.id,
  });

  const connectMutation = useMutation({
    mutationFn: async (provider: IntegrationProvider) => {
      const segment = PROVIDER_URL_SEGMENT[provider];
      const data = await apiFetch<{ auth_url: string }>(
        `${API_URL}/integrations/${segment}/auth`
      );
      // Redirect browser to OAuth provider
      window.location.href = data.auth_url;
    },
    onError: (error: Error, provider) => {
      toast.error(`Failed to connect ${PROVIDER_NAMES[provider]}: ${error.message}`);
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: async (provider: IntegrationProvider) => {
      const segment = PROVIDER_URL_SEGMENT[provider];
      await apiFetch(`${API_URL}/integrations/${segment}/disconnect`, {
        method: "DELETE",
      });
    },
    onSuccess: (_data, provider) => {
      toast.success(`${PROVIDER_NAMES[provider]} disconnected`);
      queryClient.invalidateQueries({ queryKey: ["integrations"] });
    },
    onError: (error: Error, provider) => {
      toast.error(`Failed to disconnect ${PROVIDER_NAMES[provider]}: ${error.message}`);
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
