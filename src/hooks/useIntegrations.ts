import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import type { IntegrationConnection, IntegrationProvider } from "@/types/integrations";

const PROVIDER_NAMES: Record<IntegrationProvider, string> = {
  google_drive: "Google Drive",
  slack: "Slack",
  notion: "Notion",
  monday: "Monday.com",
};

export function useIntegrations() {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const connectionsQuery = useQuery<IntegrationConnection[]>({
    queryKey: ["integrations", user?.id],
    queryFn: async () => {
      // Integration connections not yet configured — return empty
      return [];
    },
    enabled: !!user?.id,
  });

  const connectMutation = useMutation({
    mutationFn: async (provider: IntegrationProvider) => {
      // Temporarily show "Coming Soon" instead of hitting OAuth routes
      toast.info(`${PROVIDER_NAMES[provider]} integration coming soon!`);
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: async (provider: IntegrationProvider) => {
      toast.info(`${PROVIDER_NAMES[provider]} integration coming soon!`);
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
