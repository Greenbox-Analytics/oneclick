import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { toast } from "sonner";
import type { SlackChannel, NotificationSetting } from "@/types/integrations";

export function useSlackChannels(enabled: boolean = true) {
  const { user } = useAuth();

  return useQuery<SlackChannel[]>({
    queryKey: ["slack-channels", user?.id],
    queryFn: async () => {
      const data = await apiFetch<{ channels: SlackChannel[] }>(
        `${API_URL}/integrations/slack/channels`
      );
      return data.channels;
    },
    enabled: !!user?.id && enabled,
  });
}

export function useSlackSettings(enabled: boolean = true) {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const settingsQuery = useQuery<NotificationSetting[]>({
    queryKey: ["slack-settings", user?.id],
    queryFn: async () => {
      const data = await apiFetch<{ settings: NotificationSetting[] }>(
        `${API_URL}/integrations/slack/settings`
      );
      return data.settings;
    },
    enabled: !!user?.id && enabled,
  });

  const updateMutation = useMutation({
    mutationFn: async (params: {
      event_type: string;
      enabled: boolean;
      channel_id?: string;
    }) => {
      return apiFetch(`${API_URL}/integrations/slack/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["slack-settings"] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to update setting: ${error.message}`);
    },
  });

  return {
    settings: settingsQuery.data || [],
    isLoading: settingsQuery.isLoading,
    updateSetting: updateMutation.mutate,
    isUpdating: updateMutation.isPending,
  };
}
