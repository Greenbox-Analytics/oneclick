import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

interface ProjectNotificationSetting {
  id: string;
  project_id: string;
  event_type: string;
  enabled: boolean;
}

export function useProjectSlackChannel(projectId: string | undefined) {
  const queryClient = useQueryClient();

  const channelQuery = useQuery<string | null>({
    queryKey: ["project-slack-channel", projectId],
    queryFn: async () => {
      if (!projectId) return null;
      const { data } = await supabase
        .from("projects")
        .select("slack_channel_id")
        .eq("id", projectId)
        .single();
      return data?.slack_channel_id || null;
    },
    enabled: !!projectId,
  });

  const updateChannel = useMutation({
    mutationFn: async ({ channelId }: { channelId: string | null }) => {
      if (!projectId) throw new Error("No project ID");
      const { error } = await supabase
        .from("projects")
        .update({ slack_channel_id: channelId })
        .eq("id", projectId);
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["project-slack-channel", projectId] });
      toast.success("Slack channel updated");
    },
    onError: (err: Error) => {
      toast.error(`Failed to update channel: ${err.message}`);
    },
  });

  return {
    channelId: channelQuery.data,
    isLoading: channelQuery.isLoading,
    updateChannel: updateChannel.mutate,
  };
}

export function useProjectNotificationSettings(projectId: string | undefined) {
  const queryClient = useQueryClient();

  const settingsQuery = useQuery<ProjectNotificationSetting[]>({
    queryKey: ["project-notification-settings", projectId],
    queryFn: async () => {
      if (!projectId) return [];
      const { data } = await supabase
        .from("project_notification_settings")
        .select("*")
        .eq("project_id", projectId);
      return data || [];
    },
    enabled: !!projectId,
  });

  const toggleEvent = useMutation({
    mutationFn: async ({ eventType, enabled }: { eventType: string; enabled: boolean }) => {
      if (!projectId) throw new Error("No project ID");
      const { error } = await supabase
        .from("project_notification_settings")
        .upsert(
          { project_id: projectId, event_type: eventType, enabled },
          { onConflict: "project_id,event_type" }
        );
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["project-notification-settings", projectId] });
    },
  });

  const isEventEnabled = (eventType: string) => {
    const setting = settingsQuery.data?.find((s) => s.event_type === eventType);
    return setting?.enabled ?? false;
  };

  return {
    settings: settingsQuery.data || [],
    isLoading: settingsQuery.isLoading,
    toggleEvent: toggleEvent.mutate,
    isEventEnabled,
  };
}
