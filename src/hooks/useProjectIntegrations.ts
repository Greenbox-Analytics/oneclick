import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";

interface ProjectNotificationSetting {
  id: string;
  project_id: string;
  event_type: string;
  enabled: boolean;
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
