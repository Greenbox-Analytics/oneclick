import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";

export interface SlackNotification {
  id: string;
  user_id: string;
  project_id: string | null;
  channel_id: string;
  sender_name: string;
  sender_avatar_url: string | null;
  message_text: string;
  slack_ts: string;
  is_read: boolean;
  created_at: string;
  project?: { id: string; name: string };
}

export function useSlackNotifications(unreadOnly = false) {
  const { user } = useAuth();

  return useQuery<SlackNotification[]>({
    queryKey: ["slack-notifications", user?.id, unreadOnly],
    queryFn: async () => {
      if (!user?.id) return [];
      let query = supabase
        .from("slack_notifications")
        .select("*, projects(id, name)")
        .eq("user_id", user.id)
        .gte("created_at", new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString())
        .order("created_at", { ascending: false })
        .limit(50);

      if (unreadOnly) {
        query = query.eq("is_read", false);
      }

      const { data } = await query;
      return (data || []).map((n) => ({
        ...n,
        project: n.projects || null,
      }));
    },
    enabled: !!user?.id,
    refetchInterval: 30000,
  });
}

export function useSlackUnreadCount() {
  const { data } = useSlackNotifications(true);
  return data?.length || 0;
}

export function useMarkSlackRead() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (notificationId: string) => {
      const { error } = await supabase
        .from("slack_notifications")
        .update({ is_read: true })
        .eq("id", notificationId);
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["slack-notifications"] });
    },
  });
}

export function useMarkAllSlackRead() {
  const { user } = useAuth();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      if (!user?.id) return;
      const { error } = await supabase
        .from("slack_notifications")
        .update({ is_read: true })
        .eq("user_id", user.id)
        .eq("is_read", false);
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["slack-notifications"] });
    },
  });
}
