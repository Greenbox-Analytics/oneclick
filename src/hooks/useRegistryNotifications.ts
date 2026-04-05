import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL } from "@/lib/apiFetch";

export interface RegistryNotification {
  id: string;
  user_id: string;
  work_id: string | null;
  type: string;
  title: string;
  message: string;
  read: boolean;
  metadata: Record<string, unknown>;
  created_at: string;
}

export function useRegistryNotifications(unreadOnly = false) {
  const { user } = useAuth();
  return useQuery<RegistryNotification[]>({
    queryKey: ["registry-notifications", user?.id, unreadOnly],
    queryFn: async () => {
      if (!user?.id) return [];
      const res = await fetch(
        `${API_URL}/registry/notifications?user_id=${user.id}&unread_only=${unreadOnly}`
      );
      if (!res.ok) return [];
      const data = await res.json();
      return data.notifications || [];
    },
    enabled: !!user?.id,
    refetchInterval: 30000,
  });
}

export function useUnreadCount() {
  const { data } = useRegistryNotifications(true);
  return data?.length || 0;
}

export function useMarkNotificationRead() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (notificationId: string) => {
      await fetch(
        `${API_URL}/registry/notifications/${notificationId}/read?user_id=${user!.id}`,
        { method: "POST" }
      );
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["registry-notifications"] }),
  });
}

export function useMarkAllRead() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      await fetch(
        `${API_URL}/registry/notifications/read-all?user_id=${user!.id}`,
        { method: "POST" }
      );
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["registry-notifications"] }),
  });
}
