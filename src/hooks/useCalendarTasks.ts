import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { BoardTask } from "@/types/integrations";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export function useCalendarTasks(start: string, end: string) {
  const { user } = useAuth();

  const query = useQuery<BoardTask[]>({
    queryKey: ["board-tasks-calendar", start, end],
    queryFn: async () => {
      if (!user?.id) return [];
      const params = new URLSearchParams({ start, end });
      const data = await apiFetch<{ tasks: BoardTask[] }>(`${API_URL}/boards/calendar?${params}`);
      return data.tasks;
    },
    enabled: !!user?.id && !!start && !!end,
    placeholderData: keepPreviousData,
  });

  return {
    tasks: query.data || [],
    isLoading: query.isLoading,
  };
}
