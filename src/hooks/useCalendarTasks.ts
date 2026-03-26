import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { BoardTask } from "@/types/integrations";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

export function useCalendarTasks(start: string, end: string) {
  const { user } = useAuth();

  const query = useQuery<BoardTask[]>({
    queryKey: ["board-tasks-calendar", start, end],
    queryFn: async () => {
      if (!user?.id) return [];
      const params = new URLSearchParams({
        user_id: user.id,
        start,
        end,
      });
      const res = await fetch(`${API_URL}/boards/calendar?${params}`);
      if (!res.ok) throw new Error("Failed to fetch calendar tasks");
      const data = await res.json();
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
