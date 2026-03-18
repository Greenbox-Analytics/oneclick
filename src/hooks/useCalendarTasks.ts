import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import {
  startOfMonth,
  endOfMonth,
  startOfWeek,
  endOfWeek,
  format,
} from "date-fns";
import type { BoardTask } from "@/types/integrations";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

export function useCalendarTasks(year: number, month: number) {
  const { user } = useAuth();

  const monthDate = new Date(year, month, 1);
  const gridStart = format(startOfWeek(startOfMonth(monthDate)), "yyyy-MM-dd");
  const gridEnd = format(endOfWeek(endOfMonth(monthDate)), "yyyy-MM-dd");

  const query = useQuery<BoardTask[]>({
    queryKey: ["board-tasks-calendar", year, month],
    queryFn: async () => {
      if (!user?.id) return [];
      const params = new URLSearchParams({
        user_id: user.id,
        start: gridStart,
        end: gridEnd,
      });
      const res = await fetch(`${API_URL}/boards/calendar?${params}`);
      if (!res.ok) throw new Error("Failed to fetch calendar tasks");
      const data = await res.json();
      return data.tasks;
    },
    enabled: !!user?.id,
  });

  return {
    tasks: query.data || [],
    isLoading: query.isLoading,
    gridStart,
    gridEnd,
  };
}
