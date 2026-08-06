import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import type { BoardTask } from "@/types/integrations";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export function useCalendarTasks(start: string, end: string, boardId?: string) {
  const { user } = useAuth();

  const query = useQuery<BoardTask[]>({
    queryKey: ["board-tasks-calendar", start, end, boardId],
    queryFn: async () => {
      if (!user?.id) return [];
      const params = new URLSearchParams({ start, end });
      if (boardId) params.set("board_id", boardId);
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

export type CalendarFeed = {
  scope: string;
  /** Full branded name — this is what the calendar is called in Google/Apple. */
  name: string;
  /** Set only for team feeds; the picker labels rows by it. */
  team_name: string | null;
  url: string;
  webcal_url: string;
};

/** The user's private .ics subscription URLs — one per calendar (everything, personal, each team). */
export function useCalendarFeeds() {
  const { user } = useAuth();

  const query = useQuery<CalendarFeed[]>({
    queryKey: ["calendar-feeds", user?.id],
    queryFn: async () => {
      const data = await apiFetch<{ feeds: CalendarFeed[] }>(`${API_URL}/boards/calendar/feeds`);
      return data.feeds;
    },
    enabled: !!user?.id,
    staleTime: Infinity,
  });

  return { feeds: query.data || [], isLoading: query.isLoading };
}
