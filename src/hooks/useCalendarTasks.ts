import { useQuery, useMutation, useQueryClient, keepPreviousData } from "@tanstack/react-query";
import { toast } from "sonner";
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

/** A read-only event from a subscribed external calendar. Never a task. */
export type ExternalEvent = {
  uid: string;
  title: string;
  date: string;
  time: string | null;
  all_day: boolean;
  source_id: string;
  source_name: string;
};

export type CalendarSubscription = {
  id: string;
  name: string;
  last_fetched_at: string | null;
  last_error: string | null;
};

/**
 * Events from imported calendars. Deliberately a separate query from
 * useCalendarTasks: these hit third-party servers, so a slow or dead feed must
 * never hold up (or fail) the user's own tasks.
 */
export function useExternalEvents(start: string, end: string) {
  const { user } = useAuth();

  const query = useQuery<ExternalEvent[]>({
    queryKey: ["calendar-external", start, end],
    queryFn: async () => {
      const params = new URLSearchParams({ start, end });
      const data = await apiFetch<{ events: ExternalEvent[] }>(`${API_URL}/boards/calendar/external?${params}`);
      return data.events;
    },
    enabled: !!user?.id && !!start && !!end,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60 * 1000,
    retry: false,
  });

  return { events: query.data || [] };
}

export function useCalendarSubscriptions() {
  const { user } = useAuth();
  const qc = useQueryClient();

  const query = useQuery<CalendarSubscription[]>({
    queryKey: ["calendar-subscriptions"],
    queryFn: async () => {
      const data = await apiFetch<{ subscriptions: CalendarSubscription[] }>(
        `${API_URL}/boards/calendar/subscriptions`,
      );
      return data.subscriptions;
    },
    enabled: !!user?.id,
  });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["calendar-subscriptions"] });
    qc.invalidateQueries({ queryKey: ["calendar-external"] });
  };

  const add = useMutation({
    mutationFn: (body: { url: string; name?: string }) =>
      apiFetch<CalendarSubscription>(`${API_URL}/boards/calendar/subscriptions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: (sub) => {
      invalidate();
      toast.success(`Imported "${sub.name}"`);
    },
    onError: (e: Error) => toast.error(e.message),
  });

  const remove = useMutation({
    mutationFn: (id: string) =>
      apiFetch(`${API_URL}/boards/calendar/subscriptions/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      invalidate();
      toast.success("Calendar removed");
    },
    onError: (e: Error) => toast.error(e.message),
  });

  return {
    subscriptions: query.data || [],
    isLoading: query.isLoading,
    add: add.mutate,
    isAdding: add.isPending,
    remove: remove.mutate,
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
