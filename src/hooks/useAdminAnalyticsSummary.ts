import { useQuery } from "@tanstack/react-query";
import { apiFetch, API_URL } from "@/lib/apiFetch";

export interface ToolRow {
  tool: string;
  opens: number;
  completions: number;
  completion_rate: number;
  last_used: string | null;
}

export interface SparklinePoint {
  date: string;
  value: number;
}

export interface AnalyticsSummary {
  available: boolean;
  window: string;
  cohort: string;
  active_users: number;
  total_users: number;
  tool_actions: number;
  top_tool: string | null;
  top_tool_share: number;
  funnel_completion_avg: number;
  per_tool: ToolRow[];
  sparkline: SparklinePoint[];
  reason?: string;
}

interface UseArgs {
  window: "7d" | "30d" | "all";
  cohort: "testers" | "all";
}

export function useAdminAnalyticsSummary({ window, cohort }: UseArgs) {
  return useQuery<AnalyticsSummary>({
    queryKey: ["admin-analytics-summary", window, cohort],
    queryFn: async () =>
      apiFetch<AnalyticsSummary>(
        `${API_URL}/admin/analytics/summary?window=${window}&cohort=${cohort}`,
      ),
    staleTime: 60_000,
  });
}
