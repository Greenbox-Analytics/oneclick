import { useQuery } from "@tanstack/react-query";
import { apiFetch, API_URL } from "@/lib/apiFetch";

export interface ToolRow {
  tool: string;
  opens: number;
  completions: number;
  openers: number;
  converters: number;
  completion_rate: number;
  last_used: string | null;
}

export interface FunnelStep {
  label: string;
  users: number;
}

export interface ToolFunnel {
  tool: string;
  steps: FunnelStep[];
  error_rate: number;
  completed_events: number;
  failed_events: number;
}

export interface RegistryLifecycle {
  created: number;
  submitted: number;
  registered: number;
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
  funnels: ToolFunnel[];
  registry_lifecycle: RegistryLifecycle | null;
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
