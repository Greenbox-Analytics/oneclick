import { useQuery } from "@tanstack/react-query";
import { apiFetch, API_URL } from "@/lib/apiFetch";
import { useAuth } from "@/contexts/AuthContext";
import { peekCachedAnalyticsContext } from "@/hooks/useAnalyticsContext";

export interface PageRow {
  path: string;
  views: number;
  unique_visitors: number;
  avg_dwell_ms: number;
  bounce_rate: number;
}

export interface PageFlowEdge {
  from_path: string;
  to_path: string;
  count: number;
}

export interface DailyVisitorPoint {
  date: string;
  views: number;
  unique_visitors: number;
}

export interface BehaviorSummary {
  available: boolean;
  window: string;
  cohort: string;
  total_pageviews: number;
  unique_visitors: number;
  pageviews_per_visitor: number;
  top_pages: PageRow[];
  daily_visitors: DailyVisitorPoint[];
  top_flows: PageFlowEdge[];
  reason?: string | null;
}

type WindowKey = "7d" | "30d" | "all";
type Cohort = "testers" | "all";

export function useAdminBehaviorSummary(opts: { window: WindowKey; cohort: Cohort }) {
  const { user } = useAuth();
  const ctx = peekCachedAnalyticsContext(user?.id);
  // Let unknown (null cache, e.g., first paint) through; backend `require_admin`
  // is the authoritative gate. Only skip when we KNOW the user is not admin.
  const enabled = !!user && ctx?.is_admin !== false;

  return useQuery<BehaviorSummary>({
    queryKey: ["admin", "behavior-summary", opts.window, opts.cohort],
    queryFn: () =>
      apiFetch<BehaviorSummary>(
        `${API_URL}/admin/analytics/behavior?window=${opts.window}&cohort=${opts.cohort}`,
      ),
    enabled,
    staleTime: 60_000,
  });
}
