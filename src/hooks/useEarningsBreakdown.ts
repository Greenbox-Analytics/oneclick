import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export type BreakdownDimension = "country" | "month" | "year" | "format" | "vendor";

export interface BreakdownRow {
  key: string;
  net_payable: number;
  row_count: number;
  percent_of_total: number;
}

export interface BreakdownResponse {
  dimension: BreakdownDimension;
  total: number;
  row_count: number;
  rows: BreakdownRow[];
}

export function useEarningsBreakdown(
  calculationId: string | null | undefined,
  dimension: BreakdownDimension,
) {
  const { user } = useAuth();
  return useQuery<BreakdownResponse>({
    queryKey: ["oneclick-breakdown", user?.id, calculationId, dimension],
    queryFn: async () => {
      if (!calculationId) {
        return { dimension, total: 0, row_count: 0, rows: [] };
      }
      return apiFetch<BreakdownResponse>(
        `${API_URL}/oneclick/calculations/${calculationId}/breakdown?dimension=${dimension}`,
      );
    },
    enabled: !!user?.id && !!calculationId,
    // A calculation's rows are immutable once saved, so each (calculation,
    // dimension) result never goes stale — fetch each category at most once
    // per session instead of refetching when you toggle back to a tab.
    staleTime: Infinity,
  });
}
