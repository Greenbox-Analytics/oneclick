import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

// ---------------------------------------------------------------------------
// TypeScript interfaces (mirroring src/backend/oneclick/royalties/models.py)
// ---------------------------------------------------------------------------

/** A single month bucket for time-series analytics.
 *  `amount` is used in overview paid_by_month.
 *  `earned` + `paid` are used in artist/payee by_month.
 */
export interface MonthPoint {
  month: string; // "YYYY-MM"
  amount: number;
  earned: number;
  paid: number;
}

export interface TopOwed {
  payee_id: string;
  display_name: string;
  owed: number;
}

export interface OverviewOut {
  base: string;
  outstanding_total: number;
  payees_owed_count: number;
  drafted_total: number;
  draft_count: number;
  paid_total: number;
  paid_last_30d: number;
  paid_by_month: MonthPoint[];
  top_owed: TopOwed[];
  unconvertible_count: number;
}

export interface ArtistAnalyticsSummary {
  earned_total: number;
  owed_now: number;
  paid_total: number;
}

export interface ArtistAnalyticsOut {
  artist_id: string;
  base: string;
  summary: ArtistAnalyticsSummary;
  by_month: MonthPoint[];
  unconvertible_count: number;
}

export interface PayeeAnalyticsSummary {
  earned_total: number;
  paid_total: number;
  owed: number;
}

export interface PayeeAnalyticsOut {
  payee_id: string;
  display_name: string;
  base: string;
  summary: PayeeAnalyticsSummary;
  by_month: MonthPoint[];
  unconvertible_count: number;
}

// ---------------------------------------------------------------------------
// Query hooks
// ---------------------------------------------------------------------------

/** GET /oneclick/royalties/analytics/overview?base={base} */
export function useRoyaltyOverview(base: string) {
  const { user } = useAuth();
  return useQuery<OverviewOut>({
    queryKey: ["royalty-analytics-overview", user?.id, base],
    queryFn: () =>
      apiFetch<OverviewOut>(
        `${API_URL}/oneclick/royalties/analytics/overview?base=${encodeURIComponent(base)}`,
      ),
    enabled: !!user?.id,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });
}

/** GET /oneclick/royalties/analytics/artist/{artistId}?base={base} */
export function useArtistRoyaltyAnalytics(artistId: string, base: string) {
  const { user } = useAuth();
  return useQuery<ArtistAnalyticsOut>({
    queryKey: ["royalty-analytics-artist", user?.id, artistId, base],
    queryFn: () =>
      apiFetch<ArtistAnalyticsOut>(
        `${API_URL}/oneclick/royalties/analytics/artist/${artistId}?base=${encodeURIComponent(base)}`,
      ),
    enabled: !!user?.id && !!artistId,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });
}

/** GET /oneclick/royalties/analytics/payee/{payeeId}?base={base} */
export function usePayeeRoyaltyAnalytics(payeeId: string, base: string) {
  const { user } = useAuth();
  return useQuery<PayeeAnalyticsOut>({
    queryKey: ["royalty-analytics-payee", user?.id, payeeId, base],
    queryFn: () =>
      apiFetch<PayeeAnalyticsOut>(
        `${API_URL}/oneclick/royalties/analytics/payee/${payeeId}?base=${encodeURIComponent(base)}`,
      ),
    enabled: !!user?.id && !!payeeId,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });
}
