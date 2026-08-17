import { useQuery, type QueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export type CreditAction = "oneclick_run" | "registry_parse" | "zoe_message" | "split_sheet";

export interface CreditToolUsage {
  action: CreditAction;
  /** The action's BASE RATE — what a run costs, barring a pathological outlier. */
  price: number | null;
  count: number;
  /** Credits actually charged: max(base, metered) summed over the period. */
  spent: number;
}

/** GET /me/credits/usage — per-tool credit spend for the current period. */
export interface CreditUsage {
  /** false when CREDITS_ENABLED is off (or on error) → hide credit surfaces. */
  enabled: boolean;
  periodStart?: string | null;
  periodEnd?: string | null;
  monthlyGrant?: number;
  /** null in org context for a non-admin member — the pool is admin-only. */
  bundleBalance?: number | null;
  reserveBalance?: number | null;
  balance?: number | null;
  overageThisPeriod?: number;
  /** Org context only — the member's ceiling and usage against it. */
  memberCap?: number | null;
  memberCapUsed?: number;
  /** Always the CALLER's own spend, in org context too. Org-wide rollups are
   * admin-only and live on GET /orgs/{id}/usage. */
  tools?: CreditToolUsage[];
}

export function useCreditUsage(enabled = true) {
  const { user } = useAuth();
  return useQuery<CreditUsage>({
    queryKey: ["credit-usage", user?.id],
    queryFn: () => apiFetch<CreditUsage>(`${API_URL}/me/credits/usage`),
    enabled: !!user?.id && enabled,
    staleTime: 30_000,
  });
}

/** Refresh every credit surface (header ticker, chips, usage card) after a
 * metered action completes or credits are added. */
export function invalidateCreditSurfaces(qc: QueryClient): void {
  qc.invalidateQueries({ queryKey: ["entitlements"] });
  qc.invalidateQueries({ queryKey: ["credit-usage"] });
}
