import { useQuery, type QueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export type CreditAction = "oneclick_run" | "registry_parse" | "zoe_message";

export interface CreditToolUsage {
  action: CreditAction;
  price: number | null;
  count: number;
  spent: number;
}

/** GET /me/credits/usage — per-tool credit spend for the current period. */
export interface CreditUsage {
  /** false when CREDITS_ENABLED is off (or on error) → hide credit surfaces. */
  enabled: boolean;
  periodStart?: string | null;
  periodEnd?: string | null;
  monthlyGrant?: number;
  bundleBalance?: number;
  reserveBalance?: number;
  overageThisPeriod?: number;
  /** Org context only — the member's ceiling and usage against it. */
  memberCap?: number | null;
  memberCapUsed?: number;
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
