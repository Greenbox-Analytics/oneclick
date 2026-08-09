// src/hooks/useAdminOrgs.ts
// Admin Organizations tab data plumbing. All routes require platform admin;
// shapes mirror subscriptions/admin_service.list_orgs_admin / get_org_pool.
import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import type { AdminLedgerEntry } from "@/hooks/useAdmin";

export interface AdminOrgRow {
  id: string;
  name: string | null;
  status: string;
  archivedAt: string | null;
  memberCount: number;
  bundleBalance: number;
  reserveBalance: number;
  monthlyDispersalCredits: number;
  activationFloor: number;
  cumulativePaidIn: number;
}

export function useAdminOrgs(): UseQueryResult<AdminOrgRow[]> {
  return useQuery({
    queryKey: ["admin", "orgs"],
    queryFn: async () =>
      (await apiFetch<{ orgs: AdminOrgRow[] }>(`${API_URL}/admin/orgs`)).orgs,
    staleTime: 30_000,
  });
}

export interface AdminOrgPool {
  orgId: string;
  status: string;
  archivedAt: string | null;
  poolBalance: number;
  cumulativePaidIn: number;
  ledger: AdminLedgerEntry[];
}

export function useAdminOrgPool(orgId: string | null): UseQueryResult<AdminOrgPool> {
  return useQuery({
    queryKey: ["admin", "orgs", orgId, "pool"],
    queryFn: () => apiFetch<AdminOrgPool>(`${API_URL}/admin/orgs/${orgId}/pool`),
    enabled: !!orgId,
    staleTime: 15_000,
  });
}

export function useAdminOrgMutations() {
  const qc = useQueryClient();
  // ["admin","orgs"] prefix-matches ["admin","orgs",orgId,"pool"] under React
  // Query's default exact:false, so one invalidation covers both.
  const invalidate = () => qc.invalidateQueries({ queryKey: ["admin", "orgs"] });

  const grantCredits = useMutation({
    mutationFn: (args: { orgId: string; amount: number; reason: string; idempotencyKey: string }) =>
      apiFetch<{ granted: number; result: { duplicate?: boolean; activated?: boolean } }>(
        `${API_URL}/admin/orgs/${args.orgId}/pool/grant`,
        {
          method: "POST",
          body: JSON.stringify({
            amount: args.amount,
            reason: args.reason,
            idempotency_key: args.idempotencyKey,
          }),
        },
      ),
    onSuccess: invalidate,
  });

  const setDispersal = useMutation({
    mutationFn: (args: { orgId: string; monthlyDispersalCredits: number }) =>
      apiFetch(`${API_URL}/admin/orgs/${args.orgId}/dispersal`, {
        method: "PUT",
        body: JSON.stringify({ monthly_dispersal_credits: args.monthlyDispersalCredits }),
      }),
    onSuccess: invalidate,
  });

  // suspend/reactivate 409 on a status that can't make the transition (a
  // pending org has never been activated, so it can't be reinstated) — the
  // caller surfaces the backend's message rather than pre-guessing it.
  const setStatus = useMutation({
    mutationFn: (args: { orgId: string; action: "suspend" | "reactivate" }) =>
      apiFetch<{ id: string; status: string }>(
        `${API_URL}/admin/orgs/${args.orgId}/${args.action}`,
        { method: "POST" },
      ),
    onSuccess: invalidate,
  });

  return { grantCredits, setDispersal, setStatus };
}
