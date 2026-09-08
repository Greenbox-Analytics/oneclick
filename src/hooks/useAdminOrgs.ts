// src/hooks/useAdminOrgs.ts
// Admin Organizations tab data plumbing. All routes require platform admin;
// shapes mirror subscriptions/admin_service.list_orgs_admin / get_org_pool.
import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import type { AdminLedgerEntry } from "@/hooks/useAdmin";
import type { OrgUsage, UsageRange } from "@/hooks/useOrgs";
import type { PartnerKeysPayload } from "@/hooks/usePartnerKeys";

export interface AdminOrgRow {
  id: string;
  name: string | null;
  status: string;
  archivedAt: string | null;
  kind?: "self_serve" | "enterprise" | null;
  /** On by default for enterprise orgs. */
  partnerApiEnabled: boolean;
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

  const setPartnerApi = useMutation({
    mutationFn: (args: { orgId: string; enabled: boolean }) =>
      apiFetch<{ org_id: string; partner_api_enabled: boolean }>(
        `${API_URL}/admin/orgs/${args.orgId}/partner-api`,
        { method: "PUT", body: JSON.stringify({ enabled: args.enabled }) },
      ),
    onSuccess: invalidate,
  });

  return { grantCredits, setDispersal, setStatus, setPartnerApi };
}

// Read-only mirrors of the org-side endpoints for the Organizations drawer:
// same payloads on admin routes, so OrgUsageAnalysis can take these instead.
export function useAdminOrgUsage(orgId?: string, range: UsageRange = "mtd"): UseQueryResult<OrgUsage> {
  return useQuery({
    queryKey: ["admin", "orgs", orgId, "usage", range],
    queryFn: () => apiFetch<OrgUsage>(`${API_URL}/admin/orgs/${orgId}/usage?range=${range}`),
    enabled: !!orgId,
    staleTime: 15_000,
  });
}

/** One hit of POST /admin/partner-keys/lookup. `status` is DERIVED here
 * (expiry folded in), unlike PartnerKey.status which is what's stored. */
export interface KeyTraceHit {
  id: string;
  org_id: string;
  org_name: string | null;
  label: string;
  key_prefix: string;
  status: "active" | "revoked" | "expired";
  expires_at: string | null;
  created_at: string;
  last_used_at: string | null;
  /** Distinct source IPs over the last 7 days, busiest first. One key
   * answering from two continents is the leak signal. */
  recent_ips: { ip: string; requests: number; last_seen: string }[];
}

/** "We found a key in the wild — whose is it?" A mutation, not a query: the
 * pasted value is secret-adjacent, so it rides in a POST body rather than a
 * URL (which lands in access logs and browser history) and never becomes a
 * cache key. Only the first 12 characters are used server-side. */
export function useAdminKeyLookup() {
  return useMutation({
    mutationFn: (key: string) =>
      apiFetch<{ keys: KeyTraceHit[] }>(`${API_URL}/admin/partner-keys/lookup`, {
        method: "POST",
        body: JSON.stringify({ key }),
      }),
  });
}

/** Kill a key from the Msanii console. Same one-way revoke the org's own
 * admins have — the row survives (its spend keeps counting), the credential
 * stops resolving on the very next request. Nothing un-revokes: the team has
 * to mint a replacement. */
export function useAdminRevokePartnerKey() {
  const qc = useQueryClient();
  return useMutation<{ status: string }, Error, { orgId: string; keyId: string }>({
    mutationFn: ({ orgId, keyId }) =>
      apiFetch(`${API_URL}/admin/orgs/${orgId}/partner-keys/${keyId}`, { method: "DELETE" }),
    // Toasts live in the panel, like every other mutation in this file.
    onSuccess: (_data, { orgId }) =>
      qc.invalidateQueries({ queryKey: ["admin", "orgs", orgId, "partner-keys"] }),
  });
}

export function useAdminPartnerKeys(orgId?: string): UseQueryResult<PartnerKeysPayload> {
  return useQuery({
    queryKey: ["admin", "orgs", orgId, "partner-keys"],
    queryFn: () => apiFetch<PartnerKeysPayload>(`${API_URL}/admin/orgs/${orgId}/partner-keys`),
    enabled: !!orgId,
    staleTime: 15_000,
  });
}
