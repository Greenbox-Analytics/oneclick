import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { ApiError, API_URL, apiFetch } from "@/lib/apiFetch";

// ---------------------------------------------------------------------------
// useIsAdmin — returns { isAdmin, loading }
// ---------------------------------------------------------------------------

export function useIsAdmin(): { isAdmin: boolean; loading: boolean } {
  const { user } = useAuth();

  const query = useQuery({
    queryKey: ["admin", "me", user?.id],
    queryFn: async () => {
      try {
        return await apiFetch<{ email: string; isAdmin: boolean }>(`${API_URL}/admin/me`);
      } catch (err: unknown) {
        // 403 means "not admin" — treat as clean negative, not hook error.
        // Status comes from ApiError; robust to backend message changes.
        if (err instanceof ApiError && err.status === 403) {
          return { email: "", isAdmin: false };
        }
        throw err;
      }
    },
    enabled: !!user?.id,
    staleTime: 5 * 60_000,
    retry: false,
  });

  if (query.isLoading || !query.data) {
    return { isAdmin: false, loading: true };
  }
  return { isAdmin: !!query.data.isAdmin, loading: false };
}

// ---------------------------------------------------------------------------
// User listing + detail
// ---------------------------------------------------------------------------

export interface AdminUserRow {
  id: string;
  email: string | null;
  tier: "free" | "pro";
  has_override: boolean;
  created_at: string | null;
}

export interface AdminUsersResponse {
  users: AdminUserRow[];
  page: number;
  per_page: number;
  /** True when this page is full and there might be more — frontend uses this
   * for the "Next →" button. The Supabase auth admin API doesn't expose a true
   * total count, so we don't fake one. */
  has_more: boolean;
}

export function useAdminUsers(
  search: string,
  page: number,
): UseQueryResult<AdminUsersResponse> {
  return useQuery({
    queryKey: ["admin", "users", { search, page }],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (search) params.set("search", search);
      params.set("page", String(page));
      params.set("per_page", "25");
      return apiFetch<AdminUsersResponse>(`${API_URL}/admin/users?${params.toString()}`);
    },
    staleTime: 30_000,
  });
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

export interface OverridePayloadInput {
  max_artists?: number | null;
  max_projects?: number | null;
  max_tasks?: number | null;
  max_storage_bytes?: number | null;
  max_split_sheets_per_month?: number | null;
  zoe_enabled?: boolean | null;
  oneclick_enabled?: boolean | null;
  registry_enabled?: boolean | null;
  integrations_allowed?: string[] | null;
  reason?: string | null;
  expires_days?: number | null;
}

export function useAdminMutations() {
  const qc = useQueryClient();

  const invalidateUser = (userId: string) => {
    qc.invalidateQueries({ queryKey: ["admin", "users"] });
    qc.invalidateQueries({ queryKey: ["admin", "users", userId, "detail"] });
    qc.invalidateQueries({ queryKey: ["entitlements", userId] });
  };

  const grantPro = useMutation({
    mutationFn: async (userId: string) =>
      apiFetch(`${API_URL}/admin/users/${userId}/grant`, { method: "POST" }),
    onSuccess: (_data, userId) => invalidateUser(userId),
  });

  const revokePro = useMutation({
    mutationFn: async (userId: string) =>
      apiFetch(`${API_URL}/admin/users/${userId}/revoke`, { method: "POST" }),
    onSuccess: (_data, userId) => invalidateUser(userId),
  });

  const applyOverride = useMutation({
    mutationFn: async (args: { userId: string; payload: OverridePayloadInput }) =>
      apiFetch(`${API_URL}/admin/users/${args.userId}/override`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(args.payload),
      }),
    onSuccess: (_data, args) => invalidateUser(args.userId),
  });

  const clearOverride = useMutation({
    mutationFn: async (userId: string) =>
      apiFetch(`${API_URL}/admin/users/${userId}/override`, { method: "DELETE" }),
    onSuccess: (_data, userId) => invalidateUser(userId),
  });

  return { grantPro, revokePro, applyOverride, clearOverride };
}

// ---------------------------------------------------------------------------
// Pro requests
// ---------------------------------------------------------------------------

export type ProRequestStatus = "new" | "contacted" | "converted" | "declined";

export interface ProRequestRow {
  id: string;
  email: string;
  message: string | null;
  user_id: string | null;
  status: ProRequestStatus;
  created_at: string;
}

export function useProRequests(status?: ProRequestStatus): UseQueryResult<ProRequestRow[]> {
  return useQuery({
    queryKey: ["admin", "pro-requests", status ?? "all"],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (status) params.set("status", status);
      const qs = params.toString();
      return apiFetch<ProRequestRow[]>(`${API_URL}/admin/pro-requests${qs ? `?${qs}` : ""}`);
    },
    staleTime: 30_000,
  });
}
