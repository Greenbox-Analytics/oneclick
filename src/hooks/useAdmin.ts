import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { peekCachedAnalyticsContext } from "@/hooks/useAnalyticsContext";
import { ApiError, API_URL, apiFetch } from "@/lib/apiFetch";

// ---------------------------------------------------------------------------
// useIsAdmin — returns { isAdmin, loading }
// ---------------------------------------------------------------------------

export function useIsAdmin(): { isAdmin: boolean; loading: boolean } {
  const { user } = useAuth();

  // Fast-path: if the cached analytics context (populated by
  // useAnalyticsContext, server-derived) says the user is NOT an admin, skip
  // the /admin/me call entirely. This silences the inevitable 403 in DevTools
  // for the 99% of users who aren't admins.
  //
  // Cache is per-user (peekCachedAnalyticsContext checks user_id internally).
  // If no cache exists yet (first sign-in of the session), fall through to
  // the real /admin/me call — useAnalyticsContext will populate the cache on
  // its own fetch and subsequent loads will fast-path correctly.
  //
  // For users who ARE admins, the cache says is_admin=true so we proceed to
  // /admin/me to get the full admin response (and to verify against the
  // canonical source — we don't trust the cache for granting admin access,
  // only for denying it).
  const cachedCtx = peekCachedAnalyticsContext(user?.id);
  const skipAdminCall = cachedCtx !== null && cachedCtx.is_admin === false;

  const query = useQuery({
    queryKey: ["admin", "me", user?.id],
    queryFn: async () => {
      try {
        return await apiFetch<{ email: string; isAdmin: boolean }>(`${API_URL}/admin/me`);
      } catch (err: unknown) {
        // 403 = "not admin", expected clean negative.
        // 500 = ADMIN_EMAILS unset on backend (intentional SP2 fail-loud) OR
        //   backend error. Either way, admin UI shouldn't block the dashboard,
        //   so treat as "not admin" and keep the app usable.
        // Network errors / anything else → same: degrade silently to non-admin.
        if (!(err instanceof ApiError && err.status === 403)) {
          // Log non-403 for debuggability without surfacing a user-facing error.
          console.warn("useIsAdmin: /admin/me failed; treating as not-admin", err);
        }
        return { email: "", isAdmin: false };
      }
    },
    enabled: !!user?.id && !skipAdminCall,
    staleTime: 5 * 60_000,
    retry: false,
  });

  if (skipAdminCall) {
    return { isAdmin: false, loading: false };
  }
  if (query.isLoading) {
    return { isAdmin: false, loading: true };
  }
  return { isAdmin: !!query.data?.isAdmin, loading: false };
}

// ---------------------------------------------------------------------------
// User listing + detail
// ---------------------------------------------------------------------------

export interface AdminUserRow {
  id: string;
  email: string | null;
  tier: "free" | "pro";
  has_override: boolean;
  is_admin: boolean;
  is_env_admin: boolean;
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
  perPage: number = 25,
): UseQueryResult<AdminUsersResponse> {
  return useQuery({
    queryKey: ["admin", "users", { search, page, perPage }],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (search) params.set("search", search);
      params.set("page", String(page));
      params.set("per_page", String(perPage));
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

  const promoteAdmin = useMutation({
    mutationFn: async (userId: string) =>
      apiFetch(`${API_URL}/admin/users/${userId}/promote`, { method: "POST" }),
    onSuccess: (_data, userId) => invalidateUser(userId),
  });

  const demoteAdmin = useMutation({
    mutationFn: async (userId: string) =>
      apiFetch(`${API_URL}/admin/users/${userId}/demote`, { method: "POST" }),
    onSuccess: (_data, userId) => invalidateUser(userId),
  });

  const recalcStorage = useMutation({
    mutationFn: async (userId: string) =>
      apiFetch<{ user_id: string; total_storage_bytes: number }>(
        `${API_URL}/admin/users/${userId}/recalc-storage`,
        { method: "POST" },
      ),
    onSuccess: (_data, userId) => invalidateUser(userId),
  });

  return { grantPro, revokePro, applyOverride, clearOverride, promoteAdmin, demoteAdmin, recalcStorage };
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
