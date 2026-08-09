import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { API_URL, apiFetch } from "@/lib/apiFetch";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TesterGrant {
  // null = pending (pre-signup) row — revoke via useRevokePendingTesterGrant, never by user_id
  user_id: string | null;
  expires_at: string | null;
  reason: string | null;
  email: string | null;
  name: string | null;
  pending?: boolean;
  grant_duration_days?: number | null;
}

// ---------------------------------------------------------------------------
// Query key
// ---------------------------------------------------------------------------

const TESTER_GRANTS_KEY = ["admin", "tester-grants"] as const;

// ---------------------------------------------------------------------------
// useTesterGrants — list all active grants
// ---------------------------------------------------------------------------

export function useTesterGrants(): UseQueryResult<TesterGrant[]> {
  return useQuery({
    queryKey: TESTER_GRANTS_KEY,
    queryFn: () => apiFetch<TesterGrant[]>(`${API_URL}/admin/tester-grants`),
    staleTime: 30_000,
  });
}

// ---------------------------------------------------------------------------
// useCreateTesterGrant — POST /admin/tester-grants
// ---------------------------------------------------------------------------

export interface CreateTesterGrantInput {
  email: string;
  expires_at?: string | null;
  reason?: string | null;
  grant_duration_days?: number | null;
  credits?: number | null;
}

export function useCreateTesterGrant() {
  const qc = useQueryClient();

  return useMutation({
    mutationFn: (input: CreateTesterGrantInput) =>
      apiFetch<TesterGrant>(`${API_URL}/admin/tester-grants`, {
        method: "POST",
        body: JSON.stringify(input),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: TESTER_GRANTS_KEY });
    },
  });
}

// ---------------------------------------------------------------------------
// useRevokeTesterGrant — DELETE /admin/tester-grants/{user_id}
// ---------------------------------------------------------------------------

export function useRevokeTesterGrant() {
  const qc = useQueryClient();

  return useMutation({
    mutationFn: (userId: string) =>
      apiFetch(`${API_URL}/admin/tester-grants/${userId}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: TESTER_GRANTS_KEY });
    },
  });
}

// ---------------------------------------------------------------------------
// useRevokePendingTesterGrant — DELETE /admin/tester-grants/pending?email=
// Revokes an unclaimed pre-signup designation (claimed/active grants use
// useRevokeTesterGrant by user_id).
// ---------------------------------------------------------------------------

export function useRevokePendingTesterGrant() {
  const qc = useQueryClient();

  return useMutation({
    mutationFn: (email: string) =>
      apiFetch(`${API_URL}/admin/tester-grants/pending?email=${encodeURIComponent(email)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: TESTER_GRANTS_KEY });
    },
  });
}

