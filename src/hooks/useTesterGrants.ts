import { useMutation, useQuery, useQueryClient, type UseQueryResult } from "@tanstack/react-query";
import { ApiError, API_URL, apiFetch } from "@/lib/apiFetch";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TesterGrant {
  user_id: string;
  expires_at: string | null;
  reason: string | null;
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

export { ApiError };
