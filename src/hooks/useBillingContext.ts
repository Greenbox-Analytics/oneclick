// src/hooks/useBillingContext.ts
// Licensing Phase B (spec §5) — the "Working as: Personal / ⟨Org⟩" switcher.
// PUT /me/billing-context sets the caller's persistent billing context
// server-side (profiles.billing_context_org_id). Credits + entitlements are
// context-dependent, so both caches are invalidated on success. The endpoint
// 404s (identical body for "org doesn't exist" and "no active seat" — no
// existence oracle) when orgId isn't an org the caller holds an active seat
// in; switching to personal (orgId: null) always succeeds.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch, API_URL } from "@/lib/apiFetch";
import { useAuth } from "@/contexts/AuthContext";
import { clearRememberedBillingContext } from "@/hooks/useEntitlements";

export interface SetBillingContextArgs {
  /** Org id to switch into, or null to switch back to personal billing. */
  orgId: string | null;
}

export function useSetBillingContext() {
  const qc = useQueryClient();
  const { user } = useAuth();
  return useMutation<unknown, Error, SetBillingContextArgs>({
    mutationFn: async ({ orgId }) =>
      apiFetch(`${API_URL}/me/billing-context`, {
        method: "PUT",
        body: JSON.stringify({ orgId }),
      }),
    onSuccess: () => {
      // This switch was the user's own choice — forget the remembered context
      // so the payer-switch notice (useEntitlements) doesn't treat the
      // refetched context as a surprise.
      clearRememberedBillingContext(user?.id);
      qc.invalidateQueries({ queryKey: ["entitlements", user?.id] });
      qc.invalidateQueries({ queryKey: ["credit-usage", user?.id] });
    },
  });
}
