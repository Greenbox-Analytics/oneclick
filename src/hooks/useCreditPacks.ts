import { useMutation, useQuery } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface CreditPack {
  key: string;
  credits: number;
  price_cents: number;
  sort_order: number;
  /** Set only when an operator has configured a recurring Stripe price for
   * this pack — makes it eligible for the org monthly top-up catalog
   * (POST /billing/org-topup-checkout). Null on every other pack. */
  recurringPriceId?: string | null;
}

/** GET /billing/credit-packs — active, purchasable packs. */
export function useCreditPacks() {
  const { user } = useAuth();
  return useQuery<{ packs: CreditPack[] }>({
    queryKey: ["credit-packs"],
    queryFn: () => apiFetch<{ packs: CreditPack[] }>(`${API_URL}/billing/credit-packs`),
    enabled: !!user?.id,
    staleTime: 5 * 60_000,
  });
}

export interface CreateTopupArgs {
  packKey: string;
  /** Licensing Phase B: when set, the pack purchase targets that org's
   * credit pool instead of the caller's personal wallet (backend requires
   * the caller to be an active admin of a non-archived org). */
  orgId?: string;
}

/** POST /billing/create-topup-session — redirects to Stripe Checkout.
 * `{ packKey, orgId? }` routes the purchase into an org's pool (orgId set) or
 * the caller's personal wallet (orgId omitted). */
export function useCreateTopupSession() {
  return useMutation<void, Error, CreateTopupArgs>({
    mutationFn: async ({ packKey, orgId }) => {
      const res = await apiFetch<{ url: string }>(`${API_URL}/billing/create-topup-session`, {
        method: "POST",
        body: JSON.stringify({ pack_key: packKey, org_id: orgId ?? null }),
      });
      window.location.href = res.url;
    },
  });
}
