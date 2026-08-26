import { useMutation, useQuery } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import type { ToolCreditPrices } from "@/lib/credits";

export interface CreditPack {
  key: string;
  /** Human name for the card ("Starter"). Null on a pack seeded without one —
   * the picker falls back to "N credits", so it still sells and still renders. */
  label?: string | null;
  credits: number;
  price_cents: number;
  sort_order: number;
  /** Set only when an operator has configured a recurring Stripe price for
   * this pack — makes it eligible for the org monthly top-up catalog
   * (POST /billing/org-topup-checkout). Null on every other pack. */
  recurringPriceId?: string | null;
}

/** Bounds + unit price for a custom ("choose your own amount") purchase. */
export interface CustomCreditConfig {
  minCredits: number;
  maxCredits: number;
  /** Cents per credit — fractional if CREDIT_OVERAGE_USD isn't a whole cent. */
  perCreditCents: number;
}

export interface CreditPacksResponse {
  packs: CreditPack[];
  custom: CustomCreditConfig;
  /** Live per-action credit prices, used to say what an amount typically buys.
   * OMITTED by the backend when `credit_prices` reads empty — the picker then
   * shows no usage subtitle rather than quoting zeros. */
  prices?: ToolCreditPrices;
}

/** GET /billing/credit-packs — bundles, custom-amount bounds, tool prices. */
export function useCreditPacks() {
  const { user } = useAuth();
  return useQuery<CreditPacksResponse>({
    queryKey: ["credit-packs"],
    queryFn: () => apiFetch<CreditPacksResponse>(`${API_URL}/billing/credit-packs`),
    enabled: !!user?.id,
    staleTime: 5 * 60_000,
  });
}

export interface CreateTopupArgs {
  /** A catalog bundle. Mutually exclusive with `credits` — the backend 422s
   * when both or neither are set. */
  packKey?: string;
  /** A custom amount, in credits. The server prices it; there is deliberately
   * no way to send an amount of MONEY. */
  credits?: number;
  /** Licensing Phase B: when set, the purchase — bundle or custom — targets
   * that org's credit pool instead of the caller's personal wallet (backend
   * requires the caller to be an active admin of a non-archived org). */
  orgId?: string;
}

/** POST /billing/create-topup-session — redirects to Stripe Checkout.
 * `{ packKey }` or `{ credits }` picks the product; `orgId` routes either
 * purchase into an org's pool. */
export function useCreateTopupSession() {
  return useMutation<void, Error, CreateTopupArgs>({
    mutationFn: async ({ packKey, credits, orgId }) => {
      const res = await apiFetch<{ url: string }>(`${API_URL}/billing/create-topup-session`, {
        method: "POST",
        body: JSON.stringify({
          pack_key: packKey ?? null,
          credits: credits ?? null,
          org_id: orgId ?? null,
        }),
      });
      window.location.href = res.url;
    },
  });
}
