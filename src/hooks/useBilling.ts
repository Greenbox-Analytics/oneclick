import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch, API_URL } from "@/lib/apiFetch";
import { useAuth } from "@/contexts/AuthContext";

/**
 * Create a Stripe Checkout Session and return the URL to redirect the user to.
 * Pattern:
 *   const { mutateAsync: createCheckout } = useCreateCheckoutSession();
 *   const url = await createCheckout({ plan: "monthly" });
 *   window.location.href = url;
 *
 * Optional `cancel_path` / `success_path` let callers route returns to a
 * non-default page (e.g., onboarding). Backend whitelists to relative paths.
 */
export interface CheckoutArgs {
  plan: "monthly" | "annual";
  cancel_path?: string;
  success_path?: string;
}

export function useCreateCheckoutSession() {
  return useMutation<string, Error, CheckoutArgs | "monthly" | "annual">({
    mutationFn: async (arg) => {
      // Back-compat: callers can still pass just the plan string.
      const body = typeof arg === "string" ? { plan: arg } : arg;
      const res = await apiFetch(`${API_URL}/billing/create-checkout-session`, {
        method: "POST",
        body: JSON.stringify(body),
      });
      return (res as { url: string }).url;
    },
  });
}

/**
 * Create a Stripe Customer Portal Session and return the URL to redirect to.
 * Throws (404-shaped ApiError) if the user has no stripe_customer_id.
 */
export function useCreatePortalSession() {
  return useMutation<string, Error, void>({
    mutationFn: async () => {
      const res = await apiFetch(`${API_URL}/billing/create-portal-session`, {
        method: "POST",
      });
      return (res as { url: string }).url;
    },
  });
}

/**
 * Toggle pay-per-use overage opt-in (credits system). Sparse update — only the
 * fields you pass are written. Invalidates entitlements so the new state shows.
 * Backend 400s if a free-tier user tries to ENABLE credit overage.
 */
export interface BillingPrefs {
  overage_enabled?: boolean;
  overage_cap_credits?: number | null;
  storage_overage_enabled?: boolean;
}

export function useSetBillingPrefs() {
  const qc = useQueryClient();
  const { user } = useAuth();
  return useMutation<unknown, Error, BillingPrefs>({
    mutationFn: async (prefs) =>
      apiFetch(`${API_URL}/me/billing-prefs`, {
        method: "POST",
        body: JSON.stringify(prefs),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["entitlements", user?.id] });
    },
  });
}
