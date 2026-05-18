import { useMutation } from "@tanstack/react-query";
import { apiFetch, API_URL } from "@/lib/apiFetch";

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
