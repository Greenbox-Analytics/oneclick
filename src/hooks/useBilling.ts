import { useMutation } from "@tanstack/react-query";
import { apiFetch, API_URL } from "@/lib/apiFetch";

/**
 * Create a Stripe Checkout Session and return the URL to redirect the user to.
 * Pattern:
 *   const { mutateAsync: createCheckout } = useCreateCheckoutSession();
 *   const url = await createCheckout("monthly");
 *   window.location.href = url;
 */
export function useCreateCheckoutSession() {
  return useMutation<string, Error, "monthly" | "annual">({
    mutationFn: async (plan) => {
      const res = await apiFetch(`${API_URL}/billing/create-checkout-session`, {
        method: "POST",
        body: JSON.stringify({ plan }),
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
