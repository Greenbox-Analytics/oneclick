import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { apiFetch, API_URL, ApiError } from "@/lib/apiFetch";
import { clearPendingPlan } from "@/lib/pendingPlan";
import { useAuth } from "@/contexts/AuthContext";

/**
 * Create a Stripe Checkout Session and return the URL to redirect the user to.
 * Pattern:
 *   const { mutateAsync: createCheckout } = useCreateCheckoutSession();
 *   const url = await createCheckout({ plan: "basic_monthly" });
 *   window.location.href = url;
 *
 * Optional `cancel_path` / `success_path` let callers route returns to a
 * non-default page (e.g., onboarding). Backend whitelists to relative paths.
 *
 * `plan` values map to Stripe prices server-side (billing_router.py
 * PLAN_TO_ENV): <tier>_<period>, i.e. basic = the $30 plan, pro = the $50 one.
 */
export type CheckoutPlan = "basic_monthly" | "basic_annual" | "pro_monthly" | "pro_annual";

export interface CheckoutArgs {
  plan: CheckoutPlan;
  cancel_path?: string;
  success_path?: string;
}

export function useCreateCheckoutSession() {
  return useMutation<string, Error, CheckoutArgs | CheckoutPlan>({
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
 * Which Customer Portal surface to open: the home (no args) or a deep-linked
 * flow. `subscription_cancel` is Stripe's cancel confirmation page with the
 * navigation hidden; it redirects to /profile?portal=canceled the moment the
 * user confirms — the only Portal surface that redirects on its own.
 */
export interface PortalArgs {
  flow?: "subscription_cancel";
}

/**
 * Create a Stripe Customer Portal Session and return the URL to redirect to.
 * Throws (404-shaped ApiError) if the user has no stripe_customer_id; the
 * cancel flow also throws 409 (nothing to cancel / already ending) and 502
 * (Stripe refused the flow).
 */
export function useCreatePortalSession() {
  return useMutation<string, Error, PortalArgs | void>({
    mutationFn: async (args) => {
      // `void` lets callers omit the argument; narrow it before reading.
      const flow = (args as PortalArgs | undefined)?.flow;
      const res = await apiFetch(`${API_URL}/billing/create-portal-session`, {
        method: "POST",
        // A body only for a flow: the home session sends nothing, as it always has.
        ...(flow ? { body: JSON.stringify({ flow }) } : {}),
      });
      return (res as { url: string }).url;
    },
  });
}

/**
 * "Manage subscription" click handler: opens the Stripe Customer Portal, or
 * toasts when the user has no Stripe customer on file (the endpoint 404s
 * without a stripe_customer_id).
 */
export function useOpenBillingPortal() {
  const { mutateAsync: createPortal, isPending } = useCreatePortalSession();
  const openPortal = async () => {
    try {
      const url = await createPortal();
      window.location.href = url;
    } catch {
      toast.error("No billing portal on file. For billing, contact support.");
    }
  };
  return { openPortal, isPending };
}

/**
 * "Cancel plan" click handler: opens the Portal's cancel FLOW (see PortalArgs).
 * A 409 means the card was stale — nothing to cancel, or already set to end —
 * so refetch entitlements and say why, no portal. Any other refusal (Stripe
 * rejecting the flow, a Portal configuration with cancellation off) falls back
 * to the Portal home, where "Cancel plan" still exists: the feature degrades,
 * it never dead-ends.
 */
export function useOpenCancelFlow() {
  const qc = useQueryClient();
  const { user } = useAuth();
  const { mutateAsync: createPortal, isPending } = useCreatePortalSession();
  const { openPortal } = useOpenBillingPortal();
  const openCancelFlow = async () => {
    try {
      const url = await createPortal({ flow: "subscription_cancel" });
      window.location.href = url;
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        qc.invalidateQueries({ queryKey: ["entitlements", user?.id] });
        toast.info(e.message, { id: "cancel-flow" });
        return;
      }
      if (e instanceof ApiError && e.status === 404) {
        toast.error("No billing portal on file. For billing, contact support.", { id: "cancel-flow" });
        return;
      }
      toast.info("Couldn't open the cancel page — opening your billing portal instead.", { id: "cancel-flow" });
      await openPortal();
    }
  };
  return { openCancelFlow, isPending };
}

/**
 * Ask the backend to mirror the user's live subscription from Stripe onto
 * their row (cancel flag, period) and resolve once it has. Every Customer
 * Portal return calls this (usePortalReturn) so the plan card is right within
 * one round trip instead of waiting on the `customer.subscription.updated`
 * webhook — which still lands and writes the same truth. `{ synced: false }`
 * means there was nothing live to mirror; a 502 means Stripe couldn't be read
 * and the webhook is the fallback.
 */
export function useSyncSubscription() {
  return useMutation<{ synced: boolean }, Error, void>({
    mutationFn: () => apiFetch<{ synced: boolean }>(`${API_URL}/billing/sync-subscription`, { method: "POST" }),
  });
}

/**
 * The 409 `create-checkout-session` returns when the user already holds a live
 * subscription (one live personal subscription per user, 2026-09-10). Its
 * structured detail is `{ code: "subscription_exists", reason, tier }`.
 */
export function isSubscriptionConflict(err: unknown): boolean {
  return (
    err instanceof ApiError &&
    err.status === 409 &&
    (err.detail as { code?: unknown } | null | undefined)?.code === "subscription_exists"
  );
}

/**
 * Shared recovery for every checkout caller (Pricing, Onboarding, the
 * dashboard's resume strip, Billing's plan card). The endpoint refused a
 * second subscription; the caller's entitlements were simply stale (a second
 * tab, the 60s cache, a degraded read). So: forget any remembered plan (a
 * "Finish upgrading" nudge would only 409 again), refetch entitlements, say
 * why, and take them to the Customer Portal — where plan changes for a
 * subscriber actually happen. Resolves true when it handled the error:
 *
 *   catch (e) { if (await handleConflict(e)) return; toast.error(...); }
 */
export function useSubscriptionConflict(): (err: unknown) => Promise<boolean> {
  const qc = useQueryClient();
  const { user } = useAuth();
  const { openPortal } = useOpenBillingPortal();
  return async (err) => {
    if (!isSubscriptionConflict(err)) return false;
    clearPendingPlan(user?.id);
    qc.invalidateQueries({ queryKey: ["entitlements", user?.id] });
    toast.info("You already have an active subscription — change plans from your billing portal.", {
      id: "subscription-exists",
    });
    await openPortal();
    return true;
  };
}

/**
 * Toggle pay-per-use overage opt-in (credits system). Sparse update — only the
 * fields you pass are written. Invalidates entitlements so the new state shows.
 * Backend 400s if a free-tier user tries to ENABLE credit overage.
 */
export interface BillingPrefs {
  overage_enabled?: boolean;
  overage_cap_credits?: number | null;
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
