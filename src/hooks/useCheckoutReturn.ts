// src/hooks/useCheckoutReturn.ts
// Closes the loop on a subscription purchase. Stripe Checkout returns to
// /profile?stripe_session_id=...&welcome=true while the webhook that writes
// the subscription row is still in flight, so the paid tier is not there the
// instant the browser gets back — poll entitlements, then say something
// either way rather than leaving the user wondering whether they were
// charged. Sibling of useTopupReturn (credit purchases).
//
// The return signal is LATCHED once per mount. useSearchParams re-memoizes on
// location.search, so an effect keyed on the live params is torn down by its
// own URL cleanup — that is how the old inline version in Profile.tsx left
// "Activating your subscription…" up forever: the tier flipped, a dep
// changed, and the cleanup cleared the only timer that ever hid the overlay.
//
// Invariant that makes a stuck overlay unreachable: `activating` goes true in
// exactly ONE place, immediately before a timeout is armed whose handler
// settles. The only other thing that clears the timers is unmount, which
// takes the overlay down with the component.
import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { useAnalytics, type Plan } from "@/hooks/useAnalytics";
import { refreshAnalyticsContext } from "@/hooks/useAnalyticsContext";
import { useEntitlements } from "@/hooks/useEntitlements";
import { clearPendingPlan } from "@/lib/pendingPlan";
import { isPaidTier, tierLabel } from "@/lib/tiers";

const POLL_INTERVAL_MS = 1_000;
const POLL_TIMEOUT_MS = 10_000;

/**
 * Handle the `?welcome=true&stripe_session_id=` return from Stripe Checkout.
 * Mount once on the page the subscription success URL points at (`/profile`;
 * the legacy `/subscription` route redirects there with its query intact).
 * `activating` drives the full-screen "Activating your subscription…" overlay.
 */
export function useCheckoutReturn(): { activating: boolean } {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const { captureCheckoutCompleted } = useAnalytics();
  const { data: ent } = useEntitlements();

  // Frozen at mount: stripping the params below must not re-key the effect.
  const [sessionId] = useState(() =>
    searchParams.get("welcome") === "true" ? searchParams.get("stripe_session_id") : null,
  );
  const [activating, setActivating] = useState(false);

  // Read through a ref: including `ent` in the deps would restart the poll on
  // every refetch, which is the poll itself.
  const entRef = useRef(ent);
  entRef.current = ent;
  // Never reset — makes `settle` idempotent across StrictMode's double run.
  const settledRef = useRef(false);

  useEffect(() => {
    if (!sessionId) return;

    // Strip first, `replace` so a refresh or Back can't replay the welcome.
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete("welcome");
        next.delete("stripe_session_id");
        return next;
      },
      { replace: true },
    );
    // Reaching the success URL means Checkout completed. Forget the resume
    // intent NOW, not on "paid": a late webhook must never leave a "Finish
    // upgrading" button that would start a second paid checkout.
    clearPendingPlan(user?.id);

    const timers: {
      interval?: ReturnType<typeof setInterval>;
      timeout?: ReturnType<typeof setTimeout>;
    } = {};

    const settle = (paid: boolean) => {
      if (settledRef.current) return;
      settledRef.current = true;
      clearInterval(timers.interval);
      clearTimeout(timers.timeout);
      // Overlay down BEFORE any side effect, so nothing below can strand it.
      setActivating(false);
      if (paid) {
        const e = entRef.current;
        captureCheckoutCompleted((e?.subscription?.planPeriod as Plan | undefined) ?? "monthly");
        // Refresh the analytics-context cache so banners reading the 5-min
        // localStorage cache don't keep showing "you're on Free" post-upgrade.
        if (user?.id) void refreshAnalyticsContext(user.id, user.email);
        toast.success(`Welcome to ${tierLabel(e?.tier)}! Your subscription is active.`, {
          id: "checkout-welcome",
        });
        return;
      }
      // Not a failure: webhooks can lag. Never imply the payment didn't go
      // through — the next entitlements read will show the plan.
      queryClient.invalidateQueries({ queryKey: ["entitlements"] });
      toast.info("Almost there — your subscription is still activating. Refresh in a moment.", {
        id: "checkout-activating",
      });
    };

    if (isPaidTier(entRef.current?.tier)) {
      settle(true);
      return;
    }

    setActivating(true); // the ONLY place this goes true; a timeout is armed right below
    timers.interval = setInterval(() => {
      if (isPaidTier(entRef.current?.tier)) {
        settle(true);
        return;
      }
      // A poll tick must never cancel the fetch it is waiting for: the default
      // (`cancelRefetch: true`) restarts an in-flight fetch, so one slower than
      // the tick would never land while polling.
      queryClient.invalidateQueries({ queryKey: ["entitlements"] }, { cancelRefetch: false });
    }, POLL_INTERVAL_MS);
    timers.timeout = setTimeout(() => settle(false), POLL_TIMEOUT_MS);

    return () => {
      clearInterval(timers.interval);
      clearTimeout(timers.timeout);
    };
    // Keyed on the latched id alone: this must run once per return.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  return { activating };
}
