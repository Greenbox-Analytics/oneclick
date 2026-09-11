// src/hooks/usePortalReturn.ts
// The return from the Stripe Customer Portal. Two signals, both on /profile
// (billing_router.create_portal_session mints the URLs):
//
//   ?portal=canceled — where the Portal's cancel FLOW redirects the moment the
//     user confirms. The cancel has already happened at Stripe; only the row
//     may still say "renews". So: overlay, then say when the plan ends — never
//     "Renews …" flashing at someone who just canceled, and no second "Cancel
//     plan" button while we wait.
//   ?portal=return — the Portal home's "Return to Msanii" link. Nothing is
//     known about what changed there (a cancel, "Renew plan", nothing at all).
//     No overlay, no toast.
//
// Either way the row is brought up to date the same way: POST
// /billing/sync-subscription mirrors the subscription from Stripe onto the
// row NOW, then entitlements are refetched — one round trip, no waiting on the
// `customer.subscription.updated` webhook (which still lands and writes the
// same truth). A short poll stays underneath as the fallback for a sync that
// fails: it observes the webhook instead. Poll ticks refetch with
// `cancelRefetch: false` — the default cancels an in-flight fetch and starts
// over, so a fetch slower than the tick would never land while polling.
//
// Same discipline as useCheckoutReturn: the signal is LATCHED once per mount
// (useSearchParams re-memoizes on location.search, so an effect keyed on the
// live params would tear itself down when it cleans the URL), `syncing` goes
// true in exactly ONE place immediately before a timeout is armed whose
// handler settles, and settle is idempotent through a never-reset ref.
import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { useSyncSubscription } from "@/hooks/useBilling";
import { useEntitlements } from "@/hooks/useEntitlements";
import { hasLiveSubscription, tierLabel } from "@/lib/tiers";
import { fmtDate } from "@/lib/utils";

const POLL_INTERVAL_MS = 1_000;
const CANCEL_TIMEOUT_MS = 10_000;
const RETURN_BURST_TICKS = 5;
// One id for the three mutually exclusive outcomes: sonner updates in place.
const TOAST_ID = "subscription-canceled";

type PortalSignal = "canceled" | "return";

/**
 * Handle the `?portal=canceled|return` return from the Customer Portal. Mount
 * once on `/profile`. `syncing` drives the "Updating your plan…" overlay.
 */
export function usePortalReturn(): { syncing: boolean } {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const { data: ent } = useEntitlements();
  const { mutateAsync: syncSubscription } = useSyncSubscription();

  // Frozen at mount: stripping the param below must not re-key the effect.
  const [signal] = useState<PortalSignal | null>(() => {
    const value = searchParams.get("portal");
    return value === "canceled" || value === "return" ? value : null;
  });
  const [syncing, setSyncing] = useState(false);

  // Read through a ref: including `ent` in the deps would restart the poll on
  // every refetch, which is the poll itself.
  const entRef = useRef(ent);
  entRef.current = ent;
  // Never reset — makes `settle` idempotent across StrictMode's double run.
  const settledRef = useRef(false);
  // Never reset either: one sync per return, not one per effect run.
  const syncStartedRef = useRef(false);

  useEffect(() => {
    if (!signal) return;

    // Strip first, `replace` so a refresh or Back can't replay the return.
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete("portal");
        return next;
      },
      { replace: true },
    );

    const timers: {
      interval?: ReturnType<typeof setInterval>;
      timeout?: ReturnType<typeof setTimeout>;
    } = {};
    // "The row just changed, fetch it now": may cancel an in-flight fetch.
    const refetch = () => queryClient.invalidateQueries({ queryKey: ["entitlements"] });
    // A poll tick must never cancel the fetch it is waiting for.
    const poll = () => queryClient.invalidateQueries({ queryKey: ["entitlements"] }, { cancelRefetch: false });
    const syncThenRefetch = () => {
      if (syncStartedRef.current) return;
      syncStartedRef.current = true;
      // A failed sync is nothing the user can act on: the poll observes the
      // webhook instead.
      syncSubscription().then(
        () => refetch(),
        () => undefined,
      );
    };

    if (signal === "return") {
      syncThenRefetch();
      let ticks = 0;
      timers.interval = setInterval(() => {
        poll();
        if (++ticks >= RETURN_BURST_TICKS) clearInterval(timers.interval);
      }, POLL_INTERVAL_MS);
      return () => clearInterval(timers.interval);
    }

    const settle = (outcome: "ending" | "gone" | "timeout") => {
      if (settledRef.current) return;
      settledRef.current = true;
      clearInterval(timers.interval);
      clearTimeout(timers.timeout);
      // Overlay down BEFORE any side effect, so nothing below can strand it.
      setSyncing(false);
      const e = entRef.current;
      if (outcome === "ending") {
        const end = e?.subscription?.currentPeriodEnd;
        const when = end ? `on ${fmtDate(end)}` : "at the close of this billing period";
        toast.success(`Your ${tierLabel(e?.tier)} plan is set to end ${when}. You keep full access until then.`, {
          id: TOAST_ID,
        });
        return;
      }
      if (outcome === "gone") {
        toast.info("Your subscription has been canceled. You're now on the Free plan.", { id: TOAST_ID });
        return;
      }
      // Reaching ?portal=canceled means Stripe completed the cancel — only the
      // row is behind. Never imply it failed.
      refetch();
      toast.info("Your cancellation went through. The end date will show here shortly — refresh if you don't see it.", {
        id: TOAST_ID,
      });
    };

    // A missing or degraded read must NOT settle: without a trustworthy
    // subscription it would read as "gone" before the first fetch resolves.
    const check = (): boolean => {
      const e = entRef.current;
      if (!e || e.degraded) return false;
      if (e.subscription?.cancelAtPeriodEnd) {
        settle("ending");
        return true;
      }
      if (!hasLiveSubscription(e)) {
        settle("gone"); // the Portal configured to cancel immediately
        return true;
      }
      return false;
    };

    if (check()) return;

    syncThenRefetch();
    setSyncing(true); // the ONLY place this goes true; a timeout is armed right below
    timers.interval = setInterval(() => {
      if (!check()) poll();
    }, POLL_INTERVAL_MS);
    timers.timeout = setTimeout(() => settle("timeout"), CANCEL_TIMEOUT_MS);

    return () => {
      clearInterval(timers.interval);
      clearTimeout(timers.timeout);
    };
    // Keyed on the latched signal alone: this must run once per return.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signal]);

  return { syncing };
}
