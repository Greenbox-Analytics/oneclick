// src/hooks/useTopupReturn.ts
// Closes the loop on a credit purchase: Stripe Checkout returns to
// ?topup=success|canceled, and without this the buyer lands on a page showing
// their OLD balance with no acknowledgement that anything happened (React
// Query wouldn't refetch entitlements for up to 60s).
//
// Mirrors the subscription-return pattern in pages/Profile.tsx: the credits are
// granted by a webhook, so the balance is not there the instant the browser
// gets back — poll, then say something either way rather than leaving the user
// wondering whether their card was charged.
import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { invalidateCreditSurfaces } from "@/hooks/useCreditUsage";
import { useEntitlements } from "@/hooks/useEntitlements";

const POLL_INTERVAL_MS = 1_000;
const POLL_TIMEOUT_MS = 10_000;

/**
 * Handle the `?topup=` return from Stripe Checkout. Mount once per page that
 * a credit purchase can return to (`/profile`, `/teams` — the backend picks
 * the return path from whether the purchase targeted a pool).
 */
export function useTopupReturn(): void {
  const [searchParams, setSearchParams] = useSearchParams();
  const queryClient = useQueryClient();
  const { data: ent } = useEntitlements();

  const status = searchParams.get("topup");

  // Purchased credits land in the RESERVE bucket (kind='purchase'), so that is
  // what moves. Read through a ref: including it in the effect's deps would
  // restart the poll on every entitlements refetch, which is the poll itself.
  //
  // One case reads nothing: an admin buying into a POOL while their own
  // billing context is personal — their personal reserve won't move. That
  // falls through to the timeout branch, whose copy ("payment received, your
  // credits will appear shortly") is true either way. Never assert failure
  // from a balance that didn't move.
  const reserveRef = useRef<number | null | undefined>(undefined);
  reserveRef.current = ent?.credits?.reserveBalance;

  useEffect(() => {
    if (!status) return;

    // Strip the param immediately — a refresh (or a back-navigation) must not
    // replay the toast, and the URL is user-visible.
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete("topup");
        return next;
      },
      { replace: true }
    );

    if (status !== "success") {
      toast.info("Checkout cancelled — no credits were purchased.");
      return;
    }

    // Baseline for "did the grant land?". Entitlements may still be loading on
    // arrival, so it is captured lazily from the first reading we actually get
    // rather than pinned to `undefined` and never comparable.
    let before = reserveRef.current ?? null;
    let settled = false;

    invalidateCreditSurfaces(queryClient);

    const interval = setInterval(() => {
      // The balance moves only once the webhook has granted; until then this
      // is just a refetch of the same numbers.
      const now = reserveRef.current;
      if (before == null) {
        before = now ?? null;
      } else if (now != null && now > before) {
        settled = true;
        clearInterval(interval);
        clearTimeout(timeout);
        toast.success(`Credits added — ${(now - before).toLocaleString()} credits are ready to use.`);
        return;
      }
      invalidateCreditSurfaces(queryClient);
    }, POLL_INTERVAL_MS);

    const timeout = setTimeout(() => {
      clearInterval(interval);
      if (settled) return;
      // Not a failure: webhooks can lag, and delayed payment methods settle
      // hours later. Never imply the payment didn't go through.
      invalidateCreditSurfaces(queryClient);
      toast.success("Payment received — your credits will appear shortly.");
    }, POLL_TIMEOUT_MS);

    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
    // Deliberately keyed on `status` alone: this must run once per return.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);
}
