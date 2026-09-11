import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Sparkles, X } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { useEntitlements } from "@/hooks/useEntitlements";
import { useCreateCheckoutSession, useSubscriptionConflict } from "@/hooks/useBilling";
import { peekCachedAnalyticsContext } from "@/hooks/useAnalyticsContext";
import { clearPendingPlan, planLabel, readPendingPlan, stashPendingPlan, type PendingPlan } from "@/lib/pendingPlan";
import { isPaidTier } from "@/lib/tiers";
import { cn } from "@/lib/utils";

const DISMISS_KEY = "msanii.upgrade_banner.dismissed";

const STRIP_CLASS = cn(
  "flex items-center justify-between gap-3 px-4 py-2 mb-6 rounded-md",
  "bg-gradient-to-r from-primary/[0.08] to-primary/[0.02]",
  "border border-primary/15",
);

/**
 * Subtle one-line strip near the top of the Dashboard for free-tier users.
 *
 * Two variants, checked in this order:
 *
 * 1. **Finish upgrading** — the user started a paid checkout recently and
 *    didn't complete it (src/lib/pendingPlan.ts). Gated on ENTITLEMENTS, the
 *    server's word: while they're loading or degraded it says nothing, and
 *    once they read paid it clears the memory — so it can never tell a
 *    paying user they "didn't finish". It ignores the permanent dismiss flag
 *    below: that flag means "stop selling me Pro", this is "finish what you
 *    started", it's newer, self-expiring, and has its own X (which forgets
 *    the plan and hides the strip for this visit without touching the flag).
 * 2. **You're on the Free plan** — the pre-existing nudge:
 *    - Only shown when cached analytics context says `plan === "free"` AND
 *      entitlements don't already read paid (the cache lags upgrades).
 *    - Dismissible — sets a localStorage flag (per-user via user_id suffix)
 *      so the banner stays gone until the user clears storage or upgrades.
 *    - Returns null while the cache is empty (first sign-in of the session) —
 *      we'd rather not flash the banner before we know the plan than show it
 *      for a user who's actually on Pro.
 */
export function UpgradeBanner({ className }: { className?: string }) {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { data: ent } = useEntitlements();
  const { mutateAsync: createCheckout, isPending: isStartingCheckout } = useCreateCheckoutSession();
  const handleConflict = useSubscriptionConflict();
  const [dismissed, setDismissed] = useState(false);
  const [pending, setPending] = useState<PendingPlan | null>(null);
  // X on the pending strip hides everything for this mount — dismissing one
  // banner only to have another appear underneath is not a dismissal.
  const [pendingHidden, setPendingHidden] = useState(false);

  const userId = user?.id;
  const dismissStorageKey = userId ? `${DISMISS_KEY}.${userId}` : null;

  useEffect(() => {
    if (!dismissStorageKey) return;
    setDismissed(localStorage.getItem(dismissStorageKey) === "1");
  }, [dismissStorageKey]);

  useEffect(() => {
    setPending(readPendingPlan(userId));
  }, [userId]);

  const entReady = !!ent && !ent.degraded;
  const paid = entReady && isPaidTier(ent.tier);

  // Server truth wins: once entitlements say paid, the remembered intent is
  // moot (they finished on another device, or an admin granted the tier).
  useEffect(() => {
    if (!userId || !pending || !paid) return;
    clearPendingPlan(userId);
    setPending(null);
  }, [userId, pending, paid]);

  if (!user) return null;

  if (pending && !pendingHidden) {
    // Loading, degraded, or paid (being cleared above): say nothing.
    if (!entReady || paid) return null;

    const resume = async () => {
      try {
        const url = await createCheckout(pending.plan);
        stashPendingPlan(user.id, pending.plan); // fresh window if they abandon again
        window.location.href = url;
      } catch (e) {
        // Entitlements said free but the server knows better (a second tab,
        // the 60s cache): the handler forgets the plan and opens the portal.
        if (await handleConflict(e)) return;
        toast.error("Couldn't start checkout. Try again or contact support.");
      }
    };
    const forget = () => {
      clearPendingPlan(user.id);
      setPending(null);
      setPendingHidden(true);
    };

    return (
      <div className={cn(STRIP_CLASS, className)}>
        <div className="flex items-center gap-2 text-sm min-w-0 flex-wrap">
          <Sparkles className="w-3.5 h-3.5 text-primary shrink-0" />
          <span className="text-muted-foreground">
            You didn&apos;t finish upgrading to {planLabel(pending.plan)}.
          </span>
          <button
            type="button"
            onClick={resume}
            disabled={isStartingCheckout}
            className="text-primary hover:underline font-medium shrink-0 disabled:opacity-60 disabled:no-underline"
          >
            {isStartingCheckout ? "Starting checkout…" : "Finish upgrading"}
          </button>
        </div>
        <button
          type="button"
          onClick={forget}
          aria-label="Dismiss"
          className="text-muted-foreground/60 hover:text-foreground shrink-0"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  }

  if (pendingHidden || dismissed) return null;
  // The analytics cache can lag a fresh upgrade by up to 5 minutes; never
  // tell someone the server already calls paid that they're on Free.
  if (paid) return null;

  const ctx = peekCachedAnalyticsContext(user.id);
  if (!ctx || ctx.plan !== "free") return null;

  const dismiss = () => {
    if (dismissStorageKey) localStorage.setItem(dismissStorageKey, "1");
    setDismissed(true);
  };

  return (
    <div className={cn(STRIP_CLASS, className)}>
      <div className="flex items-center gap-2 text-sm min-w-0">
        <Sparkles className="w-3.5 h-3.5 text-primary shrink-0" />
        <span className="text-muted-foreground truncate">
          You're on the Free plan.
        </span>
        <button
          type="button"
          onClick={() => navigate("/pricing")}
          className="text-primary hover:underline font-medium shrink-0"
        >
          See what Pro unlocks →
        </button>
      </div>
      <button
        type="button"
        onClick={dismiss}
        aria-label="Dismiss"
        className="text-muted-foreground/60 hover:text-foreground shrink-0"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
