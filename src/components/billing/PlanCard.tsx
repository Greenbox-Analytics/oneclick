// src/components/billing/PlanCard.tsx
import { Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useEntitlements } from "@/hooks/useEntitlements";
import {
  useCreateCheckoutSession,
  useOpenBillingPortal,
  useOpenCancelFlow,
  useSubscriptionConflict,
} from "@/hooks/useBilling";
import { useIsAdmin } from "@/hooks/useAdmin";
import { AdminBadge } from "@/components/admin/AdminBadge";
import { isPaidTier, tierLabel, usd, ENTERPRISE_LABEL, TIER_PRICES, type TierKey } from "@/lib/tiers";
import { fmtDate, fmtDaysLeft } from "@/lib/utils";
import { orgContext } from "@/lib/credits";
import { planLabel, readPendingPlan, stashPendingPlan } from "@/lib/pendingPlan";

const priceLabel = (tier: string, period: string | null): { amount: string; unit: string } => {
  const key: TierKey = tier === "basic" || tier === "pro" ? tier : "free";
  // Free has no annual price to state — it always reads "/ month".
  if (key !== "free" && period === "annual") return { amount: usd(TIER_PRICES[key].annual), unit: "/ year" };
  return { amount: usd(TIER_PRICES[key].monthly), unit: "/ month" };
};

export function PlanCard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { data: ent } = useEntitlements();
  const { isAdmin } = useIsAdmin();
  const { openPortal, isPending: isOpeningPortal } = useOpenBillingPortal();
  const { openCancelFlow, isPending: isOpeningCancel } = useOpenCancelFlow();
  const { mutateAsync: createCheckout, isPending: isStartingCheckout } = useCreateCheckoutSession();
  const handleConflict = useSubscriptionConflict();

  const sub = ent?.subscription;
  const managedByOrg = orgContext(ent);
  // "Ends Oct 10, 2026 · 12 days left" once a cancel is scheduled; the row
  // otherwise reads "Renews <date>".
  const endsIn = sub?.cancelAtPeriodEnd ? fmtDaysLeft(sub.currentPeriodEnd) : "";

  // Org billing context (Licensing Phase B, spec §5): the org's pool pays, so
  // there's no plan to upgrade or price to show — just who's managing it, plus
  // a heads-up if the member also keeps a personal subscription running
  // alongside it (we never auto-cancel a personal plan).
  if (managedByOrg) {
    return (
      <Card className="p-6">
        <div className="flex items-start justify-between gap-3.5">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">Plan</h2>
            <div className="text-[13.5px] text-muted-foreground mt-0.5">Manage your subscription</div>
          </div>
          <div className="flex gap-1.5">
            <AdminBadge />
            <Badge className="uppercase">{tierLabel(ent?.tier ?? "free")}</Badge>
          </div>
        </div>

        <div className="mt-4 bg-background border border-border rounded-xl px-[18px] py-4">
          <div className="text-sm font-semibold">
            Billing is managed by {managedByOrg.orgName}
            {managedByOrg.kind === "self_serve" ? " — a team." : ` — an ${ENTERPRISE_LABEL} organization.`}
          </div>
          <p className="text-[12.5px] text-muted-foreground mt-1 max-w-[440px]">
            While you&apos;re working as {managedByOrg.orgName}, its credit pool pays for your AI work. Your own plan
            above still applies to your personal workspace.
          </p>
        </div>

        {sub?.stripeSubscriptionId && (
          <div className="flex items-center justify-between gap-4 flex-wrap mt-4 px-4 py-3.5 border border-border rounded-xl bg-background">
            <p className="text-[12.5px] text-muted-foreground max-w-[420px]">
              You&apos;re covered by {managedByOrg.orgName} — you can cancel or keep your personal plan.
            </p>
            <Button variant="outline" size="sm" onClick={openPortal} disabled={isOpeningPortal}>
              {isOpeningPortal && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Manage subscription
            </Button>
          </div>
        )}
      </Card>
    );
  }

  const tier = ent?.tier ?? "free";
  const isPaid = isPaidTier(tier);
  const adminGranted = isPaid && !sub?.stripeSubscriptionId; // Paid tier without Stripe = admin/manual grant
  const { amount, unit } = priceLabel(tier, sub?.planPeriod ?? null);

  // A paid checkout the user started but didn't finish (src/lib/pendingPlan.ts).
  // Only consulted once entitlements are in and trustworthy AND say free —
  // the memory is intent, never a claim about what they have.
  const pending = ent && !ent.degraded && !isPaid ? readPendingPlan(user?.id) : null;
  const resumeCheckout = async () => {
    if (!pending || !user) return;
    try {
      const url = await createCheckout(pending.plan);
      stashPendingPlan(user.id, pending.plan); // fresh window if they abandon again
      window.location.href = url;
    } catch (e) {
      // Stale entitlements (the server already has a live subscription):
      // the handler forgets the plan and opens the portal instead.
      if (await handleConflict(e)) return;
      toast.error("Couldn't start checkout. Try again or contact support.");
    }
  };

  return (
    <Card className="p-6">
      <div className="flex items-start justify-between gap-3.5">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Plan</h2>
          <div className="text-[13.5px] text-muted-foreground mt-0.5">Manage your subscription</div>
        </div>
        <div className="flex gap-1.5">
          <AdminBadge />
          <Badge className="uppercase">{tierLabel(tier)}</Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-7 items-end mt-1">
        <div>
          <div className="text-[26px] font-bold tracking-tight mt-4">
            {amount} <span className="text-sm font-normal text-muted-foreground">{unit}</span>
          </div>
          <div className="mt-3.5">
            <div className="flex items-center justify-between text-sm py-2.5">
              <span className="text-muted-foreground">Status</span>
              <Badge
                variant="outline"
                className="border-primary/30 text-primary capitalize bg-primary/10"
              >
                {ent?.status ?? "—"}
              </Badge>
            </div>
            {sub?.currentPeriodEnd && (
              <div className="flex items-center justify-between text-sm py-2.5 border-t border-border/60">
                <span className="text-muted-foreground">{sub.cancelAtPeriodEnd ? "Ends" : "Renews"}</span>
                <span className="tabular-nums">
                  {fmtDate(sub.currentPeriodEnd)}
                  {sub.cancelAtPeriodEnd && endsIn ? (
                    <span className="text-muted-foreground"> · {endsIn}</span>
                  ) : null}
                </span>
              </div>
            )}
            {adminGranted && (
              <div className="flex items-center justify-between text-sm py-2.5 border-t border-border/60">
                <span className="text-muted-foreground">Access</span>
                <span>Granted by admin</span>
              </div>
            )}
          </div>
        </div>

        <div>
          <div className="flex gap-2.5 flex-wrap">
            {isPaid ? (
              // No Stripe subscription = admin grant, so there is no portal to
              // open — create-portal-session 404s without a stripe_customer_id.
              // The adminGranted note below is the whole story for these users.
              !adminGranted && (
                <Button variant="outline" size="sm" onClick={openPortal} disabled={isOpeningPortal}>
                  {isOpeningPortal && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Manage subscription
                </Button>
              )
            ) : pending ? (
              <Button size="sm" onClick={resumeCheckout} disabled={isStartingCheckout}>
                {isStartingCheckout && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Finish upgrading to {planLabel(pending.plan)}
              </Button>
            ) : (
              <Button size="sm" onClick={() => navigate("/pricing")}>
                Upgrade
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={() => navigate("/pricing")}>
              View plans
            </Button>
            {isPaid && !adminGranted && !sub?.cancelAtPeriodEnd && (
              // Deep-links into the Portal's cancel flow, which redirects back
              // here the moment the user confirms (the Portal home never does).
              // Gone once the cancel is scheduled: reactivating is "Renew plan"
              // on the Portal home, behind Manage subscription.
              <Button
                variant="ghost"
                size="sm"
                className="text-muted-foreground"
                onClick={openCancelFlow}
                disabled={isOpeningCancel}
              >
                {isOpeningCancel && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Cancel plan
              </Button>
            )}
          </div>
          {adminGranted && (
            <p className="text-xs text-muted-foreground/70 mt-3 max-w-[360px]">
              {tierLabel(tier)} access via admin grant, not a paid subscription. For billing,{" "}
              <a href="mailto:tech@greenboxanalytics.ca">contact support</a>.
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}
