// src/components/billing/PlanCard.tsx
import { Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useEntitlements } from "@/hooks/useEntitlements";
import { useCreatePortalSession } from "@/hooks/useBilling";
import { useIsAdmin } from "@/hooks/useAdmin";
import { AdminBadge } from "@/components/admin/AdminBadge";
import { isPaidTier, tierLabel, ENTERPRISE_LABEL } from "@/lib/tiers";
import { fmtDate } from "@/lib/utils";

const priceLabel = (tier: string, period: string | null): { amount: string; unit: string } => {
  if (tier === "pro_max") return period === "annual" ? { amount: "$500", unit: "/ year" } : { amount: "$50", unit: "/ month" };
  if (tier === "pro") return period === "annual" ? { amount: "$250", unit: "/ year" } : { amount: "$25", unit: "/ month" };
  return { amount: "$0", unit: "/ month" };
};

export function PlanCard() {
  const navigate = useNavigate();
  const { data: ent } = useEntitlements();
  const { isAdmin } = useIsAdmin();
  const { mutateAsync: createPortal, isPending: isOpeningPortal } = useCreatePortalSession();

  const sub = ent?.subscription;
  // Key org identity off billingContext (present regardless of CREDITS_ENABLED —
  // Licensing follow-ups Task 3), falling back to credits.managedByOrg for
  // safety so this keeps working if billingContext is ever missing.
  const managedByOrg =
    ent?.billingContext?.type === "org" ? ent.billingContext : ent?.credits?.managedByOrg;

  const openPortal = async () => {
    try {
      const url = await createPortal();
      window.location.href = url;
    } catch {
      toast.error("No billing portal on file. For billing, contact support.");
    }
  };

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
            <Badge className="uppercase">{ENTERPRISE_LABEL}</Badge>
          </div>
        </div>

        <div className="mt-4 bg-background border border-border rounded-xl px-[18px] py-4">
          <div className="text-sm font-semibold">Billing is managed by {managedByOrg.orgName}.</div>
          <p className="text-[12.5px] text-muted-foreground mt-1 max-w-[440px]">
            Your organization covers your credits and access here — there&apos;s nothing to upgrade or pay for.
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
                <span className="text-muted-foreground">
                  {sub.cancelAtPeriodEnd ? "Cancels" : "Renews"}
                </span>
                <span className="tabular-nums">{fmtDate(sub.currentPeriodEnd)}</span>
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
              <Button variant="outline" size="sm" onClick={openPortal} disabled={isOpeningPortal}>
                {isOpeningPortal && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Manage subscription
              </Button>
            ) : (
              <Button size="sm" onClick={() => navigate("/pricing")}>
                Upgrade
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={() => navigate("/pricing")}>
              View plans
            </Button>
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
