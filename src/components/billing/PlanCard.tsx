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

const fmtDate = (iso?: string | null): string => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? "—"
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
};

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

  const tier = ent?.tier ?? "free";
  const isPaid = tier === "pro" || tier === "pro_max";
  const sub = ent?.subscription;
  const adminGranted = isPaid && !sub?.stripeSubscriptionId; // Pro without Stripe = admin/manual grant
  const { amount, unit } = priceLabel(tier, sub?.planPeriod ?? null);

  const openPortal = async () => {
    try {
      const url = await createPortal();
      window.location.href = url;
    } catch {
      toast.error("No billing portal on file. For billing, contact support.");
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
          <Badge className="uppercase">{tier === "pro_max" ? "Pro Max" : tier}</Badge>
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
              <Button size="sm" onClick={() => navigate("/subscription")}>
                Upgrade
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={() => navigate("/subscription")}>
              View plans
            </Button>
          </div>
          {adminGranted && (
            <p className="text-xs text-muted-foreground/70 mt-3 max-w-[360px]">
              Pro access via admin grant, not a paid subscription. For billing,{" "}
              <a href="mailto:tech@greenboxanalytics.ca">contact support</a>.
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}
