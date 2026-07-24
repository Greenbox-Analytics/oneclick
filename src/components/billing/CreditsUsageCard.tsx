// src/components/billing/CreditsUsageCard.tsx
// The "Credits & usage" card from the Account & Billing mockup: a donut ring of
// remaining credits + a per-tool cost/usage breakdown + a pay-per-use toggle.
// Renders nothing when the credits system is off (backend `enabled:false`).
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Coins, Plus, Send } from "lucide-react";
import { toast } from "sonner";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { useEntitlements } from "@/hooks/useEntitlements";
import { useCreditUsage, type CreditAction } from "@/hooks/useCreditUsage";
import { useCreditPacks } from "@/hooks/useCreditPacks";
import { useSetBillingPrefs } from "@/hooks/useBilling";
import { CreditRing, type RingSegment } from "@/components/billing/CreditRing";
import { TopUpCreditsDialog } from "@/components/billing/TopUpCreditsDialog";
import { isPaidTier, tierLabel, ENTERPRISE_LABEL } from "@/lib/tiers";
import { fmtDate } from "@/lib/utils";

type ToolMeta = { label: string; color: string; note?: string };
const TOOL_META: Record<CreditAction, ToolMeta> = {
  oneclick_run: { label: "OneClick run", color: "var(--t-oneclick)" },
  registry_parse: { label: "Registry parse", color: "var(--t-registry)", note: "cache hits free" },
  zoe_message: { label: "Zoe message", color: "var(--t-zoe)" },
};
// Ring/list order matches the mockup.
const ORDER: CreditAction[] = ["oneclick_run", "registry_parse", "zoe_message"];

const fmtDay = (iso?: string | null): string => {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

interface Row {
  label: string;
  color: string;
  note?: string;
  spent: number;
  count: number;
  price: number | null;
}

export function CreditsUsageCard() {
  const { data: usage, isLoading } = useCreditUsage();
  const { data: ent } = useEntitlements();
  const setPrefs = useSetBillingPrefs();
  const { data: packsData } = useCreditPacks();
  const [topUpOpen, setTopUpOpen] = useState(false);
  const navigate = useNavigate();

  if (isLoading || !usage) return null;
  if (!usage.enabled) return null; // flag off → no credit surfaces

  // Key org identity off billingContext (present regardless of CREDITS_ENABLED —
  // Licensing follow-ups Task 3), falling back to credits.managedByOrg for
  // safety. In practice credits-off + org context never reaches here — the
  // `usage.enabled` early-return above already hides this card when credits
  // are off — but this keeps the two signals consistent going forward.
  const managedByOrg =
    ent?.billingContext?.type === "org" ? ent.billingContext : ent?.credits?.managedByOrg;

  // Org billing context (Licensing Phase B, spec §5): credits are a seat
  // allocation from the org's pool, not a personal monthly grant — no pack
  // picker, no pay-per-use toggle. Seat balance comes straight off
  // entitlements (context-aware) rather than the per-tool usage breakdown,
  // which is a personal-wallet concept.
  if (managedByOrg) {
    const seatBalance = ent?.credits?.balance ?? 0;
    return (
      <Card className="overflow-hidden">
        <div className="flex items-start justify-between gap-4 px-6 pt-[22px] pb-1.5">
          <div>
            <div className="flex items-center gap-2.5 text-[15px] font-semibold">
              <Coins className="w-[18px] h-[18px] text-muted-foreground" />
              Credits &amp; usage
            </div>
            <div className="text-[13.5px] text-muted-foreground mt-0.5">
              Your credits from {managedByOrg.orgName}
            </div>
          </div>
          <Badge className="uppercase">{ENTERPRISE_LABEL}</Badge>
        </div>

        <div className="flex items-center justify-between gap-4 flex-wrap px-6 pt-3.5 pb-[22px]">
          <div>
            <div className="text-[32px] font-bold tracking-tight tabular-nums">
              {seatBalance.toLocaleString()}{" "}
              <span className="text-sm font-normal text-muted-foreground">credits available</span>
            </div>
            <p className="text-[12.5px] text-muted-foreground mt-1 max-w-[420px]">
              Allocated by your organization. Running low? Ask your admin for more.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="gap-1.5 flex-none"
            onClick={() => navigate("/organization")}
          >
            <Send className="w-3.5 h-3.5" />
            Request more credits
          </Button>
        </div>
      </Card>
    );
  }

  const showAddCredits = usage.enabled && (packsData?.packs?.length ?? 0) > 0;

  const grant = usage.monthlyGrant ?? 0;
  const bundle = usage.bundleBalance ?? 0;
  const reserve = usage.reserveBalance ?? 0;
  const byAction = new Map((usage.tools ?? []).map((t) => [t.action, t]));

  const rows: Row[] = ORDER.map((action) => {
    const t = byAction.get(action);
    const meta = TOOL_META[action];
    return {
      label: meta.label,
      color: meta.color,
      note: meta.note,
      spent: t?.spent ?? 0,
      count: t?.count ?? 0,
      price: t?.price ?? null,
    };
  });
  // Split sheets are not credit-metered — show activity, free.
  rows.push({
    label: "Split sheet",
    color: "var(--t-split)",
    note: "not metered",
    spent: 0,
    count: ent?.usage?.splitSheetsThisPeriod ?? 0,
    price: null,
  });

  const used = Math.max(0, grant - bundle);
  const maxSpent = Math.max(0, ...rows.map((r) => r.spent));
  const segments: RingSegment[] = ORDER.map((action) => ({
    value: byAction.get(action)?.spent ?? 0,
    color: TOOL_META[action].color,
  }));

  const tier = ent?.tier ?? "free";
  const isPaid = isPaidTier(tier);
  const overageOn = ent?.credits?.overageEnabled ?? false;

  const toggleOverage = (next: boolean) =>
    setPrefs.mutate(
      { overage_enabled: next },
      {
        onError: (e) => toast.error(e instanceof Error ? e.message : "Couldn't update pay-per-use."),
        onSuccess: () => toast.success(next ? "Pay-per-use enabled." : "Pay-per-use disabled."),
      },
    );

  return (
    <Card className="overflow-hidden">
      {/* header */}
      <div className="flex items-start justify-between gap-4 px-6 pt-[22px] pb-1.5">
        <div>
          <div className="flex items-center gap-2.5 text-[15px] font-semibold">
            <Coins className="w-[18px] h-[18px] text-muted-foreground" />
            Credits &amp; usage
          </div>
          <div className="text-[13.5px] text-muted-foreground mt-0.5">
            Billing period · {fmtDay(usage.periodStart)} – {fmtDate(usage.periodEnd)}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {showAddCredits && (
            <Button variant="outline" size="sm" className="gap-1.5" onClick={() => setTopUpOpen(true)}>
              <Plus className="w-3.5 h-3.5" />
              Add credits
            </Button>
          )}
          <Badge className="uppercase">{tierLabel(tier)}</Badge>
        </div>
      </div>

      {/* ring + breakdown */}
      <div className="grid grid-cols-1 sm:grid-cols-[240px_1fr] gap-8 px-6 pt-3.5 pb-1.5">
        <CreditRing
          left={bundle}
          grant={grant}
          used={used}
          segments={segments}
          resetLabel={fmtDate(usage.periodEnd)}
        />
        <div>
          <div className="text-[11px] font-semibold tracking-[0.11em] uppercase text-muted-foreground/70 mb-3.5">
            Cost &amp; usage per tool
          </div>
          <div>
            {rows.map((r) => {
              const pct = maxSpent ? Math.round((r.spent / maxSpent) * 100) : 0;
              const unit = r.price == null ? "free" : `${r.price} cr`;
              return (
                <div key={r.label} className="py-[13px] border-t border-border/60 first:border-t-0">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2.5 text-[14.5px] font-medium">
                      <span
                        className="w-[9px] h-[9px] rounded-[3px] flex-none"
                        style={{ background: r.color }}
                      />
                      {r.label}
                      {r.note && (
                        <span className="text-xs text-muted-foreground/70 font-normal">· {r.note}</span>
                      )}
                    </div>
                    <div className="text-sm font-semibold text-right tabular-nums">
                      {r.spent > 0 ? `${r.spent} cr` : "—"}
                      <small className="block text-[11px] font-normal text-muted-foreground mt-px">
                        {r.count} · {unit}
                        {unit !== "free" ? " ea" : ""}
                      </small>
                    </div>
                  </div>
                  <div className="h-[7px] rounded-[5px] bg-muted mt-2.5 overflow-hidden">
                    <span
                      className="block h-full rounded-[5px]"
                      style={{ width: `${Math.max(pct, r.spent > 0 ? 4 : 0)}%`, background: r.color }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          {reserve > 0 && (
            <p className="text-xs text-muted-foreground mt-3">
              + {reserve.toLocaleString()} bonus credits (don&apos;t expire).
            </p>
          )}
        </div>
      </div>

      {/* pay-per-use */}
      {isPaid && (
        <div className="flex items-center justify-between gap-4 mx-6 mt-4 mb-[22px] px-4 py-3.5 border border-border rounded-xl bg-background">
          <div>
            <div className="text-sm font-semibold">Pay-per-use</div>
            <div className="text-[12.5px] text-muted-foreground mt-0.5 max-w-[520px]">
              Keep working past your monthly credits — overage is billed on your next invoice at
              $0.02 / credit.
              {(usage.overageThisPeriod ?? 0) > 0 && ` (${usage.overageThisPeriod} cr this period)`}
            </div>
          </div>
          <Switch checked={overageOn} onCheckedChange={toggleOverage} disabled={setPrefs.isPending} />
        </div>
      )}

      <TopUpCreditsDialog open={topUpOpen} onOpenChange={setTopUpOpen} />
    </Card>
  );
}
