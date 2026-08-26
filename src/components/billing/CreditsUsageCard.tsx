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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { creditStanding, orgContext, POOL_ONLY_LABEL, TOOL_META } from "@/lib/credits";
import { useEntitlements } from "@/hooks/useEntitlements";
import { useCreditUsage, type CreditAction } from "@/hooks/useCreditUsage";
import { useCreditPacks } from "@/hooks/useCreditPacks";
import { useSetBillingPrefs } from "@/hooks/useBilling";
import { CreditRing, type RingSegment } from "@/components/billing/CreditRing";
import { TopUpCreditsDialog } from "@/components/billing/TopUpCreditsDialog";
import { isPaidTier, tierLabel, ENTERPRISE_LABEL } from "@/lib/tiers";
import { fmtDate, fmtDay } from "@/lib/utils";

// Ring/list order matches the mockup.
const ORDER: CreditAction[] = ["oneclick_run", "registry_parse", "zoe_message", "split_sheet"];

export function CreditsUsageCard() {
  const { data: usage, isLoading } = useCreditUsage();
  const { data: ent } = useEntitlements();
  const setPrefs = useSetBillingPrefs();
  const { data: packsData } = useCreditPacks();
  const [topUpOpen, setTopUpOpen] = useState(false);
  // Pay-per-use monthly spend limit — draft-until-blur (see commitLimit below).
  // Declared before the early returns so hook order stays stable.
  const [limitDraft, setLimitDraft] = useState<string | null>(null);
  const navigate = useNavigate();

  if (isLoading || !usage) return null;
  if (!usage.enabled) return null; // flag off → no credit surfaces

  const managedByOrg = orgContext(ent);

  // Org billing context (Licensing Phase B, spec §5): what a member has is
  // a monthly LIMIT on the org's shared pool, not a personal grant and not an
  // allocation they hold — no pack picker, no pay-per-use toggle. Both numbers
  // come straight off entitlements (context-aware) rather than the per-tool
  // usage breakdown, which is a personal-wallet concept.
  if (managedByOrg) {
    const cap = ent?.credits?.memberCap ?? null;
    const capUsed = ent?.credits?.memberCapUsed ?? 0;
    // null unless the caller is an org ADMIN — the pool is the org's money.
    const poolBalance = ent?.credits?.balance ?? null;
    const standing = creditStanding(ent?.credits);
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
          <Badge className="uppercase">{managedByOrg.kind === "self_serve" ? "Team" : ENTERPRISE_LABEL}</Badge>
        </div>

        <div className="flex items-center justify-between gap-4 flex-wrap px-6 pt-3.5 pb-[22px]">
          <div>
            {standing ? (
              <div className="text-[32px] font-bold tracking-tight tabular-nums">
                {standing.remaining.toLocaleString()}{" "}
                <span className="text-sm font-normal text-muted-foreground">credits left this month</span>
              </div>
            ) : (
              // Uncapped member: no ceiling of their own and no visibility of
              // the pool, so there is genuinely no number to headline.
              <div className="text-[22px] font-semibold tracking-tight">{POOL_ONLY_LABEL}</div>
            )}
            <p className="text-[12.5px] text-muted-foreground mt-1 max-w-[440px]">
              {cap != null ? (
                <>
                  You&apos;ve used {capUsed.toLocaleString()} of your {cap.toLocaleString()} monthly limit.
                  {/* The pool balance is admin-only — a member sees their limit and nothing else. */}
                  {poolBalance != null && <> {managedByOrg.orgName} has {poolBalance.toLocaleString()} in the shared pool.</>}{" "}
                  Need more? Ask your admin to raise your limit.
                </>
              ) : (
                <>
                  You draw from the shared pool with no personal limit
                  {poolBalance != null && <>, which holds {poolBalance.toLocaleString()} credits</>}. Running low? Ask
                  your admin to top the pool up.
                </>
              )}
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            className="gap-1.5 flex-none"
            onClick={() => navigate("/teams")}
          >
            <Send className="w-3.5 h-3.5" />
            Request a higher limit
          </Button>
        </div>
      </Card>
    );
  }

  const showAddCredits = (packsData?.packs?.length ?? 0) > 0;

  const grant = usage.monthlyGrant ?? 0;
  const bundle = usage.bundleBalance ?? 0;
  const reserve = usage.reserveBalance ?? 0;
  const byAction = new Map((usage.tools ?? []).map((t) => [t.action, t]));

  const rows = ORDER.map((action) => ({
    ...TOOL_META[action],
    spent: byAction.get(action)?.spent ?? 0,
    count: byAction.get(action)?.count ?? 0,
    price: byAction.get(action)?.price ?? null,
  }));
  const used = Math.max(0, grant - bundle);
  const maxSpent = Math.max(0, ...rows.map((r) => r.spent));
  const segments: RingSegment[] = rows.map((r) => ({ value: r.spent, color: r.color }));

  const tier = ent?.tier ?? "free";
  const isPaid = isPaidTier(tier);
  const overageOn = ent?.credits?.overageEnabled ?? false;
  // Rate comes from the backend (same getter the Stripe biller uses) so the
  // quote here can't drift from what a user is actually charged.
  const overageRate = ent?.credits?.overageUsdPerCredit;

  const toggleOverage = (next: boolean) =>
    setPrefs.mutate(
      { overage_enabled: next },
      {
        onError: (e) => toast.error(e instanceof Error ? e.message : "Couldn't update pay-per-use."),
        onSuccess: () => toast.success(next ? "Pay-per-use enabled." : "Pay-per-use disabled."),
      },
    );

  // Monthly spend limit on pay-per-use (backend `overage_cap_credits`; the
  // credit walls tell users to "raise it in Billing settings" — this is that
  // control). Draft-until-blur so we don't fire a request per keystroke.
  const overageCap = ent?.credits?.overageCapCredits ?? null;
  const limitValue = limitDraft ?? (overageCap != null ? String(overageCap) : "");
  const commitLimit = () => {
    if (limitDraft == null) return;
    const trimmed = limitDraft.trim();
    const parsed = trimmed === "" ? null : Number(trimmed);
    if (trimmed !== "" && (!Number.isFinite(parsed) || parsed <= 0)) {
      toast.error("Enter a positive number of credits, or leave it empty for no limit.");
      return;
    }
    if (parsed === overageCap) {
      setLimitDraft(null);
      return;
    }
    setPrefs.mutate(
      { overage_cap_credits: parsed },
      {
        onSuccess: () => {
          setLimitDraft(null);
          toast.success(parsed == null ? "Monthly limit removed." : "Monthly limit saved.");
        },
        onError: (e) => toast.error(e instanceof Error ? e.message : "Couldn't save the limit."),
      },
    );
  };

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
              // The base rate IS the price (spec 2026-08-17 §2), so quote it
              // flat. Only surface an average when the metered tail actually
              // moved the total — i.e. the period's spend exceeds count x base,
              // which happens on pathological runs and nowhere else.
              const unit =
                r.price == null
                  ? "free"
                  : r.count > 0 && r.spent > r.count * r.price
                    ? `${r.price} cr ea · ${Math.round(r.spent / r.count)} cr avg`
                    : `${r.price} cr ea`;
              return (
                <div key={r.label} className="py-[13px] border-t border-border/60 first:border-t-0">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2.5 text-[14.5px] font-medium">
                      <span
                        className="w-[9px] h-[9px] rounded-[3px] flex-none"
                        style={{ background: r.color }}
                      />
                      {r.label}
                    </div>
                    <div className="text-sm font-semibold text-right tabular-nums">
                      {r.spent > 0 ? `${r.spent} cr` : "—"}
                      <small className="block text-[11px] font-normal text-muted-foreground mt-px">
                        {r.count} · {unit}
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
          <p className="text-xs text-muted-foreground mt-3">
            Unused monthly credits don&apos;t carry over. Credits you buy separately never expire.
          </p>
          {/* Grandfathered rate (2026-08-15 tiers): live grant above the tier's
              listed grant means this user kept their old monthly credits, which
              expire with the already-paid period (grandfathered_until). Say so
              here — this card is the surface that shows "of {grant} / mo", and
              silence would make the drop at renewal read as a surprise
              downgrade. Guard tierGrant > 0: -1 means unlimited. */}
          {(ent?.caps?.monthlyCredits ?? 0) > 0 && grant > ent.caps.monthlyCredits && (
            <p className="text-xs text-muted-foreground mt-1.5">
              You&apos;re keeping your current {grant.toLocaleString()} credits a month until your
              next renewal. After that, your plan includes{" "}
              {ent.caps.monthlyCredits.toLocaleString()} a month.
            </p>
          )}
        </div>
      </div>

      {/* pay-per-use */}
      {isPaid && (
        <div className="mx-6 mt-4 mb-[22px] px-4 py-3.5 border border-border rounded-xl bg-background">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-sm font-semibold">Pay-per-use</div>
              <div className="text-[12.5px] text-muted-foreground mt-0.5 max-w-[520px]">
                Keep working past your monthly credits — overage is billed on your next invoice
                {overageRate ? ` at US$${overageRate.toFixed(2)} / credit` : ""}.
                {(usage.overageThisPeriod ?? 0) > 0 && ` (${usage.overageThisPeriod} cr this period)`}
              </div>
            </div>
            <Switch checked={overageOn} onCheckedChange={toggleOverage} disabled={setPrefs.isPending} />
          </div>
          {overageOn && (
            <div className="mt-3 pt-3 border-t border-border/60">
              <Label htmlFor="overage-cap" className="text-xs">
                Monthly limit (credits)
              </Label>
              <div className="flex items-center gap-3 mt-1.5">
                <Input
                  id="overage-cap"
                  type="number"
                  min={1}
                  className="h-8 w-32"
                  placeholder="No limit"
                  value={limitValue}
                  onChange={(e) => setLimitDraft(e.target.value)}
                  onBlur={commitLimit}
                  disabled={setPrefs.isPending}
                />
                <p className="text-[12px] text-muted-foreground">
                  Pay-per-use stops when you hit it. Leave empty for no limit.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      <TopUpCreditsDialog open={topUpOpen} onOpenChange={setTopUpOpen} />
    </Card>
  );
}
