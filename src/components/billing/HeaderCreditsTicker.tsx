// src/components/billing/HeaderCreditsTicker.tsx
// Credits pill in the page header (every authed page, left of the icon row).
// Click opens a usage popover with a per-tool breakdown and a way to buy more;
// at zero the click goes straight to top-up. Renders nothing while entitlements
// load or when the credits system is off — same guard as CreditsChip.
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertCircle, AlertTriangle, Coins } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn, fmtDay } from "@/lib/utils";
import { useEntitlements } from "@/hooks/useEntitlements";
import { useCreditUsage, type CreditAction } from "@/hooks/useCreditUsage";
import { TopUpCreditsDialog } from "@/components/billing/TopUpCreditsDialog";
import { creditStanding, POOL_ONLY_LABEL } from "@/lib/credits";

// Popover labels are activity phrasing ("runs"), unlike CreditsUsageCard's
// per-tool rows, which quote the base rate — hence not shared.
const TOOLS: { action: CreditAction; label: string; color: string }[] = [
  { action: "oneclick_run", label: "OneClick runs", color: "var(--t-oneclick)" },
  { action: "registry_parse", label: "Contract parses", color: "var(--t-registry)" },
  { action: "zoe_message", label: "Zoe analyses", color: "var(--t-zoe)" },
  { action: "split_sheet", label: "Split sheets", color: "var(--t-split)" },
];

export function HeaderCreditsTicker() {
  const navigate = useNavigate();
  const { data: ent } = useEntitlements();
  const [open, setOpen] = useState(false);
  const [topUpOpen, setTopUpOpen] = useState(false);
  // Per-tool breakdown is only needed once the popover opens.
  const { data: usage } = useCreditUsage(open);

  const credits = ent?.credits;
  if (!credits) return null; // credits off / still loading

  const managedByOrg =
    ent?.billingContext?.type === "org" ? ent.billingContext : credits.managedByOrg;
  const isOrgAdmin = managedByOrg?.role === "admin";

  // null = an uncapped org member: no cap of their own, and the pool balance is
  // admin-only. There is no figure and no meter to draw — the pill says where
  // their credits come from instead, and never claims they are out.
  const standing = creditStanding(credits);
  const remaining = standing?.remaining ?? 0;
  const total = standing?.total ?? 0;
  const pct = total > 0 ? Math.min(1, remaining / total) : 0;
  const out = standing != null && remaining <= 0;
  const low = standing != null && !out && pct <= 0.1; // matches the credit-wall trigger

  const resetDay = fmtDay(credits.periodEnd);
  const title = !standing
    ? POOL_ONLY_LABEL
    : out
      ? "You're out of credits"
      : `${remaining.toLocaleString()} of ${total.toLocaleString()} credits left${resetDay ? ` — resets ${resetDay}` : ""}`;

  // Members buy nothing on a pool — their remedy is asking the org.
  const openBuy = () => {
    if (managedByOrg && !isOrgAdmin) navigate("/teams");
    else setTopUpOpen(true);
  };

  const pill = (
    <button
      type="button"
      title={title}
      aria-label={title}
      onClick={out ? openBuy : undefined}
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-[12.5px] tabular-nums transition-colors",
        out
          ? "border-destructive/40 bg-destructive/10 text-destructive hover:bg-destructive/15"
          : low
            ? "border-amber-500/50 bg-amber-500/10 hover:bg-amber-500/20"
            : "border-border bg-muted/40 hover:bg-muted",
      )}
    >
      {out ? (
        <AlertCircle className="w-3.5 h-3.5 flex-none" />
      ) : low ? (
        <AlertTriangle className="w-3.5 h-3.5 flex-none text-amber-600" />
      ) : (
        <Coins className="w-3.5 h-3.5 flex-none text-muted-foreground" />
      )}
      <span className="whitespace-nowrap">
        {standing ? (
          <>
            <span className="font-semibold">{remaining.toLocaleString()}</span>
            {out ? (
              <span className="font-normal opacity-75"> credits · buy more</span>
            ) : (
              <span className="font-normal text-muted-foreground hidden xl:inline"> credits</span>
            )}
          </>
        ) : (
          <span className="font-medium text-muted-foreground">Shared pool</span>
        )}
      </span>
      {!out && standing && (
        <span className="hidden sm:block w-[42px] h-1 rounded-full bg-border overflow-hidden">
          <span
            className={cn("block h-full rounded-full", low ? "bg-amber-500" : "bg-primary")}
            style={{ width: `${Math.round(pct * 100)}%` }}
          />
        </span>
      )}
    </button>
  );

  const byAction = new Map((usage?.tools ?? []).map((t) => [t.action, t]));

  return (
    <>
      {out ? (
        pill
      ) : (
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>{pill}</PopoverTrigger>
          <PopoverContent align="end" className="w-[288px] p-3.5">
            <div className="flex items-baseline justify-between gap-2">
              {standing ? (
                <div className="text-[22px] font-bold tabular-nums leading-none">
                  {remaining.toLocaleString()}{" "}
                  <span className="text-[12.5px] font-medium text-muted-foreground">
                    {credits.memberCap != null
                      ? `of your ${credits.memberCap.toLocaleString()} / month limit`
                      : "left"}
                  </span>
                </div>
              ) : (
                // Uncapped org member: no personal ceiling, and the pool is the
                // org's business. Say where the credits come from, full stop.
                <div className="text-[13px] font-medium leading-snug">{POOL_ONLY_LABEL}</div>
              )}
              {resetDay && (
                <div className="text-[11.5px] text-muted-foreground whitespace-nowrap">
                  Resets {resetDay}
                </div>
              )}
            </div>

            <div className="flex h-[7px] rounded-full bg-muted overflow-hidden mt-3 mb-2">
              {total > 0 &&
                TOOLS.map((t) => {
                  const spent = byAction.get(t.action)?.spent ?? 0;
                  return spent > 0 ? (
                    <span
                      key={t.action}
                      className="block h-full"
                      style={{ width: `${(spent / total) * 100}%`, background: t.color }}
                    />
                  ) : null;
                })}
            </div>

            {TOOLS.map((t) => (
              <div key={t.action} className="flex items-center gap-2 py-[5px] text-[12.5px]">
                <span
                  className="w-[7px] h-[7px] rounded-[2px] flex-none"
                  style={{ background: t.color }}
                />
                <span className="flex-1">{t.label}</span>
                <span className="text-muted-foreground tabular-nums">
                  {byAction.get(t.action)?.count ?? 0}
                </span>
              </div>
            ))}

            <div className="flex gap-2 border-t border-border mt-2.5 pt-3">
              <Button
                size="sm"
                className="flex-1"
                onClick={() => {
                  setOpen(false);
                  openBuy();
                }}
              >
                {managedByOrg ? (isOrgAdmin ? "Buy credits" : "Request credits") : "Add credits"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="flex-1"
                onClick={() => {
                  setOpen(false);
                  navigate("/profile#credits-usage");
                }}
              >
                Usage details
              </Button>
            </div>
          </PopoverContent>
        </Popover>
      )}
      <TopUpCreditsDialog
        open={topUpOpen}
        onOpenChange={setTopUpOpen}
        orgId={isOrgAdmin ? managedByOrg?.orgId : undefined}
        orgName={isOrgAdmin ? managedByOrg?.orgName : undefined}
      />
    </>
  );
}
