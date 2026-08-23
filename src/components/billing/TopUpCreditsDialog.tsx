// src/components/billing/TopUpCreditsDialog.tsx
// "Add credits" flow off the Credits & usage card: pick a bundle or choose your
// own amount, then redirect to Stripe Checkout. Credits never expire and are
// spent after the monthly grant runs out — framed for non-technical musicians,
// not devs, which is why every bundle carries a "what this typically buys" line
// derived from live credit_prices instead of a bare credit count.
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { ApiError } from "@/lib/apiFetch";
import { usageSummary } from "@/lib/credits";
import {
  useCreditPacks,
  useCreateTopupSession,
  type CreditPack,
  type CustomCreditConfig,
} from "@/hooks/useCreditPacks";

interface TopUpCreditsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Licensing Phase B: when set, purchased credits — bundle or custom —
   * land in this org's pool instead of the caller's personal wallet
   * (admin-only, enforced server-side). */
  orgId?: string;
  /** Display name for the org, used in copy when `orgId` is set. */
  orgName?: string;
}

const perCredit = (pack: CreditPack): number => pack.price_cents / pack.credits / 100;

/** Whole dollars would misprice the ladder — the entry rung is $5.50. */
const usd = (cents: number): string =>
  (cents / 100).toLocaleString("en-US", { style: "currency", currency: "USD" });

/** Cents, 2 decimals max, trailing zeros trimmed — "1.83¢", "1.6¢", "2¢".
 * Cents rather than dollars because a 4-decimal dollar figure reads like a
 * float, and rounder dollar forms collapse rungs that only separate at the
 * fourth decimal ($0.0183 vs $0.0175 both round to $0.018). */
const perCreditLabel = (dollars: number): string =>
  `${(dollars * 100).toFixed(2).replace(/\.?0+$/, "")}¢/credit`;

/** Slider granularity. Fine enough to land on a round number, coarse enough
 * that dragging doesn't re-render per credit. Typed values are NOT snapped to
 * it — the backend takes any whole number in range, and snapping what someone
 * typed would buy them a different amount than the one on screen. */
const STEP = 100;

const clamp = (value: number, cfg: CustomCreditConfig): number =>
  Math.min(cfg.maxCredits, Math.max(cfg.minCredits, Math.floor(value)));

export function TopUpCreditsDialog({ open, onOpenChange, orgId, orgName }: TopUpCreditsDialogProps) {
  const { data, isLoading } = useCreditPacks();
  const {
    mutate: startCheckout,
    isPending,
    error,
    variables: pendingArgs,
  } = useCreateTopupSession();
  const pendingKey = pendingArgs?.packKey;

  const packs = data?.packs ?? [];
  const prices = data?.prices;
  const custom = data?.custom;

  const [customCredits, setCustomCredits] = useState<number | null>(null);
  // Seed once the config arrives — a sensible default beats an empty input,
  // and `null` until then stops the slider flashing at 0.
  useEffect(() => {
    if (custom && customCredits === null) {
      setCustomCredits(clamp(custom.minCredits * 4, custom));
    }
  }, [custom, customCredits]);

  const bestKey =
    packs.length > 0
      ? packs.reduce((best, p) => (perCredit(p) < perCredit(best) ? p : best), packs[0]).key
      : null;

  const errorMessage = error
    ? error instanceof ApiError
      ? error.message
      : "Couldn't start checkout. Please try again."
    : null;

  // The number shown, priced, and bought must be the same one — quoting a
  // total the checkout won't charge is the one thing this dialog can't do.
  const customValid =
    custom != null &&
    customCredits != null &&
    Number.isInteger(customCredits) &&
    customCredits >= custom.minCredits &&
    customCredits <= custom.maxCredits;
  const customPriceCents =
    customValid && custom && customCredits != null
      ? Math.round(customCredits * custom.perCreditCents)
      : 0;
  const customBuys = customValid && customCredits != null ? usageSummary(customCredits, prices) : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Add credits</DialogTitle>
          <DialogDescription>
            {orgId
              ? orgName
                ? `Top up ${orgName}'s credit pool with a one-time purchase.`
                : "Top up the shared credit pool with a one-time purchase."
              : "Top up with a one-time purchase — no change to your subscription."}
          </DialogDescription>
        </DialogHeader>

        <div>
          {isLoading && (
            <div className="text-sm text-muted-foreground py-6 text-center">Loading packs…</div>
          )}
          {!isLoading && packs.length === 0 && (
            <div className="text-sm text-muted-foreground py-6 text-center">
              No credit packs are available right now.
            </div>
          )}
          {packs.map((pack) => {
            const buys = usageSummary(pack.credits, prices);
            return (
              <div
                key={pack.key}
                className="flex items-center justify-between gap-3 py-3.5 border-t border-border/60 first:border-t-0"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2 text-[14.5px] font-medium">
                    {pack.label ?? `${pack.credits.toLocaleString()} credits`}
                    {pack.key === bestKey && (
                      <Badge
                        variant="outline"
                        className="border-primary/30 text-primary bg-primary/10 text-[10.5px] uppercase tracking-wide"
                      >
                        Best value
                      </Badge>
                    )}
                  </div>
                  <div className="text-[13px] text-muted-foreground mt-0.5">
                    {pack.credits.toLocaleString()} credits · {usd(pack.price_cents)} ·{" "}
                    {perCreditLabel(perCredit(pack))}
                  </div>
                  {buys && (
                    <div className="text-[12.5px] text-muted-foreground/80 mt-0.5 leading-snug">
                      {buys}
                    </div>
                  )}
                </div>
                <Button
                  size="sm"
                  className="shrink-0"
                  onClick={() => startCheckout({ packKey: pack.key, orgId })}
                  disabled={isPending}
                >
                  {isPending && pendingKey === pack.key && (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  )}
                  Buy
                </Button>
              </div>
            );
          })}
        </div>

        {custom && customCredits !== null && (
          <div className="border-t border-border/60 pt-4">
            <div className="text-[14.5px] font-medium">Or choose your own amount</div>
            <div className="flex items-center gap-3 mt-3">
              <Slider
                value={[customValid ? customCredits : custom.minCredits]}
                min={custom.minCredits}
                max={custom.maxCredits}
                step={STEP}
                onValueChange={([v]) => setCustomCredits(v)}
                className="flex-1"
                aria-label="Credits to buy"
              />
              <Input
                type="number"
                inputMode="numeric"
                min={custom.minCredits}
                max={custom.maxCredits}
                value={customCredits}
                // Clamp on blur, not on change: correcting mid-typing fights
                // the user — typing "3000" would jump at "3".
                onChange={(e) => setCustomCredits(Number(e.target.value))}
                onBlur={() => setCustomCredits((c) => clamp(Number(c) || custom.minCredits, custom))}
                className="w-28 h-9"
                aria-label="Credits to buy"
              />
            </div>
            {customValid ? (
              <>
                <div className="text-[13px] text-muted-foreground mt-2">
                  {usd(customPriceCents)} · {perCreditLabel(custom.perCreditCents / 100)}
                </div>
                {customBuys && (
                  <div className="text-[12.5px] text-muted-foreground/80 mt-0.5 leading-snug">
                    {customBuys}
                  </div>
                )}
              </>
            ) : (
              <div className="text-[13px] text-muted-foreground mt-2">
                Enter a whole number between {custom.minCredits.toLocaleString()} and{" "}
                {custom.maxCredits.toLocaleString()} credits.
              </div>
            )}
            <Button
              className="w-full mt-3"
              onClick={() => customCredits != null && startCheckout({ credits: customCredits, orgId })}
              disabled={isPending || !customValid}
            >
              {isPending && pendingArgs?.credits != null && (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              )}
              {customValid && customCredits != null
                ? `Buy ${customCredits.toLocaleString()} credits — ${usd(customPriceCents)}`
                : "Buy credits"}
            </Button>
          </div>
        )}

        {errorMessage && <div className="text-sm text-destructive">{errorMessage}</div>}

        <p className="text-xs text-muted-foreground/70">
          {orgId
            ? "Credits never expire and are shared across every seat."
            : "Credits never expire and are used after your monthly credits run out."}{" "}
          Usage figures are typical — you&apos;re only charged for what each run actually costs.
        </p>
      </DialogContent>
    </Dialog>
  );
}
