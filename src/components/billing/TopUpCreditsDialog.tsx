// src/components/billing/TopUpCreditsDialog.tsx
// "Add credits" flow off the Credits & usage card: pick a one-time pack,
// redirect to Stripe Checkout. Packs never expire and are spent after the
// monthly grant runs out — framed for non-technical musicians, not devs.
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
import { ApiError } from "@/lib/apiFetch";
import { useCreditPacks, useCreateTopupSession, type CreditPack } from "@/hooks/useCreditPacks";

interface TopUpCreditsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Licensing Phase B: when set, purchased credits land in this org's pool
   * instead of the caller's personal wallet (admin-only, enforced server-side). */
  orgId?: string;
  /** Display name for the org, used in copy when `orgId` is set. */
  orgName?: string;
}

const perCredit = (pack: CreditPack): number => pack.price_cents / pack.credits / 100;

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
  const bestKey =
    packs.length > 0
      ? packs.reduce((best, p) => (perCredit(p) < perCredit(best) ? p : best), packs[0]).key
      : null;

  const errorMessage = error
    ? error instanceof ApiError
      ? error.message
      : "Couldn't start checkout. Please try again."
    : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Add credits</DialogTitle>
          <DialogDescription>
            {orgId
              ? `Top up ${orgName ?? "your organization"}'s credit pool with a one-time pack.`
              : "Top up with a one-time pack — no change to your subscription."}
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
          {packs.map((pack) => (
            <div
              key={pack.key}
              className="flex items-center justify-between gap-3 py-3.5 border-t border-border/60 first:border-t-0"
            >
              <div>
                <div className="flex items-center gap-2 text-[14.5px] font-medium">
                  {pack.credits.toLocaleString()} credits
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
                  ${(pack.price_cents / 100).toFixed(0)} · ${perCredit(pack).toFixed(3)}/credit
                </div>
              </div>
              <Button
                size="sm"
                onClick={() => startCheckout({ packKey: pack.key, orgId })}
                disabled={isPending}
              >
                {isPending && pendingKey === pack.key && (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                )}
                Buy
              </Button>
            </div>
          ))}
        </div>

        {errorMessage && <div className="text-sm text-destructive">{errorMessage}</div>}

        <p className="text-xs text-muted-foreground/70">
          {orgId
            ? "Credits never expire and are shared across every seat in the organization."
            : "Credits never expire and are used after your monthly credits run out."}
        </p>
      </DialogContent>
    </Dialog>
  );
}
