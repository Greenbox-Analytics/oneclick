// src/components/oneclick/payments/PayoutRuns.tsx
import { CheckCheck, Clock, PieChart, X, Send } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useMarkPayoutPaid, useCancelPayout } from "@/hooks/useRoyalties";
import type { PayoutOut } from "@/hooks/useRoyalties";
import { PartyAvatar, StatusBadge, fmtMoney, fmtDate } from "./shared";
import { useToast } from "@/hooks/use-toast";

interface PayoutRunsProps {
  payouts: PayoutOut[];
  onOpenDetail: (id: string) => void;
}

// ---------------------------------------------------------------------------
// Helper: pull display name from snapshot
// ---------------------------------------------------------------------------
function getPayeeName(payout: PayoutOut): string {
  const snap = payout.breakdown_snapshot as Record<string, unknown>;
  const payee = snap?.payee as Record<string, unknown> | undefined;
  return (payee?.display_name as string | undefined) ?? payout.payee_id ?? "Unknown";
}

// ---------------------------------------------------------------------------
// RunCard
// ---------------------------------------------------------------------------
function RunCard({ payout, onOpenDetail }: { payout: PayoutOut; onOpenDetail: (id: string) => void }) {
  const { toast } = useToast();
  const markPaid = useMarkPayoutPaid();
  const cancelPayout = useCancelPayout();

  const isDraft = payout.status === "draft";
  const isPaid = payout.status === "paid";
  const payeeName = getPayeeName(payout);

  const Ic = isPaid ? CheckCheck : Clock;
  const iconTone = isPaid
    ? "bg-[hsl(var(--pay-paid-bg))] text-[hsl(var(--pay-paid-fg))]"
    : "bg-[hsl(var(--pay-sched-bg))] text-[hsl(var(--pay-sched-fg))]";

  const badge = isPaid ? (
    <StatusBadge kind="paid">
      <CheckCheck className="h-3 w-3" /> Paid
    </StatusBadge>
  ) : (
    <StatusBadge kind="sched">
      <Clock className="h-3 w-3" /> Draft
    </StatusBadge>
  );

  const orphanBadge =
    payout.orphan_state && payout.orphan_state !== "none" ? (
      <StatusBadge kind={payout.orphan_state === "orphaned" ? "out" : "partial"}>
        {payout.orphan_state === "orphaned" ? "Orphaned" : "Partial"}
      </StatusBadge>
    ) : null;

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-card">
      <div className="grid grid-cols-[auto_1fr_auto] items-center gap-3.5 px-[18px] py-[15px]">
        {/* Icon */}
        <span className={cn("flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-[10px]", iconTone)}>
          <Ic className="h-[18px] w-[18px]" />
        </span>

        {/* Info */}
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <PartyAvatar id={payout.payee_id} name={payeeName} size={22} />
            <span className="text-[15px] font-bold tracking-tight">{payeeName}</span>
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[12.5px] text-muted-foreground">
            {badge}
            {orphanBadge}
            <span className="h-[3px] w-[3px] rounded-full bg-border" />
            <span>{fmtDate(payout.created_at)}</span>
          </div>
        </div>

        {/* Amount + actions */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="text-right">
            <div className="font-mono text-[18px] font-bold tabular-nums tracking-tight">
              {fmtMoney(payout.total_amount, payout.pay_currency)}
            </div>
            <div className="mt-px text-[11.5px] text-muted-foreground">{payout.pay_currency}</div>
          </div>

          {/* Analysis button — always visible */}
          <button
            className="inline-flex items-center gap-1.5 rounded-lg bg-secondary px-2.5 py-1.5 text-xs font-semibold text-primary hover:brightness-95"
            onClick={() => onOpenDetail(payout.id)}
          >
            <PieChart className="h-3.5 w-3.5" /> Analysis
          </button>

          {/* Draft actions */}
          {isDraft && (
            <>
              <Button
                size="sm"
                variant="outline"
                disabled={markPaid.isPending}
                onClick={() =>
                  markPaid.mutate(payout.id, {
                    onError: () =>
                      toast({ variant: "destructive", title: "Failed to mark paid" }),
                  })
                }
              >
                <CheckCheck className="mr-1.5 h-3.5 w-3.5" /> Mark paid
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="text-destructive hover:text-destructive"
                disabled={cancelPayout.isPending}
                onClick={() =>
                  cancelPayout.mutate(payout.id, {
                    onError: () =>
                      toast({ variant: "destructive", title: "Failed to cancel payout" }),
                  })
                }
              >
                <X className="mr-1 h-3.5 w-3.5" /> Cancel
              </Button>
            </>
          )}

          {/* Paid action */}
          {isPaid && (
            <Button
              size="sm"
              variant="outline"
              onClick={() =>
                toast({ description: "Receipt download — coming soon." })
              }
            >
              Receipt
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// PayoutRuns
// ---------------------------------------------------------------------------

export function PayoutRuns({ payouts, onOpenDetail }: PayoutRunsProps) {
  const byDate = (a: PayoutOut, b: PayoutOut) =>
    new Date(b.created_at).getTime() - new Date(a.created_at).getTime();

  const drafts = payouts.filter((r) => r.status === "draft").sort(byDate);
  const paid = payouts.filter((r) => r.status === "paid").sort(byDate);

  if (payouts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-border bg-card p-12 text-muted-foreground shadow-sm">
        <Send className="h-8 w-8 opacity-30" />
        <p className="text-sm font-medium">No payouts yet</p>
        <p className="text-xs">Create a payout from the Parties tab to get started.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {drafts.length > 0 && (
        <div className="px-0.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
          Drafts
        </div>
      )}
      {drafts.map((r) => (
        <RunCard key={r.id} payout={r} onOpenDetail={onOpenDetail} />
      ))}

      {paid.length > 0 && (
        <div className={cn("px-0.5 text-[11px] font-bold uppercase tracking-wider text-muted-foreground", drafts.length > 0 && "mt-2.5")}>
          Paid
        </div>
      )}
      {paid.map((r) => (
        <RunCard key={r.id} payout={r} onOpenDetail={onOpenDetail} />
      ))}
    </div>
  );
}
