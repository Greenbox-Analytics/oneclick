// src/components/oneclick/payments/PayoutRuns.tsx
import { useState } from "react";
import { CheckCheck, Clock, PieChart, X, Send, MoreHorizontal, Undo2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useMarkPayoutPaid, useCancelPayout, useRevertPayout } from "@/hooks/useRoyalties";
import type { PayoutOut, PayeeSummary } from "@/hooks/useRoyalties";
import { PartyAvatar, StatusBadge, SelectBox, fmtMoney, fmtDate } from "./shared";
import { PayWithPayPalDialog, isPaypalEnabled } from "./PayWithPayPalDialog";
import { ReceiptDialog } from "./ReceiptDialog";
import { useToast } from "@/hooks/use-toast";

interface PayoutRunsProps {
  payouts: PayoutOut[];
  payees?: PayeeSummary[];
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
function RunCard({
  payout,
  onOpenDetail,
  onPayWithPaypal,
  onReceipt,
  selected,
  onToggleSelect,
}: {
  payout: PayoutOut;
  onOpenDetail: (id: string) => void;
  onPayWithPaypal?: (payout: PayoutOut) => void;
  onReceipt: (payout: PayoutOut) => void;
  selected?: boolean;
  onToggleSelect?: () => void;
}) {
  const { toast } = useToast();
  const markPaid = useMarkPayoutPaid();
  const cancelPayout = useCancelPayout();
  const revert = useRevertPayout();

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
        {/* Selection checkbox (drafts only) + status icon */}
        <span className="flex items-center gap-2.5">
          {onToggleSelect && <SelectBox on={!!selected} onClick={onToggleSelect} />}
          <span className={cn("flex h-[38px] w-[38px] shrink-0 items-center justify-center rounded-[10px]", iconTone)}>
            <Ic className="h-[18px] w-[18px]" />
          </span>
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

        {/* Amount + actions — one evenly-spaced cluster on the right */}
        <div className="flex items-center gap-7 shrink-0">
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

          {/* Draft actions: Pay with PayPal + overflow menu (Mark paid / Cancel) */}
          {isDraft && (
            <>
              {onPayWithPaypal && (
                <Button size="sm" onClick={() => onPayWithPaypal(payout)}>
                  <span className="mr-1.5 inline-flex h-[18px] w-[18px] items-center justify-center rounded-[4px] bg-white">
                    <img src="/paypal.png" alt="" className="h-3 w-3" />
                  </span>
                  Pay with PayPal
                </Button>
              )}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-8 w-8 shrink-0 text-muted-foreground"
                    aria-label="More payout actions"
                    disabled={markPaid.isPending || cancelPayout.isPending}
                  >
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-44">
                  {/* Manual fallback for payments made outside PayPal (bank
                      transfer, unsupported currency, payee without PayPal). */}
                  <DropdownMenuItem
                    disabled={markPaid.isPending}
                    onClick={() =>
                      markPaid.mutate(payout.id, {
                        onError: () =>
                          toast({ variant: "destructive", title: "Failed to mark paid" }),
                      })
                    }
                  >
                    <CheckCheck className="mr-2 h-4 w-4" /> Mark as paid
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    className="text-destructive focus:text-destructive"
                    disabled={cancelPayout.isPending}
                    onClick={() =>
                      cancelPayout.mutate(payout.id, {
                        onError: () =>
                          toast({ variant: "destructive", title: "Failed to cancel payout" }),
                      })
                    }
                  >
                    <X className="mr-2 h-4 w-4" /> Cancel payout
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </>
          )}

          {/* Paid actions: Receipt + overflow menu (Revert to draft — manual only) */}
          {isPaid && (
            <>
              <Button size="sm" variant="outline" onClick={() => onReceipt(payout)}>
                Receipt
              </Button>
              {/* PayPal payouts moved real money — reverting our status wouldn't
                  undo that, so revert is offered for manual completions only. */}
              {payout.payment_method !== "paypal" && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-8 w-8 shrink-0 text-muted-foreground"
                      aria-label="More payout actions"
                      disabled={revert.isPending}
                    >
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-48">
                    <DropdownMenuItem
                      disabled={revert.isPending}
                      onClick={() =>
                        revert.mutate(payout.id, {
                          onError: () =>
                            toast({ variant: "destructive", title: "Failed to revert payout" }),
                        })
                      }
                    >
                      <Undo2 className="mr-2 h-4 w-4" /> Revert to draft
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// PayoutRuns
// ---------------------------------------------------------------------------

export function PayoutRuns({ payouts, payees = [], onOpenDetail }: PayoutRunsProps) {
  const [paypalPayout, setPaypalPayout] = useState<PayoutOut | null>(null);
  const [receiptPayout, setReceiptPayout] = useState<PayoutOut | null>(null);
  // Default to the actionable view (Drafts); fall back to Completed when there's
  // nothing to action so the tab never opens on an empty list.
  const [subView, setSubView] = useState<"drafts" | "completed">(() =>
    payouts.some((r) => r.status === "draft") ? "drafts" : "completed",
  );

  const byDate = (a: PayoutOut, b: PayoutOut) =>
    new Date(b.created_at).getTime() - new Date(a.created_at).getTime();

  const drafts = payouts.filter((r) => r.status === "draft").sort(byDate);
  const paid = payouts.filter((r) => r.status === "paid").sort(byDate);
  const activeList = subView === "drafts" ? drafts : paid;

  // Bulk selection — drafts only (Mark as Paid / Cancel).
  const { toast } = useToast();
  const bulkMarkPaid = useMarkPayoutPaid();
  const bulkCancel = useCancelPayout();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [confirmAction, setConfirmAction] = useState<null | "markPaid" | "cancel">(null);
  const [bulkBusy, setBulkBusy] = useState(false);

  const draftIds = drafts.map((r) => r.id);
  const selectedDraftIds = draftIds.filter((id) => selectedIds.has(id));
  const selectedCount = selectedDraftIds.length;
  const allSelected = draftIds.length > 0 && selectedCount === draftIds.length;

  const toggleSelect = (id: string) =>
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const toggleSelectAll = () => setSelectedIds(allSelected ? new Set() : new Set(draftIds));
  // Selection is draft-scoped — clear it when switching to Completed.
  const switchView = (v: "drafts" | "completed") => {
    setSubView(v);
    setSelectedIds(new Set());
  };
  const runBulk = async () => {
    const action = confirmAction;
    setConfirmAction(null);
    if (!action || selectedDraftIds.length === 0) return;
    const ids = [...selectedDraftIds];
    const mut = action === "markPaid" ? bulkMarkPaid : bulkCancel;
    setBulkBusy(true);
    const results = await Promise.allSettled(ids.map((id) => mut.mutateAsync(id)));
    setBulkBusy(false);
    setSelectedIds(new Set());
    const failed = results.filter((r) => r.status === "rejected").length;
    if (failed > 0)
      toast({
        variant: "destructive",
        title: `${failed} of ${ids.length} ${action === "cancel" ? "cancellations" : "payments"} failed`,
      });
  };

  const paypalReady = isPaypalEnabled();
  const payeeById = new Map(payees.map((p) => [p.id, p]));

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
      {/* Segmented toggle: to-action drafts vs. most-recent completed payouts */}
      <div className="inline-flex items-center gap-1 self-start rounded-lg bg-secondary p-1">
        {(
          [
            ["drafts", "Drafts", drafts.length],
            ["completed", "Completed", paid.length],
          ] as const
        ).map(([value, label, count]) => (
          <button
            key={value}
            type="button"
            onClick={() => switchView(value)}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-semibold transition-colors",
              subView === value
                ? "bg-card text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {label}
            {count > 0 && <span className="ml-1.5 tabular-nums opacity-70">{count}</span>}
          </button>
        ))}
      </div>

      {/* Select-all + bulk actions (drafts only) */}
      {subView === "drafts" && drafts.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-muted/30 px-3 py-2">
          <button
            type="button"
            onClick={toggleSelectAll}
            className="flex items-center gap-2 text-xs font-semibold text-muted-foreground hover:text-foreground"
          >
            <SelectBox on={allSelected} onClick={toggleSelectAll} />
            {selectedCount > 0 ? `${selectedCount} selected` : "Select all"}
          </button>
          {selectedCount > 0 && (
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="ghost"
                className="text-muted-foreground"
                disabled={bulkBusy}
                onClick={() => setSelectedIds(new Set())}
              >
                Clear
              </Button>
              <Button size="sm" variant="outline" disabled={bulkBusy} onClick={() => setConfirmAction("markPaid")}>
                <CheckCheck className="mr-1.5 h-3.5 w-3.5" /> Mark as Paid
              </Button>
              <Button
                size="sm"
                variant="ghost"
                className="text-destructive hover:text-destructive"
                disabled={bulkBusy}
                onClick={() => setConfirmAction("cancel")}
              >
                <X className="mr-1 h-3.5 w-3.5" /> Cancel
              </Button>
            </div>
          )}
        </div>
      )}

      {activeList.length === 0 ? (
        <div className="rounded-xl border border-border bg-card px-4 py-8 text-center text-xs text-muted-foreground">
          {subView === "drafts" ? "No draft payouts to action." : "No completed payouts yet."}
        </div>
      ) : (
        activeList.map((r) => (
          <RunCard
            key={r.id}
            payout={r}
            onOpenDetail={onOpenDetail}
            onPayWithPaypal={subView === "drafts" && paypalReady ? setPaypalPayout : undefined}
            onReceipt={setReceiptPayout}
            selected={subView === "drafts" ? selectedIds.has(r.id) : undefined}
            onToggleSelect={subView === "drafts" ? () => toggleSelect(r.id) : undefined}
          />
        ))
      )}

      {paypalPayout && (
        <PayWithPayPalDialog
          payout={paypalPayout}
          payee={payeeById.get(paypalPayout.payee_id)}
          onClose={() => setPaypalPayout(null)}
        />
      )}

      {receiptPayout && (
        <ReceiptDialog payout={receiptPayout} onClose={() => setReceiptPayout(null)} />
      )}

      <AlertDialog open={confirmAction !== null} onOpenChange={(o) => { if (!o) setConfirmAction(null); }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {confirmAction === "cancel"
                ? `Cancel ${selectedCount} payout${selectedCount === 1 ? "" : "s"}?`
                : `Mark ${selectedCount} payout${selectedCount === 1 ? "" : "s"} as paid?`}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {confirmAction === "cancel"
                ? "This permanently removes the selected draft payouts. This can't be undone."
                : "This records the selected payouts as paid outside PayPal (e.g. bank transfer). Only do this if the money has actually been sent — it won't send anything."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep</AlertDialogCancel>
            <AlertDialogAction
              className={confirmAction === "cancel" ? "bg-destructive text-destructive-foreground hover:bg-destructive/90" : undefined}
              onClick={runBulk}
            >
              {confirmAction === "cancel" ? "Cancel payouts" : "Mark as paid"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
