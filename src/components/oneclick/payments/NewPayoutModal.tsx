// src/components/oneclick/payments/NewPayoutModal.tsx
import { useState } from "react";
import { Send, Check as CheckIcon, CheckCheck, Info } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { useCreatePayout } from "@/hooks/useRoyalties";
import type { PayeeSummary } from "@/hooks/useRoyalties";
import { PartyAvatar, fmtMoney } from "./shared";
import { useToast } from "@/hooks/use-toast";

interface NewPayoutModalProps {
  payees: PayeeSummary[];
  initialIds: string[];
  base: string;
  onClose: () => void;
}

function SelectBox({ on, partial, onClick }: { on: boolean; partial?: boolean; onClick: () => void }) {
  return (
    <span
      role="checkbox"
      aria-checked={on}
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      className={cn(
        "inline-flex h-[18px] w-[18px] shrink-0 cursor-pointer items-center justify-center rounded-[5px] border-[1.5px] transition-colors",
        on || partial
          ? "border-primary bg-primary text-primary-foreground"
          : "border-border bg-background",
      )}
    >
      {on && !partial && <CheckIcon className="h-3 w-3" />}
      {partial && <span className="h-0.5 w-2 rounded bg-primary-foreground" />}
    </span>
  );
}

export function NewPayoutModal({ payees, base, initialIds, onClose }: NewPayoutModalProps) {
  // One idempotency key per modal open — stable for the lifetime of this modal.
  const [idempotencyKey] = useState(() => crypto.randomUUID());
  const [note, setNote] = useState("");
  const [done, setDone] = useState(false);
  const [resultCount, setResultCount] = useState(0);

  const { toast } = useToast();
  const createPayout = useCreatePayout();

  // Eligible = payees with owed > 0
  const eligible = payees.filter((p) => p.owed > 0);

  const [sel, setSel] = useState<string[]>(() => {
    const eligibleIds = new Set(eligible.map((p) => p.id));
    return initialIds.filter((id) => eligibleIds.has(id));
  });

  const toggle = (id: string) =>
    setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  const allOn = sel.length === eligible.length && eligible.length > 0;
  const partialOn = sel.length > 0 && !allOn;

  // Sort: selected first, then by owed desc
  const sortedEligible = [...eligible].sort((a, b) => {
    const aOn = sel.includes(a.id) ? 0 : 1;
    const bOn = sel.includes(b.id) ? 0 : 1;
    if (aOn !== bOn) return aOn - bOn;
    return b.owed - a.owed;
  });

  const handleCreate = () => {
    createPayout.mutate(
      { payee_ids: sel, idempotency_key: idempotencyKey, note: note || undefined },
      {
        onSuccess: (created) => {
          const count = Array.isArray(created) ? created.length : sel.length;
          setResultCount(count);
          setDone(true);
        },
        onError: (err) => {
          toast({
            variant: "destructive",
            title: "Failed to create invoices",
            description: err instanceof Error ? err.message : "An error occurred.",
          });
        },
      },
    );
  };

  return (
    <Dialog open onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="max-w-lg gap-0 overflow-y-auto p-0 max-h-[90vh]">
        <DialogHeader className="border-b border-border px-5 py-4">
          <DialogTitle>{done ? "Invoices created" : "New payout"}</DialogTitle>
        </DialogHeader>

        {done ? (
          <div className="px-6 py-8 text-center">
            <div className="mx-auto mb-3.5 flex h-[60px] w-[60px] items-center justify-center rounded-full bg-[hsl(var(--pay-paid-bg))] text-[hsl(var(--pay-paid-fg))]">
              <CheckCheck className="h-7 w-7" />
            </div>
            <div className="text-base font-bold">
              {resultCount} draft invoice{resultCount !== 1 ? "s" : ""} created
            </div>
            <p className="mx-auto mt-2 max-w-[42ch] text-[13px] text-muted-foreground">
              Each invoice is in <strong>draft</strong> status. Mark them paid once you have
              confirmed the transfers in your payment processor.
            </p>
            <div className="mt-6 flex justify-end">
              <Button onClick={onClose}>Done</Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-4 px-5 py-4">
            {/* Info note */}
            <div className="flex items-start gap-2 rounded-[10px] border border-[hsl(217_70%_88%)] bg-[hsl(217_80%_95%)] px-3 py-2 text-[12px] leading-relaxed text-[hsl(217_50%_32%)] dark:border-[hsl(217_40%_28%)] dark:bg-[hsl(217_45%_18%)] dark:text-[hsl(217_80%_80%)]">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              Each payee is paid in their own payout currency. Conversion from the reporting
              currency happens at invoice creation time.
            </div>

            {/* Payee selector */}
            <div>
              <div className="mb-2 flex items-center justify-between">
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-lg px-1.5 py-1 text-[13px] font-semibold text-primary hover:bg-secondary"
                  onClick={() => setSel(allOn ? [] : eligible.map((p) => p.id))}
                >
                  <SelectBox
                    on={allOn}
                    partial={partialOn}
                    onClick={() => setSel(allOn ? [] : eligible.map((p) => p.id))}
                  />
                  {allOn ? "Clear all" : "Select all"}
                </button>
                <span className="text-[12.5px] text-muted-foreground">
                  {sel.length} of {eligible.length} selected
                </span>
              </div>

              {eligible.length === 0 ? (
                <div className="rounded-xl border border-border bg-card p-6 text-center text-[13px] text-muted-foreground">
                  No payees with outstanding balances.
                </div>
              ) : (
                <div className="flex max-h-[300px] flex-col gap-[7px] overflow-y-auto p-0.5">
                  {sortedEligible.map((payee) => {
                    const on = sel.includes(payee.id);
                    return (
                      <div
                        key={payee.id}
                        onClick={() => toggle(payee.id)}
                        className={cn(
                          "grid cursor-pointer grid-cols-[24px_1fr_auto] items-center gap-3 rounded-xl border p-3 transition-colors",
                          on
                            ? "border-primary bg-secondary/50"
                            : "border-border hover:border-primary/50",
                        )}
                      >
                        <SelectBox on={on} onClick={() => toggle(payee.id)} />
                        <span className="flex min-w-0 items-center gap-2.5">
                          <PartyAvatar id={payee.id} name={payee.display_name} size={30} />
                          <span className="min-w-0">
                            <div className="text-[13.5px] font-semibold">{payee.display_name}</div>
                            <div className="text-[11.5px] text-muted-foreground">
                              {payee.payout_currency}
                            </div>
                          </span>
                        </span>
                        <span className="text-right">
                          <div className="font-mono text-[13.5px] font-bold tabular-nums">
                            {fmtMoney(payee.owed, base)}
                          </div>
                          {payee.payout_currency !== base && (
                            <div className="text-[11px] font-medium text-muted-foreground font-mono">
                              {fmtMoney(payee.owed_native, payee.payout_currency)}
                            </div>
                          )}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Optional note */}
            <div>
              <label className="mb-1.5 block text-[12.5px] font-medium text-muted-foreground">
                Note (optional)
              </label>
              <Textarea
                placeholder="e.g. Q2 2026 quarterly payout"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                rows={2}
                className="resize-none text-[13px]"
              />
            </div>

            {/* Footer */}
            <div className="flex items-center gap-2">
              <Button variant="outline" className="mr-auto" onClick={onClose}>
                Cancel
              </Button>
              <Button
                disabled={sel.length === 0 || createPayout.isPending}
                onClick={handleCreate}
              >
                {createPayout.isPending ? (
                  "Creating…"
                ) : (
                  <>
                    <Send className="mr-1.5 h-4 w-4" />
                    Create invoice{sel.length !== 1 ? "s" : ""} ({sel.length})
                  </>
                )}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
