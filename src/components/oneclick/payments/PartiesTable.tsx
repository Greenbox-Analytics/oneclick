// src/components/oneclick/payments/PartiesTable.tsx
import { AlertTriangle, ChevronRight, Check as CheckIcon, Send } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import type { PayeeSummary } from "@/hooks/useRoyalties";
import { PartyAvatar, StatusBadge, partyStatusKind, STATUS_ICON, fmtMoney } from "./shared";

interface PartiesTableProps {
  parties: PayeeSummary[];
  /** Reporting (base) currency — amounts in PayeeSummary.earned/paid/owed are already in this currency */
  base: string;
  selection: string[];
  toggleSel: (id: string) => void;
  onOpenParty: (id: string) => void;
  onPaySelected: () => void;
  onClear: () => void;
}

function SelectBox({ on, disabled, onClick }: { on: boolean; disabled?: boolean; onClick: () => void }) {
  return (
    <span
      role="checkbox"
      aria-checked={on}
      onClick={(e) => { e.stopPropagation(); if (!disabled) onClick(); }}
      className={cn(
        "inline-flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-[5px] border-[1.5px] transition-colors",
        on ? "border-primary bg-primary text-primary-foreground" : "border-border bg-background",
        disabled ? "cursor-not-allowed opacity-40" : "cursor-pointer",
      )}
    >
      {on && <CheckIcon className="h-3 w-3" />}
    </span>
  );
}

// 5 cols on mobile (earned/paid hidden via `hidden md:block`), 7 on md+.
const COLS =
  "grid grid-cols-[26px_minmax(140px,1.6fr)_1fr_110px_24px] md:grid-cols-[26px_minmax(180px,1.6fr)_1fr_1fr_1fr_132px_24px] items-center gap-3.5 px-4 md:px-[18px]";

export function PartiesTable({
  parties,
  base,
  selection,
  toggleSel,
  onOpenParty,
  onPaySelected,
  onClear,
}: PartiesTableProps) {
  // Sum owed (already in base) for selected payees
  const owedSum = parties
    .filter((p) => selection.includes(p.id))
    .reduce((s, p) => s + p.owed, 0);

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
      {/* Header row */}
      <div className={cn(COLS, "border-b border-border bg-muted/40 py-[11px] text-[10.5px] font-bold uppercase tracking-wider text-muted-foreground")}>
        <span />
        <span>Party</span>
        <span className="hidden text-right md:block">Earned</span>
        <span className="hidden text-right md:block">Paid</span>
        <span className="text-right">Outstanding</span>
        <span className="text-right">Status</span>
        <span />
      </div>

      {parties.map((payee) => {
        const canSelect = payee.owed > 0;
        const sel = selection.includes(payee.id);
        const kind = partyStatusKind(payee.status);
        const StatusIcon = STATUS_ICON[kind];
        const statusLabel = kind === "out" ? "Owed" : kind === "sched" ? "Scheduled" : "Settled";
        return (
          <button
            key={payee.id}
            type="button"
            onClick={() => onOpenParty(payee.id)}
            className={cn(COLS, "w-full border-b border-border py-3 text-left transition-colors last:border-b-0 hover:bg-muted/55")}
          >
            <span className="flex items-center justify-center">
              <SelectBox on={sel} disabled={!canSelect} onClick={() => canSelect && toggleSel(payee.id)} />
            </span>

            {/* Party name + payout currency chip */}
            <span className="flex min-w-0 items-center gap-2.5">
              <PartyAvatar id={payee.id} name={payee.display_name} />
              <span className="min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="truncate text-[13.5px] font-semibold">{payee.display_name}</span>
                  {payee.collision && (
                    <AlertTriangle
                      className="h-3.5 w-3.5 shrink-0 text-[hsl(var(--pay-out-fg))]"
                      title="Name collision — multiple payees share this name"
                    />
                  )}
                </div>
                <div className="mt-0.5 flex items-center gap-1.5 text-[11.5px] text-muted-foreground">
                  <span className="rounded-[5px] bg-muted px-1.5 py-px text-[10px] font-bold text-muted-foreground">
                    {payee.payout_currency}
                  </span>
                  {payee.project_count > 0 && (
                    <span className="text-[11px]">
                      {payee.project_count} project{payee.project_count !== 1 ? "s" : ""}
                    </span>
                  )}
                </div>
              </span>
            </span>

            {/* Earned (base) */}
            <span className="hidden text-right md:block">
              <span className="font-mono text-[13.5px] tabular-nums text-muted-foreground">
                {fmtMoney(payee.earned, base, { dp: 0 })}
              </span>
            </span>

            {/* Paid (base) */}
            <span className="hidden text-right md:block">
              <span className="font-mono text-[13.5px] tabular-nums text-muted-foreground">
                {fmtMoney(payee.paid, base, { dp: 0 })}
              </span>
            </span>

            {/* Owed (base) — with optional native sub-line */}
            <span className="text-right">
              <span className="font-mono text-[14px] font-bold tabular-nums">
                {fmtMoney(payee.owed, base, { dp: 0 })}
              </span>
              {payee.payout_currency !== base && payee.owed_native > 0 && (
                <div className="mt-px font-mono text-[11px] text-muted-foreground">
                  {fmtMoney(payee.owed_native, payee.payout_currency, { dp: 0 })}
                </div>
              )}
            </span>

            {/* Status badge */}
            <span className="flex justify-end">
              <StatusBadge kind={kind}>
                <StatusIcon className="h-3 w-3" /> {statusLabel}
              </StatusBadge>
            </span>

            <span className="flex justify-end text-muted-foreground">
              <ChevronRight className="h-[15px] w-[15px]" />
            </span>
          </button>
        );
      })}

      {/* Footer: selection summary */}
      {selection.length > 0 && (
        <div className="flex items-center gap-3.5 border-t border-border bg-muted/50 px-[18px] py-[13px]">
          <span className="text-[13px] font-semibold">{selection.length} selected</span>
          <span className="text-[13px] text-muted-foreground">·</span>
          <span className="font-mono text-[13px] font-bold tabular-nums">
            {fmtMoney(owedSum, base, { dp: 0 })} owed
          </span>
          <span className="flex-1" />
          <Button variant="outline" size="sm" onClick={onClear}>Clear</Button>
          <Button size="sm" onClick={onPaySelected}>
            <Send className="mr-1.5 h-4 w-4" /> Pay selected
          </Button>
        </div>
      )}
    </div>
  );
}
