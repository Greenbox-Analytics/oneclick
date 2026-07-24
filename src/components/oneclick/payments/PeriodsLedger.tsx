// src/components/oneclick/payments/PeriodsLedger.tsx
import { useMemo } from "react";
import { Calendar } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PeriodLedger, PeriodLedgerRow, PeriodCell } from "@/hooks/useRoyalties";
import { PartyAvatar, fmtMoney } from "./shared";

interface PeriodsLedgerProps {
  ledger: PeriodLedger;
  onOpenParty: (id: string) => void;
}

const DOT: Record<string, string> = {
  paid:     "bg-[hsl(150_55%_45%)]",
  sched:    "bg-[hsl(217_70%_58%)]",
  owed:     "bg-[hsl(0_72%_55%)]",
  // aliases so server strings map directly
  settled:  "bg-[hsl(150_55%_45%)]",
  scheduled: "bg-[hsl(217_70%_58%)]",
};

function dotClass(state: string): string {
  return DOT[state] ?? DOT.owed;
}

/**
 * Build a stable ordered list of period columns from all cells across all rows.
 * Each entry: { key: royalty_statement_id, label: period_start–period_end or fallback }
 */
function buildColumns(rows: PeriodLedgerRow[]): { key: string; label: string }[] {
  const seen = new Map<string, string>();
  rows.forEach((row) => {
    row.cells.forEach((cell) => {
      if (!seen.has(cell.royalty_statement_id)) {
        const label = periodLabel(cell);
        seen.set(cell.royalty_statement_id, label);
      }
    });
  });
  // Sort by key (statement id) to keep columns in a stable chronological order
  return [...seen.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, label]) => ({ key, label }));
}

function periodLabel(cell: PeriodCell): string {
  if (cell.period_start && cell.period_end) {
    const start = new Date(cell.period_start + "T00:00:00");
    const end   = new Date(cell.period_end   + "T00:00:00");
    const mo = (d: Date) => d.toLocaleDateString("en-US", { month: "short" });
    if (start.getFullYear() === end.getFullYear()) {
      return `${mo(start)}–${mo(end)} ${end.getFullYear()}`;
    }
    return `${mo(start)} ${start.getFullYear()}–${mo(end)} ${end.getFullYear()}`;
  }
  // Fallback: just show the statement id shortened
  return cell.royalty_statement_id.slice(0, 8);
}

export function PeriodsLedger({ ledger, onOpenParty }: PeriodsLedgerProps) {
  const { base, rows } = ledger;

  const columns = useMemo(() => buildColumns(rows), [rows]);

  // Build a lookup: rowIndex → statementId → cell
  const cellMap = useMemo(() => {
    return rows.map((row) => {
      const m: Record<string, PeriodCell> = {};
      row.cells.forEach((cell) => { m[cell.royalty_statement_id] = cell; });
      return m;
    });
  }, [rows]);

  // Column totals (sum of cell.earned across all rows)
  const colTotals = useMemo(() => {
    const totals: Record<string, number> = {};
    columns.forEach((col) => { totals[col.key] = 0; });
    rows.forEach((row) => {
      row.cells.forEach((cell) => {
        totals[cell.royalty_statement_id] = (totals[cell.royalty_statement_id] ?? 0) + cell.earned;
      });
    });
    return totals;
  }, [columns, rows]);

  const grandTotal = useMemo(() => rows.reduce((s, r) => s + r.total, 0), [rows]);

  if (rows.length === 0) {
    return (
      <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
        <div className="flex items-center gap-3 border-b border-border px-[18px] py-[15px]">
          <Calendar className="h-4 w-4 text-primary" />
          <span className="text-[15px] font-bold tracking-tight">Quarterly royalty ledger</span>
        </div>
        <div className="flex flex-col items-center justify-center gap-2 py-14 text-muted-foreground">
          <Calendar className="h-8 w-8 opacity-30" />
          <p className="text-sm">No period data yet</p>
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
      <div className="flex items-center gap-3 border-b border-border px-[18px] py-[15px]">
        <Calendar className="h-4 w-4 text-primary" />
        <span className="text-[15px] font-bold tracking-tight">Quarterly royalty ledger</span>
        <span className="rounded-full bg-muted px-2.5 py-0.5 text-xs font-semibold text-muted-foreground">{columns.length} period{columns.length !== 1 ? "s" : ""}</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse">
          <thead>
            <tr>
              <th className="sticky left-0 z-[1] border-b border-border bg-muted/40 px-4 py-3 text-left text-[10.5px] font-bold uppercase tracking-wide text-muted-foreground">
                Party
              </th>
              {columns.map((col) => (
                <th
                  key={col.key}
                  className="border-b border-border bg-muted/40 px-4 py-3 text-right text-[10.5px] font-bold uppercase tracking-wide text-muted-foreground"
                >
                  {col.label}
                </th>
              ))}
              <th className="border-b border-border bg-muted/40 px-4 py-3 text-right text-[10.5px] font-bold uppercase tracking-wide text-muted-foreground">
                Total ({base})
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => (
              <tr
                key={row.payee_id}
                onClick={() => onOpenParty(row.payee_id)}
                className="group cursor-pointer hover:bg-muted/45"
              >
                <td className="sticky left-0 z-[1] border-b border-border bg-card px-4 py-3 group-hover:bg-muted/45">
                  <span className="flex items-center gap-2.5">
                    <PartyAvatar id={row.payee_id} name={row.display_name} size={24} />
                    <span>
                      <div className="text-[13px] font-semibold">{row.display_name}</div>
                    </span>
                  </span>
                </td>
                {columns.map((col) => {
                  const cell = cellMap[rowIdx][col.key];
                  return (
                    <td key={col.key} className="border-b border-border px-4 py-3 text-right font-mono text-[13px] tabular-nums">
                      {cell && cell.earned > 0 ? (
                        <span className="inline-flex flex-col items-end gap-0.5">
                          <span>{fmtMoney(cell.earned, base)}</span>
                          <span
                            className={cn("h-[7px] w-[7px] rounded-full", dotClass(cell.state))}
                            title={cell.state === "settled" ? "Paid" : cell.state === "scheduled" ? "Draft" : "Unpaid"}
                          />
                        </span>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                  );
                })}
                <td className="border-b border-border px-4 py-3 text-right font-mono text-[13px] font-bold tabular-nums">
                  {fmtMoney(row.total, base)}
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="[&>td]:border-t-2 [&>td]:border-border [&>td]:bg-muted/50 [&>td]:px-4 [&>td]:py-3 [&>td]:text-right [&>td]:font-mono [&>td]:text-[13px] [&>td]:font-bold [&>td]:tabular-nums">
              <td className="sticky left-0 z-[1] !text-left">Total earned</td>
              {columns.map((col) => (
                <td key={col.key}>{fmtMoney(colTotals[col.key] ?? 0, base)}</td>
              ))}
              <td>{fmtMoney(grandTotal, base)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
      <div className="flex flex-wrap items-center gap-4 border-t border-border px-4 py-3 text-[11.5px] text-muted-foreground">
        <span className="font-semibold">Settlement state:</span>
        <span className="flex items-center gap-1.5"><span className={cn("h-2 w-2 rounded-full", DOT.paid)} /> Paid</span>
        <span className="flex items-center gap-1.5"><span className={cn("h-2 w-2 rounded-full", DOT.sched)} /> Draft</span>
        <span className="flex items-center gap-1.5"><span className={cn("h-2 w-2 rounded-full", DOT.owed)} /> Unpaid</span>
        <span className="ml-auto">All values in {base}</span>
      </div>
    </div>
  );
}
