// src/components/oneclick/payments/PayoutDetailModal.tsx
import { useState } from "react";
import { CheckCheck, Clock, PieChart, Receipt, Download, Music, Users, Banknote, AlertTriangle, Loader2 } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { API_URL } from "@/lib/apiFetch";
import type { PayoutOut } from "@/hooks/useRoyalties";
import { StatusBadge, fmtMoney, fmtDate, idToColor, downloadPdf } from "./shared";
import { useToast } from "@/hooks/use-toast";

interface PayoutDetailModalProps {
  payout: PayoutOut;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Typed snapshot shapes
// ---------------------------------------------------------------------------

interface SnapshotLine {
  song: string;
  role?: string;
  royalty_type?: string;
  percentage?: number;
  amount_owed: number;       // statement currency
  amount_pay_ccy: number;    // pay currency
}

interface SnapshotStatement {
  royalty_statement_id: string;
  period_start?: string;
  period_end?: string;
  statement_currency: string;
  statement_total?: number;    // total of the full statement (≠ payee slice)
  payee_subtotal_owed: number; // payee's slice in statement currency
  payee_subtotal_pay_ccy: number;
  lines: SnapshotLine[];
}

interface SnapshotProject {
  project_id: string;
  name: string;
  statements: SnapshotStatement[];
}

interface SnapshotPayee {
  id: string;
  display_name: string;
  payout_currency: string;
}

interface BreakdownSnapshot {
  payee: SnapshotPayee;
  fx: { rate_date: string; rates_used: Record<string, number> };
  projects: SnapshotProject[];
  total_pay_ccy: number;
}

function castSnapshot(raw: Record<string, unknown>): BreakdownSnapshot | null {
  if (!raw?.payee || !raw?.projects) return null;
  return raw as unknown as BreakdownSnapshot;
}

// ---------------------------------------------------------------------------
// Donut chart (CSS conic-gradient — same visual as old seed modal)
// ---------------------------------------------------------------------------

interface DonutSlice {
  label: string;
  amount: number;
  color: string;
}

function Donut({ slices, total, payCur }: { slices: DonutSlice[]; total: number; payCur: string }) {
  let acc = 0;
  const stops = slices
    .map((s) => {
      const start = total > 0 ? (acc / total) * 100 : 0;
      acc += s.amount;
      const end = total > 0 ? (acc / total) * 100 : 100;
      return `${s.color} ${start}% ${end}%`;
    })
    .join(", ");

  return (
    <div
      className="relative mx-auto h-[168px] w-[168px] rounded-full shadow-sm"
      style={{ background: `conic-gradient(${stops || "#e5e7eb 0% 100%"})` }}
    >
      <div className="absolute inset-[30px] flex flex-col items-center justify-center rounded-full bg-card">
        <span className="font-mono text-[19px] font-bold tracking-tight">
          {fmtMoney(total, payCur, { dp: 0 })}
        </span>
        <span className="text-[11px] font-semibold tracking-wide text-muted-foreground">
          {payCur}
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// PayoutDetailModal
// ---------------------------------------------------------------------------

export function PayoutDetailModal({ payout, onClose }: PayoutDetailModalProps) {
  const { toast } = useToast();
  const [exporting, setExporting] = useState(false);

  const handleExport = async () => {
    setExporting(true);
    try {
      await downloadPdf(
        `${API_URL}/oneclick/royalties/payouts/${payout.id}/breakdown.pdf`,
        `Payout_Breakdown_${payout.id}.pdf`,
      );
    } catch {
      toast({ variant: "destructive", title: "Couldn't export the breakdown. Please try again." });
    } finally {
      setExporting(false);
    }
  };
  const snap = castSnapshot(payout.breakdown_snapshot);
  const isDraft = payout.status === "draft";
  const payCur = payout.pay_currency;

  // Badge
  const badge = isDraft ? (
    <StatusBadge kind="sched">
      <Clock className="h-3 w-3" /> Draft
    </StatusBadge>
  ) : (
    <StatusBadge kind="paid">
      <CheckCheck className="h-3 w-3" /> Paid
    </StatusBadge>
  );

  const iconTone = isDraft
    ? "bg-[hsl(var(--pay-sched-bg))] text-[hsl(var(--pay-sched-fg))]"
    : "bg-[hsl(var(--pay-paid-bg))] text-[hsl(var(--pay-paid-fg))]";

  const Ic = isDraft ? Clock : CheckCheck;

  // Derived counts from snapshot (fallback to payout top-level if snapshot missing)
  const projects = snap?.projects ?? [];
  const allStatements = projects.flatMap((p) => p.statements);
  const totalPayCcy = snap?.total_pay_ccy ?? payout.total_amount;
  const payeeName = snap?.payee.display_name ?? payout.payee_id;

  // Donut slices — one per project
  const projectSlices: DonutSlice[] = projects.map((proj) => {
    const projTotal = proj.statements.reduce(
      (s, st) => s + st.payee_subtotal_pay_ccy,
      0,
    );
    return { label: proj.name, amount: projTotal, color: idToColor(proj.project_id) };
  });

  // Flat table rows
  interface TableRow {
    projectName: string;
    statementId: string;
    period_start?: string;
    period_end?: string;
    stmtCur: string;
    statementTotal?: number;
    song: string;
    role?: string;
    royaltyType?: string;
    percentage?: number;
    amount_owed: number;
    amount_pay_ccy: number;
  }

  const tableRows: TableRow[] = [];
  projects.forEach((proj) => {
    proj.statements.forEach((stmt) => {
      stmt.lines.forEach((line) => {
        tableRows.push({
          projectName: proj.name,
          statementId: stmt.royalty_statement_id,
          period_start: stmt.period_start,
          period_end: stmt.period_end,
          stmtCur: stmt.statement_currency,
          statementTotal: stmt.statement_total,
          song: line.song,
          role: line.role,
          royaltyType: line.royalty_type,
          percentage: line.percentage,
          amount_owed: line.amount_owed,
          amount_pay_ccy: line.amount_pay_ccy,
        });
      });
    });
  });

  const stats = [
    {
      label: "Projects",
      value: String(projects.length),
      ic: PieChart,
      big: false,
    },
    {
      label: "Statements",
      value: String(allStatements.length),
      ic: Music,
      big: false,
    },
    {
      label: "Payees",
      value: "1",
      ic: Users,
      big: false,
    },
    {
      label: "Total payout",
      value: fmtMoney(totalPayCcy, payCur),
      ic: Banknote,
      big: true,
    },
  ];

  return (
    <Dialog open onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="max-w-4xl gap-0 overflow-y-auto p-0 max-h-[92vh]">
        <DialogHeader className="border-b border-border px-5 py-4">
          <DialogTitle>Payout analysis</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-[18px] p-5">
          {/* Header */}
          <div className="flex items-center gap-3.5">
            <span
              className={cn(
                "flex h-[42px] w-[42px] shrink-0 items-center justify-center rounded-[10px]",
                iconTone,
              )}
            >
              <Ic className="h-[18px] w-[18px]" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="text-[19px] font-bold tracking-tight">{payeeName}</div>
              <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[12.5px] text-muted-foreground">
                {badge}
                <span className="h-[3px] w-[3px] rounded-full bg-border" />
                <span>Paid in {payCur}</span>
                <span className="h-[3px] w-[3px] rounded-full bg-border" />
                <span>Created {fmtDate(payout.created_at)}</span>
                {payout.paid_at && (
                  <>
                    <span className="h-[3px] w-[3px] rounded-full bg-border" />
                    <span>Paid {fmtDate(payout.paid_at)}</span>
                  </>
                )}
                {payout.orphan_state && payout.orphan_state !== "none" && (
                  <>
                    <span className="h-[3px] w-[3px] rounded-full bg-border" />
                    <StatusBadge kind={payout.orphan_state === "orphaned" ? "out" : "partial"}>
                      {payout.orphan_state}
                    </StatusBadge>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Stat cards */}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {stats.map((s) => {
              const Si = s.ic;
              return (
                <div key={s.label} className="rounded-xl border border-border p-3.5">
                  <div className="flex items-center justify-between gap-2 text-muted-foreground">
                    <span className="text-xs font-medium">{s.label}</span>
                    <Si className="h-[15px] w-[15px]" />
                  </div>
                  <div
                    className={cn(
                      "mt-2 font-bold tabular-nums tracking-tight",
                      s.big ? "font-mono text-[23px] text-primary" : "text-[26px]",
                    )}
                  >
                    {s.value}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Distribution donut — only when we have snapshot data */}
          {snap && projects.length > 0 && (
            <div className="overflow-hidden rounded-xl border border-border">
              <div className="flex items-center gap-2 border-b border-border bg-muted/40 px-4 py-3 text-sm font-bold tracking-tight">
                <PieChart className="h-4 w-4 text-primary" /> Royalty distribution
              </div>
              <div className="grid items-center gap-6 p-5 md:grid-cols-[200px_1fr]">
                <Donut slices={projectSlices} total={totalPayCcy} payCur={payCur} />
                <div className="flex flex-col">
                  {projectSlices.map((slice) => {
                    const pct = totalPayCcy > 0 ? (slice.amount / totalPayCcy) * 100 : 0;
                    return (
                      <div
                        key={slice.label}
                        className="grid grid-cols-[10px_1fr_auto_auto] items-center gap-3 border-b border-dashed border-border py-2.5 text-[13px] last:border-b-0"
                      >
                        <span
                          className="h-2.5 w-2.5 rounded-[3px]"
                          style={{ background: slice.color }}
                        />
                        <span className="flex min-w-0 flex-col font-semibold">
                          {slice.label}
                        </span>
                        <span className="font-mono tabular-nums text-muted-foreground">
                          {pct.toFixed(1)}%
                        </span>
                        <span className="min-w-[88px] text-right font-mono font-bold tabular-nums">
                          {fmtMoney(slice.amount, payCur)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Breakdown table */}
          {snap && tableRows.length > 0 ? (
            <div className="overflow-hidden rounded-xl border border-border">
              <div className="flex items-center gap-2 border-b border-border bg-muted/40 px-4 py-3 text-sm font-bold tracking-tight">
                <Receipt className="h-4 w-4 text-primary" /> Royalty breakdown
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  {tableRows.length} lines
                </span>
              </div>
              <div className="max-h-[360px] overflow-auto">
                <table className="w-full min-w-[760px] border-collapse">
                  <thead>
                    <tr className="[&>th]:sticky [&>th]:top-0 [&>th]:z-[1] [&>th]:border-b [&>th]:border-border [&>th]:bg-card [&>th]:px-3.5 [&>th]:py-2.5 [&>th]:text-left [&>th]:text-[10.5px] [&>th]:font-bold [&>th]:uppercase [&>th]:tracking-wide [&>th]:text-muted-foreground">
                      <th>Project</th>
                      <th>Song</th>
                      <th>Role</th>
                      <th>Type</th>
                      <th className="!text-right">Stmt total</th>
                      <th className="!text-right">Share</th>
                      <th className="!text-right">Owed (stmt ccy)</th>
                      <th className="!text-right">Pay ({payCur})</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableRows.map((r, i) => (
                      <tr
                        key={i}
                        className="[&>td]:border-b [&>td]:border-border [&>td]:px-3.5 [&>td]:py-2.5 [&>td]:text-[12.5px] hover:bg-muted/45"
                      >
                        <td className="text-muted-foreground">{r.projectName}</td>
                        <td className="font-semibold">{r.song}</td>
                        <td className="text-muted-foreground">{r.role ?? "—"}</td>
                        <td>
                          {r.royaltyType && (
                            <span className="rounded-full bg-secondary px-2 py-0.5 text-[11px] font-semibold text-primary">
                              {r.royaltyType}
                            </span>
                          )}
                        </td>
                        {/* statement_total — the FULL statement total, distinct from payee slice */}
                        <td className="text-right font-mono tabular-nums">
                          {r.statementTotal != null
                            ? fmtMoney(r.statementTotal, r.stmtCur)
                            : "—"}
                        </td>
                        <td className="text-right font-mono tabular-nums">
                          {r.percentage != null ? `${r.percentage.toFixed(1)}%` : "—"}
                        </td>
                        {/* amount_owed is in statement currency */}
                        <td className="text-right font-mono tabular-nums">
                          {fmtMoney(r.amount_owed, r.stmtCur)}
                        </td>
                        <td className="text-right font-mono font-bold tabular-nums">
                          {fmtMoney(r.amount_pay_ccy, payCur)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr className="[&>td]:sticky [&>td]:bottom-0 [&>td]:border-t-2 [&>td]:border-border [&>td]:bg-muted/50 [&>td]:px-3.5 [&>td]:py-3 [&>td]:text-[13px] [&>td]:font-bold">
                      <td colSpan={7}>Total payout · {payeeName}</td>
                      <td className="text-right font-mono tabular-nums">
                        {fmtMoney(totalPayCcy, payCur)}
                      </td>
                    </tr>
                  </tfoot>
                </table>
              </div>
              <p className="flex items-center gap-1.5 border-t border-border px-4 py-2.5 text-[11.5px] text-muted-foreground">
                <AlertTriangle className="h-[13px] w-[13px] shrink-0" />
                Stmt total = the full statement revenue; Owed = payee's slice at their percentage.
                Amounts in the statement currency are converted to {payCur} at the FX rate captured
                for this payout (date: {snap?.fx?.rate_date ?? payout.fx_rate_date}).
              </p>
            </div>
          ) : (
            !snap && (
              <div className="rounded-xl border border-border bg-muted/30 p-5 text-center text-[13px] text-muted-foreground">
                Breakdown snapshot not available for this payout.
              </div>
            )
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-2 border-t border-border px-5 py-4">
          <Button variant="ghost" className="mr-auto" disabled={exporting} onClick={handleExport}>
            {exporting ? (
              <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
            ) : (
              <Download className="mr-1.5 h-4 w-4" />
            )}
            Export breakdown
          </Button>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
