import { useMemo, useState } from "react";
import { Send, Coins, CheckCheck, Hourglass, Users, Calendar, Search, Loader2, LayoutDashboard } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useRoyaltyPayees, useRoyaltyPeriods, useRoyaltyPayouts, useMarkPayoutPaid } from "@/hooks/useRoyalties";
import type { PeriodLedger } from "@/hooks/useRoyalties";
import { useReportingCurrency } from "@/hooks/useReportingCurrency";
import { CurrencySelect, money } from "./shared";
import { PartiesTable } from "./PartiesTable";
import { PeriodsLedger } from "./PeriodsLedger";
import { NewPayoutModal } from "./NewPayoutModal";
import { PayoutRuns } from "./PayoutRuns";
import { PayoutDetailModal } from "./PayoutDetailModal";
import { PartyDrawer } from "./PartyDrawer";
import { OverviewDashboard } from "./analytics/OverviewDashboard";

// ---------------------------------------------------------------------------
// PaymentTracking — orchestrator (Task 14 read-path rewrite)
//
// Tasks 15/16 will wire up: PartyDrawer, NewPayoutModal, PayoutDetailModal,
// PayoutRuns. Those modals are stubbed/gated here so this file compiles cleanly
// while consuming real hook data.
// ---------------------------------------------------------------------------

export function PaymentTracking() {
  const [view, setView] = useState("overview");
  const [baseCur, setBaseCur] = useReportingCurrency();
  const [selection, setSelection] = useState<string[]>([]);
  const [query, setQuery] = useState("");

  const [drawerId, setDrawerId] = useState<string | null>(null);
  const [payoutOpen, setPayoutOpen] = useState<{ initialIds: string[] } | null>(null);
  const [detailRunId, setDetailRunId] = useState<string | null>(null);
  const markPaid = useMarkPayoutPaid();

  // ── Server data ──────────────────────────────────────────────────────────
  const { data: payees = [], isLoading: payeesLoading, isFetching: payeesFetching } = useRoyaltyPayees(baseCur);
  const { data: periods, isLoading: periodsLoading } = useRoyaltyPeriods(baseCur);
  const { data: payouts = [] } = useRoyaltyPayouts();

  // ── Derived totals — amounts already in base; no client FX ───────────────
  const totals = useMemo(() => {
    let earned = 0, paid = 0, owed = 0, awaitingCount = 0;
    payees.forEach((p) => {
      earned += p.earned;
      paid   += p.paid;
      owed   += p.owed;
      if (p.owed > 0) awaitingCount++;
    });
    const runsPaid = payouts.filter((r) => r.status === "paid").length;
    return { earned, paid, owed, awaitingCount, runsPaid };
  }, [payees, payouts]);

  // ── Filtered + sorted parties list ───────────────────────────────────────
  const filteredParties = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = needle
      ? payees.filter((p) => p.display_name.toLowerCase().includes(needle))
      : payees;
    return [...filtered].sort((a, b) => b.owed - a.owed);
  }, [payees, query]);

  const toggleSel = (id: string) =>
    setSelection((s) => s.includes(id) ? s.filter((x) => x !== id) : [...s, id]);

  // Only treat as "first load" skeleton when there's truly no prior data.
  // With keepPreviousData, payeesLoading stays false during a currency switch
  // (the stale data is still present), so the table never flashes to empty.
  const isLoading = payeesLoading && payees.length === 0;
  const isEmpty = !payeesLoading && !isLoading && payees.length === 0;

  // Safe ledger fallback while loading
  const safeLedger: PeriodLedger = periods ?? { base: baseCur, rows: [] };

  return (
    <div className="flex flex-col gap-5">
      {/* Head */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-primary">OneClick · Royalties</div>
          <h1 className="mt-1.5 text-2xl font-bold tracking-tight">Royalty Tracking</h1>
          <p className="text-sm text-muted-foreground">
            Track royalties across quarters and pay every party in bulk, in any currency.
          </p>
        </div>
        {/* TODO Task 15: open NewPayoutModal */}
        <Button onClick={() => setPayoutOpen({ initialIds: selection })}>
          <Send className="w-4 h-4 mr-1.5" /> New payout
        </Button>
      </div>

      <StatBand totals={totals} base={baseCur} isLoading={isLoading} />

      <Tabs value={view} onValueChange={setView}>
        <div className="flex flex-wrap items-center gap-3">
          <TabsList>
            <TabsTrigger value="overview" className="gap-1.5"><LayoutDashboard className="w-4 h-4" /> Overview</TabsTrigger>
            <TabsTrigger value="parties" className="gap-1.5"><Users className="w-4 h-4" /> Parties</TabsTrigger>
            <TabsTrigger value="runs" className="gap-1.5"><Send className="w-4 h-4" /> Payouts</TabsTrigger>
            <TabsTrigger value="periods" className="gap-1.5"><Calendar className="w-4 h-4" /> Periods</TabsTrigger>
          </TabsList>
          <span className="flex-1" />
          <div className="flex items-center gap-1.5">
            <CurrencySelect value={baseCur} onChange={setBaseCur} />
            {payeesFetching && !isLoading && (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
            )}
          </div>
          {view === "parties" && (
            <div className="relative max-w-[260px]">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                className="pl-9"
                placeholder="Search parties…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
          )}
        </div>

        {/* ── Overview tab ── */}
        <TabsContent value="overview" className="mt-4">
          <OverviewDashboard base={baseCur} />
        </TabsContent>

        {/* ── Parties tab ── */}
        <TabsContent value="parties" className="mt-4">
          {isLoading ? (
            <PartiesTableSkeleton />
          ) : isEmpty ? (
            <EmptyState />
          ) : (
            <PartiesTable
              parties={filteredParties}
              base={baseCur}
              selection={selection}
              toggleSel={toggleSel}
              onOpenParty={(id) => setDrawerId(id)}
              onPaySelected={() => setPayoutOpen({ initialIds: selection })}
              onClear={() => setSelection([])}
            />
          )}
        </TabsContent>

        {/* ── Payouts tab ── */}
        <TabsContent value="runs" className="mt-4">
          <PayoutRuns payouts={payouts} onOpenDetail={setDetailRunId} />
        </TabsContent>

        {/* ── Periods tab ── */}
        <TabsContent value="periods" className="mt-4">
          {periodsLoading ? (
            <div className="flex items-center justify-center rounded-2xl border border-border bg-card p-12 shadow-sm">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <PeriodsLedger ledger={safeLedger} onOpenParty={setDrawerId} />
          )}
        </TabsContent>
      </Tabs>

      {/* NewPayoutModal */}
      {payoutOpen && (
        <NewPayoutModal
          payees={payees}
          base={baseCur}
          initialIds={payoutOpen.initialIds}
          onClose={() => setPayoutOpen(null)}
        />
      )}

      {/* PayoutDetailModal */}
      {detailRunId && (() => {
        const p = payouts.find((x) => x.id === detailRunId);
        return p ? (
          <PayoutDetailModal
            payout={p}
            onClose={() => setDetailRunId(null)}
            onMarkPaid={(id) => { markPaid.mutate(id); setDetailRunId(null); }}
          />
        ) : null;
      })()}

      {/* PartyDrawer */}
      {drawerId && (
        <PartyDrawer
          payeeId={drawerId}
          base={baseCur}
          onClose={() => setDrawerId(null)}
          onPayout={(ids) => {
            setDrawerId(null);
            setPayoutOpen({ initialIds: ids });
          }}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// StatBand
// ---------------------------------------------------------------------------

type StatTotals = { earned: number; paid: number; owed: number; awaitingCount: number; runsPaid: number };

function StatBand({ totals, base, isLoading }: { totals: StatTotals; base: string; isLoading: boolean }) {
  const f = (n: number) => money(n, base, { dp: 0 });
  const cards = [
    {
      ic: Coins,
      tone: "bg-secondary text-primary",
      label: "Total earned to date",
      num: f(totals.earned),
      sub: "Across all parties & periods",
    },
    {
      ic: CheckCheck,
      tone: "bg-[hsl(var(--pay-paid-bg))] text-[hsl(var(--pay-paid-fg))]",
      label: "Paid out so far",
      num: f(totals.paid),
      sub: `${totals.runsPaid} payout${totals.runsPaid !== 1 ? "s" : ""} completed`,
    },
    {
      ic: Hourglass,
      tone: "bg-[hsl(var(--pay-out-bg))] text-[hsl(var(--pay-out-fg))]",
      label: "Outstanding",
      num: f(totals.owed),
      sub: `${totals.awaitingCount} part${totals.awaitingCount !== 1 ? "ies" : "y"} awaiting payout`,
    },
  ];
  return (
    <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-3">
      {cards.map((c) => {
        const Ic = c.ic;
        return (
          <div key={c.label} className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <div className="flex items-center gap-2">
              <span className={cn("flex h-[30px] w-[30px] items-center justify-center rounded-lg", c.tone)}>
                <Ic className="h-4 w-4" />
              </span>
              <span className="text-[12.5px] font-medium text-muted-foreground">{c.label}</span>
            </div>
            {isLoading ? (
              <div className="mt-2 h-[27px] w-24 animate-pulse rounded bg-muted" />
            ) : (
              <div className="mt-2 font-mono text-[27px] font-bold tabular-nums tracking-tight">{c.num}</div>
            )}
            <div className="mt-0.5 text-xs text-muted-foreground">{c.sub}</div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Loading skeleton for the parties table
// ---------------------------------------------------------------------------

function PartiesTableSkeleton() {
  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="flex items-center gap-4 border-b border-border px-[18px] py-4 last:border-b-0">
          <div className="h-[18px] w-[18px] animate-pulse rounded-[5px] bg-muted" />
          <div className="h-[30px] w-[30px] animate-pulse rounded-full bg-muted" />
          <div className="flex flex-1 flex-col gap-1.5">
            <div className="h-[14px] w-32 animate-pulse rounded bg-muted" />
            <div className="h-[11px] w-20 animate-pulse rounded bg-muted" />
          </div>
          <div className="hidden h-[13px] w-20 animate-pulse rounded bg-muted md:block" />
          <div className="hidden h-[13px] w-20 animate-pulse rounded bg-muted md:block" />
          <div className="h-[14px] w-20 animate-pulse rounded bg-muted" />
          <div className="h-[22px] w-20 animate-pulse rounded-full bg-muted" />
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-border bg-card py-16 text-muted-foreground shadow-sm">
      <Coins className="h-10 w-10 opacity-25" />
      <p className="text-sm font-medium text-foreground">No royalty data yet</p>
      <p className="max-w-[32ch] text-center text-xs">
        Run OneClick on a royalty statement to populate royalty tracking.
      </p>
    </div>
  );
}
