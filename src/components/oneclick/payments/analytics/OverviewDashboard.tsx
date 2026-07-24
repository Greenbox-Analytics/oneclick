// src/components/oneclick/payments/analytics/OverviewDashboard.tsx
//
// Portfolio Overview dashboard — the default sub-view in the Royalty Tracking tab.
// Shows KPI cards, a paid-over-time area chart, and a top-outstanding bar chart.

import { TrendingUp, BarChart2, AlertCircle } from "lucide-react";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import { useRoyaltyOverview, type TopOwed } from "@/hooks/useRoyaltyAnalytics";
import { fmtMoney, idToColor } from "../shared";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface OverviewDashboardProps {
  base: string;
}

// ---------------------------------------------------------------------------
// Chart configs
// ---------------------------------------------------------------------------

const areaChartConfig: ChartConfig = {
  amount: {
    label: "Paid",
    color: "hsl(var(--primary))",
  },
};

const barChartConfig: ChartConfig = {
  owed: {
    label: "Outstanding",
    color: "hsl(var(--pay-out-bg))",
  },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format a YYYY-MM string to a short month label, e.g. "Jan '25" */
function fmtMonth(ym: string): string {
  const [y, m] = ym.split("-");
  const date = new Date(Number(y), Number(m) - 1, 1);
  return date.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}

/** True when every amount in the series is 0 or negative */
function allZero(values: number[]): boolean {
  return values.every((v) => !v || v <= 0);
}

// ---------------------------------------------------------------------------
// Skeleton block
// ---------------------------------------------------------------------------

function ChartSkeleton({ label }: { label: string }) {
  return (
    <div className="flex h-[200px] flex-col items-center justify-center gap-2 rounded-xl border border-border bg-card p-4 shadow-sm">
      <div className="h-[14px] w-32 animate-pulse rounded bg-muted" />
      <p className="text-xs text-muted-foreground">{label}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty chart state
// ---------------------------------------------------------------------------

function EmptyChart({ icon: Icon, message }: { icon: React.ElementType; message: string }) {
  return (
    <div className="flex h-[200px] flex-col items-center justify-center gap-2 text-muted-foreground">
      <Icon className="h-8 w-8 opacity-25" />
      <p className="text-sm">{message}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function OverviewDashboard({ base }: OverviewDashboardProps) {
  const { data, isLoading, isError } = useRoyaltyOverview(base);

  // ── Paid-by-month chart data ─────────────────────────────────────────────
  const monthPoints = data?.paid_by_month ?? [];
  const areaData = monthPoints.map((p) => ({
    month: fmtMonth(p.month),
    amount: p.amount,
  }));
  const showAreaChart = areaData.length > 0 && !allZero(areaData.map((d) => d.amount));

  // ── Top owed bar chart data ───────────────────────────────────────────────
  const topOwed: TopOwed[] = data?.top_owed ?? [];
  const showBarChart = topOwed.length > 0 && !allZero(topOwed.map((t) => t.owed));

  return (
    <div className="flex flex-col gap-5">
      {/* ── Unconvertible note ──────────────────────────────────────────────── */}
      {data && data.unconvertible_count > 0 && (
        <p className="text-xs text-muted-foreground">
          {data.unconvertible_count} amount{data.unconvertible_count !== 1 ? "s" : ""} can&apos;t
          be shown in {base} due to missing exchange rates.
        </p>
      )}

      {/* ── Charts row ─────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Paid over time — area chart */}
        <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
          <div className="mb-3 flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-semibold">Paid over time</span>
          </div>
          {isLoading ? (
            <ChartSkeleton label="Loading…" />
          ) : isError ? (
            <EmptyChart icon={AlertCircle} message="Couldn't load chart" />
          ) : !showAreaChart ? (
            <EmptyChart icon={TrendingUp} message="Not enough royalty data yet" />
          ) : (
            <ChartContainer config={areaChartConfig} className="h-[200px] w-full">
              <AreaChart
                data={areaData}
                margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="hsl(var(--border))" strokeDasharray="3 3" />
                <XAxis
                  dataKey="month"
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  tickFormatter={(v: number) => fmtMoney(v, base)}
                  axisLine={false}
                  tickLine={false}
                  width={72}
                />
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      formatter={(value) => [fmtMoney(Number(value), base), "Paid"]}
                    />
                  }
                />
                <Area
                  type="monotone"
                  dataKey="amount"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  fill="url(#areaGradient)"
                  dot={false}
                />
              </AreaChart>
            </ChartContainer>
          )}
        </div>

        {/* Top outstanding — horizontal bar chart */}
        <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
          <div className="mb-3 flex items-center gap-2">
            <BarChart2 className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-semibold">Top outstanding</span>
          </div>
          {isLoading ? (
            <ChartSkeleton label="Loading…" />
          ) : isError ? (
            <EmptyChart icon={AlertCircle} message="Couldn't load chart" />
          ) : !showBarChart ? (
            <EmptyChart icon={BarChart2} message="Not enough royalty data yet" />
          ) : (
            <ChartContainer config={barChartConfig} className="h-[200px] w-full">
              <BarChart
                layout="vertical"
                data={topOwed}
                margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
              >
                <CartesianGrid horizontal={false} stroke="hsl(var(--border))" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  tickFormatter={(v: number) => fmtMoney(v, base)}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="display_name"
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={false}
                  tickLine={false}
                  width={100}
                />
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      formatter={(value, _name, item) => [
                        fmtMoney(Number(value), base),
                        item.payload?.display_name ?? "Outstanding",
                      ]}
                    />
                  }
                />
                <Bar dataKey="owed" radius={[0, 4, 4, 0]}>
                  {topOwed.map((entry) => (
                    <Cell key={entry.payee_id} fill={idToColor(entry.payee_id)} />
                  ))}
                </Bar>
              </BarChart>
            </ChartContainer>
          )}
        </div>
      </div>
    </div>
  );
}
