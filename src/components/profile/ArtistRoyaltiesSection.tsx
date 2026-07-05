// src/components/profile/ArtistRoyaltiesSection.tsx
//
// "Royalties" section card on the Artist Profile page.
// Shows an owed-now KPI, earned/paid totals, and an earned-vs-paid
// by-month area chart — mirroring the PayeeTrendChart pattern.
//
// Hidden entirely when there is no royalty activity yet (all-zero summary
// and empty by_month). isError → muted notice, never a misleading $0.

import { TrendingUp, Hourglass, AlertCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Legend } from "recharts";
import { useArtistRoyaltyAnalytics } from "@/hooks/useRoyaltyAnalytics";
import { useReportingCurrency } from "@/hooks/useReportingCurrency";
import { fmtMoney } from "@/components/oneclick/payments/shared";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ArtistRoyaltiesSectionProps {
  artistId: string;
}

// ---------------------------------------------------------------------------
// Chart config
// ---------------------------------------------------------------------------

const chartConfig: ChartConfig = {
  earned: {
    label: "Earned (by statement period)",
    color: "hsl(var(--primary))",
  },
  paid: {
    label: "Paid (by payment date)",
    color: "hsl(150 55% 45%)",
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

/** True when every earned+paid value in the series is 0 or negative */
function allZero(points: { earned: number; paid: number }[]): boolean {
  return points.every((p) => (!p.earned || p.earned <= 0) && (!p.paid || p.paid <= 0));
}

/** True when the summary is all-zero */
function summaryIsEmpty(summary: { earned_total: number; owed_now: number; paid_total: number }): boolean {
  return !summary.earned_total && !summary.owed_now && !summary.paid_total;
}

// ---------------------------------------------------------------------------
// Skeleton
// ---------------------------------------------------------------------------

function Skeleton() {
  return (
    <Card className="border border-border shadow-sm hover:shadow-md transition-shadow overflow-hidden">
      <div className="h-0.5 bg-emerald-500/40" />
      <CardHeader className="bg-gradient-to-r from-emerald-500/5 to-transparent">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-emerald-500" />
          <CardTitle>Royalties</CardTitle>
        </div>
        <CardDescription>Earned and paid royalties for this artist</CardDescription>
      </CardHeader>
      <CardContent className="pt-6 space-y-4">
        {/* KPI skeleton */}
        <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="rounded-xl border border-border bg-card p-4">
              <div className="h-[12px] w-28 animate-pulse rounded bg-muted mb-2" />
              <div className="h-[27px] w-24 animate-pulse rounded bg-muted" />
            </div>
          ))}
        </div>
        {/* Chart skeleton */}
        <div className="flex h-[200px] items-center justify-center rounded-xl border border-border bg-card/50 p-4">
          <div className="h-[120px] w-full animate-pulse rounded-lg bg-muted/60" />
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ArtistRoyaltiesSection({ artistId }: ArtistRoyaltiesSectionProps) {
  const [base] = useReportingCurrency();
  const { data, isLoading, isError } = useArtistRoyaltyAnalytics(artistId, base);

  // Loading state
  if (isLoading) {
    return <Skeleton />;
  }

  // Error state — muted notice, no $0 headline
  if (isError) {
    return (
      <Card className="border border-border shadow-sm overflow-hidden">
        <div className="h-0.5 bg-emerald-500/40" />
        <CardHeader className="bg-gradient-to-r from-emerald-500/5 to-transparent">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-emerald-500" />
            <CardTitle>Royalties</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="pt-4 pb-6">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <AlertCircle className="h-4 w-4 opacity-50" />
            <span>Couldn&apos;t load royalty data</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  // No data yet — render nothing (don't show an empty section)
  if (!data) return null;

  const { summary, by_month, unconvertible_count } = data;

  // Hidden when all activity is zero and no monthly data
  if (summaryIsEmpty(summary) && (by_month.length === 0 || allZero(by_month))) {
    return null;
  }

  const chartData = by_month.map((p) => ({
    month: fmtMonth(p.month),
    earned: p.earned,
    paid: p.paid,
  }));
  const showChart = by_month.length > 0 && !allZero(by_month);

  return (
    <Card className="border border-border shadow-sm hover:shadow-md transition-shadow overflow-hidden">
      <div className="h-0.5 bg-emerald-500/40" />
      <CardHeader className="bg-gradient-to-r from-emerald-500/5 to-transparent">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-emerald-500" />
          <CardTitle>Royalties</CardTitle>
        </div>
        <CardDescription>Earned and paid royalties for this artist ({base})</CardDescription>
      </CardHeader>
      <CardContent className="pt-6 space-y-4">
        {/* KPI cards */}
        <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-3">
          {/* Owed now */}
          <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <div className="flex items-center gap-2">
              <span className="flex h-[30px] w-[30px] items-center justify-center rounded-lg bg-[hsl(var(--pay-out-bg))] text-[hsl(var(--pay-out-fg))]">
                <Hourglass className="h-4 w-4" />
              </span>
              <span className="text-[12.5px] font-medium text-muted-foreground">Owed now</span>
            </div>
            <div className="mt-2 font-mono text-[27px] font-bold tabular-nums tracking-tight">
              {fmtMoney(summary.owed_now, base, { dp: 0 })}
            </div>
            <div className="mt-0.5 text-xs text-muted-foreground">Outstanding balance</div>
          </div>

          {/* Earned total */}
          <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <div className="flex items-center gap-2">
              <span className="flex h-[30px] w-[30px] items-center justify-center rounded-lg bg-primary/10 text-primary">
                <TrendingUp className="h-4 w-4" />
              </span>
              <span className="text-[12.5px] font-medium text-muted-foreground">Earned total</span>
            </div>
            <div className="mt-2 font-mono text-[27px] font-bold tabular-nums tracking-tight">
              {fmtMoney(summary.earned_total, base, { dp: 0 })}
            </div>
            <div className="mt-0.5 text-xs text-muted-foreground">By statement period</div>
          </div>

          {/* Paid total */}
          <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <div className="flex items-center gap-2">
              <span className="flex h-[30px] w-[30px] items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-600">
                <TrendingUp className="h-4 w-4" />
              </span>
              <span className="text-[12.5px] font-medium text-muted-foreground">Paid total</span>
            </div>
            <div className="mt-2 font-mono text-[27px] font-bold tabular-nums tracking-tight">
              {fmtMoney(summary.paid_total, base, { dp: 0 })}
            </div>
            <div className="mt-0.5 text-xs text-muted-foreground">By payment date</div>
          </div>
        </div>

        {/* Unconvertible note */}
        {unconvertible_count > 0 && (
          <p className="text-xs text-muted-foreground">
            {unconvertible_count} amount{unconvertible_count !== 1 ? "s" : ""} can&apos;t be shown in {base} due to missing exchange rates.
          </p>
        )}

        {/* Earned vs. Paid chart */}
        {showChart ? (
          <div className="rounded-xl border border-border bg-card p-4">
            <div className="mb-1 flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
              <span className="text-[13px] font-semibold">Earned vs. Paid over time</span>
            </div>
            <p className="mb-3 text-[11px] text-muted-foreground">
              Earned is by statement period · Paid is by payment date
            </p>
            <ChartContainer config={chartConfig} className="h-[180px] w-full">
              <AreaChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="artistRoyEarnedGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0.01} />
                  </linearGradient>
                  <linearGradient id="artistRoyPaidGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(150 55% 45%)" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="hsl(150 55% 45%)" stopOpacity={0.01} />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  vertical={false}
                  stroke="hsl(var(--border))"
                  strokeDasharray="3 3"
                />
                <XAxis
                  dataKey="month"
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                  tickFormatter={(v: number) => fmtMoney(v, base, { dp: 0 })}
                  axisLine={false}
                  tickLine={false}
                  width={68}
                />
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      formatter={(value, name) => [
                        fmtMoney(Number(value), base),
                        name === "earned"
                          ? "Earned (statement period)"
                          : "Paid (payment date)",
                      ]}
                    />
                  }
                />
                <Legend
                  iconType="circle"
                  iconSize={8}
                  wrapperStyle={{ fontSize: 11 }}
                  formatter={(value) =>
                    value === "earned"
                      ? "Earned (by statement period)"
                      : "Paid (by payment date)"
                  }
                />
                <Area
                  type="monotone"
                  dataKey="earned"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  fill="url(#artistRoyEarnedGrad)"
                  dot={false}
                />
                <Area
                  type="monotone"
                  dataKey="paid"
                  stroke="hsl(150 55% 45%)"
                  strokeWidth={2}
                  fill="url(#artistRoyPaidGrad)"
                  dot={false}
                />
              </AreaChart>
            </ChartContainer>
          </div>
        ) : (
          <div className="flex h-[100px] flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-card/50 text-muted-foreground">
            <TrendingUp className="h-6 w-6 opacity-20" />
            <p className="text-[12.5px]">No monthly royalty data yet</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
