// src/components/oneclick/payments/analytics/PayeeTrendChart.tsx
//
// Earned-vs-paid trend chart for a single collaborator (payee).
// Rendered inside PartyDrawer below the balances/payment-history sections.

import { TrendingUp, AlertCircle } from "lucide-react";
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
  Legend,
} from "recharts";
import { usePayeeRoyaltyAnalytics } from "@/hooks/useRoyaltyAnalytics";
import { fmtMoney } from "../shared";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface PayeeTrendChartProps {
  payeeId: string;
  base: string;
}

// ---------------------------------------------------------------------------
// Chart config
// ---------------------------------------------------------------------------

const trendChartConfig: ChartConfig = {
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

/** True when every value across all series is 0 or negative */
function allZero(points: { earned: number; paid: number }[]): boolean {
  return points.every((p) => (!p.earned || p.earned <= 0) && (!p.paid || p.paid <= 0));
}

// ---------------------------------------------------------------------------
// Small skeleton
// ---------------------------------------------------------------------------

function TrendSkeleton() {
  return (
    <div className="flex h-[180px] flex-col items-center justify-center gap-2 rounded-xl border border-border bg-card/50 p-4">
      <div className="h-[12px] w-28 animate-pulse rounded bg-muted" />
      <div className="h-[120px] w-full animate-pulse rounded-lg bg-muted/60" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

function EmptyChart({ message }: { message: string }) {
  return (
    <div className="flex h-[160px] flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-card/50 text-muted-foreground">
      <TrendingUp className="h-7 w-7 opacity-20" />
      <p className="text-[12.5px]">{message}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function PayeeTrendChart({ payeeId, base }: PayeeTrendChartProps) {
  const { data, isLoading, isError } = usePayeeRoyaltyAnalytics(payeeId, base);

  if (isLoading) {
    return <TrendSkeleton />;
  }

  // Error: show a muted indicator — no misleading $0 line
  if (isError) {
    return (
      <div className="flex h-[120px] items-center justify-center gap-2 rounded-xl border border-dashed border-border bg-card/50 text-muted-foreground">
        <AlertCircle className="h-4 w-4 opacity-50" />
        <span className="text-[12px]">Couldn&apos;t load trend data</span>
      </div>
    );
  }

  const byMonth = data?.by_month ?? [];

  // Empty or all-zero: no empty axes
  if (byMonth.length === 0 || allZero(byMonth)) {
    return <EmptyChart message="Not enough royalty data yet" />;
  }

  const chartData = byMonth.map((p) => ({
    month: fmtMonth(p.month),
    earned: p.earned,
    paid: p.paid,
  }));

  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="mb-1 flex items-center gap-2">
        <TrendingUp className="h-4 w-4 text-muted-foreground" />
        <span className="text-[13px] font-semibold">Earned vs. Paid over time</span>
      </div>
      <p className="mb-3 text-[11px] text-muted-foreground">
        Earned is by statement period · Paid is by payment date
      </p>
      <ChartContainer config={trendChartConfig} className="h-[180px] w-full">
        <AreaChart
          data={chartData}
          margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="payeeTrendEarnedGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.2} />
              <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0.01} />
            </linearGradient>
            <linearGradient id="payeeTrendPaidGrad" x1="0" y1="0" x2="0" y2="1">
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
            tickFormatter={(v: number) => fmtMoney(v, base)}
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
            fill="url(#payeeTrendEarnedGrad)"
            dot={false}
          />
          <Area
            type="monotone"
            dataKey="paid"
            stroke="hsl(150 55% 45%)"
            strokeWidth={2}
            fill="url(#payeeTrendPaidGrad)"
            dot={false}
          />
        </AreaChart>
      </ChartContainer>
    </div>
  );
}
