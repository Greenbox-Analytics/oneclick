// src/components/orgs/UsageDetailDialog.tsx
// The breakdown behind one row of a usage table — a member, an API key or a
// folder. Purely presentational: the caller hands it a UsageSubject already
// built from the payload it fetched, so this file never fetches anything.
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Tile, ToolMix } from "./usageTableBits";
import { TOOLS } from "@/lib/usageTools";
import { bucketSeries, windowLabel, type UsageSubject } from "@/lib/orgUsage";
import type { UsageRange } from "@/hooks/useOrgs";

const chartConfig: ChartConfig = Object.fromEntries(TOOLS.map((t) => [t.id, { label: t.label, color: t.color }]));

export function UsageDetailDialog({
  subject,
  range,
  since,
  windowTotal,
  onClose,
}: {
  subject: UsageSubject | null;
  range: UsageRange;
  since: string | null;
  /** Credits spent across the whole window, for the "share of total" tile. */
  windowTotal: number;
  onClose: () => void;
}) {
  const data = subject ? bucketSeries(subject.series, range, since).map((b) => ({ label: b.label, ...b.tools })) : [];
  const share = subject && windowTotal ? `${Math.round((subject.total / windowTotal) * 100)}%` : "—";

  return (
    <Dialog open={!!subject} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl">
        {subject && (
          <>
            <DialogHeader>
              <DialogTitle>{subject.title}</DialogTitle>
              <DialogDescription>
                {subject.subtitle} · {windowLabel(range, since)}
              </DialogDescription>
            </DialogHeader>

            <div className="grid grid-cols-3 gap-3">
              <Tile label="Credits used" value={subject.total.toLocaleString()} />
              <Tile label="Runs" value={subject.runs.toLocaleString()} />
              <Tile
                label="Share of total"
                value={share}
                sub={windowTotal ? `of ${windowTotal.toLocaleString()} credits in this window` : undefined}
              />
            </div>

            <div>
              <h4 className="mb-2 text-[13px] font-semibold">Credits over time</h4>
              {subject.series.length === 0 ? (
                <p className="py-6 text-center text-[13px] text-muted-foreground">No credits used in this window.</p>
              ) : (
                <ChartContainer config={chartConfig} className="h-[200px] w-full">
                  <AreaChart data={data} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                    <CartesianGrid vertical={false} strokeDasharray="3 3" />
                    <XAxis dataKey="label" tickLine={false} axisLine={false} fontSize={11} minTickGap={16} />
                    <YAxis tickLine={false} axisLine={false} fontSize={11} width={44} allowDecimals={false} />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <ChartLegend content={<ChartLegendContent />} />
                    {TOOLS.filter((t) => subject.tools[t.id] > 0).map((t) => (
                      <Area
                        key={t.id}
                        stackId="credits"
                        type="monotone"
                        dataKey={t.id}
                        stroke={t.color}
                        fill={t.color}
                        fillOpacity={0.35}
                      />
                    ))}
                  </AreaChart>
                </ChartContainer>
              )}
            </div>

            <div>
              <h4 className="mb-2 text-[13px] font-semibold">By tool</h4>
              <ToolMix tools={subject.tools} total={subject.total} />
            </div>

            <dl className="grid grid-cols-2 gap-x-6 gap-y-3">
              {subject.facts.map((f) => (
                <div key={f.label}>
                  <dt className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                    {f.label}
                  </dt>
                  <dd className="text-[13px]">{f.value}</dd>
                </div>
              ))}
            </dl>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
