// src/components/orgs/OrgUsageAnalysis.tsx
// Admin console: the team's credit usage over a chosen window — four tiles, a
// stacked per-tool chart, the tool mix, then per-member and per-key tables.
// Everything below the header follows the one selected range. The members
// table above keeps its own MTD numbers: the range is part of the query key.
// Admin-only by construction (AdminConsole mounts it; the endpoint 403s others).
import { useState } from "react";
import { BarChart3, Download, Loader2 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FoldersTable, KeysTable, MembersTable, Tile, ToolMix } from "./usageTableBits";
import { UsageDetailDialog } from "./UsageDetailDialog";
import { useToast } from "@/hooks/use-toast";
import { API_URL } from "@/lib/apiFetch";
import { downloadPdf } from "@/lib/downloadPdf";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { useOrgUsage, type UsageRange } from "@/hooks/useOrgs";
import { usePartnerKeys } from "@/hooks/usePartnerKeys";
import { TOOLS } from "@/lib/usageTools";
import {
  USAGE_RANGES,
  bucketSeries,
  byToolTotals,
  delta,
  folderUsageRows,
  keyUsageRows,
  memberRows,
  topTool,
  totals,
  windowLabel,
  type UsageSubject,
} from "@/lib/orgUsage";

const chartConfig: ChartConfig = Object.fromEntries(TOOLS.map((t) => [t.id, { label: t.label, color: t.color }]));

// What "vs previous" is measured against, per range. "all" has no previous window.
const VS_LABEL: Partial<Record<UsageRange, string>> = {
  "7d": "the previous 7 days",
  "14d": "the previous 14 days",
  mtd: "the previous period",
  "1y": "the previous 365 days",
};

/** The admin Organizations drawer renders this card read-only against the
 * /admin routes, so the two data hooks are injectable.
 *
 * `useUsage` / `useKeys` MUST be a stable reference for the life of the mount —
 * pass a module-level hook, never one built inline or chosen from state.
 * Each implementation calls a different number of hooks (the default
 * `usePartnerKeys` calls `useAuth` + `useQuery`; `useAdminPartnerKeys` calls
 * only `useQuery`), so swapping a prop mid-mount changes the hook count and
 * breaks the rules of hooks. Both are called unconditionally below. */
export function OrgUsageAnalysis({
  orgId,
  partnerApiEnabled,
  useUsage = useOrgUsage,
  useKeys = usePartnerKeys,
  reportPath,
}: {
  orgId: string;
  partnerApiEnabled: boolean;
  useUsage?: typeof useOrgUsage;
  useKeys?: typeof usePartnerKeys;
  /** Where the PDF comes from — the admin drawer points this at /admin. */
  reportPath?: string;
}) {
  const [range, setRange] = useState<UsageRange>("mtd");
  const [selected, setSelected] = useState<UsageSubject | null>(null);
  const { data: usage, isSuccess, isError } = useUsage(orgId, range);
  // The keys endpoint 403s an org without API access — don't ask.
  const { data: keysData } = useKeys(partnerApiEnabled ? orgId : undefined);
  const keys = keysData?.keys ?? [];

  const series = usage?.series ?? [];
  const sums = totals(series);
  const tools = byToolTotals(series);
  const top = topTool(tools);
  // Render off the range the payload was computed for, not the one just picked.
  const shown = usage?.range ?? range;
  const chartData = usage ? bucketSeries(series, shown, usage.since).map((b) => ({ label: b.label, ...b.tools })) : [];
  const members = memberRows(usage?.seats ?? [], keys);
  const keyRows = keyUsageRows(usage?.byKey ?? []);
  const folderRows = folderUsageRows(usage?.byFolder ?? []);
  const activeMembers = members.filter((m) => m.total > 0).length;
  // The backend only lists days with spend, so an empty series IS an empty window.
  const empty = isSuccess && series.length === 0;
  const n = (v: number) => (isSuccess ? v.toLocaleString() : "—");

  const { toast } = useToast();
  const [downloading, setDownloading] = useState(false);
  const path = reportPath ?? `/orgs/${orgId}/usage/report.pdf`;
  const download = async () => {
    setDownloading(true);
    try {
      await downloadPdf(`${API_URL}${path}?range=${range}`, `usage-report-${range}.pdf`);
    } catch {
      toast({ title: "Couldn't download the report", description: "Please try again.", variant: "destructive" });
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Card className="flex flex-col gap-4 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h3 className="flex items-center gap-2 text-[15px] font-semibold">
            <BarChart3 className="h-4 w-4" /> Usage
          </h3>
          <p className="text-[13px] text-muted-foreground">
            {usage ? windowLabel(shown, usage.since) : isError ? "Couldn't load" : "Loading…"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div
            role="radiogroup"
            aria-label="Usage range"
            className="inline-flex rounded-lg border border-border bg-muted/40 p-0.5"
          >
            {USAGE_RANGES.map((r) => (
              <button
                key={r.id}
                type="button"
                role="radio"
                aria-checked={range === r.id}
                title={r.title}
                onClick={() => setRange(r.id)}
                className={`rounded-md px-2.5 py-1 text-[12px] font-semibold transition-colors ${range === r.id ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}
              >
                {r.label}
              </button>
            ))}
          </div>
          <Button size="sm" variant="outline" onClick={download} disabled={!isSuccess || downloading}>
            {downloading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            Download PDF
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Tile
          label="Credits used"
          value={n(sums.credits)}
          change={isSuccess ? delta(sums.credits, usage?.previous?.credits) : null}
          vs={VS_LABEL[shown]}
        />
        <Tile
          label="Runs"
          value={n(sums.runs)}
          change={isSuccess ? delta(sums.runs, usage?.previous?.runs) : null}
          vs={VS_LABEL[shown]}
        />
        <Tile
          label="Most used tool"
          value={isSuccess ? (top?.label ?? "—") : "—"}
          sub={top ? `${Math.round(top.share * 100)}% of credits` : undefined}
        />
        <Tile label="Active members" value={n(activeMembers)} sub={usage ? `of ${usage.seats.length}` : undefined} />
      </div>

      {isError ? (
        <p className="py-6 text-center text-[13px] text-muted-foreground">
          Couldn&apos;t load usage. Please try refreshing.
        </p>
      ) : !isSuccess ? (
        <div className="flex justify-center py-8">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      ) : empty ? (
        <p className="py-6 text-center text-[13px] text-muted-foreground">No usage in this period.</p>
      ) : (
        <>
          <div>
            <h4 className="mb-2 text-[13px] font-semibold">Credits over time</h4>
            <ChartContainer config={chartConfig} className="h-[220px] w-full">
              <BarChart data={chartData} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis dataKey="label" tickLine={false} axisLine={false} fontSize={11} minTickGap={16} />
                <YAxis tickLine={false} axisLine={false} fontSize={11} width={44} allowDecimals={false} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <ChartLegend content={<ChartLegendContent />} />
                {TOOLS.filter((t) => tools[t.id] > 0).map((t) => (
                  <Bar key={t.id} dataKey={t.id} stackId="credits" fill={t.color} />
                ))}
              </BarChart>
            </ChartContainer>
          </div>
          <div>
            <h4 className="mb-2 text-[13px] font-semibold">By tool</h4>
            <ToolMix tools={tools} total={sums.credits} />
          </div>
        </>
      )}

      {partnerApiEnabled ? (
        <Tabs defaultValue="members">
          <TabsList>
            <TabsTrigger value="members">By member</TabsTrigger>
            <TabsTrigger value="keys">By key</TabsTrigger>
            <TabsTrigger value="folders">By folder</TabsTrigger>
          </TabsList>
          <TabsContent value="members">
            <MembersTable rows={members} loaded={isSuccess} showKeys onSelect={setSelected} />
          </TabsContent>
          <TabsContent value="keys">
            <KeysTable rows={keyRows} loaded={isSuccess} onSelect={setSelected} />
          </TabsContent>
          <TabsContent value="folders">
            <FoldersTable rows={folderRows} loaded={isSuccess} onSelect={setSelected} />
          </TabsContent>
        </Tabs>
      ) : (
        <MembersTable rows={members} loaded={isSuccess} showKeys={false} onSelect={setSelected} />
      )}

      <UsageDetailDialog
        subject={selected}
        range={shown}
        since={usage?.since ?? null}
        windowTotal={sums.credits}
        onClose={() => setSelected(null)}
      />
    </Card>
  );
}
