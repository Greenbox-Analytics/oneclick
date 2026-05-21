import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ExternalLink } from "lucide-react";
import {
  useAdminBehaviorSummary,
  type DailyVisitorPoint,
} from "@/hooks/useAdminBehaviorSummary";

type WindowKey = "7d" | "30d" | "all";
type Cohort = "testers" | "all";

const WINDOW_LABELS: Record<WindowKey, string> = {
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  all: "All time",
};

function formatDuration(ms: number): string {
  if (!ms || ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rest = Math.round(s - m * 60);
  return `${m}m ${rest}s`;
}

function Sparkline({ data }: { data: DailyVisitorPoint[] }) {
  if (data.length === 0) return <div className="text-xs text-muted-foreground">No data</div>;
  const width = 220;
  const height = 40;
  const max = Math.max(...data.map((d) => d.unique_visitors), 1);
  const stepX = width / Math.max(data.length - 1, 1);
  const points = data
    .map((d, i) => `${i * stepX},${height - (d.unique_visitors / max) * height}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-10">
      <polyline
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        points={points}
        className="text-primary"
      />
    </svg>
  );
}

export function BehaviorAnalyticsCard() {
  const [window, setWindow] = useState<WindowKey>("7d");
  const [cohort, setCohort] = useState<Cohort>("all");
  const { data, isLoading, error } = useAdminBehaviorSummary({ window, cohort });
  const dashboardUrl = import.meta.env.VITE_POSTHOG_DASHBOARD_URL as string | undefined;

  return (
    <Card className="mb-6">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-lg">
          Page Behavior · {WINDOW_LABELS[window]}
        </CardTitle>
        <div className="flex items-center gap-2">
          <Select value={cohort} onValueChange={(v) => setCohort(v as Cohort)}>
            <SelectTrigger className="w-[120px]"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All users</SelectItem>
              <SelectItem value="testers">Testers</SelectItem>
            </SelectContent>
          </Select>
          <Select value={window} onValueChange={(v) => setWindow(v as WindowKey)}>
            <SelectTrigger className="w-[140px]"><SelectValue /></SelectTrigger>
            <SelectContent>
              {(Object.keys(WINDOW_LABELS) as WindowKey[]).map((w) => (
                <SelectItem key={w} value={w}>{WINDOW_LABELS[w]}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          {dashboardUrl && (
            <Button asChild variant="outline" size="sm">
              <a href={dashboardUrl} target="_blank" rel="noreferrer">
                View in PostHog <ExternalLink className="ml-1 w-3.5 h-3.5" />
              </a>
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-[10px] text-muted-foreground -mt-2 mb-3">
          Dwell/bounce data available from the deploy that landed pathname normalization.
          Pre-deploy events use the legacy property name and don't join the dwell CTE.
        </p>
        <p className="text-[10px] text-muted-foreground mb-3">
          Cohorts mix dev + prod events (shared PostHog project).
        </p>
        {isLoading && <div className="h-24 animate-pulse bg-muted rounded" />}
        {error && (
          <div className="text-sm text-destructive">
            Couldn't load behavior analytics. Try again later.
          </div>
        )}
        {data && !data.available && (
          <div className="text-sm text-muted-foreground">
            Analytics unavailable. {data.reason || "Configure POSTHOG_PERSONAL_API_KEY to see this card."}
          </div>
        )}
        {data && data.available && (
          <>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <KPI label="Total pageviews" value={data.total_pageviews.toLocaleString()} />
              <KPI label="Unique visitors" value={data.unique_visitors.toLocaleString()} />
              <KPI
                label="Pageviews / visitor"
                value={data.pageviews_per_visitor.toFixed(2)}
                title="Window-level average, not per-session — we don't track session boundaries."
              />
            </div>

            <div className="mb-4">
              <div className="text-xs text-muted-foreground mb-1">Daily unique visitors</div>
              <Sparkline data={data.daily_visitors} />
            </div>

            <div className="grid grid-cols-2 gap-6">
              <div>
                <div className="text-sm font-medium mb-2">Top pages</div>
                <table className="w-full text-sm">
                  <thead className="text-left text-muted-foreground">
                    <tr>
                      <th className="pb-1">Path</th>
                      <th>Views</th>
                      <th>Unique</th>
                      <th>Avg dwell</th>
                      <th>Bounce</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.top_pages.map((r) => (
                      <tr key={r.path} className="border-t">
                        <td className="py-2 truncate max-w-[160px]" title={r.path}>{r.path}</td>
                        <td>{r.views}</td>
                        <td>{r.unique_visitors}</td>
                        <td>{formatDuration(r.avg_dwell_ms)}</td>
                        <td>{Math.round(r.bounce_rate * 100)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-[10px] text-muted-foreground mt-2">
                  Bounce = sessions where dwell on this page was &lt; 10s.
                </p>
              </div>
              <div>
                <div className="text-sm font-medium mb-2">Top flows</div>
                <table className="w-full text-sm">
                  <thead className="text-left text-muted-foreground">
                    <tr>
                      <th className="pb-1">From → To</th>
                      <th>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.top_flows.map((e, i) => (
                      <tr key={`${e.from_path}-${e.to_path}-${i}`} className="border-t">
                        <td className="py-2 truncate max-w-[260px]">
                          <span className="text-muted-foreground">{e.from_path}</span>
                          {" → "}
                          {e.to_path}
                        </td>
                        <td>{e.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function KPI({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div title={title}>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-2xl font-semibold">{value}</div>
    </div>
  );
}
