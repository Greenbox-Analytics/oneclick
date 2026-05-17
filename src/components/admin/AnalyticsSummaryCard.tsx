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
import { useAdminAnalyticsSummary } from "@/hooks/useAdminAnalyticsSummary";

type WindowKey = "7d" | "30d" | "all";

const WINDOW_LABELS: Record<WindowKey, string> = {
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  "all": "All time",
};

function formatPct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

function formatRelative(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  const diffMs = Date.now() - date.getTime();
  const mins = Math.round(diffMs / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return `${days}d ago`;
}

export function AnalyticsSummaryCard() {
  const [window, setWindow] = useState<WindowKey>("7d");
  const cohort = "testers" as const;
  const { data, isLoading, error } = useAdminAnalyticsSummary({ window, cohort });
  const dashboardUrl = import.meta.env.VITE_POSTHOG_DASHBOARD_URL as string | undefined;

  return (
    <Card className="mb-6">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-lg">
          Tool Usage — Testers · {WINDOW_LABELS[window]}
        </CardTitle>
        <div className="flex items-center gap-2">
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
        {isLoading && <div className="h-24 animate-pulse bg-muted rounded" />}
        {error && (
          <div className="text-sm text-destructive">
            Couldn't load analytics. Try again later.
          </div>
        )}
        {data && !data.available && (
          <div className="text-sm text-muted-foreground">
            Analytics unavailable. {data.reason || "Configure POSTHOG_PERSONAL_API_KEY to see this card."}
          </div>
        )}
        {data && data.available && (
          <>
            <div className="grid grid-cols-4 gap-4 mb-4">
              <KPI label="Active testers" value={`${data.active_users} / ${data.total_users}`} />
              <KPI label="Tool actions" value={data.tool_actions.toLocaleString()} />
              <KPI
                label="Top tool"
                value={data.top_tool ? `${data.top_tool} (${formatPct(data.top_tool_share)})` : "—"}
              />
              <KPI label="Funnel completion (avg)" value={formatPct(data.funnel_completion_avg)} />
            </div>
            <table className="w-full text-sm">
              <thead className="text-left text-muted-foreground">
                <tr>
                  <th className="pb-1">Tool</th>
                  <th>Opens</th>
                  <th>Completions</th>
                  <th>Completion rate</th>
                  <th>Last used</th>
                </tr>
              </thead>
              <tbody>
                {data.per_tool.map((r) => (
                  <tr key={r.tool} className="border-t">
                    <td className="py-2">{r.tool}</td>
                    <td>{r.opens}</td>
                    <td>{r.completions}</td>
                    <td>{formatPct(r.completion_rate)}</td>
                    <td>{formatRelative(r.last_used)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function KPI({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-2xl font-semibold">{value}</div>
    </div>
  );
}
