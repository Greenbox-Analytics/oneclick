import { Fragment, useState } from "react";
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
import { useAdminAnalyticsSummary, type ToolFunnel } from "@/hooks/useAdminAnalyticsSummary";

type WindowKey = "7d" | "30d" | "all";

const WINDOW_LABELS: Record<WindowKey, string> = {
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  "all": "All time",
};

function formatPct(n: number): string {
  return `${Math.round(n * 100)}%`;
}

// Below this, ratios are too noisy to show as a rate. Applied with DIFFERENT units on
// purpose: drop-off % is gated on the prior step's USER count (it's a per-user funnel),
// while the error badge is gated on ATTEMPT count (completed_events + failed_events, since
// error rate is attempt-level). Same threshold, two units — intentional, not a bug.
const MIN_N = 5;

function errBadgeClass(rate: number): string {
  if (rate >= 0.25) return "text-red-600";
  if (rate >= 0.1) return "text-amber-600";
  return "text-muted-foreground";
}

function FunnelStrip({ funnel }: { funnel: ToolFunnel }) {
  const first = funnel.steps[0]?.users ?? 0;
  const attempts = funnel.completed_events + funnel.failed_events;
  return (
    <div className="py-2 border-t">
      <div className="flex items-center justify-between mb-1">
        <span className="font-medium">{funnel.tool}</span>
        {attempts >= MIN_N ? (
          <span className={`text-xs ${errBadgeClass(funnel.error_rate)}`}>
            err {formatPct(funnel.error_rate)}
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">err — (n&lt;{MIN_N})</span>
        )}
      </div>
      <div className="flex items-end gap-2">
        {funnel.steps.map((step, i) => {
          const prev = i > 0 ? funnel.steps[i - 1].users : null;
          const showDrop = prev !== null && prev >= MIN_N;
          const drop = prev ? 1 - step.users / prev : 0;
          return (
            <Fragment key={step.label}>
              {i > 0 && (
                <span className="text-xs text-muted-foreground pb-5">
                  {showDrop ? `↓${Math.round(drop * 100)}%` : "→"}
                </span>
              )}
              <div className="flex-1">
                <div className="text-xs text-muted-foreground">{step.label}</div>
                <div className="h-5 bg-muted rounded">
                  <div
                    className="h-full bg-primary/30 rounded"
                    style={{ width: `${first ? (step.users / first) * 100 : 0}%` }}
                  />
                </div>
                <div className="text-sm font-medium mt-0.5">{step.users}</div>
              </div>
            </Fragment>
          );
        })}
      </div>
    </div>
  );
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
                  <th title="Times the tool was opened (raw event count).">Opens</th>
                  <th title="Total completed actions — a user can complete several per open. See legend below for what counts.">
                    Completions
                  </th>
                  <th title="Distinct testers who opened the tool in this window.">Openers</th>
                  <th title="Openers who completed at least one action (a subset of Openers).">
                    Converted
                  </th>
                  <th title="Converted ÷ Openers — the share of testers who opened and then completed at least one action. Bounded at 100%.">
                    Completion rate
                  </th>
                  <th>Last used</th>
                </tr>
              </thead>
              <tbody>
                {data.per_tool.map((r) => (
                  <tr key={r.tool} className="border-t">
                    <td className="py-2">{r.tool}</td>
                    <td>{r.opens}</td>
                    <td>{r.completions}</td>
                    <td>{r.openers}</td>
                    <td>{r.converters}</td>
                    <td>{formatPct(r.completion_rate)}</td>
                    <td>{formatRelative(r.last_used)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-xs text-muted-foreground leading-relaxed">
              <span className="font-medium">What counts as a completion:</span>{" "}
              <span className="font-medium text-foreground">oneclick</span> — a royalty calculation
              finished · <span className="font-medium text-foreground">zoe</span> — Zoe returned an
              answer · <span className="font-medium text-foreground">splitsheet</span> — a split
              sheet was generated · <span className="font-medium text-foreground">registry</span> — a
              work reached Registered status. Opens &amp; Completions are raw event counts (a tester
              can complete several actions per open); Completion rate is a per-tester funnel
              (Converted ÷ Openers).
            </p>
            {(data.funnels.length > 0 || data.registry_lifecycle) && (
              <div className="mt-5">
                <div className="text-sm font-medium mb-1">Funnel &amp; reliability</div>
                {/* Funnel strips render only when there are session funnels; the registry
                    strip is gated independently so registry activity shows even when the
                    session tools are quiet. */}
                {data.funnels.map((f) => (
                  <FunnelStrip key={f.tool} funnel={f} />
                ))}
                {data.registry_lifecycle && (
                  <div className="mt-3 pt-2 border-t">
                    <div className="text-xs text-muted-foreground mb-1">
                      Registry lifecycle (works)
                    </div>
                    <div className="flex items-center gap-3 text-sm">
                      <span>
                        created <b>{data.registry_lifecycle.created}</b>
                      </span>
                      <span className="text-muted-foreground">→</span>
                      <span>
                        submitted <b>{data.registry_lifecycle.submitted}</b>
                      </span>
                      <span className="text-muted-foreground">→</span>
                      <span>
                        registered <b>{data.registry_lifecycle.registered}</b>
                      </span>
                    </div>
                  </div>
                )}
                <p className="mt-3 text-xs text-muted-foreground leading-relaxed">
                  Steps are distinct testers (ordered). Error rate is failed ÷ completed-or-failed
                  attempts (not all attempts). Percentages are hidden below {MIN_N}. The registry
                  strip counts <b>works</b> (created/submitted/registered events); the registry row
                  in the table above counts <b>users</b> — different units, not a discrepancy.
                  Boards/calendar have no funnel (no completion events).
                </p>
              </div>
            )}
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
