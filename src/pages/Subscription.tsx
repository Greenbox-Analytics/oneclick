import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Music, Check, Lock, Sparkles, Calculator, FileText } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import {
  useEntitlements,
  type Entitlements,
  type IntegrationName,
} from "@/hooks/useEntitlements";
import { useArtistsList } from "@/hooks/useArtistsList";
import { useProjectsList } from "@/hooks/useProjectsList";
import { useBoards } from "@/hooks/useBoards";

// IMPORTANT: hook return shapes (verified against actual files):
//   useArtistsList()  → { artists, isLoading }
//   useProjectsList(artistIds?, projectIds?) → { projects, contracts, isLoading }
//                       NOTE: enabled=false unless artistIds OR projectIds non-empty
//                       APPLIES TO contractsQuery only — projectsQuery is always
//                       on for the calling user. So: call with no args to get all
//                       user projects in ONE request (vs N+1 fan-out with artistIds).
//   useBoards()       → { columns, tasks, isLoading, ...mutations }
//                       tasks is BoardTask[] (already resolved, never undefined).
//                       columnsQuery.data is BoardColumn[] (kanban columns), not boards.
//                       What "max_boards" actually counts is under-specified in SP1;
//                       we OMIT the boards card here until that's clarified.

const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
};

const formatPeriodEnd = (iso: string): string => {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });
};

const daysUntil = (iso: string): number => {
  const d = new Date(iso);
  const ms = d.getTime() - Date.now();
  return Math.max(0, Math.ceil(ms / (1000 * 60 * 60 * 24)));
};

interface QuotaCardProps {
  label: string;
  current: number;
  cap: number;
  formatter?: (n: number) => string;
}

const QuotaCard = ({ label, current, cap, formatter }: QuotaCardProps) => {
  const fmt = formatter ?? ((n: number) => String(n));
  if (cap === -1) {
    return (
      <Card className="p-5">
        <div className="text-sm text-muted-foreground mb-2">{label}</div>
        <div className="text-2xl font-semibold tracking-tight">{fmt(current)}</div>
        <div className="text-xs text-muted-foreground mt-1">Unlimited</div>
      </Card>
    );
  }
  const pct = cap > 0 ? (current / cap) * 100 : 0;
  const nearLimit = pct >= 80 && pct < 100;
  const overCap = pct >= 100;
  return (
    <Card className="p-5">
      <div className="text-sm text-muted-foreground mb-2 flex items-center justify-between">
        <span>{label}</span>
        {nearLimit && <span className="text-xs text-amber-600 dark:text-amber-500">Near limit</span>}
        {overCap && <span className="text-xs text-destructive">Over limit</span>}
      </div>
      <div className="text-2xl font-semibold tracking-tight">
        {fmt(current)} <span className="text-base text-muted-foreground font-normal">/ {fmt(cap)}</span>
      </div>
      <Progress value={Math.min(100, pct)} className="mt-3" />
    </Card>
  );
};

const INTEGRATION_LABELS: Record<IntegrationName, string> = {
  google_drive: "Google Drive",
  slack: "Slack",
  notion: "Notion",
  monday: "Monday",
};

const Subscription = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const entQuery = useEntitlements();
  const { artists } = useArtistsList();
  // No-args useProjectsList() fires GET /projects once for all the user's projects.
  // Don't pass artistIds — that would trigger an N+1 fan-out (one /artists/{id}/projects
  // request per artist). The hook's projectsQuery.enabled is just !!user?.id (always on);
  // the args-gated enabled clause is for contractsQuery, which we don't need here.
  const { projects } = useProjectsList();
  // useBoards() returns { columns, tasks, isLoading, ...mutations } — tasks is already
  // a resolved BoardTask[] (never undefined). The spec's tasksQuery.data pattern does not
  // apply to this hook. Boards card is omitted — see module-level comment.
  const { tasks } = useBoards();

  const ent: Entitlements | undefined = entQuery.data;
  const isPro = ent?.tier === "pro";

  // Suppress unused-var lint warning — user is consumed by useEntitlements indirectly
  void user;

  const artistsCount = artists?.length ?? 0;
  const projectsCount = projects?.length ?? 0;
  const tasksCount = tasks?.length ?? 0;
  // boardsCount intentionally omitted — see import-block comment.

  const isOverCap = useMemo(() => {
    if (!ent) return false;
    return (
      (ent.caps.maxArtists !== -1 && artistsCount > ent.caps.maxArtists) ||
      (ent.caps.maxProjects !== -1 && projectsCount > ent.caps.maxProjects) ||
      (ent.caps.maxTasks !== -1 && tasksCount > ent.caps.maxTasks) ||
      (ent.caps.maxStorageBytes !== -1 && ent.usage.totalStorageBytes > ent.caps.maxStorageBytes) ||
      (ent.caps.maxSplitSheetsPerMonth !== -1 && ent.usage.splitSheetsThisPeriod > ent.caps.maxSplitSheetsPerMonth)
    );
  }, [ent, artistsCount, projectsCount, tasksCount]);

  if (entQuery.isLoading || !ent) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => navigate("/")}
            >
              <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-lg font-semibold tracking-tight">Msanii</span>
            </div>
            <Button variant="ghost" onClick={() => navigate("/dashboard")}>
              Back to dashboard
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-5xl">
        <h1 className="text-3xl font-semibold tracking-tight mb-6">Subscription &amp; usage</h1>

        {isOverCap && (
          <div className="mb-6 rounded-lg border border-destructive/20 bg-destructive/5 p-4 flex items-center justify-between">
            <div className="flex-1">
              <div className="font-medium text-sm">You're over your Free tier limits</div>
              <div className="text-sm text-muted-foreground mt-1">
                Some create actions are blocked until you reduce your usage or upgrade.
              </div>
            </div>
            <Button onClick={() => navigate("/pricing")}>Upgrade to Pro</Button>
          </div>
        )}

        {/* Tier card */}
        <Card className="p-6 mb-8 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-xl font-semibold">{isPro ? "Pro" : "Free"} plan</h2>
              {isPro && <Badge>Pro</Badge>}
              {ent.degraded && (
                <Badge variant="outline" className="text-xs">Account state temporarily unavailable</Badge>
              )}
            </div>
            <div className="text-muted-foreground">
              {isPro ? "$25 / month" : "$0 / month"}
            </div>
            {ent.usage.periodEnd && (
              <div className="text-xs text-muted-foreground mt-1">
                Period resets {formatPeriodEnd(ent.usage.periodEnd)} (in {daysUntil(ent.usage.periodEnd)} days)
              </div>
            )}
          </div>
          {isPro ? (
            <Button variant="outline" disabled>Manage plan (coming soon)</Button>
          ) : (
            <Button onClick={() => navigate("/pricing")}>Upgrade to Pro →</Button>
          )}
        </Card>

        {/* Resource limits */}
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-3">
          Resource limits
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
          <QuotaCard label="Artists" current={artistsCount} cap={ent.caps.maxArtists} />
          <QuotaCard label="Projects" current={projectsCount} cap={ent.caps.maxProjects} />
          <QuotaCard label="Tasks" current={tasksCount} cap={ent.caps.maxTasks} />
          <Card className="p-5 col-span-2">
            <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
              Storage
            </div>
            <div className="text-2xl font-semibold tracking-tight">
              {formatBytes(ent.usage.totalStorageBytes)}
              {ent.caps.maxStorageBytes !== -1 && (
                <span className="text-base text-muted-foreground font-normal">
                  {" / "}{formatBytes(ent.caps.maxStorageBytes)}
                </span>
              )}
            </div>
            {ent.caps.maxStorageBytes === -1 ? (
              <div className="text-xs text-muted-foreground mt-1">Unlimited</div>
            ) : (
              <Progress
                value={Math.min(100, (ent.usage.totalStorageBytes / ent.caps.maxStorageBytes) * 100)}
                className="mt-3"
              />
            )}
          </Card>
        </div>

        {/* This period */}
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-3">
          This period
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
          <Card className="p-5">
            <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
              <FileText className="w-4 h-4" /> Split sheets
            </div>
            <div className="text-2xl font-semibold tracking-tight">
              {ent.usage.splitSheetsThisPeriod}
              {ent.caps.maxSplitSheetsPerMonth !== -1 && (
                <span className="text-base text-muted-foreground font-normal">
                  {" / "}{ent.caps.maxSplitSheetsPerMonth}
                </span>
              )}
            </div>
          </Card>
          <Card className="p-5">
            <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
              <Sparkles className="w-4 h-4" /> Zoe queries
            </div>
            <div className="text-2xl font-semibold tracking-tight">
              {ent.usage.zoeQueriesThisPeriod}
            </div>
            {!ent.features.zoeEnabled && (
              <div className="text-xs text-muted-foreground mt-1">Pro feature</div>
            )}
          </Card>
          <Card className="p-5">
            <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
              <Calculator className="w-4 h-4" /> OneClick runs
            </div>
            <div className="text-2xl font-semibold tracking-tight">
              {ent.usage.oneclickRunsThisPeriod}
            </div>
            {!ent.features.oneclickEnabled && (
              <div className="text-xs text-muted-foreground mt-1">Pro feature</div>
            )}
          </Card>
        </div>

        {/* Integrations */}
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-3">
          Integrations
        </h3>
        <Card className="p-5">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {(Object.keys(INTEGRATION_LABELS) as IntegrationName[]).map((name) => {
              const enabled = ent.features.integrationsAllowed.includes(name);
              return (
                <div
                  key={name}
                  className={`flex items-center gap-2 px-3 py-2 rounded-md border ${
                    enabled ? "border-border" : "border-dashed border-border/50 opacity-60"
                  }`}
                >
                  {enabled ? (
                    <Check className="w-4 h-4 text-primary flex-shrink-0" />
                  ) : (
                    <Lock className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                  )}
                  <span className="text-sm">{INTEGRATION_LABELS[name]}</span>
                </div>
              );
            })}
          </div>
        </Card>
      </main>
    </div>
  );
};

export default Subscription;
