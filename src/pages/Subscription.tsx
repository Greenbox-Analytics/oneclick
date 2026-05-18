import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Music, Check, Lock, Sparkles, FileText } from "lucide-react";
import { toast } from "sonner";
import { useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import {
  useEntitlements,
  type Entitlements,
  type IntegrationName,
} from "@/hooks/useEntitlements";
import { useArtistsList } from "@/hooks/useArtistsList";
import { useProjectsList } from "@/hooks/useProjectsList";
import { useBoards } from "@/hooks/useBoards";
import { useCreateCheckoutSession, useCreatePortalSession } from "@/hooks/useBilling";
import { useAnalytics, type Plan } from "@/hooks/useAnalytics";
import { peekCachedAnalyticsContext, refreshAnalyticsContext } from "@/hooks/useAnalyticsContext";

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

  const [searchParams, setSearchParams] = useSearchParams();
  const stripeSessionId = searchParams.get("stripe_session_id");
  const welcome = searchParams.get("welcome") === "true";
  const [plan, setPlan] = useState<"monthly" | "annual">("monthly");
  const [isPolling, setIsPolling] = useState(false);
  const { mutateAsync: createCheckout, isPending: isStartingCheckout } = useCreateCheckoutSession();
  const { mutateAsync: createPortal, isPending: isOpeningPortal } = useCreatePortalSession();
  const queryClient = useQueryClient();
  const { captureCheckoutStarted, captureCheckoutCompleted } = useAnalytics();
  const checkoutCompletedFiredRef = useRef(false);

  const ent: Entitlements | undefined = entQuery.data;
  const isPro = ent?.tier === "pro";
  const hasStripeSubscription = Boolean(ent?.subscription?.stripeSubscriptionId);
  const isProViaOverride = isPro && !hasStripeSubscription;
  // Tester status comes from /me/analytics-context (filters tier_overrides
  // by reason LIKE 'tester%'). Cached value is good enough here — if a tester
  // grant was issued just now, the user will see it after the next refresh
  // or sign-in.
  const analyticsCtx = user?.id ? peekCachedAnalyticsContext(user.id) : null;
  const isTester = analyticsCtx?.is_tester === true;
  const testerExpiresAt = analyticsCtx?.tester_expires_at ?? null;

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

  // Post-Checkout polling — fires when returning from Stripe with ?welcome=true&stripe_session_id=...
  useEffect(() => {
    if (!welcome || !stripeSessionId) return;
    if (isPro) {
      if (!checkoutCompletedFiredRef.current) {
        checkoutCompletedFiredRef.current = true;
        const completedPlan = (ent?.subscription?.planPeriod as Plan | undefined) ?? "monthly";
        captureCheckoutCompleted(completedPlan);
      }
      // Refresh the analytics-context cache so the dashboard UpgradeBanner
      // (which reads from the 5-min localStorage cache) doesn't keep showing
      // "you're on Free" for up to 5 minutes after upgrade.
      if (user?.id) {
        void refreshAnalyticsContext(user.id, user.email);
      }
      setSearchParams({});
      toast.success("Welcome to Pro! Your subscription is active.");
      return;
    }
    setIsPolling(true);
    const interval = setInterval(() => {
      queryClient.invalidateQueries({ queryKey: ["entitlements"] });
    }, 1000);
    const timeout = setTimeout(() => {
      clearInterval(interval);
      setIsPolling(false);
      toast.info("Subscription is processing. Refresh in a moment if it doesn't show up.");
    }, 10_000);
    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [welcome, stripeSessionId, isPro]);

  const handleSubscribe = async () => {
    try {
      const url = await createCheckout(plan);
      captureCheckoutStarted(plan);
      window.location.href = url;
    } catch {
      toast.error("Couldn't start checkout. Try again or contact support.");
    }
  };

  const handleManageBilling = async () => {
    try {
      const url = await createPortal();
      window.location.href = url;
    } catch (err: unknown) {
      const status = (err as { status?: number })?.status;
      if (status === 404) {
        toast.error("No Stripe subscription on file. If you believe this is an error, contact support.");
      } else {
        toast.error("Couldn't open billing portal. Try again or contact support.");
      }
    }
  };

  if (entQuery.isLoading || !ent) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Post-Checkout polling overlay */}
      {isPolling && (
        <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur flex items-center justify-center">
          <div className="text-center">
            <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            <div className="text-sm">Activating your Pro subscription…</div>
          </div>
        </div>
      )}

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

        {/* Over-cap banner (SP3) */}
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

        {/* Cancel-scheduled banner */}
        {ent.subscription?.cancelAtPeriodEnd && ent.subscription?.currentPeriodEnd && (
          <div className="mb-6 rounded-lg border border-amber-500/20 bg-amber-500/5 p-4">
            <div className="font-medium text-sm">Subscription scheduled to end</div>
            <div className="text-sm text-muted-foreground mt-1">
              Your Pro access ends on {new Date(ent.subscription.currentPeriodEnd).toLocaleDateString()}.
              Reactivate via Manage Billing if you change your mind.
            </div>
          </div>
        )}

        {/* Override-only Pro banner — distinguishes tester grants from generic admin grants */}
        {isProViaOverride && isTester && (
          <div className="mb-6 rounded-lg border border-primary/30 bg-primary/5 p-4">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-primary" />
              <div className="font-medium text-sm">Beta tester access</div>
            </div>
            <div className="text-sm text-muted-foreground mt-1">
              You have full Pro access as a beta tester
              {testerExpiresAt ? ` until ${formatPeriodEnd(testerExpiresAt)}` : " (no expiration set)"}.
              Thanks for helping us shape Msanii! For questions or feedback,{" "}
              <a href="mailto:tech@greenboxanalytics.ca" className="underline">
                contact us
              </a>
              .
            </div>
          </div>
        )}
        {isProViaOverride && !isTester && (
          <div className="mb-6 rounded-lg border border-border bg-card p-4">
            <div className="font-medium text-sm">Pro access granted by admin</div>
            <div className="text-sm text-muted-foreground mt-1">
              You have Pro access via an admin grant, not a paid subscription.
              For billing questions,{" "}
              <a href="mailto:tech@greenboxanalytics.ca" className="underline">
                contact support
              </a>
              .
            </div>
          </div>
        )}

        {/* Subscribe CTA for Free users */}
        {!isPro && (
          <Card className="p-8 mb-6">
            <h2 className="text-xl font-semibold mb-2">Upgrade to Pro</h2>
            <p className="text-sm text-muted-foreground mb-6">
              Unlimited artists, projects, tasks, storage, split sheets, plus Zoe AI, OneClick, Registry, and all
              integrations.
            </p>
            <Tabs value={plan} onValueChange={(v) => setPlan(v as "monthly" | "annual")} className="mb-6">
              <TabsList className="grid w-full max-w-xs grid-cols-2">
                <TabsTrigger value="monthly">Monthly — $25/mo</TabsTrigger>
                <TabsTrigger value="annual">Annual — $250/yr</TabsTrigger>
              </TabsList>
            </Tabs>
            <Button onClick={handleSubscribe} disabled={isStartingCheckout} size="lg">
              {isStartingCheckout ? "Starting checkout…" : "Subscribe to Pro"}
            </Button>
          </Card>
        )}

        {/* Manage Billing button for Stripe-Pro users */}
        {isPro && hasStripeSubscription && (
          <div className="flex justify-end mb-4">
            <Button variant="outline" onClick={handleManageBilling} disabled={isOpeningPortal}>
              {isOpeningPortal ? "Opening…" : "Manage billing"}
            </Button>
          </div>
        )}

        {/* Tier card */}
        <Card className="p-6 mb-8 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h2 className="text-xl font-semibold">{isPro ? "Pro" : "Free"} plan</h2>
              {isPro && <Badge>Pro</Badge>}
              {isTester && <Badge variant="secondary">Tester</Badge>}
              {ent.degraded && (
                <Badge variant="outline" className="text-xs">
                  Account state temporarily unavailable
                </Badge>
              )}
            </div>
            <div className="text-muted-foreground">
              {isPro
                ? ent.subscription?.planPeriod === "annual"
                  ? "$250 / year"
                  : "$25 / month"
                : "$0 / month"}
            </div>
            {isPro && hasStripeSubscription && ent.subscription?.currentPeriodEnd && (
              <div className="text-xs text-muted-foreground mt-1">
                {ent.subscription.cancelAtPeriodEnd ? "Access ends" : "Renews"}{" "}
                {formatPeriodEnd(ent.subscription.currentPeriodEnd)} (in{" "}
                {daysUntil(ent.subscription.currentPeriodEnd)} days)
              </div>
            )}
            {!isPro && ent.usage.periodEnd && (
              <div className="text-xs text-muted-foreground mt-1">
                Period resets {formatPeriodEnd(ent.usage.periodEnd)} (in {daysUntil(ent.usage.periodEnd)} days)
              </div>
            )}
          </div>
        </Card>

        {/* Resource limits */}
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-3">Resource limits</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
          <QuotaCard label="Artists" current={artistsCount} cap={ent.caps.maxArtists} />
          <QuotaCard label="Projects" current={projectsCount} cap={ent.caps.maxProjects} />
          <QuotaCard label="Tasks" current={tasksCount} cap={ent.caps.maxTasks} />
          <Card className="p-5 col-span-2">
            <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">Storage</div>
            <div className="text-2xl font-semibold tracking-tight">
              {formatBytes(ent.usage.totalStorageBytes)}
              {ent.caps.maxStorageBytes !== -1 && (
                <span className="text-base text-muted-foreground font-normal">
                  {" / "}
                  {formatBytes(ent.caps.maxStorageBytes)}
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
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-3">This period</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
          <Card className="p-5">
            <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
              <FileText className="w-4 h-4" /> Split sheets
            </div>
            <div className="text-2xl font-semibold tracking-tight">
              {ent.usage.splitSheetsThisPeriod}
              {ent.caps.maxSplitSheetsPerMonth !== -1 && (
                <span className="text-base text-muted-foreground font-normal">
                  {" / "}
                  {ent.caps.maxSplitSheetsPerMonth}
                </span>
              )}
            </div>
          </Card>
          <Card className="p-5">
            <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
              <Sparkles className="w-4 h-4" /> Zoe queries
            </div>
            <div className="text-2xl font-semibold tracking-tight">{ent.usage.zoeQueriesThisPeriod}</div>
            {!ent.features.zoeEnabled && <div className="text-xs text-muted-foreground mt-1">Pro feature</div>}
          </Card>
          <QuotaCard
            label="OneClick runs this period"
            current={ent.usage.oneclickRunsThisPeriod ?? 0}
            cap={ent.caps.maxOneclickRunsPerMonth}
          />
        </div>

        {/* Integrations */}
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide mb-3">Integrations</h3>
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
