// src/pages/AdminUsers.tsx
// Msanii admin console — sidebar shell over four views (Overview, Users,
// Organizations, Beta testers) plus a ⌘K palette.
//
// The shell owns the cross-view selection state (selected user / org, the
// users search + filter) so a row on Overview or a hit in the palette can
// deep-link into another view's drawer.
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Beaker,
  Building2,
  ChevronRight,
  Home,
  LineChart,
  Search,
  Users,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { useAdminUsers } from "@/hooks/useAdmin";
import { useAdminOrgs, type AdminOrgRow } from "@/hooks/useAdminOrgs";
import { useTesterGrants } from "@/hooks/useTesterGrants";
import { isPaidTier, tierLabel } from "@/lib/tiers";
import { AnalyticsSummaryCard } from "@/components/admin/AnalyticsSummaryCard";
import { BehaviorAnalyticsCard } from "@/components/admin/BehaviorAnalyticsCard";
import { AdminOrgsPanel } from "@/components/admin/AdminOrgsPanel";
import { AdminUsersPanel, type UserFilter } from "@/components/admin/AdminUsersPanel";
import { AdminTestersPanel } from "@/components/admin/AdminTestersPanel";
import { AdminPalette } from "@/components/admin/AdminPalette";
import { PanelHeader, QueueRow, shortDate, StatTile, Tag } from "@/components/admin/ui";

type View = "overview" | "users" | "orgs" | "testers";

const TITLES: Record<View, [string, string]> = {
  overview: ["Overview", "Everything that needs an admin decision today"],
  users: ["Users", "Search, filter, and manage accounts in place"],
  orgs: ["Organizations", "Licenses, members, and shared credit pools"],
  testers: ["Beta testers", "Grant, revoke, and track tester access"],
};

const AdminConsole = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [view, setView] = useState<View>("overview");
  const [paletteOpen, setPaletteOpen] = useState(false);

  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState<UserFilter>("all");
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);
  const [selectedOrgId, setSelectedOrgId] = useState<string | null>(null);

  const orgsQuery = useAdminOrgs();
  const grantsQuery = useTesterGrants();
  const orgs = orgsQuery.data ?? [];
  const grants = grantsQuery.data ?? [];
  const pendingOrgs = orgs.filter((o) => o.status === "pending");
  const activeTesters = grants.filter((g) => !g.pending);
  const pendingInvites = grants.filter((g) => g.pending);

  const posthogUrl = import.meta.env.VITE_POSTHOG_DASHBOARD_URL || "https://us.posthog.com";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen(true);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const openUser = (id: string) => {
    setSelectedUserId(id);
    setView("users");
  };
  const openOrg = (id: string) => {
    setSelectedOrgId(id);
    setView("orgs");
  };

  const nav: { id: View; label: string; icon: typeof Home; count?: number; alert?: number }[] = [
    { id: "overview", label: "Overview", icon: Home, alert: pendingOrgs.length },
    { id: "users", label: "Users", icon: Users },
    { id: "orgs", label: "Organizations", icon: Building2, count: orgs.length },
    { id: "testers", label: "Beta testers", icon: Beaker, count: activeTesters.length },
  ];

  const [title, subtitle] = TITLES[view];

  return (
    <div className="flex h-screen overflow-hidden bg-muted/40">
      {/* Sidebar — collapses to an icon rail below md */}
      <nav className="flex w-16 shrink-0 flex-col border-r border-border bg-card md:w-[236px]">
        <div className="flex items-center gap-2.5 px-3 py-4 md:px-4">
          <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-primary text-[13px] font-bold text-primary-foreground">
            M
          </div>
          <div className="hidden md:block">
            <b className="text-[15px] tracking-tight">Msanii</b>
            <span className="block text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Admin
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-0.5 px-2">
          {nav.map((n) => (
            <NavButton
              key={n.id}
              icon={n.icon}
              label={n.label}
              active={view === n.id}
              onClick={() => setView(n.id)}
              trailing={
                n.alert ? (
                  <span className="ml-auto rounded-full bg-amber-100 px-1.5 font-mono text-[11px] font-semibold text-amber-700 dark:bg-amber-950 dark:text-amber-300">
                    {n.alert}
                  </span>
                ) : n.count !== undefined ? (
                  <span className="ml-auto font-mono text-[11px] text-muted-foreground">
                    {n.count}
                  </span>
                ) : null
              }
            />
          ))}
        </div>

        <div className="mt-4 flex flex-col gap-0.5 px-2">
          <div className="hidden px-2 pb-1 pt-2 text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground md:block">
            Shortcuts
          </div>
          <NavButton
            icon={Zap}
            label="Overrides"
            onClick={() => {
              setFilter("override");
              setView("users");
            }}
          />
          <NavButton
            icon={LineChart}
            label="Analytics ↗"
            onClick={() => window.open(posthogUrl, "_blank", "noopener,noreferrer")}
          />
          <NavButton icon={ArrowLeft} label="Back to app" onClick={() => navigate("/dashboard")} />
        </div>

        <div className="mt-auto flex items-center gap-2.5 border-t border-border/60 p-3">
          <div className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-primary text-[11px] font-semibold text-primary-foreground">
            {(user?.email ?? "?").slice(0, 2).toUpperCase()}
          </div>
          <div className="hidden min-w-0 md:block">
            <div className="truncate text-[12.5px] font-semibold leading-tight">{user?.email}</div>
            <div className="text-[11px] text-muted-foreground">Platform admin</div>
          </div>
        </div>
      </nav>

      {/* Main */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-border bg-card px-5 py-2.5">
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold tracking-tight">{title}</h1>
            <p className="truncate text-xs text-muted-foreground">{subtitle}</p>
          </div>
          <div className="flex-1" />
          <button
            onClick={() => setPaletteOpen(true)}
            className="hidden items-center gap-2 rounded-lg border border-border bg-muted/60 px-2.5 py-1.5 text-[13px] text-muted-foreground hover:border-primary/40 sm:flex sm:w-56"
          >
            <Search className="h-4 w-4 shrink-0" />
            <span className="truncate">Search users, orgs…</span>
            <span className="ml-auto rounded border border-border border-b-2 px-1 font-mono text-[10.5px]">
              ⌘K
            </span>
          </button>
          <Button onClick={() => setView("testers")}>Grant tester access</Button>
        </header>

        <div className="min-h-0 flex-1 overflow-auto p-5">
          <div className="mx-auto max-w-[1180px]">
            {view === "overview" && (
              <Overview
                orgs={orgs}
                pendingOrgs={pendingOrgs}
                activeTesterCount={activeTesters.length}
                pendingInviteCount={pendingInvites.length}
                onOpenOrg={openOrg}
                onOpenUser={openUser}
                onGoTo={setView}
              />
            )}
            {view === "users" && (
              <AdminUsersPanel
                search={search}
                onSearchChange={setSearch}
                page={page}
                onPageChange={setPage}
                filter={filter}
                onFilterChange={setFilter}
                selectedUserId={selectedUserId}
                onSelectUser={setSelectedUserId}
                onOpenOrg={openOrg}
              />
            )}
            {view === "orgs" && (
              <AdminOrgsPanel selectedOrgId={selectedOrgId} onSelectOrg={setSelectedOrgId} />
            )}
            {view === "testers" && <AdminTestersPanel onSelectUser={openUser} />}
          </div>
        </div>
      </div>

      <AdminPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        onGoUser={openUser}
        onGoOrg={openOrg}
        actions={[
          { label: "Grant beta tester access", run: () => setView("testers") },
          { label: "Review organizations", run: () => setView("orgs") },
          { label: "Browse all users", run: () => setView("users") },
          { label: "Back to overview", run: () => setView("overview") },
        ]}
      />
    </div>
  );
};

function NavButton({
  icon: Icon,
  label,
  active,
  onClick,
  trailing,
}: {
  icon: typeof Home;
  label: string;
  active?: boolean;
  onClick: () => void;
  trailing?: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-current={active || undefined}
      title={label}
      className={cn(
        "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-[13.5px] font-medium",
        active
          ? "bg-secondary font-semibold text-primary"
          : "text-muted-foreground hover:bg-muted/70",
      )}
    >
      <Icon className="h-4 w-4 shrink-0" />
      <span className="hidden truncate md:inline">{label}</span>
      <span className="hidden md:contents">{trailing}</span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------

function Overview({
  orgs,
  pendingOrgs,
  activeTesterCount,
  pendingInviteCount,
  onOpenOrg,
  onOpenUser,
  onGoTo,
}: {
  orgs: AdminOrgRow[];
  pendingOrgs: AdminOrgRow[];
  activeTesterCount: number;
  pendingInviteCount: number;
  onOpenOrg: (id: string) => void;
  onOpenUser: (id: string) => void;
  onGoTo: (v: View) => void;
}) {
  // Page 1 of the unfiltered listing — same query key the Users view uses, so
  // this warms that table's cache rather than adding a request. The auth admin
  // API returns newest-first and exposes no total count, which is why there is
  // no "total users" tile here.
  const recentQuery = useAdminUsers("", 1);
  const recent = (recentQuery.data?.users ?? []).slice(0, 5);
  const poolTotal = orgs.reduce((sum, o) => sum + o.bundleBalance + o.reserveBalance, 0);
  const lowestPools = [...orgs]
    .sort((a, b) => a.bundleBalance + a.reserveBalance - (b.bundleBalance + b.reserveBalance))
    .slice(0, 5);

  return (
    <>
      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile
          label="Organizations"
          value={orgs.length.toLocaleString()}
          hint={`${pendingOrgs.length} awaiting activation`}
        />
        <StatTile
          label="Credits in pools"
          value={poolTotal.toLocaleString()}
          hint={`across ${orgs.length} organization${orgs.length === 1 ? "" : "s"}`}
        />
        <StatTile
          label="Active testers"
          value={activeTesterCount.toLocaleString()}
          hint={`${pendingInviteCount} invite${pendingInviteCount === 1 ? "" : "s"} awaiting first sign-in`}
        />
        <StatTile
          label="Pending invites"
          value={pendingInviteCount.toLocaleString()}
          hint="activate on first verified sign-in"
        />
      </div>

      <Card className="mb-4 overflow-hidden p-0">
        <PanelHeader
          title="Needs your attention"
          subtitle="Licenses waiting on credits before they activate"
        >
          {pendingOrgs.length > 0 ? (
            <Tag tone="warn">{pendingOrgs.length} open</Tag>
          ) : (
            <Tag tone="ok">Clear</Tag>
          )}
        </PanelHeader>
        {pendingOrgs.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-muted-foreground">
            Nothing needs an admin decision right now.
          </div>
        )}
        {pendingOrgs.map((o) => (
          <QueueRow key={o.id}>
            <Tag tone="warn">Activation</Tag>
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13.5px] font-medium">{o.name ?? "Organization"}</div>
              <div className="text-xs text-muted-foreground">
                {o.cumulativePaidIn.toLocaleString()} of {o.activationFloor.toLocaleString()} credits
                paid in · {o.memberCount} member{o.memberCount === 1 ? "" : "s"}
              </div>
            </div>
            <Button size="sm" variant="outline" onClick={() => onOpenOrg(o.id)}>
              Review
            </Button>
          </QueueRow>
        ))}
      </Card>

      <div className="mb-4 grid gap-3 lg:grid-cols-2">
        <Card className="overflow-hidden p-0">
          <PanelHeader title="Newest users">
            <Button size="sm" variant="ghost" onClick={() => onGoTo("users")}>
              View all
            </Button>
          </PanelHeader>
          {recentQuery.isLoading && (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">Loading…</div>
          )}
          {recent.map((u) => (
            <QueueRow key={u.id} onClick={() => onOpenUser(u.id)}>
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13.5px] font-medium">
                  {u.name ?? u.email ?? u.id}
                </div>
                <div className="truncate text-xs text-muted-foreground">
                  {u.name ? u.email : `Joined ${shortDate(u.created_at)}`}
                </div>
              </div>
              <Tag tone={isPaidTier(u.tier) ? "paid" : "neutral"}>{tierLabel(u.tier)}</Tag>
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            </QueueRow>
          ))}
        </Card>

        <Card className="overflow-hidden p-0">
          <PanelHeader title="Lowest credit pools">
            <Button size="sm" variant="ghost" onClick={() => onGoTo("orgs")}>
              View all
            </Button>
          </PanelHeader>
          {lowestPools.length === 0 && (
            <div className="px-4 py-8 text-center text-sm text-muted-foreground">
              No organizations yet.
            </div>
          )}
          {lowestPools.map((o) => (
            <QueueRow key={o.id} onClick={() => onOpenOrg(o.id)}>
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13.5px] font-medium">{o.name ?? "Organization"}</div>
                <div className="text-xs text-muted-foreground">
                  {o.memberCount} member{o.memberCount === 1 ? "" : "s"} ·{" "}
                  {o.monthlyDispersalCredits.toLocaleString()}/mo
                </div>
              </div>
              <span className="font-mono font-semibold tabular-nums">
                {(o.bundleBalance + o.reserveBalance).toLocaleString()}
              </span>
            </QueueRow>
          ))}
        </Card>
      </div>

      <AnalyticsSummaryCard />
      <BehaviorAnalyticsCard />
    </>
  );
}

export default AdminConsole;
