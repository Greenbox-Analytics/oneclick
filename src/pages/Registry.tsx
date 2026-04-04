import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import {
  useWorks,
  useMyCollaborations,
  useMyInvites,
  useAcceptFromDashboard,
  useDeclineInvitation,
  type Work,
  type DashboardInvite,
} from "@/hooks/useRegistry";
import {
  useRegistryNotifications,
  type RegistryNotification,
} from "@/hooks/useRegistryNotifications";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Music,
  Shield,
  Bell,
  Search,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Check,
  X,
  Clock,
  Users,
  Loader2,
  FileText,
} from "lucide-react";

/* ------------------------------------------------------------------ */
/* Constants                                                          */
/* ------------------------------------------------------------------ */

const STATUS_COLORS: Record<string, string> = {
  draft: "bg-gray-500/15 text-gray-400",
  pending_approval: "bg-amber-500/15 text-amber-400",
  registered: "bg-emerald-500/15 text-emerald-400",
};

const STATUS_LABELS: Record<string, string> = {
  draft: "Draft",
  pending_approval: "Pending",
  registered: "Registered",
};

/* ------------------------------------------------------------------ */
/* Helper: time-ago                                                   */
/* ------------------------------------------------------------------ */

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(dateStr).toLocaleDateString();
}

/* ------------------------------------------------------------------ */
/* Sub-components                                                     */
/* ------------------------------------------------------------------ */

function SummaryCard({
  label,
  count,
  subtitle,
  colors,
}: {
  label: string;
  count: number;
  subtitle: string;
  colors: { bg: string; border: string; text: string };
}) {
  return (
    <Card className={`${colors.bg} border ${colors.border}`}>
      <CardContent className="p-4">
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className={`text-3xl font-bold mt-1 ${colors.text}`}>{count}</p>
        <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Action Required Tab                                                */
/* ------------------------------------------------------------------ */

function ActionRequiredTab({
  invites,
  isLoading,
}: {
  invites: DashboardInvite[];
  isLoading: boolean;
}) {
  const navigate = useNavigate();
  const acceptMut = useAcceptFromDashboard();
  const declineMut = useDeclineInvitation();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-primary" />
      </div>
    );
  }

  if (invites.length === 0) {
    return (
      <div className="text-center py-16">
        <Check className="w-12 h-12 text-emerald-400 mx-auto mb-3 opacity-60" />
        <p className="text-muted-foreground">You're all caught up — no pending invitations.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {invites.map((inv) => {
        const workTitle = inv.works_registry?.title || "Unknown work";
        const workId = inv.works_registry?.id;
        const busy = acceptMut.isPending || declineMut.isPending;

        return (
          <Card key={inv.id} className="border-amber-500/20 bg-amber-500/5">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <div className="mt-0.5 rounded-full bg-amber-500/15 p-2">
                  <Bell className="w-4 h-4 text-amber-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground">
                    You've been invited as <span className="text-amber-400">{inv.role}</span> on{" "}
                    <span className="font-semibold">{workTitle}</span>
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Invited by {inv.name || "someone"} &middot; {timeAgo(inv.invited_at)}
                  </p>

                  <div className="flex items-center gap-3 mt-3">
                    <Button
                      size="sm"
                      onClick={() => acceptMut.mutate(inv.id)}
                      disabled={busy}
                    >
                      <Check className="w-3.5 h-3.5 mr-1" /> Accept
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => declineMut.mutate(inv.id)}
                      disabled={busy}
                    >
                      <X className="w-3.5 h-3.5 mr-1" /> Decline
                    </Button>
                    {workId && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => navigate(`/tools/registry/${workId}`)}
                      >
                        <ExternalLink className="w-3.5 h-3.5 mr-1" /> View Work
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* My Works Tab                                                       */
/* ------------------------------------------------------------------ */

function MyWorksTab({
  works,
  isLoading,
  searchQuery,
}: {
  works: Work[];
  isLoading: boolean;
  searchQuery: string;
}) {
  const navigate = useNavigate();
  const [collapsedYears, setCollapsedYears] = useState<Set<number>>(new Set());

  const filtered = useMemo(() => {
    if (!searchQuery.trim()) return works;
    const q = searchQuery.toLowerCase();
    return works.filter(
      (w) =>
        w.title.toLowerCase().includes(q) ||
        (w.isrc && w.isrc.toLowerCase().includes(q)) ||
        (w.iswc && w.iswc.toLowerCase().includes(q))
    );
  }, [works, searchQuery]);

  // Group by year
  const groupedByYear = useMemo(() => {
    const map = new Map<number, Work[]>();
    for (const w of filtered) {
      const year = new Date(w.created_at).getFullYear();
      if (!map.has(year)) map.set(year, []);
      map.get(year)!.push(w);
    }
    return Array.from(map.entries()).sort(([a], [b]) => b - a);
  }, [filtered]);

  const toggleYear = (year: number) => {
    setCollapsedYears((prev) => {
      const next = new Set(prev);
      if (next.has(year)) next.delete(year);
      else next.add(year);
      return next;
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-primary" />
      </div>
    );
  }

  if (filtered.length === 0) {
    return (
      <div className="text-center py-16">
        <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-3 opacity-50" />
        <p className="text-muted-foreground">
          {searchQuery ? "No works match your search." : "No works registered yet."}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {groupedByYear.map(([year, yearWorks]) => {
        const collapsed = collapsedYears.has(year);

        // Sub-group by project
        const byProject = new Map<string, Work[]>();
        for (const w of yearWorks) {
          const key = w.project_id || "no-project";
          if (!byProject.has(key)) byProject.set(key, []);
          byProject.get(key)!.push(w);
        }

        return (
          <div key={year}>
            <button
              onClick={() => toggleYear(year)}
              className="flex items-center gap-2 mb-2 text-sm font-semibold text-muted-foreground hover:text-foreground transition-colors"
            >
              {collapsed ? (
                <ChevronRight className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
              {year}
              <span className="text-xs font-normal">({yearWorks.length} works)</span>
            </button>

            {!collapsed && (
              <div className="space-y-3 ml-6">
                {Array.from(byProject.entries()).map(([projId, projWorks]) => (
                  <div key={projId}>
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                        Project
                      </span>
                      {projId !== "no-project" && (
                        <Button
                          variant="link"
                          size="sm"
                          className="h-auto p-0 text-xs text-primary"
                          onClick={() => navigate(`/projects/${projId}`)}
                        >
                          Open project <ExternalLink className="w-3 h-3 ml-1" />
                        </Button>
                      )}
                    </div>

                    <div className="space-y-1.5">
                      {projWorks.map((w) => (
                        <Card
                          key={w.id}
                          className="hover:border-primary/50 transition-colors cursor-pointer"
                          onClick={() => navigate(`/tools/registry/${w.id}`)}
                        >
                          <CardContent className="p-3 flex items-center justify-between">
                            <div className="flex items-center gap-3 min-w-0">
                              <Music className="w-4 h-4 text-muted-foreground shrink-0" />
                              <span className="text-sm font-medium truncate">{w.title}</span>
                              <Badge
                                className={
                                  STATUS_COLORS[w.status] || "bg-gray-500/15 text-gray-400"
                                }
                              >
                                {STATUS_LABELS[w.status] || w.status}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-4 text-xs text-muted-foreground shrink-0">
                              {w.isrc && <span>ISRC: {w.isrc}</span>}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Collaborations Tab                                                 */
/* ------------------------------------------------------------------ */

function CollaborationsTab({
  works,
  isLoading,
}: {
  works: Work[];
  isLoading: boolean;
}) {
  const navigate = useNavigate();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-primary" />
      </div>
    );
  }

  if (works.length === 0) {
    return (
      <div className="text-center py-16">
        <Users className="w-12 h-12 text-muted-foreground mx-auto mb-3 opacity-50" />
        <p className="text-muted-foreground">No collaborations yet.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      {works.map((w) => (
        <Card
          key={w.id}
          className="hover:border-primary/50 transition-colors cursor-pointer"
          onClick={() => navigate(`/tools/registry/${w.id}`)}
        >
          <CardContent className="p-4">
            <div className="flex items-start justify-between">
              <div className="min-w-0">
                <p className="text-sm font-medium truncate">{w.title}</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {w.work_type.replace("_", " ").toUpperCase()}
                </p>
              </div>
              <Badge className={STATUS_COLORS[w.status] || "bg-gray-500/15 text-gray-400"}>
                {STATUS_LABELS[w.status] || w.status}
              </Badge>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* All Activity Tab                                                   */
/* ------------------------------------------------------------------ */

function AllActivityTab({
  notifications,
  isLoading,
}: {
  notifications: RegistryNotification[];
  isLoading: boolean;
}) {
  const navigate = useNavigate();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-primary" />
      </div>
    );
  }

  if (notifications.length === 0) {
    return (
      <div className="text-center py-16">
        <Clock className="w-12 h-12 text-muted-foreground mx-auto mb-3 opacity-50" />
        <p className="text-muted-foreground">No activity yet.</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {notifications.map((n) => (
        <Card key={n.id} className="hover:border-primary/30 transition-colors">
          <CardContent className="p-3 flex items-start gap-3">
            <div className="mt-0.5 rounded-full bg-muted p-1.5">
              <Clock className="w-3.5 h-3.5 text-muted-foreground" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">{n.title}</span>
                <span className="text-xs text-muted-foreground">{timeAgo(n.created_at)}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">{n.message}</p>
            </div>
            {n.work_id && (
              <Button
                size="sm"
                variant="ghost"
                className="shrink-0"
                onClick={() => navigate(`/tools/registry/${n.work_id}`)}
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </Button>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main Page Component                                                */
/* ------------------------------------------------------------------ */

const Registry = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");

  // Data hooks
  const worksQuery = useWorks();
  const collabQuery = useMyCollaborations();
  const invitesQuery = useMyInvites();
  const notificationsQuery = useRegistryNotifications();

  const works = worksQuery.data || [];
  const collabs = collabQuery.data || [];
  const invites = invitesQuery.data || [];
  const notifications = notificationsQuery.data || [];

  // Summary counts
  const myWorksCount = works.length;
  const registeredCount = works.filter((w) => w.status === "registered").length;
  const pendingCount = works.filter((w) => w.status === "pending_approval").length;
  const collabCount = collabs.length;

  const uniqueProjects = new Set(works.map((w) => w.project_id).filter(Boolean));

  // Default tab: action-required if invites exist, otherwise my-works
  const defaultTab = invites.length > 0 ? "action-required" : "my-works";

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div
            className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/dashboard")}
          >
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/tools")}>
            Back to Tools
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Page Title */}
        <div className="mb-6">
          <h2 className="text-3xl font-bold text-foreground flex items-center gap-3">
            <Shield className="w-8 h-8 text-primary" /> Rights Registry
          </h2>
          <p className="text-muted-foreground mt-1">
            Track your ownership, stakes, and collaborations across all projects
          </p>
        </div>

        {/* Search */}
        <div className="relative max-w-md mb-6">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search works by title, ISRC, or ISWC..."
            className="pl-9"
          />
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
          <SummaryCard
            label="My Works"
            count={myWorksCount}
            subtitle={`Across ${uniqueProjects.size} project${uniqueProjects.size !== 1 ? "s" : ""}`}
            colors={{
              bg: "bg-purple-500/10",
              border: "border-purple-500/20",
              text: "text-purple-400",
            }}
          />
          <SummaryCard
            label="Registered"
            count={registeredCount}
            subtitle="Fully confirmed"
            colors={{
              bg: "bg-emerald-500/10",
              border: "border-emerald-500/20",
              text: "text-emerald-400",
            }}
          />
          <SummaryCard
            label="Pending"
            count={pendingCount}
            subtitle="Awaiting approval"
            colors={{
              bg: "bg-amber-500/10",
              border: "border-amber-500/20",
              text: "text-amber-400",
            }}
          />
          <SummaryCard
            label="Collaborations"
            count={collabCount}
            subtitle="Works you contribute to"
            colors={{
              bg: "bg-blue-500/10",
              border: "border-blue-500/20",
              text: "text-blue-400",
            }}
          />
        </div>

        {/* Tabs */}
        <Tabs defaultValue={defaultTab}>
          <TabsList className="mb-4">
            <TabsTrigger value="action-required" className="gap-1.5">
              <Bell className="w-3.5 h-3.5" />
              Action Required
              {invites.length > 0 && (
                <span className="ml-1 inline-flex items-center justify-center w-5 h-5 text-[10px] font-bold rounded-full bg-amber-500 text-white">
                  {invites.length}
                </span>
              )}
            </TabsTrigger>
            <TabsTrigger value="my-works" className="gap-1.5">
              <Music className="w-3.5 h-3.5" />
              My Works
            </TabsTrigger>
            <TabsTrigger value="collaborations" className="gap-1.5">
              <Users className="w-3.5 h-3.5" />
              Collaborations
            </TabsTrigger>
            <TabsTrigger value="activity" className="gap-1.5">
              <Clock className="w-3.5 h-3.5" />
              All Activity
            </TabsTrigger>
          </TabsList>

          <TabsContent value="action-required">
            <ActionRequiredTab
              invites={invites}
              isLoading={invitesQuery.isLoading}
            />
          </TabsContent>

          <TabsContent value="my-works">
            <MyWorksTab
              works={works}
              isLoading={worksQuery.isLoading}
              searchQuery={searchQuery}
            />
          </TabsContent>

          <TabsContent value="collaborations">
            <CollaborationsTab
              works={collabs}
              isLoading={collabQuery.isLoading}
            />
          </TabsContent>

          <TabsContent value="activity">
            <AllActivityTab
              notifications={notifications}
              isLoading={notificationsQuery.isLoading}
            />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Registry;
