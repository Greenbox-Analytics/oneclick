import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Search,
  Plus,
  Layers,
  CheckCircle,
  Clock,
  AlertTriangle,
  Users,
  Music,
  ChevronRight,
  ChevronDown,
  Folder,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useWorks, useMyCollaborations, type Work } from "@/hooks/useRegistry";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { cn } from "@/lib/utils";
import { StatCard } from "./StatCard";
import { Segmented } from "./Segmented";
import { WorkRow, type DashboardWork } from "./WorkRow";
import { RegistryAvatar } from "./RegistryAvatar";
import AddWorkDialog from "@/components/project/AddWorkDialog";

interface ProjectInfo {
  id: string;
  name: string;
  artist_id: string;
  created_at: string;
}

interface ArtistInfo {
  id: string;
  name: string;
}

function computeIssues(work: Work): string[] {
  const issues: string[] = [];
  if (work.release_date && !work.isrc) issues.push("Missing ISRC");
  return issues;
}

function enrichWorks(
  works: Work[],
  ownership: "owner" | "collaborator",
  projectsById: Record<string, ProjectInfo>,
  artistsById: Record<string, ArtistInfo>
): DashboardWork[] {
  return works.map((w) => {
    const project = projectsById[w.project_id || ""];
    const artist = project ? artistsById[project.artist_id] : artistsById[w.artist_id];
    return {
      ...w,
      released: !!w.release_date,
      artist: artist ? { id: artist.id, name: artist.name } : undefined,
      project: project ? { id: project.id, name: project.name } : undefined,
      issues: computeIssues(w),
      ownership,
      canEdit: ownership === "owner",
    };
  });
}

export function RegistryDashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();

  // Data
  const worksQuery = useWorks();
  const collabsQuery = useMyCollaborations();
  // Stabilize array refs so memoized groupings below don't recompute every render.
  const works = useMemo(() => worksQuery.data || [], [worksQuery.data]);
  const collabs = useMemo(() => collabsQuery.data || [], [collabsQuery.data]);

  // Join: load projects + artists for everything we'll display
  const allProjectIds = useMemo(() => {
    const ids = new Set<string>();
    [...works, ...collabs].forEach((w) => {
      if (w.project_id) ids.add(w.project_id);
    });
    return Array.from(ids);
  }, [works, collabs]);

  const projectsQuery = useQuery<ProjectInfo[]>({
    queryKey: ["registry-dashboard-projects", user?.id, allProjectIds.join(",")],
    queryFn: async () => {
      if (allProjectIds.length === 0) return [];
      const { data, error } = await supabase
        .from("projects")
        .select("id, name, artist_id, created_at")
        .in("id", allProjectIds);
      if (error) throw error;
      return (data || []) as ProjectInfo[];
    },
    enabled: !!user?.id && allProjectIds.length > 0,
  });

  const projectsById = useMemo(() => {
    const map: Record<string, ProjectInfo> = {};
    for (const p of projectsQuery.data || []) map[p.id] = p;
    return map;
  }, [projectsQuery.data]);

  const allArtistIds = useMemo(() => {
    const ids = new Set<string>();
    [...works, ...collabs].forEach((w) => {
      if (w.artist_id) ids.add(w.artist_id);
    });
    Object.values(projectsById).forEach((p) => ids.add(p.artist_id));
    return Array.from(ids);
  }, [works, collabs, projectsById]);

  const artistsQuery = useQuery<ArtistInfo[]>({
    queryKey: ["registry-dashboard-artists", user?.id, allArtistIds.join(",")],
    queryFn: async () => {
      if (allArtistIds.length === 0) return [];
      const { data, error } = await supabase
        .from("artists")
        .select("id, name")
        .in("id", allArtistIds);
      if (error) throw error;
      return (data || []) as ArtistInfo[];
    },
    enabled: !!user?.id && allArtistIds.length > 0,
  });

  const artistsById = useMemo(() => {
    const map: Record<string, ArtistInfo> = {};
    for (const a of artistsQuery.data || []) map[a.id] = a;
    return map;
  }, [artistsQuery.data]);

  const myWorks = useMemo(
    () => enrichWorks(works, "owner", projectsById, artistsById),
    [works, projectsById, artistsById]
  );
  const sharedWorks = useMemo(
    () => enrichWorks(collabs, "collaborator", projectsById, artistsById),
    [collabs, projectsById, artistsById]
  );

  // UI state
  const [view, setView] = useState<"mine" | "shared">("mine");
  const [statFilter, setStatFilter] = useState<
    null | "released" | "unreleased" | "attention"
  >(null);
  const [sortBy, setSortBy] = useState<"added" | "title" | "release">("added");
  const [searchQuery, setSearchQuery] = useState("");
  const [closedArtists, setClosedArtists] = useState<Record<string, boolean>>({});
  const [closedProjects, setClosedProjects] = useState<Record<string, boolean>>({});
  const [addWorkOpen, setAddWorkOpen] = useState(false);

  const q = searchQuery.trim().toLowerCase();

  const match = (w: DashboardWork) => {
    if (!q) return true;
    return (
      w.title.toLowerCase().includes(q) ||
      (w.isrc && w.isrc.toLowerCase().includes(q)) ||
      (w.artist?.name || "").toLowerCase().includes(q) ||
      (w.project?.name || "").toLowerCase().includes(q)
    );
  };

  const matchStat = (w: DashboardWork) => {
    if (statFilter === "released") return w.released;
    if (statFilter === "unreleased") return !w.released;
    if (statFilter === "attention") return (w.issues?.length || 0) > 0;
    return true;
  };

  const sortWorks = (list: DashboardWork[]) => {
    const copy = [...list];
    if (sortBy === "title") copy.sort((a, b) => a.title.localeCompare(b.title));
    else if (sortBy === "release")
      copy.sort((a, b) => (b.release_date || "9999").localeCompare(a.release_date || "9999"));
    else copy.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
    return copy;
  };

  // Counts off full sets so stat cards stay stable
  const totalWorks = myWorks.length + sharedWorks.length;
  const released = [...myWorks, ...sharedWorks].filter((w) => w.released).length;
  const unreleased = totalWorks - released;
  const attention = [...myWorks, ...sharedWorks].filter((w) => (w.issues?.length || 0) > 0).length;
  const sharedCount = sharedWorks.length;

  // Filtered lists for current view
  const minFiltered = sortWorks(myWorks.filter((w) => match(w) && matchStat(w)));
  const sharedFiltered = sortWorks(sharedWorks.filter((w) => match(w) && matchStat(w)));

  // Group My Works by artist → year → project
  const grouped = useMemo(() => {
    const byArtist: Record<
      string,
      { artist: ArtistInfo; projects: Record<string, DashboardWork[]> }
    > = {};
    for (const w of minFiltered) {
      const aid = w.artist?.id || w.artist_id || "unknown";
      const pid = w.project?.id || "no-project";
      if (!byArtist[aid]) {
        byArtist[aid] = {
          artist: w.artist || { id: aid, name: "Unknown artist" },
          projects: {},
        };
      }
      if (!byArtist[aid].projects[pid]) byArtist[aid].projects[pid] = [];
      byArtist[aid].projects[pid].push(w);
    }
    return byArtist;
  }, [minFiltered]);

  const forceOpen = !!q || !!statFilter;
  const artistIds = Object.keys(grouped);
  const artistOpenDefault = artistIds.length <= 2;
  const isArtistOpen = (aid: string) =>
    forceOpen ? true : aid in closedArtists ? !closedArtists[aid] : artistOpenDefault;
  const isProjectOpen = (pid: string, count: number) => {
    if (forceOpen) return true;
    const def = count <= 2;
    return pid in closedProjects ? !closedProjects[pid] : def;
  };

  const yearOf = (pid: string) => {
    const p = projectsById[pid];
    return p ? new Date(p.created_at).getFullYear() : new Date().getFullYear();
  };

  const isLoading = worksQuery.isLoading || collabsQuery.isLoading;

  return (
    <div>
      {/* Title block */}
      <div className="mb-6">
        <div className="text-[11px] tracking-widest uppercase font-semibold text-primary mb-2">
          Rights Registry
        </div>
        <h2 className="text-3xl font-bold tracking-tight">Your rights, in one place</h2>
        <p className="text-muted-foreground mt-1">
          Every work you own or contribute to — released and unreleased — with full
          traceability of contracts and ownership.
        </p>
      </div>

      {/* Stat cards */}
      <div
        data-walkthrough="registry-summary"
        className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3 mb-6"
      >
        <StatCard
          icon={Layers}
          num={totalWorks}
          label="Total works involved"
          tint={{ bg: "bg-primary/15", fg: "text-primary" }}
          active={!statFilter}
          onClick={() => setStatFilter(null)}
        />
        <StatCard
          icon={CheckCircle}
          num={released}
          label="Released tracks"
          tint={{ bg: "bg-emerald-500/15", fg: "text-emerald-400" }}
          active={statFilter === "released"}
          onClick={() => setStatFilter(statFilter === "released" ? null : "released")}
        />
        <StatCard
          icon={Clock}
          num={unreleased}
          label="Unreleased tracks"
          tint={{ bg: "bg-amber-500/15", fg: "text-amber-400" }}
          active={statFilter === "unreleased"}
          onClick={() => setStatFilter(statFilter === "unreleased" ? null : "unreleased")}
        />
        <StatCard
          icon={AlertTriangle}
          num={attention}
          label="Need attention"
          tint={{ bg: "bg-amber-500/15", fg: "text-amber-400" }}
          active={statFilter === "attention"}
          onClick={() => setStatFilter(statFilter === "attention" ? null : "attention")}
        />
        <StatCard
          icon={Users}
          num={sharedCount}
          label="Shared with you"
          tint={{ bg: "bg-blue-500/15", fg: "text-blue-400" }}
          active={view === "shared" && !statFilter}
          onClick={() => {
            setView("shared");
            setStatFilter(null);
          }}
        />
      </div>

      {/* Toolbar */}
      <div
        data-walkthrough="registry-tabs"
        className="flex items-center gap-3 flex-wrap mb-6"
      >
        <Segmented
          value={view}
          onChange={(v) => setView(v)}
          options={[
            { value: "mine", label: "My Works", icon: Music, count: myWorks.length },
            { value: "shared", label: "Shared with Me", icon: Users, count: sharedWorks.length },
          ]}
        />
        <div className="relative flex-1 max-w-md min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            data-walkthrough="registry-search"
            placeholder="Search works, ISRC, artist, or project…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <Select value={sortBy} onValueChange={(v) => setSortBy(v as typeof sortBy)}>
          <SelectTrigger className="w-[170px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="added">Recently added</SelectItem>
            <SelectItem value="title">Title A–Z</SelectItem>
            <SelectItem value="release">Release date</SelectItem>
          </SelectContent>
        </Select>
        <Button className="ml-auto" onClick={() => setAddWorkOpen(true)}>
          <Plus className="w-4 h-4 mr-2" /> Add work
        </Button>
      </div>

      {/* Active filter notice */}
      {statFilter && (
        <div className="flex items-center gap-2 mb-4">
          <Badge variant="outline" className="font-semibold">
            {statFilter === "released"
              ? "Showing released tracks"
              : statFilter === "unreleased"
                ? "Showing unreleased tracks"
                : "Showing works that need attention"}
          </Badge>
          <button
            type="button"
            className="text-xs font-semibold text-primary hover:underline"
            onClick={() => setStatFilter(null)}
          >
            Clear filter
          </button>
        </div>
      )}

      {/* Body */}
      <div data-walkthrough="registry-action">
        {view === "mine" ? (
          <MyWorksList
            grouped={grouped}
            artistIds={artistIds}
            isLoading={isLoading}
            isEmpty={minFiltered.length === 0}
            yearOf={yearOf}
            isArtistOpen={isArtistOpen}
            isProjectOpen={isProjectOpen}
            toggleArtist={(aid) =>
              setClosedArtists((c) => ({
                ...c,
                [aid]: aid in c ? !c[aid] : artistOpenDefault,
              }))
            }
            toggleProject={(pid, count) =>
              setClosedProjects((c) => ({ ...c, [pid]: pid in c ? !c[pid] : count <= 2 }))
            }
            sortWorks={sortWorks}
            onOpenWork={(workId) => navigate(`/tools/registry/${workId}`)}
            onOpenProject={(projectId) => navigate(`/projects/${projectId}`)}
            onAddWork={() => setAddWorkOpen(true)}
            searchActive={!!q || !!statFilter}
          />
        ) : (
          <SharedList
            works={sharedFiltered}
            isLoading={isLoading}
            onOpenWork={(workId) => navigate(`/tools/registry/${workId}`)}
            onOpenProject={(projectId) => navigate(`/projects/${projectId}`)}
            searchActive={!!q || !!statFilter}
          />
        )}
      </div>

      {/* Add work — registry-launched wizard (destination step prepended) */}
      {addWorkOpen && (
        <AddWorkDialog
          open={addWorkOpen}
          onOpenChange={setAddWorkOpen}
          projectId=""
          artistId=""
        />
      )}
    </div>
  );
}

interface MyWorksListProps {
  grouped: Record<string, { artist: ArtistInfo; projects: Record<string, DashboardWork[]> }>;
  artistIds: string[];
  isLoading: boolean;
  isEmpty: boolean;
  yearOf: (pid: string) => number;
  isArtistOpen: (aid: string) => boolean;
  isProjectOpen: (pid: string, count: number) => boolean;
  toggleArtist: (aid: string) => void;
  toggleProject: (pid: string, count: number) => void;
  sortWorks: (list: DashboardWork[]) => DashboardWork[];
  onOpenWork: (workId: string) => void;
  onOpenProject: (projectId: string) => void;
  onAddWork: () => void;
  searchActive: boolean;
}

function MyWorksList({
  grouped,
  artistIds,
  isLoading,
  isEmpty,
  yearOf,
  isArtistOpen,
  isProjectOpen,
  toggleArtist,
  toggleProject,
  sortWorks,
  onOpenWork,
  onOpenProject,
  onAddWork,
  searchActive,
}: MyWorksListProps) {
  if (isLoading) {
    return (
      <div className="py-16 text-center text-muted-foreground text-sm">Loading…</div>
    );
  }
  if (isEmpty) {
    return (
      <div className="rounded-lg border border-dashed bg-card p-12 text-center">
        <Music className="w-10 h-10 text-muted-foreground/40 mx-auto mb-3" />
        <p className="text-muted-foreground mb-4">
          {searchActive ? "No works match your filters." : "You don't own any works yet."}
        </p>
        {!searchActive && (
          <Button onClick={onAddWork}>
            <Plus className="w-4 h-4 mr-2" /> Add your first work
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {artistIds.map((aid) => {
        const group = grouped[aid];
        const projects = group.projects;
        const pids = Object.keys(projects).sort((a, b) => yearOf(b) - yearOf(a));
        const artistWorks = pids.reduce((n, pid) => n + projects[pid].length, 0);
        const open = isArtistOpen(aid);

        // years within this artist, desc
        const byYear: Record<number, string[]> = {};
        for (const pid of pids) {
          const y = yearOf(pid);
          if (!byYear[y]) byYear[y] = [];
          byYear[y].push(pid);
        }
        const years = Object.keys(byYear)
          .map(Number)
          .sort((a, b) => b - a);

        return (
          <div
            key={aid}
            className="rounded-xl border bg-card overflow-hidden"
          >
            <button
              type="button"
              onClick={() => toggleArtist(aid)}
              className="w-full flex items-center gap-3 px-4 py-3 hover:bg-muted/40 text-left"
            >
              {open ? (
                <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />
              ) : (
                <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
              )}
              <RegistryAvatar name={group.artist.name} size={26} />
              <span className="text-sm font-bold tracking-tight truncate">
                {group.artist.name}
              </span>
              <Badge variant="outline" className="text-[11px] font-semibold">
                {artistWorks}
              </Badge>
              <span className="ml-auto text-xs text-muted-foreground">
                {pids.length} {pids.length === 1 ? "project" : "projects"}
              </span>
            </button>
            {open && (
              <div className="flex flex-col gap-2 px-3 pb-4">
                {years.map((year) => (
                  <div key={year}>
                    <div className="flex items-center gap-2 px-2 pt-2 pb-1">
                      <span className="text-sm font-bold tabular-nums">{year}</span>
                      <Badge variant="outline" className="text-[10px] font-semibold">
                        {byYear[year].length}{" "}
                        {byYear[year].length === 1 ? "project" : "projects"}
                      </Badge>
                      <span className="flex-1 h-px bg-border" />
                    </div>
                    {byYear[year].map((pid) => {
                      const list = sortWorks(projects[pid]);
                      const pOpen = isProjectOpen(pid, pids.length);
                      const rel = list.filter((w) => w.released).length;
                      const unrel = list.length - rel;
                      const projectName = list[0]?.project?.name || "Unknown project";
                      return (
                        <div key={pid} className="border-t first:border-t-0 border-border/40">
                          <button
                            type="button"
                            onClick={() => toggleProject(pid, pids.length)}
                            className="w-full flex items-center gap-2 px-2 py-2 hover:bg-muted/40 rounded-md text-left"
                          >
                            {pOpen ? (
                              <ChevronDown className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                            ) : (
                              <ChevronRight className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                            )}
                            <Folder className="w-3.5 h-3.5 text-muted-foreground" />
                            <span className="text-xs font-semibold truncate">{projectName}</span>
                            <Badge variant="outline" className="text-[10px] font-semibold">
                              {list.length}
                            </Badge>
                            <span className="ml-auto text-xs text-muted-foreground inline-flex items-center gap-2">
                              {rel > 0 && <span>{rel} released</span>}
                              {rel > 0 && unrel > 0 && (
                                <span className="w-1 h-1 rounded-full bg-border" />
                              )}
                              {unrel > 0 && <span>{unrel} unreleased</span>}
                            </span>
                          </button>
                          {pOpen && (
                            <div className="flex flex-col gap-2 pl-7 pr-1 pb-2">
                              {list.map((w) => (
                                <WorkRow
                                  key={w.id}
                                  work={w}
                                  onOpen={onOpenWork}
                                  onOpenProject={onOpenProject}
                                />
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
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

interface SharedListProps {
  works: DashboardWork[];
  isLoading: boolean;
  onOpenWork: (workId: string) => void;
  onOpenProject: (projectId: string) => void;
  searchActive: boolean;
}

function SharedList({ works, isLoading, onOpenWork, onOpenProject, searchActive }: SharedListProps) {
  if (isLoading) {
    return (
      <div className="py-16 text-center text-muted-foreground text-sm">Loading…</div>
    );
  }
  if (works.length === 0) {
    return (
      <div className="rounded-lg border border-dashed bg-card p-12 text-center">
        <Users className="w-10 h-10 text-muted-foreground/40 mx-auto mb-3" />
        <p className="text-muted-foreground">
          {searchActive
            ? "No shared works match your filters."
            : "Nothing has been shared with you yet."}
        </p>
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      {works.map((w) => (
        <WorkRow key={w.id} work={w} onOpen={onOpenWork} onOpenProject={onOpenProject} />
      ))}
    </div>
  );
}
