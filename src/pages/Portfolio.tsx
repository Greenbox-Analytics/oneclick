import { useState, useEffect, useMemo, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";
import { usePortfolioData } from "@/hooks/usePortfolioData";
import type { ProjectCard, SharedProjectCard } from "@/hooks/usePortfolioData";
import { ProjectFormDialog } from "@/components/ProjectFormDialog";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Music,
  ArrowLeft,
  Folder,
  Search,
  X,
  User,
  LogOut,
  ArrowUpDown,
  Plus,
  Users,
  FileText,
  Calendar,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

const ROLE_COLORS: Record<string, string> = {
  owner: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  admin: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  editor: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  viewer: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
};

const Portfolio = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { user } = useAuth();
  const { toast } = useToast();

  // Profile for header
  const [profile, setProfile] = useState<{ full_name: string | null; avatar_url: string | null } | null>(null);

  useEffect(() => {
    const fetchProfile = async () => {
      if (!user) return;
      const { data } = await supabase
        .from("profiles")
        .select("full_name, avatar_url")
        .eq("id", user.id)
        .single();
      if (data) setProfile(data);
    };
    fetchProfile();
  }, [user]);

  const getInitials = () => {
    if (profile?.full_name) {
      return profile.full_name.trim().split(/\s+/).map((n) => n[0]).join("").toUpperCase().slice(0, 2);
    }
    return user?.email?.substring(0, 2).toUpperCase() || "U";
  };

  // Filter state
  const initialArtistId = searchParams.get("artist");
  const [selectedArtistIds, setSelectedArtistIds] = useState<string[]>(
    initialArtistId ? [initialArtistId] : []
  );
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sortOrder, setSortOrder] = useState<"alpha" | "newest" | "oldest">("alpha");

  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.portfolio, {
    onComplete: () => markToolCompleted("portfolio"),
  });

  useEffect(() => {
    if (!onboardingLoading && !statuses.portfolio && walkthrough.phase === "idle") {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.portfolio]);

  // Artist search state
  const [artistSearchInput, setArtistSearchInput] = useState("");
  const [showArtistSuggestions, setShowArtistSuggestions] = useState(false);
  const artistSearchRef = useRef<HTMLDivElement>(null);

  // Project create dialog
  const [projectDialogOpen, setProjectDialogOpen] = useState(false);
  const [defaultArtistIdForProject, setDefaultArtistIdForProject] = useState<string | undefined>();

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(searchInput), 300);
    return () => clearTimeout(timer);
  }, [searchInput]);

  // Close artist suggestions on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (artistSearchRef.current && !artistSearchRef.current.contains(e.target as Node)) {
        setShowArtistSuggestions(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const { myProjects, sharedProjects, allArtists, isLoading, refetchProjects } = usePortfolioData({
    selectedArtistIds,
    searchQuery: debouncedSearch,
    sortOrder,
  });

  const hasActiveFilters = selectedArtistIds.length > 0 || debouncedSearch;

  const clearFilters = () => {
    setSelectedArtistIds([]);
    setSearchInput("");
    setDebouncedSearch("");
    setArtistSearchInput("");
  };

  const handleAddProject = (artistId?: string) => {
    setDefaultArtistIdForProject(artistId);
    setProjectDialogOpen(true);
  };

  const handleSaveProject = async (data: { name: string; description: string; artist_id: string }) => {
    const { data: existing } = await supabase
      .from("projects")
      .select("id")
      .eq("artist_id", data.artist_id)
      .ilike("name", data.name.trim())
      .limit(1);
    if (existing && existing.length > 0) {
      toast({
        title: "Duplicate project name",
        description: `A project named "${data.name.trim()}" already exists for this artist.`,
        className: "bg-white text-black border border-border",
      });
      return;
    }
    const { error } = await supabase
      .from("projects")
      .insert({ name: data.name, description: data.description || null, artist_id: data.artist_id });
    if (error) throw error;
    toast({ title: "Success", description: "Project created" });
    refetchProjects();
  };

  // Artist search suggestions
  const artistSuggestions = useMemo(() => {
    if (!artistSearchInput.trim()) return [];
    const query = artistSearchInput.toLowerCase();
    return allArtists
      .filter(
        (a) => a.name.toLowerCase().includes(query) && !selectedArtistIds.includes(a.id)
      )
      .slice(0, 5);
  }, [artistSearchInput, allArtists, selectedArtistIds]);

  const addArtistFilter = (artistId: string) => {
    setSelectedArtistIds((prev) => [...prev, artistId]);
    setArtistSearchInput("");
    setShowArtistSuggestions(false);
  };

  const removeArtistFilter = (artistId: string) => {
    setSelectedArtistIds((prev) => prev.filter((id) => id !== artistId));
  };

  // Group my projects by year → artist for display
  const projectsByYearAndArtist = useMemo(() => {
    // First group by year
    const yearMap = new Map<number, Map<string, ProjectCard[]>>();
    for (const p of myProjects) {
      const year = new Date(p.created_at).getFullYear();
      if (!yearMap.has(year)) yearMap.set(year, new Map());
      const artistMap = yearMap.get(year)!;
      const key = p.artist_name;
      if (!artistMap.has(key)) artistMap.set(key, []);
      artistMap.get(key)!.push(p);
    }
    // Sort years descending, artists alphabetically within each year
    return Array.from(yearMap.entries())
      .sort(([a], [b]) => b - a)
      .map(([year, artistMap]) => ({
        year,
        artists: Array.from(artistMap.entries())
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([name, projects]) => ({ name, projects })),
        totalProjects: Array.from(artistMap.values()).reduce((sum, arr) => sum + arr.length, 0),
      }));
  }, [myProjects]);

  // Group shared projects by year → artist
  const sharedByYearAndArtist = useMemo(() => {
    const yearMap = new Map<number, Map<string, SharedProjectCard[]>>();
    for (const p of sharedProjects) {
      const year = new Date(p.created_at).getFullYear();
      if (!yearMap.has(year)) yearMap.set(year, new Map());
      const artistMap = yearMap.get(year)!;
      const key = p.artist_name;
      if (!artistMap.has(key)) artistMap.set(key, []);
      artistMap.get(key)!.push(p);
    }
    return Array.from(yearMap.entries())
      .sort(([a], [b]) => b - a)
      .map(([year, artistMap]) => ({
        year,
        artists: Array.from(artistMap.entries())
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([name, projects]) => ({ name, projects })),
        totalProjects: Array.from(artistMap.values()).reduce((sum, arr) => sum + arr.length, 0),
      }));
  }, [sharedProjects]);

  // Track collapsed years
  const [collapsedYears, setCollapsedYears] = useState<Set<number>>(new Set());
  const toggleYear = (year: number) => {
    setCollapsedYears((prev) => {
      const next = new Set(prev);
      if (next.has(year)) next.delete(year);
      else next.add(year);
      return next;
    });
  };

  const sortLabel = sortOrder === "alpha" ? "A-Z" : sortOrder === "newest" ? "Newest" : "Oldest";

  return (
    <div className="min-h-screen bg-background">
      {/* === HEADER === */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/dashboard")}
          >
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>

          <div className="flex items-center gap-2">
            <ToolHelpButton onClick={walkthrough.replay} />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-10 w-10 rounded-full bg-primary hover:bg-primary/90">
                  <Avatar className="h-10 w-10">
                    <AvatarImage src={profile?.avatar_url || ""} alt={profile?.full_name || ""} />
                    <AvatarFallback className="bg-primary text-primary-foreground">{getInitials()}</AvatarFallback>
                  </Avatar>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56" align="end" forceMount>
                <DropdownMenuLabel className="font-normal">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none">{profile?.full_name || "User"}</p>
                    <p className="text-xs leading-none text-muted-foreground">{user?.email}</p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => navigate("/profile")}>
                  <User className="mr-2 h-4 w-4" />
                  <span>Profile</span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => navigate("/")}>
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* === PAGE TITLE === */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => navigate("/dashboard")}>
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div>
              <h2 className="text-3xl font-bold text-foreground">Portfolio</h2>
              <p className="text-muted-foreground">Your projects organized by artist</p>
            </div>
          </div>
          <Button data-walkthrough="portfolio-add" onClick={() => handleAddProject()} className="gap-2">
            <Plus className="w-4 h-4" />
            Create Project
          </Button>
        </div>

        {/* === FILTER BAR === */}
        <div data-walkthrough="portfolio-filters" className="sticky top-0 z-10 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 pb-4 mb-6 border-b border-border">
          <div className="flex flex-wrap items-end gap-4">
            {/* Artist search with suggestions */}
            <div className="w-64 relative" ref={artistSearchRef}>
              <Label className="text-xs text-muted-foreground mb-1 block">Artist</Label>
              <Input
                placeholder="Search artists..."
                value={artistSearchInput}
                onChange={(e) => {
                  setArtistSearchInput(e.target.value);
                  setShowArtistSuggestions(true);
                }}
                onFocus={() => setShowArtistSuggestions(true)}
                className="h-9"
              />
              {showArtistSuggestions && artistSuggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-popover border border-border rounded-md shadow-lg z-20 overflow-hidden">
                  {artistSuggestions.map((artist) => (
                    <button
                      key={artist.id}
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted/50 transition-colors text-left"
                      onClick={() => addArtistFilter(artist.id)}
                    >
                      <Avatar className="h-5 w-5">
                        <AvatarImage src={artist.avatar || ""} />
                        <AvatarFallback className="text-[10px] bg-primary/10">
                          {artist.name.slice(0, 2).toUpperCase()}
                        </AvatarFallback>
                      </Avatar>
                      {artist.name}
                    </button>
                  ))}
                </div>
              )}
              {selectedArtistIds.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1.5">
                  {selectedArtistIds.map((id) => {
                    const artist = allArtists.find((a) => a.id === id);
                    return (
                      <Badge key={id} variant="secondary" className="gap-1 pr-1">
                        {artist?.name || "Unknown"}
                        <button
                          onClick={() => removeArtistFilter(id)}
                          className="ml-0.5 rounded-full hover:bg-muted p-0.5"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </Badge>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Text search */}
            <div className="flex-1 min-w-[200px]">
              <Label className="text-xs text-muted-foreground mb-1 block">Search</Label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search projects..."
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  className="pl-9 h-9"
                />
              </div>
            </div>

            {/* Sort dropdown */}
            <div>
              <Label className="text-xs text-muted-foreground mb-1 block">Sort</Label>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" className="h-9 gap-1.5">
                    <ArrowUpDown className="w-3.5 h-3.5" />
                    {sortLabel}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => setSortOrder("alpha")}>
                    A-Z (Alphabetical)
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setSortOrder("newest")}>
                    Newest First
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setSortOrder("oldest")}>
                    Oldest First
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            {/* Clear filters */}
            {hasActiveFilters && (
              <Button variant="ghost" size="sm" onClick={clearFilters} className="h-9">
                <X className="w-4 h-4 mr-1" /> Clear
              </Button>
            )}
          </div>
        </div>

        {/* === LOADING STATE === */}
        {isLoading && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-40 rounded-xl" />
            ))}
          </div>
        )}

        {/* === EMPTY STATE === */}
        {!isLoading && myProjects.length === 0 && sharedProjects.length === 0 && (
          <div className="text-center py-16 text-muted-foreground">
            <Folder className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-medium mb-1">No projects found</h3>
            <p className="text-sm">
              {hasActiveFilters ? "Try adjusting your filters" : "Add artists and projects to get started"}
            </p>
          </div>
        )}

        {/* === MY PROJECTS SECTION === */}
        {!isLoading && myProjects.length > 0 && (
          <section className="mb-10" data-walkthrough="portfolio-year">
            <div className="flex items-center gap-3 mb-4">
              <h3 className="text-xl font-semibold text-foreground">My Projects</h3>
              <Badge variant="secondary">{myProjects.length}</Badge>
            </div>

            {projectsByYearAndArtist.map(({ year, artists, totalProjects }) => (
              <div key={year} className="mb-6">
                {/* Year header — collapsible */}
                <button
                  onClick={() => toggleYear(year)}
                  className="flex items-center gap-2 mb-3 group cursor-pointer hover:opacity-80 transition-opacity"
                >
                  {collapsedYears.has(year) ? (
                    <ChevronRight className="w-4 h-4 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-muted-foreground" />
                  )}
                  <span className="text-lg font-semibold text-foreground">{year}</span>
                  <Badge variant="outline" className="text-xs">
                    {totalProjects} project{totalProjects !== 1 ? "s" : ""}
                  </Badge>
                </button>

                {/* Artists and projects within this year */}
                {!collapsedYears.has(year) && artists.map(({ name: artistName, projects }) => (
                  <div key={artistName} className="mb-5 ml-6">
                    <div className="flex items-center gap-2 mb-3">
                      <Avatar className="h-6 w-6">
                        <AvatarImage src={projects[0]?.artist_avatar || ""} />
                        <AvatarFallback className="text-[10px] bg-primary/10">
                          {artistName.slice(0, 2).toUpperCase()}
                        </AvatarFallback>
                      </Avatar>
                      <span className="text-sm font-medium text-muted-foreground">{artistName}</span>
                      <Badge variant="outline" className="text-xs">
                        {projects.length} project{projects.length !== 1 ? "s" : ""}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                      {projects.map((project) => (
                        <ProjectCardComponent key={project.id} project={project} onClick={() => navigate(`/projects/${project.id}`)} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </section>
        )}

        {/* === SHARED WITH ME SECTION === */}
        {!isLoading && sharedProjects.length > 0 && (
          <section className="mb-10">
            <div className="flex items-center gap-3 mb-4">
              <h3 className="text-xl font-semibold text-foreground">Shared with Me</h3>
              <Badge variant="secondary">{sharedProjects.length}</Badge>
            </div>

            {sharedByYearAndArtist.map(({ year, artists, totalProjects }) => (
              <div key={year} className="mb-6">
                <button
                  onClick={() => toggleYear(year + 10000)}
                  className="flex items-center gap-2 mb-3 group cursor-pointer hover:opacity-80 transition-opacity"
                >
                  {collapsedYears.has(year + 10000) ? (
                    <ChevronRight className="w-4 h-4 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-muted-foreground" />
                  )}
                  <span className="text-lg font-semibold text-foreground">{year}</span>
                  <Badge variant="outline" className="text-xs">
                    {totalProjects} project{totalProjects !== 1 ? "s" : ""}
                  </Badge>
                </button>

                {!collapsedYears.has(year + 10000) && artists.map(({ name: artistName, projects }) => (
                  <div key={artistName} className="mb-5 ml-6">
                    <div className="flex items-center gap-2 mb-3">
                      <Avatar className="h-6 w-6">
                        <AvatarFallback className="text-[10px] bg-primary/10">
                          {artistName.slice(0, 2).toUpperCase()}
                        </AvatarFallback>
                      </Avatar>
                      <span className="text-sm font-medium text-muted-foreground">{artistName}</span>
                      <Badge variant="outline" className="text-xs">
                        {projects.length} project{projects.length !== 1 ? "s" : ""}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                      {projects.map((project) => (
                        <SharedProjectCardComponent
                          key={project.id}
                          project={project}
                          onClick={() => navigate(`/projects/${project.id}`)}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </section>
        )}

        {/* === WALKTHROUGH === */}
        <ToolIntroModal
          config={TOOL_CONFIGS.portfolio}
          isOpen={walkthrough.phase === "modal"}
          onStartTour={walkthrough.startSpotlight}
          onSkip={walkthrough.skip}
        />
        <WalkthroughProvider
          isActive={walkthrough.phase === "spotlight"}
          currentStep={walkthrough.currentStep}
          currentStepIndex={walkthrough.visibleStepIndex}
          totalSteps={walkthrough.totalSteps}
          onNext={walkthrough.next}
          onSkip={walkthrough.skip}
        />
      </main>

      {/* === PROJECT FORM DIALOG === */}
      <ProjectFormDialog
        open={projectDialogOpen}
        onOpenChange={setProjectDialogOpen}
        project={null}
        artists={allArtists}
        defaultArtistId={defaultArtistIdForProject}
        onSave={handleSaveProject}
      />
    </div>
  );
};

// --- Project Card Component ---

function ProjectCardComponent({
  project,
  onClick,
}: {
  project: ProjectCard;
  onClick: () => void;
}) {
  return (
    <Card
      className="group cursor-pointer border border-border/50 hover:border-border hover:shadow-md transition-all duration-200 rounded-xl overflow-hidden"
      onClick={onClick}
    >
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="min-w-0 flex-1">
            <h4 className="font-semibold text-foreground truncate">{project.name}</h4>
            <p className="text-sm text-muted-foreground truncate">{project.artist_name}</p>
          </div>
        </div>

        {project.description && (
          <p className="text-xs text-muted-foreground/80 line-clamp-2 mb-3">{project.description}</p>
        )}

        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          {project.work_count > 0 && (
            <span className="flex items-center gap-1">
              <FileText className="w-3.5 h-3.5" />
              {project.work_count} work{project.work_count !== 1 ? "s" : ""}
            </span>
          )}
          {project.member_count > 0 && (
            <span className="flex items-center gap-1">
              <Users className="w-3.5 h-3.5" />
              {project.member_count}
            </span>
          )}
          <span className="flex items-center gap-1 ml-auto">
            <Calendar className="w-3.5 h-3.5" />
            {new Date(project.created_at).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
            })}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

// --- Shared Project Card Component ---

function SharedProjectCardComponent({
  project,
  onClick,
}: {
  project: SharedProjectCard;
  onClick: () => void;
}) {
  return (
    <Card
      className="group cursor-pointer border border-border/50 hover:border-border hover:shadow-md transition-all duration-200 rounded-xl overflow-hidden"
      onClick={onClick}
    >
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="min-w-0 flex-1">
            <h4 className="font-semibold text-foreground truncate">{project.name}</h4>
            <p className="text-sm text-muted-foreground truncate">{project.artist_name}</p>
          </div>
          <Badge
            variant="outline"
            className={`text-xs capitalize flex-shrink-0 ${ROLE_COLORS[project.role] || ""}`}
          >
            {project.role}
          </Badge>
        </div>

        {project.description && (
          <p className="text-xs text-muted-foreground/80 line-clamp-2 mb-3">{project.description}</p>
        )}

        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          {project.work_count > 0 && (
            <span className="flex items-center gap-1">
              <FileText className="w-3.5 h-3.5" />
              {project.work_count} work{project.work_count !== 1 ? "s" : ""}
            </span>
          )}
          {project.member_count > 0 && (
            <span className="flex items-center gap-1">
              <Users className="w-3.5 h-3.5" />
              {project.member_count}
            </span>
          )}
          <span className="flex items-center gap-1 ml-auto">
            <Calendar className="w-3.5 h-3.5" />
            {new Date(project.created_at).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "numeric",
            })}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

export default Portfolio;
