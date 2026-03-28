import { useState, useEffect, useMemo, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { useToast } from "@/hooks/use-toast";
import { usePortfolioData } from "@/hooks/usePortfolioData";
import type { Tables } from "@/integrations/supabase/types";
import { ContractUploadModal } from "@/components/ContractUploadModal";
import { RoyaltyStatementUploadModal } from "@/components/RoyaltyStatementUploadModal";
import { ProjectFormDialog } from "@/components/ProjectFormDialog";
import { AudioSection } from "@/components/AudioSection";
import { FileShareDialog } from "@/components/FileShareDialog";
import { useAudioData } from "@/hooks/useAudioData";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import { ToolIntroModal } from "@/components/walkthrough/ToolIntroModal";
import { ToolHelpButton } from "@/components/walkthrough/ToolHelpButton";
import { WalkthroughProvider } from "@/components/walkthrough/WalkthroughProvider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Music,
  ArrowLeft,
  Folder,
  FileText,
  Upload,
  Search,
  Trash2,
  CheckSquare,
  X,
  User,
  LogOut,
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Plus,
  Pencil,
  MoreVertical,
  Send,
} from "lucide-react";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

const ARTIST_COLORS = [
  { border: "border-l-rose-500", bg: "bg-rose-500/5", avatar: "bg-rose-500/15 text-rose-700 dark:text-rose-300" },
  { border: "border-l-sky-500", bg: "bg-sky-500/5", avatar: "bg-sky-500/15 text-sky-700 dark:text-sky-300" },
  { border: "border-l-amber-500", bg: "bg-amber-500/5", avatar: "bg-amber-500/15 text-amber-700 dark:text-amber-300" },
  { border: "border-l-emerald-500", bg: "bg-emerald-500/5", avatar: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300" },
  { border: "border-l-violet-500", bg: "bg-violet-500/5", avatar: "bg-violet-500/15 text-violet-700 dark:text-violet-300" },
  { border: "border-l-orange-500", bg: "bg-orange-500/5", avatar: "bg-orange-500/15 text-orange-700 dark:text-orange-300" },
  { border: "border-l-teal-500", bg: "bg-teal-500/5", avatar: "bg-teal-500/15 text-teal-700 dark:text-teal-300" },
  { border: "border-l-pink-500", bg: "bg-pink-500/5", avatar: "bg-pink-500/15 text-pink-700 dark:text-pink-300" },
  { border: "border-l-indigo-500", bg: "bg-indigo-500/5", avatar: "bg-indigo-500/15 text-indigo-700 dark:text-indigo-300" },
  { border: "border-l-lime-500", bg: "bg-lime-500/5", avatar: "bg-lime-500/15 text-lime-700 dark:text-lime-300" },
];

function getArtistColor(name: string) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return ARTIST_COLORS[Math.abs(hash) % ARTIST_COLORS.length];
}

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

  // Filter state — initialize from ?artist= query param if present
  const initialArtistId = searchParams.get("artist");
  const [selectedArtistIds, setSelectedArtistIds] = useState<string[]>(
    initialArtistId ? [initialArtistId] : []
  );
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [sortOrder, setSortOrder] = useState<"alpha" | "newest" | "oldest">("alpha");

  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(
    TOOL_CONFIGS.portfolio,
    statuses.portfolio,
    onboardingLoading,
    () => markToolCompleted("portfolio")
  );

  // Ref to track first folder grid for walkthrough attribute
  const firstFolderGridRef = useRef(true);

  // Artist search state
  const [artistSearchInput, setArtistSearchInput] = useState("");
  const [showArtistSuggestions, setShowArtistSuggestions] = useState(false);
  const artistSearchRef = useRef<HTMLDivElement>(null);

  // File viewer dialog state
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [selectedFileType, setSelectedFileType] = useState<string | null>(null);
  const [fileToDelete, setFileToDelete] = useState<any>(null);
  const [fileSearchQuery, setFileSearchQuery] = useState("");

  // Upload modal state
  const [uploadingFile, setUploadingFile] = useState<string | null>(null);
  const [contractUploadModalOpen, setContractUploadModalOpen] = useState(false);
  const [contractUploadProjectId, setContractUploadProjectId] = useState<string>("");
  const [royaltyStatementUploadModalOpen, setRoyaltyStatementUploadModalOpen] = useState(false);
  const [royaltyStatementUploadProjectId, setRoyaltyStatementUploadProjectId] = useState<string>("");

  // Ticket expansion state per project
  const [expandedTickets, setExpandedTickets] = useState<Record<string, boolean>>({});

  // Project CRUD state
  const [projectDialogOpen, setProjectDialogOpen] = useState(false);
  const [editingProject, setEditingProject] = useState<{ id: string; name: string; description: string | null; artist_id: string } | null>(null);
  const [defaultArtistIdForProject, setDefaultArtistIdForProject] = useState<string | undefined>();
  const [projectToDelete, setProjectToDelete] = useState<{ id: string; name: string } | null>(null);

  // File sharing state
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [filesToShare, setFilesToShare] = useState<{ file_name: string; file_path: string; file_source: "project_file" | "audio_file"; file_id: string }[]>([]);

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

  const { years, allArtists, isLoading, refetchFiles, refetchProjects } = usePortfolioData({
    selectedArtistIds,
    searchQuery: debouncedSearch,
    dateFrom: dateFrom || undefined,
    dateTo: dateTo || undefined,
    sortOrder,
  });

  // Derive IDs for audio data hook
  const allArtistIds = useMemo(() => allArtists.map((a) => a.id), [allArtists]);
  const allProjectIds = useMemo(() => {
    const ids: string[] = [];
    for (const yg of years) {
      for (const ag of yg.artists) {
        for (const p of ag.projects) ids.push(p.id);
      }
    }
    return ids;
  }, [years]);

  const audioData = useAudioData(allArtistIds, allProjectIds);

  const hasActiveFilters = selectedArtistIds.length > 0 || debouncedSearch || dateFrom || dateTo;

  const clearFilters = () => {
    setSelectedArtistIds([]);
    setSearchInput("");
    setDebouncedSearch("");
    setDateFrom("");
    setDateTo("");
    setArtistSearchInput("");
  };

  // --- Project CRUD handlers ---

  const handleAddProject = (artistId?: string) => {
    setEditingProject(null);
    setDefaultArtistIdForProject(artistId);
    setProjectDialogOpen(true);
  };

  const handleEditProject = (project: { id: string; name: string; description: string | null; artist_id: string }) => {
    setEditingProject(project);
    setDefaultArtistIdForProject(undefined);
    setProjectDialogOpen(true);
  };

  const handleSaveProject = async (data: { name: string; description: string; artist_id: string }) => {
    if (editingProject) {
      // Check for duplicate name (excluding current project)
      const { data: existing } = await supabase
        .from("projects")
        .select("id")
        .eq("artist_id", data.artist_id)
        .ilike("name", data.name.trim())
        .neq("id", editingProject.id)
        .limit(1);
      if (existing && existing.length > 0) {
        toast({ title: "Duplicate project name", description: `A project named "${data.name.trim()}" already exists for this artist.`, className: "bg-white text-black border border-border" });
        return;
      }
      const { error } = await supabase
        .from("projects")
        .update({ name: data.name, description: data.description || null })
        .eq("id", editingProject.id);
      if (error) throw error;
      toast({ title: "Success", description: "Project updated" });
    } else {
      // Check for duplicate name
      const { data: existing } = await supabase
        .from("projects")
        .select("id")
        .eq("artist_id", data.artist_id)
        .ilike("name", data.name.trim())
        .limit(1);
      if (existing && existing.length > 0) {
        toast({ title: "Duplicate project name", description: `A project named "${data.name.trim()}" already exists for this artist.`, className: "bg-white text-black border border-border" });
        return;
      }
      const { error } = await supabase
        .from("projects")
        .insert({ name: data.name, description: data.description || null, artist_id: data.artist_id });
      if (error) throw error;
      toast({ title: "Success", description: "Project created" });
    }
    refetchProjects();
  };

  const handleDeleteProject = async () => {
    if (!projectToDelete) return;
    try {
      // Delete all project files from storage and DB first
      const { data: files } = await supabase
        .from("project_files")
        .select("*")
        .eq("project_id", projectToDelete.id);

      if (files && files.length > 0) {
        const storagePaths = files
          .map(f => (f as any).file_path as string | undefined)
          .filter((p): p is string => !!p);
        if (storagePaths.length > 0) {
          await supabase.storage.from("project-files").remove(storagePaths);
        }
        await supabase.from("project_files").delete().eq("project_id", projectToDelete.id);
      }

      const { error } = await supabase.from("projects").delete().eq("id", projectToDelete.id);
      if (error) throw error;

      refetchProjects();
      refetchFiles();
      toast({ title: "Success", description: "Project deleted" });
    } catch (error: any) {
      console.error("Error deleting project:", error);
      toast({ title: "Error", description: error.message || "Failed to delete project", variant: "destructive" });
    } finally {
      setProjectToDelete(null);
    }
  };

  // Artist search suggestions (top 5 closest matches)
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

  // --- File operation handlers ---

  const normalizeFileName = (name: string) => name.trim().toLowerCase();

  const handleContractUploadClick = (projectId: string) => {
    setContractUploadProjectId(projectId);
    setContractUploadModalOpen(true);
  };

  const handleRoyaltyStatementUploadClick = (projectId: string) => {
    setRoyaltyStatementUploadProjectId(projectId);
    setRoyaltyStatementUploadModalOpen(true);
  };

  const handleFileUpload = async (
    projectId: string,
    folderCategory: string,
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const { data: existingFiles, error: checkError } = await supabase
      .from("project_files")
      .select("file_name")
      .eq("project_id", projectId);

    if (!checkError && existingFiles) {
      const hasDuplicate = existingFiles.some(
        (existing) => normalizeFileName(existing.file_name) === normalizeFileName(file.name)
      );
      if (hasDuplicate) {
        toast({
          title: "Duplicate file name",
          description: `A file named "${file.name}" already exists in this project.`,
          className: "bg-white text-black border border-border",
        });
        event.target.value = "";
        return;
      }
    }

    setUploadingFile(`${projectId}-${folderCategory}`);

    try {
      const filePath = `${projectId}/${folderCategory}/${Date.now()}_${file.name}`;
      const { error: uploadError } = await supabase.storage.from("project-files").upload(filePath, file);
      if (uploadError) throw uploadError;

      const { data: urlData } = supabase.storage.from("project-files").getPublicUrl(filePath);

      const { error: dbError } = await supabase
        .from("project_files")
        .insert({
          project_id: projectId,
          file_name: file.name,
          file_url: urlData.publicUrl,
          file_path: filePath,
          folder_category: folderCategory,
          file_size: file.size,
          file_type: file.type,
        })
        .select()
        .single();

      if (dbError) {
        const errorMessage = dbError.message?.toLowerCase() || "";
        if (errorMessage.includes("duplicate") || errorMessage.includes("unique")) {
          await supabase.storage.from("project-files").remove([filePath]);
          toast({
            title: "Duplicate file name",
            description: `A file named "${file.name}" already exists in this project.`,
            className: "bg-white text-black border border-border",
          });
          return;
        }
        throw dbError;
      }

      refetchFiles();
      toast({ title: "Success", description: "File uploaded successfully" });
    } catch (error: any) {
      console.error("Upload error:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to upload file",
        variant: "destructive",
      });
    } finally {
      setUploadingFile(null);
      event.target.value = "";
    }
  };

  const handleFileDownload = async (file: any) => {
    try {
      const { data, error } = await supabase.storage.from("project-files").download(file.file_path);
      if (error) throw error;

      const url = URL.createObjectURL(data);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.file_name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error: any) {
      toast({ title: "Error", description: "Failed to download file", variant: "destructive" });
    }
  };

  const handleFileDelete = async () => {
    if (!fileToDelete || !user) return;

    try {
      if (fileToDelete.folder_category === "contract") {
        const formData = new FormData();
        formData.append("user_id", user.id);
        const vectorDeleteResponse = await fetch(`${API_URL}/contracts/${fileToDelete.id}`, {
          method: "DELETE",
          body: formData,
        });
        if (!vectorDeleteResponse.ok) {
          const errorData = await vectorDeleteResponse.json();
          throw new Error(errorData.detail || "Failed to delete contract vectors");
        }
        refetchFiles();
        toast({ title: "Success", description: "Contract deleted successfully" });
      } else {
        const { error: storageError } = await supabase.storage
          .from("project-files")
          .remove([fileToDelete.file_path]);
        if (storageError) throw storageError;

        const { error: dbError } = await supabase.from("project_files").delete().eq("id", fileToDelete.id);
        if (dbError) throw dbError;

        refetchFiles();
        toast({ title: "Success", description: "File deleted successfully" });
      }
    } catch (error: any) {
      console.error("Error deleting file:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to delete file",
        variant: "destructive",
      });
    } finally {
      setFileToDelete(null);
    }
  };

  // Helper: get files for the dialog (with search filtering)
  const getFilesForDialog = (): Tables<"project_files">[] => {
    if (!selectedProject || !selectedFileType) return [];
    let files: Tables<"project_files">[] = [];
    for (const yearGroup of years) {
      for (const artistGroup of yearGroup.artists) {
        for (const project of artistGroup.projects) {
          if (project.id === selectedProject) {
            files = project.files[selectedFileType] || [];
            break;
          }
        }
      }
    }
    if (fileSearchQuery) {
      const q = fileSearchQuery.toLowerCase();
      files = files.filter((f) => f.file_name.toLowerCase().includes(q));
    }
    return files;
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
              <p className="text-muted-foreground">Your profile organized by year, artist, and project</p>
            </div>
          </div>
          <Button data-walkthrough="portfolio-add" onClick={() => handleAddProject()} className="gap-2">
            <Plus className="w-4 h-4" />
            Add Project
          </Button>
        </div>

        {/* === FILTER BAR (sticky) === */}
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
              {/* Selected artist chips */}
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
                  placeholder="Search projects & files..."
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  className="pl-9 h-9"
                />
              </div>
            </div>

            {/* Date range */}
            <div className="w-40">
              <Label className="text-xs text-muted-foreground mb-1 block">From</Label>
              <Input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className="h-9" />
            </div>
            <div className="w-40">
              <Label className="text-xs text-muted-foreground mb-1 block">To</Label>
              <Input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className="h-9" />
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
          <div className="space-y-4">
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
            <Skeleton className="h-12 w-full" />
          </div>
        )}

        {/* === EMPTY STATE === */}
        {!isLoading && years.length === 0 && (
          <div className="text-center py-16 text-muted-foreground">
            <Folder className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-medium mb-1">No projects found</h3>
            <p className="text-sm">
              {hasActiveFilters ? "Try adjusting your filters" : "Add artists and projects to get started"}
            </p>
          </div>
        )}

        {/* === YEAR ACCORDION === */}
        <Accordion type="multiple" defaultValue={years.map((y) => `year-${y.year}`)} className="space-y-4">
          {years.map((yearGroup, index) => (
            <AccordionItem key={yearGroup.year} value={`year-${yearGroup.year}`} data-walkthrough={index === 0 ? "portfolio-year" : undefined} className="border rounded-lg px-4">
              <AccordionTrigger className="hover:no-underline py-4">
                <div className="flex items-center gap-3">
                  <span className="text-xl font-bold">{yearGroup.year}</span>
                  <Badge variant="secondary">
                    {yearGroup.totalProjects} project{yearGroup.totalProjects !== 1 ? "s" : ""}
                  </Badge>
                </div>
              </AccordionTrigger>
              <AccordionContent className="pb-4">
                {/* === ALPHABETICAL LETTER GROUPS === */}
                <Accordion
                  type="multiple"
                  defaultValue={yearGroup.letterGroups.map((lg) => `letter-${lg.letter}-${yearGroup.year}`)}
                  className="space-y-3"
                >
                  {yearGroup.letterGroups.map((letterGroup) => (
                    <AccordionItem
                      key={letterGroup.letter}
                      value={`letter-${letterGroup.letter}-${yearGroup.year}`}
                      className="border rounded-lg overflow-hidden"
                    >
                      <AccordionTrigger className="hover:no-underline px-4 py-2.5 bg-muted/30">
                        <span className="text-sm font-bold text-primary">{letterGroup.letter}</span>
                      </AccordionTrigger>
                      <AccordionContent className="px-3 pt-2 pb-3">
                    <Accordion type="multiple" className="space-y-2">
                      {letterGroup.artists.map((artistGroup) => {
                        const artistColor = getArtistColor(artistGroup.artist.name);
                        return (
                        <AccordionItem
                          key={artistGroup.artist.id}
                          value={`artist-${artistGroup.artist.id}-${yearGroup.year}`}
                          className={`border rounded-lg px-3 border-l-4 ${artistColor.border} ${artistColor.bg}`}
                        >
                          <div className="flex items-center">
                          <AccordionTrigger className="hover:no-underline py-3 flex-1">
                            <div className="flex items-center gap-3">
                              <Avatar className="h-8 w-8">
                                <AvatarImage src={artistGroup.artist.avatar || ""} />
                                <AvatarFallback className={`text-xs ${artistColor.avatar}`}>
                                  {artistGroup.artist.name.slice(0, 2).toUpperCase()}
                                </AvatarFallback>
                              </Avatar>
                              <span
                                className="font-semibold hover:underline cursor-pointer"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  navigate(`/artists/${artistGroup.artist.id}`);
                                }}
                              >
                                {artistGroup.artist.name}
                              </span>
                              <Badge variant="outline" className="text-xs">
                                {artistGroup.projects.length} project
                                {artistGroup.projects.length !== 1 ? "s" : ""}
                              </Badge>
                            </div>
                          </AccordionTrigger>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7 ml-2"
                            title="Add project for this artist"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleAddProject(artistGroup.artist.id);
                            }}
                          >
                            <Plus className="w-4 h-4" />
                          </Button>
                          </div>
                          <AccordionContent className="pb-3 space-y-3">
                            {/* === PROJECT CARDS === */}
                            {/* === AUDIO SECTION (per artist) === */}
                            <AudioSection
                              artistId={artistGroup.artist.id}
                              folders={audioData.foldersByArtist.get(artistGroup.artist.id) || []}
                              filesByFolder={audioData.filesByFolder}
                              projectsByAudioFile={audioData.projectsByAudioFile}
                              artistProjects={artistGroup.projects.map((p) => ({ id: p.id, name: p.name }))}
                              onCreateFolder={audioData.createFolder}
                              onRenameFolder={audioData.renameFolder}
                              onDeleteFolder={audioData.deleteFolder}
                              onUploadFile={audioData.uploadAudioFile}
                              onDeleteFile={audioData.deleteAudioFile}
                              onLinkAudio={audioData.linkAudioToProject}
                              onUnlinkAudio={audioData.unlinkAudioFromProject}
                            />

                            <Separator className="my-3" />

                            {/* === PROJECT CARDS === */}
                            <Accordion type="multiple" className="space-y-3">
                            {artistGroup.projects.map((project) => {
                              const isTicketsExpanded = expandedTickets[project.id] || false;
                              const visibleTasks = isTicketsExpanded
                                ? project.tasks
                                : project.tasks.slice(0, 5);
                              const hasMoreTasks = project.tasks.length > 5;

                              return (
                                <AccordionItem key={project.id} value={`project-${project.id}`} className="border-0">
                                <Card className="group/card border-0 bg-gradient-to-br from-card to-card/80 shadow-sm hover:shadow-md transition-all duration-300 rounded-xl overflow-hidden ring-1 ring-border/50 hover:ring-border">
                                  <CardHeader className="pb-0 pt-4 px-5">
                                    <div className="flex items-start justify-between gap-3">
                                      <AccordionTrigger className="hover:no-underline p-0 flex-1 min-w-0">
                                        <div className="flex-1 min-w-0 text-left">
                                          <div className="flex items-center gap-2">
                                            <CardTitle className="text-base font-semibold tracking-tight truncate">{project.name}</CardTitle>
                                            <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 font-normal text-muted-foreground border-border/60 flex-shrink-0">
                                              {Object.values(project.files).flat().length} files
                                            </Badge>
                                          </div>
                                          {project.description && (
                                            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{project.description}</p>
                                          )}
                                          <p className="text-[11px] text-muted-foreground/70 mt-1.5">
                                            {new Date(project.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                                          </p>
                                        </div>
                                      </AccordionTrigger>
                                      <DropdownMenu>
                                        <DropdownMenuTrigger asChild>
                                          <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 group-hover/card:opacity-100 transition-opacity">
                                            <MoreVertical className="w-4 h-4" />
                                          </Button>
                                        </DropdownMenuTrigger>
                                        <DropdownMenuContent align="end">
                                          <DropdownMenuItem onClick={() => handleEditProject({
                                            id: project.id,
                                            name: project.name,
                                            description: project.description,
                                            artist_id: project.artist_id,
                                          })}>
                                            <Pencil className="w-4 h-4 mr-2" />
                                            Edit Project
                                          </DropdownMenuItem>
                                          <DropdownMenuItem
                                            onClick={() => {
                                              const allProjectFiles = Object.values(project.files)
                                                .flat()
                                                .map((f) => {
                                                  let path = (f as any).file_path || "";
                                                  if (!path && f.file_url) {
                                                    const marker = "/object/public/project-files/";
                                                    const idx = f.file_url.indexOf(marker);
                                                    if (idx >= 0) path = decodeURIComponent(f.file_url.slice(idx + marker.length));
                                                  }
                                                  return {
                                                    file_name: f.file_name,
                                                    file_path: path,
                                                    file_source: "project_file" as const,
                                                    file_id: f.id,
                                                  };
                                                });
                                              if (allProjectFiles.length > 0) {
                                                setFilesToShare(allProjectFiles);
                                                setShareDialogOpen(true);
                                              }
                                            }}
                                            disabled={Object.values(project.files).flat().length === 0}
                                          >
                                            <Send className="w-4 h-4 mr-2" />
                                            Share Files
                                          </DropdownMenuItem>
                                          <DropdownMenuItem
                                            className="text-destructive focus:text-destructive"
                                            onClick={() => setProjectToDelete({ id: project.id, name: project.name })}
                                          >
                                            <Trash2 className="w-4 h-4 mr-2" />
                                            Delete Project
                                          </DropdownMenuItem>
                                        </DropdownMenuContent>
                                      </DropdownMenu>
                                    </div>
                                  </CardHeader>
                                  <AccordionContent>
                                  <CardContent className="space-y-3 px-5 pb-5 pt-3">
                                    {/* FILE GRID (2x2) */}
                                    <div
                                      data-walkthrough={firstFolderGridRef.current ? "portfolio-folders" : undefined}
                                      ref={(el) => { if (el && firstFolderGridRef.current) firstFolderGridRef.current = false; }}
                                      className="grid grid-cols-2 gap-2"
                                    >
                                      {[
                                        { name: "Contracts", category: "contract" },
                                        { name: "Split Sheets", category: "split_sheet" },
                                        { name: "Royalty Statements", category: "royalty_statement" },
                                        { name: "Other", category: "other" },
                                      ].map((folder) => {
                                        const fileCount = (project.files[folder.category] || []).length;
                                        const isUploading =
                                          uploadingFile === `${project.id}-${folder.category}`;
                                        return (
                                          <div key={folder.category} className="relative">
                                            {folder.category !== "contract" &&
                                              folder.category !== "royalty_statement" && (
                                                <input
                                                  type="file"
                                                  id={`upload-${project.id}-${folder.category}`}
                                                  className="hidden"
                                                  onChange={(e) =>
                                                    handleFileUpload(project.id, folder.category, e)
                                                  }
                                                  disabled={isUploading}
                                                />
                                              )}
                                            <div
                                              className="p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors cursor-pointer group/folder"
                                              onClick={() => {
                                                setSelectedProject(project.id);
                                                setSelectedFileType(folder.category);
                                                setFileSearchQuery("");
                                              }}
                                            >
                                              <div className="flex items-center justify-between mb-1.5">
                                                <Folder className="w-4 h-4 text-muted-foreground" />
                                                <Button
                                                  variant="ghost"
                                                  size="icon"
                                                  className="h-5 w-5 opacity-0 group-hover/folder:opacity-100 transition-opacity"
                                                  onClick={(e) => {
                                                    e.stopPropagation();
                                                    if (folder.category === "contract")
                                                      handleContractUploadClick(project.id);
                                                    else if (folder.category === "royalty_statement")
                                                      handleRoyaltyStatementUploadClick(project.id);
                                                    else
                                                      document
                                                        .getElementById(
                                                          `upload-${project.id}-${folder.category}`
                                                        )
                                                        ?.click();
                                                  }}
                                                  disabled={isUploading}
                                                >
                                                  {isUploading ? (
                                                    <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
                                                  ) : (
                                                    <Upload className="w-3 h-3" />
                                                  )}
                                                </Button>
                                              </div>
                                              <p className="text-xs font-medium">{folder.name}</p>
                                              <p className="text-[11px] text-muted-foreground mt-0.5">
                                                {fileCount} {fileCount === 1 ? "file" : "files"}
                                              </p>
                                            </div>
                                          </div>
                                        );
                                            })}
                                    </div>

                                    {/* TICKETS SECTION */}
                                    {project.tasks.length > 0 && (
                                      <div className="pt-1">
                                        <div className="flex items-center gap-2 mb-2">
                                          <CheckSquare className="w-3.5 h-3.5 text-muted-foreground/70" />
                                          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Tickets</span>
                                          <span className="text-[10px] text-muted-foreground/60 bg-muted rounded-full px-1.5 py-0.5 leading-none">
                                            {project.tasks.length}
                                          </span>
                                        </div>
                                        <div className="space-y-1">
                                          {visibleTasks.map((task) => (
                                            <div
                                              key={task.id}
                                              className="flex items-center justify-between py-1.5 px-2.5 rounded-lg hover:bg-muted/60 transition-colors cursor-pointer group/task"
                                              onClick={() =>
                                                navigate(
                                                  `/workspace?tab=boards&taskId=${task.id}`
                                                )
                                              }
                                            >
                                              <span className="text-sm truncate flex-1 min-w-0">{task.title}</span>
                                              <div className="flex items-center gap-1.5 flex-shrink-0 ml-2">
                                                {task.priority && (
                                                  <span
                                                    className={`w-1.5 h-1.5 rounded-full ${
                                                      task.priority === "urgent"
                                                        ? "bg-red-500"
                                                        : task.priority === "high"
                                                        ? "bg-orange-500"
                                                        : task.priority === "medium"
                                                        ? "bg-yellow-500"
                                                        : "bg-green-500"
                                                    }`}
                                                    title={task.priority}
                                                  />
                                                )}
                                                {task.column_title && (
                                                  <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                                                    {task.column_title}
                                                  </span>
                                                )}
                                                {task.due_date && (
                                                  <span className="text-[10px] text-muted-foreground/70">
                                                    {new Date(task.due_date).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                                                  </span>
                                                )}
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                        {hasMoreTasks && (
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            className="w-full mt-1 text-xs text-muted-foreground h-7"
                                            onClick={() =>
                                              setExpandedTickets((prev) => ({
                                                ...prev,
                                                [project.id]: !prev[project.id],
                                              }))
                                            }
                                          >
                                            {isTicketsExpanded ? (
                                              <>
                                                <ChevronUp className="w-3 h-3 mr-1" />
                                                Show less
                                              </>
                                            ) : (
                                              <>
                                                <ChevronDown className="w-3 h-3 mr-1" />
                                                {project.tasks.length - 5} more
                                              </>
                                            )}
                                          </Button>
                                        )}
                                      </div>
                                    )}

                                    {/* LINKED AUDIO FILES */}
                                    {(audioData.audioLinksByProject.get(project.id) || []).length > 0 && (
                                      <div className="pt-1">
                                        <div className="flex items-center gap-2 mb-2">
                                          <Music className="w-3.5 h-3.5 text-muted-foreground/70" />
                                          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Linked Audio</span>
                                          <span className="text-[10px] text-muted-foreground/60 bg-muted rounded-full px-1.5 py-0.5 leading-none">
                                            {(audioData.audioLinksByProject.get(project.id) || []).length}
                                          </span>
                                        </div>
                                        <div className="space-y-0.5">
                                          {(audioData.audioLinksByProject.get(project.id) || []).map((audioFile) => (
                                            <div
                                              key={audioFile.id}
                                              className="flex items-center gap-2 py-1.5 px-2.5 rounded-lg text-sm text-muted-foreground hover:bg-muted/40 transition-colors"
                                            >
                                              <Music className="w-3 h-3 flex-shrink-0 text-emerald-500/70" />
                                              <span className="truncate">{audioFile.file_name}</span>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                  </CardContent>
                                  </AccordionContent>
                                </Card>
                                </AccordionItem>
                              );
                            })}
                            </Accordion>
                          </AccordionContent>
                        </AccordionItem>
                        );
                      })}
                    </Accordion>
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
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

      {/* === FILE VIEWER DIALOG === */}
      <Dialog
        open={selectedProject !== null && selectedFileType !== null}
        onOpenChange={(open) => {
          if (!open) {
            setSelectedProject(null);
            setSelectedFileType(null);
            setFileSearchQuery("");
          }
        }}
      >
        <DialogContent className="max-w-2xl max-h-[80vh] flex flex-col overflow-hidden">
          <DialogHeader>
            <div className="flex items-center justify-between pr-8">
              <DialogTitle className="flex items-center gap-2">
                <Folder className="w-5 h-5" />
                {selectedFileType === "contract" && "Contracts"}
                {selectedFileType === "split_sheet" && "Split Sheets"}
                {selectedFileType === "royalty_statement" && "Royalty Statements"}
                {selectedFileType === "other" && "Other Files"}
              </DialogTitle>
              <div className="relative w-48">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
                <Input
                  placeholder="Search files..."
                  value={fileSearchQuery}
                  onChange={(e) => setFileSearchQuery(e.target.value)}
                  className="pl-8 h-8 text-sm"
                />
              </div>
            </div>
          </DialogHeader>
          <div className="space-y-2 overflow-y-auto flex-1 min-h-0">
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={() => {
                if (selectedFileType === "contract") handleContractUploadClick(selectedProject!);
                else if (selectedFileType === "royalty_statement")
                  handleRoyaltyStatementUploadClick(selectedProject!);
                else document.getElementById(`upload-${selectedProject}-${selectedFileType}`)?.click();
              }}
            >
              <Upload className="w-4 h-4 mr-2" /> Upload File
            </Button>

            {getFilesForDialog().length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Folder className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>{fileSearchQuery ? "No files match your search" : "No files uploaded yet"}</p>
              </div>
            ) : (
              getFilesForDialog().map((file) => (
                <div
                  key={file.id}
                  className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <FileText className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{file.file_name}</p>
                      <p className="text-xs text-muted-foreground">
                        {file.file_size ? `${(file.file_size / 1024).toFixed(1)} KB` : "Unknown size"} ·{" "}
                        {new Date(file.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      title="Share"
                      onClick={() => {
                        let path = (file as any).file_path || "";
                        if (!path && file.file_url) {
                          const marker = "/object/public/project-files/";
                          const idx = file.file_url.indexOf(marker);
                          if (idx >= 0) path = decodeURIComponent(file.file_url.slice(idx + marker.length));
                        }
                        setFilesToShare([{
                          file_name: file.file_name,
                          file_path: path,
                          file_source: "project_file",
                          file_id: file.id,
                        }]);
                        setShareDialogOpen(true);
                      }}
                    >
                      <Send className="w-4 h-4" />
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => handleFileDownload(file)} title="Download">
                      <Upload className="w-4 h-4 rotate-180" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setFileToDelete(file)}
                      className="text-destructive hover:text-destructive"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              ))
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* === DELETE CONFIRMATION === */}
      <AlertDialog
        open={fileToDelete !== null}
        onOpenChange={(open) => {
          if (!open) setFileToDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete File?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete <strong>{fileToDelete?.file_name}</strong>. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleFileDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete File
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* === UPLOAD MODALS === */}
      <ContractUploadModal
        open={contractUploadModalOpen}
        onOpenChange={(open) => { if (!open) setContractUploadModalOpen(false); }}
        projectId={contractUploadProjectId}
        onUploadComplete={() => refetchFiles()}
      />
      <RoyaltyStatementUploadModal
        open={royaltyStatementUploadModalOpen}
        onOpenChange={(open) => { if (!open) setRoyaltyStatementUploadModalOpen(false); }}
        projectId={royaltyStatementUploadProjectId}
        onUploadComplete={() => refetchFiles()}
      />

      {/* === PROJECT FORM DIALOG === */}
      <ProjectFormDialog
        open={projectDialogOpen}
        onOpenChange={setProjectDialogOpen}
        project={editingProject}
        artists={allArtists}
        defaultArtistId={defaultArtistIdForProject}
        onSave={handleSaveProject}
      />

      {/* === DELETE PROJECT CONFIRMATION === */}
      <AlertDialog
        open={projectToDelete !== null}
        onOpenChange={(open) => { if (!open) setProjectToDelete(null); }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Project?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete <strong>{projectToDelete?.name}</strong> and all its files. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteProject}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete Project
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* === FILE SHARE DIALOG === */}
      <FileShareDialog
        open={shareDialogOpen}
        onOpenChange={setShareDialogOpen}
        files={filesToShare}
      />
    </div>
  );
};

export default Portfolio;
