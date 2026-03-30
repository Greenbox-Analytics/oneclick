import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useWorks, useMyCollaborations, useCreateWork, type Work } from "@/hooks/useRegistry";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import {
  Music, ArrowLeft, Plus, Search, Shield, Loader2, FileText, AlertCircle, Users,
} from "lucide-react";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

const WORK_TYPES = [
  { value: "single", label: "Single" },
  { value: "ep_track", label: "EP Track" },
  { value: "album_track", label: "Album Track" },
  { value: "composition", label: "Composition" },
];

const STATUS_COLORS: Record<string, string> = {
  draft: "bg-yellow-100 text-yellow-800",
  pending_approval: "bg-amber-100 text-amber-800",
  registered: "bg-green-100 text-green-800",
  disputed: "bg-red-100 text-red-800",
};

interface Artist { id: string; name: string; }
interface Project { id: string; name: string; artist_id: string; }

const Registry = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedArtistId, setSelectedArtistId] = useState<string>("all");
  const [showCreateDialog, setShowCreateDialog] = useState(false);

  // Create form state
  const [newTitle, setNewTitle] = useState("");
  const [newArtistId, setNewArtistId] = useState("");
  const [newProjectId, setNewProjectId] = useState("");
  const [newWorkType, setNewWorkType] = useState("single");
  const [newIsrc, setNewIsrc] = useState("");
  const [newIswc, setNewIswc] = useState("");

  // Fetch artists
  const artistsQuery = useQuery<Artist[]>({
    queryKey: ["registry-artists", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const res = await fetch(`${API_URL}/artists?user_id=${user.id}`);
      if (!res.ok) return [];
      const data = await res.json();
      return (Array.isArray(data) ? data : data.artists || []).map(
        (a: Record<string, unknown>) => ({ id: a.id as string, name: a.name as string })
      );
    },
    enabled: !!user?.id,
  });

  // Fetch projects for selected artist in create form
  const projectsQuery = useQuery<Project[]>({
    queryKey: ["registry-projects", newArtistId],
    queryFn: async () => {
      if (!newArtistId) return [];
      const res = await fetch(`${API_URL}/projects/${newArtistId}`);
      if (!res.ok) return [];
      const data = await res.json();
      return (Array.isArray(data) ? data : data.projects || []).map(
        (p: Record<string, unknown>) => ({
          id: p.id as string, name: p.name as string, artist_id: p.artist_id as string,
        })
      );
    },
    enabled: !!newArtistId,
  });

  const artistFilter = selectedArtistId === "all" ? undefined : selectedArtistId;
  const worksQuery = useWorks(artistFilter);
  const collabQuery = useMyCollaborations();
  const createWork = useCreateWork();

  const filteredWorks = useMemo(() => {
    const works = worksQuery.data || [];
    if (!searchQuery.trim()) return works;
    const q = searchQuery.toLowerCase();
    return works.filter(
      (w) =>
        w.title.toLowerCase().includes(q) ||
        (w.isrc && w.isrc.toLowerCase().includes(q)) ||
        (w.iswc && w.iswc.toLowerCase().includes(q))
    );
  }, [worksQuery.data, searchQuery]);

  const handleCreate = async () => {
    if (!newTitle.trim() || !newArtistId || !newProjectId) return;
    await createWork.mutateAsync({
      artist_id: newArtistId,
      project_id: newProjectId,
      title: newTitle.trim(),
      work_type: newWorkType,
      isrc: newIsrc.trim() || undefined,
      iswc: newIswc.trim() || undefined,
    });
    setShowCreateDialog(false);
    setNewTitle(""); setNewArtistId(""); setNewProjectId("");
    setNewWorkType("single"); setNewIsrc(""); setNewIswc("");
  };

  const getArtistName = (artistId: string) =>
    (artistsQuery.data || []).find((a) => a.id === artistId)?.name || "Unknown";

  const myCollabWorks = collabQuery.data || [];
  // Split into pending (need action) vs confirmed (already responded)
  const pendingWorks = myCollabWorks.filter((w) => w.status === "pending_approval");
  const allCollabWorks = myCollabWorks;

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/dashboard")}>
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-6 h-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
          </div>
          <Button variant="outline" onClick={() => navigate("/tools")}>
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Tools
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-3xl font-bold text-foreground flex items-center gap-3">
              <Shield className="w-8 h-8 text-primary" /> Rights Registry
            </h2>
            <p className="text-muted-foreground mt-1">
              Track ownership, splits, licensing, and generate proof-of-ownership documents
            </p>
          </div>
          <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
            <DialogTrigger asChild>
              <Button><Plus className="w-4 h-4 mr-2" /> Register Work</Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader><DialogTitle>Register a New Work</DialogTitle></DialogHeader>
              <div className="space-y-4 pt-2">
                <div>
                  <label className="text-sm font-medium">Title *</label>
                  <Input value={newTitle} onChange={(e) => setNewTitle(e.target.value)}
                    placeholder="Song or composition title" />
                </div>
                <div>
                  <label className="text-sm font-medium">Artist *</label>
                  <Select value={newArtistId} onValueChange={(v) => { setNewArtistId(v); setNewProjectId(""); }}>
                    <SelectTrigger><SelectValue placeholder="Select artist" /></SelectTrigger>
                    <SelectContent>
                      {(artistsQuery.data || []).map((a) => (
                        <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-sm font-medium">Project *</label>
                  <Select value={newProjectId} onValueChange={setNewProjectId} disabled={!newArtistId}>
                    <SelectTrigger><SelectValue placeholder={newArtistId ? "Select project" : "Select artist first"} /></SelectTrigger>
                    <SelectContent>
                      {(projectsQuery.data || []).map((p) => (
                        <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <label className="text-sm font-medium">Work Type</label>
                  <Select value={newWorkType} onValueChange={setNewWorkType}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {WORK_TYPES.map((t) => (
                        <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-sm font-medium">ISRC</label>
                    <Input value={newIsrc} onChange={(e) => setNewIsrc(e.target.value)}
                      placeholder="e.g. USRC17607839" />
                  </div>
                  <div>
                    <label className="text-sm font-medium">ISWC</label>
                    <Input value={newIswc} onChange={(e) => setNewIswc(e.target.value)}
                      placeholder="e.g. T-345246800-1" />
                  </div>
                </div>
                <Button onClick={handleCreate}
                  disabled={!newTitle.trim() || !newArtistId || !newProjectId || createWork.isPending}
                  className="w-full">
                  {createWork.isPending ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Plus className="w-4 h-4 mr-2" />}
                  Register Work
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Pending Your Review — works needing action */}
        {pendingWorks.length > 0 && (
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-amber-500" /> Pending Your Review
            </h3>
            <div className="flex gap-3 overflow-x-auto pb-2">
              {pendingWorks.map((work) => (
                <Card key={work.id}
                  className="min-w-[250px] hover:border-primary/50 transition-colors cursor-pointer border-amber-200 bg-amber-50/30"
                  onClick={() => navigate(`/tools/registry/${work.id}`)}>
                  <CardContent className="p-4">
                    <div className="font-medium text-sm">{work.title}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {work.work_type.replace("_", " ").toUpperCase()}
                    </div>
                    <Badge className="mt-2 bg-amber-100 text-amber-800">Needs Your Confirmation</Badge>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* My Works as Collaborator — all works user is involved in across all artists/projects */}
        {allCollabWorks.length > 0 && (
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
              <Users className="w-5 h-5 text-primary" /> Shared Works
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {allCollabWorks.map((work) => (
                <Card key={work.id}
                  className="hover:border-primary/50 transition-colors cursor-pointer"
                  onClick={() => navigate(`/tools/registry/${work.id}`)}>
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="font-medium text-sm">{work.title}</div>
                        <div className="text-xs text-muted-foreground mt-0.5">
                          {getArtistName(work.artist_id)} · {work.work_type.replace("_", " ").toUpperCase()}
                        </div>
                      </div>
                      <Badge className={STATUS_COLORS[work.status] || "bg-gray-100 text-gray-800"}>
                        {work.status.replace("_", " ")}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="flex gap-3 mb-6">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by title, ISRC, or ISWC..." className="pl-9" />
          </div>
          <Select value={selectedArtistId} onValueChange={setSelectedArtistId}>
            <SelectTrigger className="w-48"><SelectValue placeholder="All Artists" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Artists</SelectItem>
              {(artistsQuery.data || []).map((a) => (
                <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Works Grid */}
        {worksQuery.isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        ) : filteredWorks.length === 0 ? (
          <div className="text-center py-20">
            <FileText className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-semibold text-foreground mb-2">No works registered</h3>
            <p className="text-muted-foreground mb-4">
              Register your first work to start tracking ownership and rights
            </p>
            <Button onClick={() => setShowCreateDialog(true)}>
              <Plus className="w-4 h-4 mr-2" /> Register Work
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredWorks.map((work) => (
              <Card key={work.id}
                className="hover:border-primary/50 transition-colors cursor-pointer group"
                onClick={() => navigate(`/tools/registry/${work.id}`)}>
                <CardHeader className="pb-2">
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-base leading-tight">{work.title}</CardTitle>
                    <Badge className={STATUS_COLORS[work.status] || "bg-gray-100 text-gray-800"}>
                      {work.status.replace("_", " ")}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{getArtistName(work.artist_id)}</p>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span>{work.work_type.replace("_", " ").toUpperCase()}</span>
                    {work.isrc && <span>ISRC: {work.isrc}</span>}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default Registry;
