import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Music,
  ArrowLeft,
  ArrowRight,
  Plus,
  Trash2,
  Loader2,
  FileText,
  Download,
  Info,
  Save,
  Users,
  CheckCircle,
  Folder,
  BookOpen,
} from "lucide-react";
import { toast } from "sonner";

const API_URL =
  import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

const ROLES = [
  "Artist",
  "Producer",
  "Songwriter",
  "Composer",
  "Lyricist",
  "Arranger",
  "Engineer",
  "Featured Artist",
  "Other",
];

interface Contributor {
  id: string;
  name: string;
  role: string;
  publishingPercentage: string;
  masterPercentage: string;
  publisherOrLabel: string;
  ipi: string;
}

interface Artist {
  id: string;
  name: string;
}

interface Project {
  id: string;
  name: string;
}

const createEmptyContributor = (): Contributor => ({
  id: crypto.randomUUID(),
  name: "",
  role: "",
  publishingPercentage: "",
  masterPercentage: "",
  publisherOrLabel: "",
  ipi: "",
});

const STEPS = [
  { label: "Details", icon: FileText },
  { label: "Splits", icon: Users },
  { label: "Summary", icon: Download },
];

const Req = () => <span className="text-green-600">*</span>;

const SplitSheet = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [step, setStep] = useState(0);

  // Step 1
  const [workTitle, setWorkTitle] = useState("");
  const [workType, setWorkType] = useState("single");
  const [splitType, setSplitType] = useState("both");
  const [date, setDate] = useState(() => new Date().toISOString().split("T")[0]);
  const [selectedArtistId, setSelectedArtistId] = useState("");
  const [artists, setArtists] = useState<Artist[]>([]);
  const [loadingArtists, setLoadingArtists] = useState(true);

  // Step 2
  const [contributors, setContributors] = useState<Contributor[]>([
    createEmptyContributor(),
    createEmptyContributor(),
  ]);

  // Step 3
  const [format, setFormat] = useState<"pdf" | "docx">("pdf");
  const [saveToArtist, setSaveToArtist] = useState(false);
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [projects, setProjects] = useState<Project[]>([]);
  const [loadingProjects, setLoadingProjects] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [hasGenerated, setHasGenerated] = useState(false);
  const [generatedBlob, setGeneratedBlob] = useState<Blob | null>(null);

  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.splitsheet, {
    onComplete: () => markToolCompleted("splitsheet"),
  });

  useEffect(() => {
    if (!onboardingLoading && !statuses.splitsheet && walkthrough.phase === "idle") {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.splitsheet]);

  // Fetch artists
  useEffect(() => {
    if (!user?.id) return;
    setLoadingArtists(true);
    fetch(`${API_URL}/artists?user_id=${user.id}`)
      .then((res) => res.json())
      .then((data) => setArtists(data || []))
      .catch(() => setArtists([]))
      .finally(() => setLoadingArtists(false));
  }, [user?.id]);

  // Fetch projects when artist changes
  useEffect(() => {
    if (!selectedArtistId) {
      setProjects([]);
      setSelectedProjectId("");
      return;
    }
    setLoadingProjects(true);
    fetch(`${API_URL}/projects/${selectedArtistId}`)
      .then((res) => res.json())
      .then((data) => setProjects(data || []))
      .catch(() => setProjects([]))
      .finally(() => setLoadingProjects(false));
  }, [selectedArtistId]);

  // Validation
  const pubTotal = contributors.reduce(
    (sum, c) => sum + (parseFloat(c.publishingPercentage) || 0),
    0
  );
  const masterTotal = contributors.reduce(
    (sum, c) => sum + (parseFloat(c.masterPercentage) || 0),
    0
  );

  const needsPublishing = splitType === "publishing" || splitType === "both";
  const needsMaster = splitType === "master" || splitType === "both";

  const isPubValid = !needsPublishing || Math.abs(pubTotal - 100) < 0.01;
  const isMasterValid = !needsMaster || Math.abs(masterTotal - 100) < 0.01;
  const isValidTotal = isPubValid && isMasterValid;

  const allNamesPresent =
    contributors.length > 0 && contributors.every((c) => c.name.trim() !== "");
  const allRolesPresent = contributors.every((c) => c.role !== "");

  const canProceedStep1 = workTitle.trim() !== "" && selectedArtistId !== "";
  const canProceedStep2 = allNamesPresent && allRolesPresent && isValidTotal;
  const canGenerate =
    canProceedStep1 &&
    canProceedStep2 &&
    !isGenerating &&
    (!saveToArtist || selectedProjectId !== "");

  const updateContributor = (id: string, field: keyof Contributor, value: string) => {
    setContributors((prev) =>
      prev.map((c) => (c.id === id ? { ...c, [field]: value } : c))
    );
  };

  const addContributor = () =>
    setContributors((prev) => [...prev, createEmptyContributor()]);

  const removeContributor = (id: string) => {
    if (contributors.length <= 1) return;
    setContributors((prev) => prev.filter((c) => c.id !== id));
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    setHasGenerated(false);
    setGeneratedBlob(null);
    try {
      const response = await fetch(`${API_URL}/splitsheet/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          work_title: workTitle,
          work_type: workType,
          split_type: splitType,
          date,
          format,
          save_to_artist: saveToArtist,
          artist_id: selectedArtistId || null,
          project_id: saveToArtist && selectedProjectId ? selectedProjectId : null,
          contributors: contributors.map((c) => ({
            name: c.name,
            role: c.role,
            publishing_percentage: needsPublishing ? parseFloat(c.publishingPercentage) || 0 : null,
            master_percentage: needsMaster ? parseFloat(c.masterPercentage) || 0 : null,
            publisher_or_label: c.publisherOrLabel || null,
            ipi_number: c.ipi || null,
          })),
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => null);
        throw new Error(err?.detail || "Failed to generate split sheet");
      }

      const blob = await response.blob();
      setGeneratedBlob(blob);
      setHasGenerated(true);

      const savedMsg = saveToArtist ? " and saved to artist profile" : "";
      toast.success(`Split sheet generated${savedMsg} successfully`);
    } catch (err: any) {
      toast.error(err.message || "Failed to generate split sheet");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!generatedBlob) return;
    const url = window.URL.createObjectURL(generatedBlob);
    const a = document.createElement("a");
    a.href = url;
    const ext = format === "docx" ? "docx" : "pdf";
    a.download = `Split_Sheet_${workTitle.replace(/\s+/g, "_")}.${ext}`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const selectedArtistName = artists.find((a) => a.id === selectedArtistId)?.name;

  const splitTypeLabel =
    splitType === "both"
      ? "Publishing & Master"
      : splitType === "publishing"
      ? "Publishing"
      : "Master";

  const goBack = (toStep: number) => {
    setHasGenerated(false);
    setGeneratedBlob(null);
    setStep(toStep);
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground hover:text-foreground"
              onClick={() => navigate(-1)}
            >
              <ArrowLeft className="w-4 h-4 mr-1" /> Back
            </Button>
            <div className="w-px h-6 bg-border" />
            <div
              className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => navigate("/dashboard")}
            >
              <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-6 h-6 text-primary-foreground" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/docs")}
              title="Documentation"
              className="text-muted-foreground hover:text-foreground"
            >
              <BookOpen className="w-4 h-4" />
            </Button>
            <ToolHelpButton onClick={walkthrough.replay} />
            <Button variant="outline" onClick={() => navigate("/tools")}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Tools
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-3xl">
        <div className="mb-6">
          <h2 className="text-3xl font-bold text-foreground">Split Sheet Generator</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Create professional split sheet documents for your music
          </p>
        </div>

        {/* Explainer */}
        <div data-walkthrough="splitsheet-info" className="flex gap-2 items-start mb-6 px-4 py-3 rounded-lg bg-primary/5 border border-primary/20">
          <Info className="w-4 h-4 text-primary mt-0.5 shrink-0" />
          <p className="text-xs text-muted-foreground leading-relaxed">
            A <span className="font-medium text-foreground">split sheet</span> documents who owns what
            share of a song's royalties. It can cover <span className="font-medium text-foreground">publishing</span> (songwriting),{" "}
            <span className="font-medium text-foreground">master</span> (sound recording), or both.
          </p>
        </div>

        {/* Step Indicator */}
        <div data-walkthrough="splitsheet-steps" className="flex items-center justify-center mb-8 gap-2">
          {STEPS.map((s, i) => {
            const Icon = s.icon;
            const isActive = i === step;
            const isCompleted = i < step;
            return (
              <div key={i} className="flex items-center gap-2">
                {i > 0 && (
                  <div
                    className={`h-px w-8 md:w-16 ${
                      isCompleted ? "bg-primary" : "bg-border"
                    }`}
                  />
                )}
                <button
                  onClick={() => i < step && goBack(i)}
                  disabled={i > step}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : isCompleted
                      ? "bg-primary/10 text-primary cursor-pointer hover:bg-primary/20"
                      : "bg-muted text-muted-foreground cursor-not-allowed"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden md:inline">{s.label}</span>
                </button>
              </div>
            );
          })}
        </div>

        {/* ==================== Step 1: Work Details ==================== */}
        {step === 0 && (
          <div className="space-y-5">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-lg">Work Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium mb-1 block">
                      Artist <Req />
                    </label>
                    {loadingArtists ? (
                      <div className="flex items-center gap-2 text-sm text-muted-foreground py-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Loading...
                      </div>
                    ) : artists.length === 0 ? (
                      <p className="text-sm text-muted-foreground py-2">
                        No artists found.{" "}
                        <button className="text-primary underline" onClick={() => navigate("/artists/new")}>
                          Create one
                        </button>
                      </p>
                    ) : (
                      <Select value={selectedArtistId} onValueChange={setSelectedArtistId}>
                        <SelectTrigger><SelectValue placeholder="Select artist" /></SelectTrigger>
                        <SelectContent>
                          {artists.map((a) => (
                            <SelectItem key={a.id} value={a.id}>{a.name}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-1 block">
                      Work Title <Req />
                    </label>
                    <Input
                      placeholder="Song, album, or project name"
                      value={workTitle}
                      onChange={(e) => setWorkTitle(e.target.value)}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium mb-1 block">Type</label>
                    <Select value={workType} onValueChange={setWorkType}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="single">Single</SelectItem>
                        <SelectItem value="album">Album</SelectItem>
                        <SelectItem value="ep">EP</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-1 block">Date</label>
                    <Input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
                  </div>
                </div>

                {/* Split Type */}
                <div data-walkthrough="splitsheet-royalty-type">
                  <label className="text-sm font-medium mb-2 block">
                    Royalty Type <Req />
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { value: "publishing", label: "Publishing" },
                      { value: "master", label: "Master" },
                      { value: "both", label: "Both" },
                    ].map((st) => (
                      <button
                        key={st.value}
                        onClick={() => setSplitType(st.value)}
                        className={`py-2.5 px-3 rounded-lg border-2 text-sm font-medium transition-colors ${
                          splitType === st.value
                            ? "border-primary bg-primary/5 text-primary"
                            : "border-border hover:border-primary/30 text-muted-foreground"
                        }`}
                      >
                        {st.label}
                      </button>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1.5">
                    {splitType === "publishing"
                      ? "Songwriting royalties — composition & lyrics"
                      : splitType === "master"
                      ? "Sound recording royalties — master ownership"
                      : "Both publishing and master royalty splits"}
                  </p>
                </div>
              </CardContent>
            </Card>

            <div className="flex justify-end">
              <Button disabled={!canProceedStep1} onClick={() => setStep(1)}>
                Next
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </div>
          </div>
        )}

        {/* ==================== Step 2: Ownership Splits ==================== */}
        {step === 1 && (
          <div className="space-y-5">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-lg flex items-center justify-between">
                  <span>Contributors</span>
                  <div className="flex gap-3 text-xs font-medium">
                    {needsPublishing && (
                      <span className={isPubValid ? "text-green-600" : "text-red-500"}>
                        Pub: {pubTotal.toFixed(1)}%
                      </span>
                    )}
                    {needsMaster && (
                      <span className={isMasterValid ? "text-green-600" : "text-red-500"}>
                        Master: {masterTotal.toFixed(1)}%
                      </span>
                    )}
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {contributors.map((c, index) => (
                  <div key={c.id} className="border border-border rounded-lg p-3 space-y-2.5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-muted-foreground">
                        #{index + 1}
                      </span>
                      {contributors.length > 1 && (
                        <button
                          onClick={() => removeContributor(c.id)}
                          className="text-muted-foreground hover:text-destructive transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <Input
                        placeholder="Name *"
                        value={c.name}
                        onChange={(e) => updateContributor(c.id, "name", e.target.value)}
                      />
                      <Select
                        value={c.role}
                        onValueChange={(val) => updateContributor(c.id, "role", val)}
                      >
                        <SelectTrigger><SelectValue placeholder="Role *" /></SelectTrigger>
                        <SelectContent>
                          {ROLES.map((role) => (
                            <SelectItem key={role} value={role}>{role}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className={`grid gap-2 ${splitType === "both" ? "grid-cols-2" : "grid-cols-1"}`}>
                      {needsPublishing && (
                        <Input
                          type="number"
                          min="0"
                          max="100"
                          step="0.01"
                          placeholder={splitType === "both" ? "Publishing %" : "Ownership %"}
                          value={c.publishingPercentage}
                          onChange={(e) =>
                            updateContributor(c.id, "publishingPercentage", e.target.value)
                          }
                        />
                      )}
                      {needsMaster && (
                        <Input
                          type="number"
                          min="0"
                          max="100"
                          step="0.01"
                          placeholder={splitType === "both" ? "Master %" : "Ownership %"}
                          value={c.masterPercentage}
                          onChange={(e) =>
                            updateContributor(c.id, "masterPercentage", e.target.value)
                          }
                        />
                      )}
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <Input
                        placeholder="Publisher / Label"
                        value={c.publisherOrLabel}
                        onChange={(e) => updateContributor(c.id, "publisherOrLabel", e.target.value)}
                        className="text-sm"
                      />
                      <Input
                        placeholder="IPI / CAE #"
                        value={c.ipi}
                        onChange={(e) => updateContributor(c.id, "ipi", e.target.value)}
                        className="text-sm"
                      />
                    </div>
                  </div>
                ))}

                <Button variant="outline" size="sm" className="w-full" onClick={addContributor}>
                  <Plus className="w-4 h-4 mr-1" />
                  Add Contributor
                </Button>
              </CardContent>
            </Card>

            {!canProceedStep2 && (
              <p className="text-xs text-muted-foreground text-center">
                {!allNamesPresent
                  ? "All contributors need a name"
                  : !allRolesPresent
                  ? "All contributors need a role"
                  : !isPubValid
                  ? `Publishing splits must total 100% (currently ${pubTotal.toFixed(1)}%)`
                  : !isMasterValid
                  ? `Master splits must total 100% (currently ${masterTotal.toFixed(1)}%)`
                  : ""}
              </p>
            )}

            <div className="flex justify-between">
              <Button variant="outline" onClick={() => goBack(0)}>
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back
              </Button>
              <Button disabled={!canProceedStep2} onClick={() => setStep(2)}>
                Next
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </div>
          </div>
        )}

        {/* ==================== Step 3: Summary ==================== */}
        {step === 2 && (
          <div className="space-y-5">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-lg">Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Work info */}
                <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
                  <span className="text-muted-foreground">Artist</span>
                  <span className="font-medium">{selectedArtistName}</span>
                  <span className="text-muted-foreground">Title</span>
                  <span className="font-medium">{workTitle}</span>
                  <span className="text-muted-foreground">Type</span>
                  <span className="font-medium">{workType.toUpperCase()}</span>
                  <span className="text-muted-foreground">Royalty Type</span>
                  <span className="font-medium">{splitTypeLabel}</span>
                  <span className="text-muted-foreground">Date</span>
                  <span className="font-medium">{date}</span>
                </div>

                {/* Splits breakdown */}
                <div className="border-t border-border pt-3 space-y-3">
                  {needsPublishing && (
                    <div>
                      <div className="text-xs font-medium text-muted-foreground mb-1.5">
                        {splitType === "both" ? "Publishing Splits" : "Ownership Splits"}
                      </div>
                      {contributors.map((c) => (
                        <div key={c.id} className="flex justify-between text-sm py-0.5">
                          <span>
                            {c.name} <span className="text-muted-foreground">· {c.role}</span>
                          </span>
                          <span className="font-medium">
                            {parseFloat(c.publishingPercentage).toFixed(2)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                  {needsMaster && (
                    <div>
                      <div className="text-xs font-medium text-muted-foreground mb-1.5">
                        {splitType === "both" ? "Master Splits" : "Ownership Splits"}
                      </div>
                      {contributors.map((c) => (
                        <div key={c.id} className="flex justify-between text-sm py-0.5">
                          <span>
                            {c.name} <span className="text-muted-foreground">· {c.role}</span>
                          </span>
                          <span className="font-medium">
                            {parseFloat(c.masterPercentage).toFixed(2)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Format + Save */}
                <div className="border-t border-border pt-3 space-y-3">
                  <div className="flex gap-2">
                    <button
                      onClick={() => { setFormat("pdf"); setHasGenerated(false); setGeneratedBlob(null); }}
                      className={`flex-1 py-2 rounded-lg border-2 text-sm font-medium transition-colors ${
                        format === "pdf"
                          ? "border-primary bg-primary/5 text-primary"
                          : "border-border text-muted-foreground hover:border-primary/30"
                      }`}
                    >
                      PDF
                    </button>
                    <button
                      onClick={() => { setFormat("docx"); setHasGenerated(false); setGeneratedBlob(null); }}
                      className={`flex-1 py-2 rounded-lg border-2 text-sm font-medium transition-colors ${
                        format === "docx"
                          ? "border-primary bg-primary/5 text-primary"
                          : "border-border text-muted-foreground hover:border-primary/30"
                      }`}
                    >
                      Word
                    </button>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="save-to-artist"
                        checked={saveToArtist}
                        onCheckedChange={(checked) => {
                          setSaveToArtist(checked === true);
                          if (!checked) setSelectedProjectId("");
                        }}
                      />
                      <label htmlFor="save-to-artist" className="text-sm cursor-pointer">
                        Save to {selectedArtistName}'s profile
                      </label>
                    </div>

                    {saveToArtist && (
                      <div className="pl-6">
                        <label className="text-xs font-medium text-muted-foreground mb-1 block">
                          <Folder className="w-3 h-3 inline mr-1" />
                          Save to portfolio <Req />
                        </label>
                        {loadingProjects ? (
                          <div className="flex items-center gap-2 text-xs text-muted-foreground py-1">
                            <Loader2 className="w-3 h-3 animate-spin" />
                            Loading...
                          </div>
                        ) : projects.length === 0 ? (
                          <p className="text-xs text-muted-foreground py-1">
                            No projects found for this artist.
                          </p>
                        ) : (
                          <Select value={selectedProjectId} onValueChange={setSelectedProjectId}>
                            <SelectTrigger className="h-8 text-sm">
                              <SelectValue placeholder="Select a project" />
                            </SelectTrigger>
                            <SelectContent>
                              {projects.map((p) => (
                                <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="flex justify-between items-center">
              <Button variant="outline" onClick={() => goBack(1)}>
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back
              </Button>
              <div className="flex gap-2">
                {!hasGenerated ? (
                  <Button disabled={!canGenerate} onClick={handleGenerate}>
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        {saveToArtist ? (
                          <Save className="w-4 h-4 mr-2" />
                        ) : (
                          <FileText className="w-4 h-4 mr-2" />
                        )}
                        Generate
                      </>
                    )}
                  </Button>
                ) : (
                  <Button onClick={handleDownload} className="gap-2">
                    <Download className="w-4 h-4" />
                    Download
                  </Button>
                )}
              </div>
            </div>

            {hasGenerated && (
              <div className="flex items-center justify-center gap-2 text-sm text-green-600">
                <CheckCircle className="w-4 h-4" />
                Ready to download
              </div>
            )}
          </div>
        )}
        <ToolIntroModal
          config={TOOL_CONFIGS.splitsheet}
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
    </div>
  );
};

export default SplitSheet;
