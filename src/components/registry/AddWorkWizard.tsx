import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  Clock,
  ChevronRight,
  FileText,
  Folder,
  Loader2,
  Music,
  Plus,
  Search,
  Sparkles,
  Upload,
  X,
  AlertTriangle,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";
import { useCreateWork, useCreateStake, useInviteCollaborator } from "@/hooks/useRegistry";
import {
  useSpotifySearch,
  type SpotifyTrack,
} from "@/hooks/useSpotifySearch";
import {
  useParseContractSplits,
  type ParsedParty,
} from "@/hooks/useParseContractSplits";
import { Artwork } from "./Artwork";
import { RegistryAvatar } from "./RegistryAvatar";
import { RoyaltySplitsTable, type SplitRow } from "./RoyaltySplitsTable";

interface AddWorkWizardProps {
  open: boolean;
  onClose: () => void;
  /** When set, the wizard skips the destination step and uses these ids. */
  initialProjectId?: string;
  initialArtistId?: string;
}

interface ProjectOption {
  id: string;
  name: string;
  artist_id: string;
}

interface ArtistOption {
  id: string;
  name: string;
}

const TYPES = [
  ["single", "Single"],
  ["ep_track", "EP Track"],
  ["album_track", "Album Track"],
  ["composition", "Composition"],
  ["other", "Other"],
] as const;

export function AddWorkWizard({
  open,
  onClose,
  initialProjectId,
  initialArtistId,
}: AddWorkWizardProps) {
  const needsDestination = !initialProjectId;

  // -------- destination state --------
  const { user } = useAuth();
  const [selectedArtistId, setSelectedArtistId] = useState<string>(initialArtistId || "");
  const [selectedProjectId, setSelectedProjectId] = useState<string>(initialProjectId || "");

  const artistsQuery = useQuery<ArtistOption[]>({
    queryKey: ["registry-wizard-artists", user?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from("artists")
        .select("id, name")
        .order("name");
      if (error) throw error;
      return (data || []) as ArtistOption[];
    },
    enabled: open && !!user?.id && needsDestination,
  });

  const projectsQuery = useQuery<ProjectOption[]>({
    queryKey: ["registry-wizard-projects", user?.id, selectedArtistId],
    queryFn: async () => {
      if (!selectedArtistId) return [];
      const { data, error } = await supabase
        .from("projects")
        .select("id, name, artist_id")
        .eq("artist_id", selectedArtistId)
        .order("created_at", { ascending: false });
      if (error) throw error;
      return (data || []) as ProjectOption[];
    },
    enabled: open && !!user?.id && !!selectedArtistId && needsDestination,
  });

  const selectedArtist = artistsQuery.data?.find((a) => a.id === selectedArtistId);
  const selectedProject = projectsQuery.data?.find((p) => p.id === selectedProjectId);

  const artistIdInUse = needsDestination ? selectedArtistId : initialArtistId || "";
  const projectIdInUse = needsDestination ? selectedProjectId : initialProjectId || "";

  // Look up the chosen project + artist when launched from project detail
  const projectQuery = useQuery<ProjectOption | null>({
    queryKey: ["registry-wizard-fixed-project", initialProjectId],
    queryFn: async () => {
      if (!initialProjectId) return null;
      const { data, error } = await supabase
        .from("projects")
        .select("id, name, artist_id")
        .eq("id", initialProjectId)
        .maybeSingle();
      if (error) throw error;
      return (data as ProjectOption) || null;
    },
    enabled: open && !!initialProjectId,
  });
  const fixedArtistQuery = useQuery<ArtistOption | null>({
    queryKey: ["registry-wizard-fixed-artist", artistIdInUse],
    queryFn: async () => {
      if (!artistIdInUse) return null;
      const { data, error } = await supabase
        .from("artists")
        .select("id, name")
        .eq("id", artistIdInUse)
        .maybeSingle();
      if (error) throw error;
      return (data as ArtistOption) || null;
    },
    enabled: open && !!artistIdInUse,
  });

  const artistName =
    (needsDestination ? selectedArtist?.name : fixedArtistQuery.data?.name) || "";
  const projectName =
    (needsDestination ? selectedProject?.name : projectQuery.data?.name) || "";

  // -------- wizard state --------
  const [released, setReleased] = useState<null | boolean>(null);
  const [step, setStep] = useState(0);

  // Track details
  const [title, setTitle] = useState("");
  const [workType, setWorkType] = useState("single");
  const [meta, setMeta] = useState({
    isrc: "",
    upc: "",
    releaseDate: "",
    label: "",
    genre: "",
    spotifyUrl: "",
  });

  // Spotify search
  const [searchQuery, setSearchQuery] = useState("");
  const [searchEnabled, setSearchEnabled] = useState(false);
  const spotifyResults = useSpotifySearch(searchQuery, searchEnabled);
  const [chosen, setChosen] = useState<SpotifyTrack | null>(null);

  // Splits
  const [royMode, setRoyMode] = useState<null | "ai" | "manual">(null);
  const [contractFile, setContractFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const parseSplits = useParseContractSplits();
  const [splitRows, setSplitRows] = useState<SplitRow[]>([]);
  const [splitsSource, setSplitsSource] = useState<
    null | { type: "ai" | "manual"; file?: string; date: string }
  >(null);
  const [mainArtistInContract, setMainArtistInContract] = useState(true);

  // Mutations
  const createWork = useCreateWork();
  const createStake = useCreateStake();
  const inviteCollaborator = useInviteCollaborator();
  const [submitting, setSubmitting] = useState(false);

  const reset = () => {
    setSelectedArtistId(initialArtistId || "");
    setSelectedProjectId(initialProjectId || "");
    setReleased(null);
    setStep(0);
    setTitle("");
    setWorkType("single");
    setMeta({ isrc: "", upc: "", releaseDate: "", label: "", genre: "", spotifyUrl: "" });
    setSearchQuery("");
    setSearchEnabled(false);
    setChosen(null);
    setRoyMode(null);
    setContractFile(null);
    setSplitRows([]);
    setSplitsSource(null);
    setMainArtistInContract(true);
    setSubmitting(false);
  };

  useEffect(() => {
    if (!open) reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Step labels
  const innerSteps = released
    ? ["Release", "Find track", "Confirm", "Splits"]
    : ["Release", "Track details", "Splits"];
  const stepLabels = needsDestination ? ["Project", ...innerSteps] : innerSteps;

  const onDestination = needsDestination && (!selectedArtistId || !selectedProjectId);
  const royaltyStep = released ? 3 : 2;
  const onRoyalty = released !== null && step === royaltyStep;
  const currentStepIdx = onDestination
    ? 0
    : needsDestination
      ? released === null
        ? 1
        : step + 1
      : released === null
        ? 0
        : step;

  const runSearch = () => {
    setSearchQuery(title.trim());
    setSearchEnabled(true);
  };

  const chooseResult = (r: SpotifyTrack) => {
    setChosen(r);
    setMeta({
      isrc: r.isrc || "",
      upc: r.upc || "",
      releaseDate: r.release_date || "",
      label: r.label || "",
      genre: "",
      spotifyUrl: r.spotify_url || "",
    });
    setStep(2);
  };

  // Build initial split rows when entering the royalty step
  useEffect(() => {
    if (!onRoyalty) return;
    if (splitRows.length === 0 && royMode === null) {
      // Seed with just the user's row so manual mode has something to render later.
      setSplitRows([
        {
          key: "you",
          name: artistName || "You",
          role: "Primary Artist",
          isYou: true,
          master: 0,
          publishing: 0,
        },
      ]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onRoyalty]);

  const handleParse = async (
    source:
      | { kind: "upload"; file: File; displayName: string }
      | { kind: "project"; contractFileId: string; displayName: string }
  ) => {
    try {
      const result = await parseSplits.mutateAsync(
        source.kind === "upload"
          ? { file: source.file, mainArtistName: artistName }
          : { contractFileId: source.contractFileId, mainArtistName: artistName }
      );
      const today = new Date().toISOString().slice(0, 10);
      if (result.parties.length === 0) {
        toast.error("We couldn't read royalty splits from this contract");
        setRoyMode("manual");
        setSplitRows([
          {
            key: "you",
            name: artistName || "You",
            role: "Primary Artist",
            isYou: true,
            master: 0,
            publishing: 0,
          },
        ]);
        setSplitsSource(null);
        return;
      }
      setMainArtistInContract(result.main_artist_found);
      const rows: SplitRow[] = [];
      if (!result.main_artist_found) {
        rows.push({
          key: "you",
          name: artistName || "You",
          role: "Primary Artist",
          isYou: true,
          master: 0,
          publishing: 0,
        });
      }
      for (const p of result.parties) {
        rows.push({
          key: p.name,
          name: p.name,
          role: p.role,
          isYou: p.is_main_artist,
          master: Math.round(p.master_pct),
          publishing: Math.round(p.publishing_pct),
        });
      }
      setSplitRows(rows);
      setSplitsSource({ type: "ai", file: source.displayName, date: today });
    } catch (e) {
      toast.error((e as Error).message || "Parse failed");
    }
  };

  const finish = async () => {
    if (!projectIdInUse || !artistIdInUse) {
      toast.error("Pick a project first");
      return;
    }
    setSubmitting(true);
    try {
      const created = await createWork.mutateAsync({
        artist_id: artistIdInUse,
        project_id: projectIdInUse,
        title: (chosen?.title || title).trim(),
        work_type: workType,
        ...(meta.isrc ? { isrc: meta.isrc } : {}),
        ...(meta.upc ? { upc: meta.upc } : {}),
        ...(meta.releaseDate ? { release_date: meta.releaseDate } : {}),
        ...(meta.spotifyUrl ? { notes: meta.spotifyUrl } : {}),
      });
      const workId = (created as { id?: string })?.id;

      // Persist splits as ownership_stakes rows
      if (workId) {
        for (const row of splitRows) {
          const name = row.name?.trim();
          if (!name) continue;
          if (row.master > 0) {
            await createStake.mutateAsync({
              work_id: workId,
              stake_type: "master",
              holder_name: name,
              holder_role: row.role || (row.isYou ? "Primary Artist" : "Collaborator"),
              percentage: row.master,
              is_owner_stake: row.isYou === true,
            });
          }
          if (row.publishing > 0) {
            await createStake.mutateAsync({
              work_id: workId,
              stake_type: "publishing",
              holder_name: name,
              holder_role: row.role || (row.isYou ? "Primary Artist" : "Collaborator"),
              percentage: row.publishing,
              is_owner_stake: row.isYou === true,
            });
          }
        }
      }

      onClose();
    } catch (e) {
      toast.error((e as Error).message || "Failed to add work");
    } finally {
      setSubmitting(false);
    }
  };

  // -------- footer --------
  const footer = (() => {
    if (onDestination)
      return (
        <Button variant="outline" onClick={onClose}>
          Cancel
        </Button>
      );
    if (released === null) {
      return needsDestination ? (
        <Button
          variant="outline"
          onClick={() => {
            setSelectedProjectId("");
          }}
        >
          <ArrowLeft className="w-4 h-4 mr-1" /> Back
        </Button>
      ) : (
        <Button variant="outline" onClick={onClose}>
          Cancel
        </Button>
      );
    }
    if (onRoyalty) {
      return (
        <>
          <Button variant="outline" onClick={() => setStep(released ? 2 : 1)}>
            <ArrowLeft className="w-4 h-4 mr-1" /> Back
          </Button>
          <Button onClick={finish} disabled={submitting}>
            {submitting ? (
              <Loader2 className="w-4 h-4 mr-1 animate-spin" />
            ) : (
              <CheckCircle2 className="w-4 h-4 mr-1" />
            )}
            Add work
          </Button>
        </>
      );
    }
    if (released && step === 1) {
      return (
        <>
          <Button
            variant="outline"
            onClick={() => {
              setReleased(null);
              setStep(0);
              setSearchEnabled(false);
            }}
          >
            <ArrowLeft className="w-4 h-4 mr-1" /> Back
          </Button>
          <span className="text-xs text-muted-foreground">Pick a match to continue</span>
        </>
      );
    }
    if (released && step === 2) {
      return (
        <>
          <Button variant="outline" onClick={() => setStep(1)}>
            <ArrowLeft className="w-4 h-4 mr-1" /> Back to results
          </Button>
          <Button onClick={() => setStep(3)}>
            Continue <ArrowRight className="w-4 h-4 ml-1" />
          </Button>
        </>
      );
    }
    if (!released && step === 1) {
      return (
        <>
          <Button
            variant="outline"
            onClick={() => {
              setReleased(null);
              setStep(0);
            }}
          >
            <ArrowLeft className="w-4 h-4 mr-1" /> Back
          </Button>
          <Button disabled={!title.trim()} onClick={() => setStep(2)}>
            Continue <ArrowRight className="w-4 h-4 ml-1" />
          </Button>
        </>
      );
    }
    return null;
  })();

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Add a work</DialogTitle>
        </DialogHeader>

        {/* Step indicator */}
        <div className="flex items-center gap-1 px-1 mb-4">
          {stepLabels.map((label, i) => {
            const state = i < currentStepIdx ? "done" : i === currentStepIdx ? "active" : "todo";
            return (
              <div key={label} className="flex items-center gap-1 flex-1">
                <div
                  className={cn(
                    "w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-semibold shrink-0",
                    state === "done"
                      ? "bg-primary text-primary-foreground"
                      : state === "active"
                        ? "bg-primary/15 text-primary border border-primary/30"
                        : "bg-muted text-muted-foreground"
                  )}
                >
                  {state === "done" ? <CheckCircle2 className="w-3.5 h-3.5" /> : i + 1}
                </div>
                <span
                  className={cn(
                    "text-[11px] font-medium truncate",
                    state === "active" ? "text-foreground" : "text-muted-foreground"
                  )}
                >
                  {label}
                </span>
                {i < stepLabels.length - 1 && (
                  <span
                    className={cn(
                      "h-px flex-1",
                      i < currentStepIdx ? "bg-primary" : "bg-border"
                    )}
                  />
                )}
              </div>
            );
          })}
        </div>

        {/* Body */}
        <div className="max-h-[60vh] overflow-y-auto px-1 -mx-1">
          {onDestination ? (
            <DestinationStep
              artists={artistsQuery.data || []}
              projects={projectsQuery.data || []}
              selectedArtistId={selectedArtistId}
              onSelectArtist={(id) => {
                setSelectedArtistId(id);
                setSelectedProjectId("");
              }}
              onSelectProject={(id) => setSelectedProjectId(id)}
            />
          ) : released === null ? (
            <ReleaseStatusStep onPick={(val) => {
              setReleased(val);
              setStep(1);
            }} />
          ) : released && step === 1 ? (
            <SpotifyStep
              title={title}
              setTitle={setTitle}
              artistName={artistName}
              loading={spotifyResults.isFetching}
              results={spotifyResults.data}
              onSearch={runSearch}
              onPick={chooseResult}
              onFallbackManual={() => {
                setReleased(false);
                setStep(1);
              }}
            />
          ) : released && step === 2 && chosen ? (
            <ConfirmMetadataStep
              chosen={chosen}
              meta={meta}
              setMeta={setMeta}
              workType={workType}
              setWorkType={setWorkType}
              onChangeMatch={() => setStep(1)}
            />
          ) : !released && step === 1 ? (
            <UnreleasedStep
              title={title}
              setTitle={setTitle}
              workType={workType}
              setWorkType={setWorkType}
              artistName={artistName}
              meta={meta}
              setMeta={setMeta}
            />
          ) : onRoyalty ? (
            <RoyaltyStep
              royMode={royMode}
              setRoyMode={setRoyMode}
              contractFile={contractFile}
              setContractFile={setContractFile}
              fileInputRef={fileInputRef}
              parsing={parseSplits.isPending}
              splitRows={splitRows}
              setSplitRows={setSplitRows}
              splitsSource={splitsSource}
              onParse={handleParse}
              artistName={artistName}
              mainArtistInContract={mainArtistInContract}
              projectId={projectIdInUse}
            />
          ) : null}
        </div>

        <DialogFooter className="gap-2">{footer}</DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================
// Steps
// ============================================================

function DestinationStep({
  artists,
  projects,
  selectedArtistId,
  onSelectArtist,
  onSelectProject,
}: {
  artists: ArtistOption[];
  projects: ProjectOption[];
  selectedArtistId: string;
  onSelectArtist: (id: string) => void;
  onSelectProject: (id: string) => void;
}) {
  return (
    <div className="space-y-4">
      <div>
        <p className="text-base font-semibold">Where does this work belong?</p>
        <p className="text-sm text-muted-foreground">
          Link it to an artist and one of their projects.
        </p>
      </div>
      <div>
        <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Artist
        </label>
        <Select value={selectedArtistId} onValueChange={onSelectArtist}>
          <SelectTrigger className="mt-1">
            <SelectValue placeholder="Pick an artist…" />
          </SelectTrigger>
          <SelectContent>
            {artists.map((a) => (
              <SelectItem key={a.id} value={a.id}>
                {a.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {selectedArtistId && (
        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Project
          </label>
          {projects.length === 0 ? (
            <p className="text-sm text-muted-foreground mt-2">
              No projects yet for this artist. Create one from Portfolio first.
            </p>
          ) : (
            <div className="mt-2 grid grid-cols-1 gap-1.5 max-h-60 overflow-y-auto">
              {projects.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => onSelectProject(p.id)}
                  className="flex items-center gap-3 p-3 rounded-lg border hover:bg-muted/40 hover:border-primary/40 text-left"
                >
                  <Folder className="w-4 h-4 text-muted-foreground" />
                  <span className="flex-1 text-sm font-medium">{p.name}</span>
                  <ChevronRight className="w-4 h-4 text-muted-foreground" />
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ReleaseStatusStep({ onPick }: { onPick: (val: boolean) => void }) {
  return (
    <div className="space-y-4">
      <div>
        <p className="text-base font-semibold">Is this track released?</p>
        <p className="text-sm text-muted-foreground">
          We'll pull metadata from Spotify for released tracks, or let you enter it by hand
          for unreleased ones.
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <button
          type="button"
          onClick={() => onPick(true)}
          className="rounded-xl border p-5 text-left hover:border-primary/40 hover:bg-muted/40 flex flex-col gap-2"
        >
          <div className="w-10 h-10 rounded-lg bg-emerald-500/15 text-emerald-400 flex items-center justify-center">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div className="text-sm font-semibold">Yes, it's released</div>
          <div className="text-xs text-muted-foreground">
            Search streaming services and auto-fill ISRC, label, release date and more.
          </div>
        </button>
        <button
          type="button"
          onClick={() => onPick(false)}
          className="rounded-xl border p-5 text-left hover:border-primary/40 hover:bg-muted/40 flex flex-col gap-2"
        >
          <div className="w-10 h-10 rounded-lg bg-amber-500/15 text-amber-400 flex items-center justify-center">
            <Clock className="w-5 h-5" />
          </div>
          <div className="text-sm font-semibold">Not yet — it's unreleased</div>
          <div className="text-xs text-muted-foreground">
            Enter the details manually now. You can pull metadata later once it's live.
          </div>
        </button>
      </div>
    </div>
  );
}

function SpotifyStep({
  title,
  setTitle,
  artistName,
  loading,
  results,
  onSearch,
  onPick,
  onFallbackManual,
}: {
  title: string;
  setTitle: (t: string) => void;
  artistName: string;
  loading: boolean;
  results: SpotifyTrack[] | undefined;
  onSearch: () => void;
  onPick: (r: SpotifyTrack) => void;
  onFallbackManual: () => void;
}) {
  return (
    <div className="space-y-4">
      <div>
        <p className="text-base font-semibold">What's the track called?</p>
        <p className="text-sm text-muted-foreground">
          We'll search Spotify for <b>{artistName || "this artist"}</b> and show the closest
          matches.
        </p>
      </div>
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            autoFocus
            placeholder="e.g. Midnight Drive"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onSearch()}
            className="pl-9"
          />
        </div>
        <Button onClick={onSearch} disabled={!title.trim() || loading}>
          {loading ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : null}
          Search
        </Button>
      </div>

      {loading && (
        <div className="space-y-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3 p-2 rounded-lg border animate-pulse">
              <div className="w-11 h-11 rounded-lg bg-muted" />
              <div className="flex-1">
                <div className="h-3 w-1/2 bg-muted rounded mb-2" />
                <div className="h-2 w-1/3 bg-muted rounded" />
              </div>
            </div>
          ))}
        </div>
      )}

      {results && !loading && (
        <>
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              <b className="text-foreground">{results.length}</b> results for "{title.trim()}"
            </span>
            <span>Select the correct match</span>
          </div>
          <div className="space-y-1.5">
            {results.map((r) => (
              <button
                key={r.id}
                type="button"
                onClick={() => onPick(r)}
                className="w-full flex items-center gap-3 p-2 rounded-lg border hover:border-primary/40 hover:bg-muted/40 text-left"
              >
                {r.cover_url ? (
                  <img src={r.cover_url} alt="" className="w-11 h-11 rounded-md shrink-0" />
                ) : (
                  <Artwork seed={(r.title || "") + r.artist} hasArtwork size={44} />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold truncate">{r.title}</span>
                    {r.explicit && (
                      <span className="text-[9px] font-bold bg-muted text-muted-foreground rounded px-1">
                        E
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground truncate">
                    {r.artist} {r.album ? `· ${r.album}` : ""} {r.year ? `· ${r.year}` : ""}
                  </div>
                </div>
                <ChevronRight className="w-4 h-4 text-muted-foreground" />
              </button>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            Not seeing it?{" "}
            <button
              type="button"
              className="text-primary hover:underline"
              onClick={onFallbackManual}
            >
              Enter details manually instead
            </button>
            .
          </p>
        </>
      )}
    </div>
  );
}

function ConfirmMetadataStep({
  chosen,
  meta,
  setMeta,
  workType,
  setWorkType,
  onChangeMatch,
}: {
  chosen: SpotifyTrack;
  meta: { isrc: string; upc: string; releaseDate: string; label: string; genre: string; spotifyUrl: string };
  setMeta: React.Dispatch<React.SetStateAction<typeof meta>>;
  workType: string;
  setWorkType: (v: string) => void;
  onChangeMatch: () => void;
}) {
  const set = (patch: Partial<typeof meta>) => setMeta((m) => ({ ...m, ...patch }));
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-xs bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-lg px-3 py-2">
        <Sparkles className="w-3.5 h-3.5" />
        Metadata pulled from Spotify — review and edit before adding.
      </div>
      <div className="flex items-center gap-3 p-3 rounded-lg border">
        {chosen.cover_url ? (
          <img src={chosen.cover_url} alt="" className="w-14 h-14 rounded-md" />
        ) : (
          <Artwork seed={chosen.title || ""} hasArtwork size={56} />
        )}
        <div className="flex-1 min-w-0">
          <div className="text-sm font-bold truncate">{chosen.title}</div>
          <div className="text-xs text-muted-foreground truncate">
            {chosen.artist} {chosen.album ? `· ${chosen.album}` : ""}{" "}
            {chosen.year ? `· ${chosen.year}` : ""}
          </div>
        </div>
        <button
          type="button"
          className="text-xs text-primary hover:underline"
          onClick={onChangeMatch}
        >
          Change
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <WizardField label="ISRC">
          <Input
            className="font-mono"
            value={meta.isrc}
            onChange={(e) => set({ isrc: e.target.value })}
          />
        </WizardField>
        <WizardField label="Release date">
          <Input
            type="date"
            value={meta.releaseDate}
            onChange={(e) => set({ releaseDate: e.target.value })}
          />
        </WizardField>
        <WizardField label="Label">
          <Input value={meta.label} onChange={(e) => set({ label: e.target.value })} />
        </WizardField>
        <WizardField label="Genre">
          <Input value={meta.genre} onChange={(e) => set({ genre: e.target.value })} />
        </WizardField>
        <WizardField label="UPC">
          <Input
            className="font-mono"
            value={meta.upc}
            onChange={(e) => set({ upc: e.target.value })}
          />
        </WizardField>
        <WizardField label="Work type">
          <Select value={workType} onValueChange={setWorkType}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TYPES.map(([v, l]) => (
                <SelectItem key={v} value={v}>
                  {l}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </WizardField>
        <WizardField label="Spotify link" full>
          <Input value={meta.spotifyUrl} onChange={(e) => set({ spotifyUrl: e.target.value })} />
        </WizardField>
      </div>
    </div>
  );
}

function UnreleasedStep({
  title,
  setTitle,
  workType,
  setWorkType,
  artistName,
  meta,
  setMeta,
}: {
  title: string;
  setTitle: (v: string) => void;
  workType: string;
  setWorkType: (v: string) => void;
  artistName: string;
  meta: { isrc: string; upc: string; releaseDate: string; label: string; genre: string; spotifyUrl: string };
  setMeta: React.Dispatch<React.SetStateAction<typeof meta>>;
}) {
  const set = (patch: Partial<typeof meta>) => setMeta((m) => ({ ...m, ...patch }));
  return (
    <div className="space-y-4">
      <div className="flex items-start gap-2 text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-lg px-3 py-2">
        <Clock className="w-3.5 h-3.5 mt-0.5" />
        <span>
          Unreleased — enter what you know now. The rest can be filled in on the work page
          later.
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <WizardField label="Track name" required full>
          <Input
            autoFocus
            value={title}
            placeholder="Song or composition title"
            onChange={(e) => setTitle(e.target.value)}
          />
        </WizardField>
        <WizardField label="Work type">
          <Select value={workType} onValueChange={setWorkType}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TYPES.map(([v, l]) => (
                <SelectItem key={v} value={v}>
                  {l}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </WizardField>
        <WizardField label="Artist">
          <div className="h-10 rounded-md border bg-muted/40 px-3 flex items-center gap-2 text-sm">
            <RegistryAvatar name={artistName || "?"} size={20} />
            {artistName || "—"}
          </div>
        </WizardField>
        <WizardField label="ISRC" hint="Optional">
          <Input
            className="font-mono"
            value={meta.isrc}
            placeholder="e.g. USRC17607831"
            onChange={(e) => set({ isrc: e.target.value })}
          />
        </WizardField>
        <WizardField label="UPC" hint="Optional">
          <Input
            className="font-mono"
            value={meta.upc}
            placeholder="000000000000"
            onChange={(e) => set({ upc: e.target.value })}
          />
        </WizardField>
        <WizardField label="Genre">
          <Input value={meta.genre} onChange={(e) => set({ genre: e.target.value })} />
        </WizardField>
        <WizardField label="Expected label">
          <Input value={meta.label} onChange={(e) => set({ label: e.target.value })} />
        </WizardField>
        <WizardField label="Target release date" full>
          <Input
            type="date"
            value={meta.releaseDate}
            onChange={(e) => set({ releaseDate: e.target.value })}
          />
        </WizardField>
      </div>
    </div>
  );
}

interface ProjectContract {
  id: string;
  file_name: string;
  folder_category: string;
  created_at: string;
}

type AiSource = "upload" | "project";

function RoyaltyStep({
  royMode,
  setRoyMode,
  contractFile,
  setContractFile,
  fileInputRef,
  parsing,
  splitRows,
  setSplitRows,
  splitsSource,
  onParse,
  artistName,
  mainArtistInContract,
  projectId,
}: {
  royMode: null | "ai" | "manual";
  setRoyMode: (v: null | "ai" | "manual") => void;
  contractFile: File | null;
  setContractFile: (f: File | null) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  parsing: boolean;
  splitRows: SplitRow[];
  setSplitRows: (r: SplitRow[]) => void;
  splitsSource: null | { type: "ai" | "manual"; file?: string; date: string };
  onParse: (
    source:
      | { kind: "upload"; file: File; displayName: string }
      | { kind: "project"; contractFileId: string; displayName: string }
  ) => void;
  artistName: string;
  mainArtistInContract: boolean;
  projectId: string;
}) {
  const [aiSource, setAiSource] = useState<AiSource>("project");
  const [pickedContract, setPickedContract] = useState<ProjectContract | null>(null);

  const projectContractsQuery = useQuery<ProjectContract[]>({
    queryKey: ["wizard-project-contracts", projectId],
    queryFn: async () => {
      if (!projectId) return [];
      const { data, error } = await supabase
        .from("project_files")
        .select("id, file_name, folder_category, created_at")
        .eq("project_id", projectId)
        .in("folder_category", ["contract", "split_sheet"])
        .order("created_at", { ascending: false });
      if (error) throw error;
      return (data || []) as ProjectContract[];
    },
    enabled: !!projectId && royMode === "ai",
  });

  const triggerParse = () => {
    if (aiSource === "upload" && contractFile) {
      onParse({ kind: "upload", file: contractFile, displayName: contractFile.name });
    } else if (aiSource === "project" && pickedContract) {
      onParse({
        kind: "project",
        contractFileId: pickedContract.id,
        displayName: pickedContract.file_name,
      });
    }
  };

  const canParse =
    !parsing &&
    ((aiSource === "upload" && !!contractFile) ||
      (aiSource === "project" && !!pickedContract));

  return (
    <div className="space-y-4">
      <div>
        <p className="text-base font-semibold">Royalty splits</p>
        <p className="text-sm text-muted-foreground">
          Pull splits from the related contract with AI, or set them by hand.
        </p>
      </div>

      {royMode === null && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => setRoyMode("ai")}
            className="rounded-xl border p-5 text-left hover:border-primary/40 hover:bg-muted/40 flex flex-col gap-2"
          >
            <div className="w-10 h-10 rounded-lg bg-emerald-500/15 text-emerald-400 flex items-center justify-center">
              <Sparkles className="w-5 h-5" />
            </div>
            <div className="text-sm font-semibold">Pull from the contract</div>
            <div className="text-xs text-muted-foreground">
              Attach the related contract — AI extracts the royalty split for every party.
            </div>
          </button>
          <button
            type="button"
            onClick={() => setRoyMode("manual")}
            className="rounded-xl border p-5 text-left hover:border-primary/40 hover:bg-muted/40 flex flex-col gap-2"
          >
            <div className="w-10 h-10 rounded-lg bg-amber-500/15 text-amber-400 flex items-center justify-center">
              <Plus className="w-5 h-5" />
            </div>
            <div className="text-sm font-semibold">I'll add them by hand</div>
            <div className="text-xs text-muted-foreground">
              Type splits directly, or skip and add them later on the work page.
            </div>
          </button>
        </div>
      )}

      {royMode === "ai" && splitRows.filter((r) => !r.isYou).length === 0 && (
        <>
          <div className="inline-flex items-center gap-1 rounded-lg border bg-muted/40 p-1">
            <button
              type="button"
              onClick={() => setAiSource("project")}
              className={cn(
                "inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                aiSource === "project"
                  ? "bg-background text-primary shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <FileText className="w-3.5 h-3.5" /> Pick from project
            </button>
            <button
              type="button"
              onClick={() => setAiSource("upload")}
              className={cn(
                "inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                aiSource === "upload"
                  ? "bg-background text-primary shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Upload className="w-3.5 h-3.5" /> Upload new
            </button>
          </div>

          {aiSource === "project" && (
            <div>
              {projectContractsQuery.isLoading ? (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  <Loader2 className="w-4 h-4 inline mr-2 animate-spin" />
                  Loading project contracts…
                </div>
              ) : (projectContractsQuery.data || []).length === 0 ? (
                <div className="rounded-xl border border-dashed p-6 text-center">
                  <FileText className="w-6 h-6 text-muted-foreground/40 mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">
                    No contracts or split sheets uploaded to this project yet.
                  </p>
                  <p className="text-xs text-muted-foreground/70 mt-1">
                    Switch to "Upload new" or add a contract from the Files tab first.
                  </p>
                </div>
              ) : (
                <div className="max-h-60 overflow-y-auto rounded-lg border divide-y">
                  {(projectContractsQuery.data || []).map((c) => {
                    const picked = pickedContract?.id === c.id;
                    return (
                      <button
                        key={c.id}
                        type="button"
                        onClick={() => setPickedContract(c)}
                        className={cn(
                          "w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-muted/40 transition-colors",
                          picked && "bg-muted/40"
                        )}
                      >
                        <FileText
                          className={cn(
                            "w-4 h-4 shrink-0",
                            picked ? "text-primary" : "text-muted-foreground"
                          )}
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium truncate">{c.file_name}</div>
                          <div className="text-[11px] text-muted-foreground">
                            {c.folder_category === "split_sheet" ? "Split sheet" : "Contract"}
                          </div>
                        </div>
                        {picked && <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {aiSource === "upload" && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf,.pdf"
                className="hidden"
                onChange={(e) => setContractFile(e.target.files?.[0] || null)}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  "w-full rounded-xl border-2 border-dashed p-6 flex flex-col items-center gap-2 hover:border-primary/40 hover:bg-muted/40",
                  contractFile && "border-primary/40 bg-muted/40"
                )}
              >
                {contractFile ? (
                  <Sparkles className="w-6 h-6 text-primary" />
                ) : (
                  <Upload className="w-6 h-6 text-muted-foreground" />
                )}
                <div className="text-sm font-semibold">
                  {contractFile?.name || "Click to choose the related contract"}
                </div>
                <div className="text-xs text-muted-foreground">
                  {contractFile ? "Ready to analyze" : "PDF up to ~25 MB · text is read by AI"}
                </div>
              </button>
            </>
          )}

          <div className="flex items-center gap-3">
            <Button onClick={triggerParse} disabled={!canParse}>
              {parsing ? (
                <Loader2 className="w-4 h-4 mr-1 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4 mr-1" />
              )}
              {parsing ? "Reading contract…" : "Parse with AI"}
            </Button>
            <button
              type="button"
              className="text-xs text-primary hover:underline"
              onClick={() => {
                setRoyMode(null);
                setContractFile(null);
                setPickedContract(null);
              }}
            >
              Back to options
            </button>
          </div>
        </>
      )}

      {royMode === "ai" && splitRows.length > 0 && (
        <>
          {!mainArtistInContract && (
            <div className="flex items-start gap-2 text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-lg px-3 py-2">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              <span>
                We didn't see <b>{artistName}</b> in the contract. Enter your own master /
                publishing % manually in the "You" row below.
              </span>
            </div>
          )}
          <RoyaltySplitsTable
            rows={splitRows}
            onChange={setSplitRows}
            editable
            allowAddRow
            source={splitsSource}
            warnOnImbalance={false}
          />
        </>
      )}

      {royMode === "manual" && (
        <>
          <div className="flex items-start gap-2 text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-lg px-3 py-2">
            <Clock className="w-3.5 h-3.5 mt-0.5" />
            <span>
              Set your own split now, or skip — everything stays editable on the work page.
            </span>
          </div>
          <RoyaltySplitsTable
            rows={
              splitRows.length > 0
                ? splitRows
                : [
                    {
                      key: "you",
                      name: artistName || "You",
                      role: "Primary Artist",
                      isYou: true,
                      master: 0,
                      publishing: 0,
                    },
                  ]
            }
            onChange={setSplitRows}
            editable
            allowAddRow
            warnOnImbalance={false}
          />
        </>
      )}
    </div>
  );
}

function WizardField({
  label,
  required,
  hint,
  full,
  children,
}: {
  label: string;
  required?: boolean;
  hint?: string;
  full?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className={cn(full && "md:col-span-2")}>
      <label className="text-xs font-semibold text-foreground mb-1.5 block">
        {label}
        {required && <span className="text-destructive"> *</span>}
      </label>
      {children}
      {hint && <p className="text-[11px] text-muted-foreground mt-1">{hint}</p>}
    </div>
  );
}
