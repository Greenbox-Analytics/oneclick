import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
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
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { useCreateWork, useCreateStake, useInviteCollaborator } from "@/hooks/useRegistry";
import { useStorageStatus } from "@/hooks/useEntitlements";
import {
  useSpotifySearch,
  type CreditedArtist,
  type SpotifyTrack,
} from "@/hooks/useSpotifySearch";
import {
  useParseContractSplits,
  type ParsedParty,
} from "@/hooks/useParseContractSplits";
import { Artwork } from "./Artwork";
import { RegistryAvatar } from "./RegistryAvatar";
import { RoyaltySplitsTable, type SplitRow } from "./RoyaltySplitsTable";
import { mergeParsedContracts, type MergeConflict } from "./mergeParsedContracts";
import { AddWorkConfirmDialog } from "./AddWorkConfirmDialog";
import { NewArtistDialog } from "@/components/NewArtistDialog";
import { ProjectFormDialog } from "@/components/ProjectFormDialog";

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

// A contract queued for AI split parsing. Splits are often spread across
// several contracts (producer deal, feature deal, …) that only together
// account for 100% — the queue lets the user parse them all and merge.
interface QueuedContract {
  id: string;
  kind: "upload" | "project";
  file?: File;
  contractFileId?: string;
  displayName: string;
  status: "pending" | "parsing" | "done" | "error";
  error?: string;
  parties?: ParsedParty[]; // raw parse result, kept for provenance + re-merge
  mainArtistFound?: boolean;
}

export function AddWorkWizard({
  open,
  onClose,
  initialProjectId,
  initialArtistId,
}: AddWorkWizardProps) {
  const needsDestination = !initialProjectId;

  // -------- destination state --------
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [selectedArtistId, setSelectedArtistId] = useState<string>(initialArtistId || "");
  const [selectedProjectId, setSelectedProjectId] = useState<string>(initialProjectId || "");

  // Inline create-artist / create-project — lets the user spin up a missing
  // destination without leaving the wizard.
  const [newArtistOpen, setNewArtistOpen] = useState(false);
  const [newProjectOpen, setNewProjectOpen] = useState(false);

  const handleArtistCreated = (artistId: string) => {
    // Refresh the wizard's artist list, then auto-pick the new artist. The
    // dialog itself invalidates portfolio queries; this one is wizard-scoped.
    queryClient.invalidateQueries({ queryKey: ["registry-wizard-artists"] });
    setSelectedArtistId(artistId);
    setSelectedProjectId("");
  };

  const handleProjectSave = async (data: { name: string; description: string; artist_id: string }) => {
    // Direct supabase insert — the auto_create_project_owner trigger fires for
    // user-client inserts and seeds ownership, so we don't need POST /projects.
    const { data: inserted, error } = await supabase
      .from("projects")
      .insert({ artist_id: data.artist_id, name: data.name, description: data.description || null })
      .select("id")
      .single();
    if (error) {
      toast.error(error.message || "Couldn't create project");
      return;
    }
    queryClient.invalidateQueries({ queryKey: ["registry-wizard-projects"] });
    setSelectedProjectId(inserted.id);
    toast.success("Project created");
  };

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

  // Manually-entered credited artists for the unreleased flow (no Spotify to
  // pull from). Saved to featured_artists, mirroring the released path's shape.
  const [manualArtists, setManualArtists] = useState<CreditedArtist[]>([]);

  // Splits
  const [royMode, setRoyMode] = useState<null | "ai" | "manual">(null);
  const [queuedContracts, setQueuedContracts] = useState<QueuedContract[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const parseSplits = useParseContractSplits();
  const [splitRows, setSplitRows] = useState<SplitRow[]>([]);
  const [splitsSource, setSplitsSource] = useState<
    null | { type: "ai" | "manual"; file?: string; date: string }
  >(null);
  const [mainArtistInContract, setMainArtistInContract] = useState(true);
  const [mergeConflicts, setMergeConflicts] = useState<MergeConflict[]>([]);
  // True once the user hand-edits the split table — guards against a re-parse
  // silently overwriting their edits.
  const [rowsDirty, setRowsDirty] = useState(false);

  // Mutations
  const createWork = useCreateWork();
  const createStake = useCreateStake();
  const inviteCollaborator = useInviteCollaborator();
  const storageStatus = useStorageStatus();
  const [submitting, setSubmitting] = useState(false);
  // Final review dialog shown between "Add work" and the actual submit.
  const [confirmOpen, setConfirmOpen] = useState(false);

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
    setManualArtists([]);
    setRoyMode(null);
    setQueuedContracts([]);
    setSplitRows([]);
    setSplitsSource(null);
    setMainArtistInContract(true);
    setMergeConflicts([]);
    setRowsDirty(false);
    setSubmitting(false);
    setConfirmOpen(false);
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

  const chooseResult = async (r: SpotifyTrack) => {
    setChosen(r);
    // Seed meta immediately from the search result so the Confirm step renders
    // without a flash of empty fields…
    setMeta({
      isrc: r.isrc || "",
      upc: r.upc || "",
      releaseDate: r.release_date || "",
      label: r.label || "",
      genre: r.genre || "",
      spotifyUrl: r.spotify_url || "",
    });
    setStep(2);
    // …then upgrade with the richer /tracks/{id} payload (album-fetched UPC,
    // artist-fetched genre). Falls back silently on network errors.
    try {
      const full = await apiFetch<SpotifyTrack>(`${API_URL}/integrations/spotify/tracks/${r.id}`);
      setChosen((prev) => (prev?.id === r.id ? { ...prev, ...full } : prev));
      setMeta((prev) => ({
        ...prev,
        isrc: prev.isrc || full.isrc || "",
        upc: prev.upc || full.upc || "",
        releaseDate: prev.releaseDate || full.release_date || "",
        label: prev.label || full.label || "",
        genre: prev.genre || full.genre || "",
        spotifyUrl: prev.spotifyUrl || full.spotify_url || "",
      }));
    } catch {
      // Best-effort enrichment — leave the search-result values in place.
    }
  };

  // ---- multi-contract queue ----
  const patchQueued = (id: string, patch: Partial<QueuedContract>) =>
    setQueuedContracts((prev) => prev.map((q) => (q.id === id ? { ...q, ...patch } : q)));

  const addContracts = (
    items: Array<
      | { kind: "upload"; file: File; displayName: string }
      | { kind: "project"; contractFileId: string; displayName: string }
    >
  ) =>
    setQueuedContracts((prev) => [
      ...prev,
      ...items.map((item) => ({ ...item, id: crypto.randomUUID(), status: "pending" as const })),
    ]);

  const clearContracts = () => setQueuedContracts([]);

  // Rebuild the split table from every successfully parsed contract.
  const applyMerge = (done: QueuedContract[]) => {
    const withParties = done.filter((q) => (q.parties?.length ?? 0) > 0);
    if (withParties.length === 0) {
      toast.error("We couldn't read royalty splits from these contracts");
      // Manual mode renders its own default "You" row when splitRows is empty.
      setRoyMode("manual");
      setSplitRows([]);
      setSplitsSource(null);
      setMergeConflicts([]);
      return;
    }
    const merged = mergeParsedContracts(withParties);
    setMainArtistInContract(merged.mainArtistFoundAny);
    setSplitRows(merged.rows);
    setMergeConflicts(merged.conflicts);
    setSplitsSource({
      type: "ai",
      file:
        withParties.length === 1
          ? withParties[0].displayName
          : `${withParties.length} contracts`,
      date: new Date().toISOString().slice(0, 10),
    });
    setRowsDirty(false);
  };

  const removeContract = (id: string) => {
    const removed = queuedContracts.find((q) => q.id === id);
    const remaining = queuedContracts.filter((q) => q.id !== id);
    setQueuedContracts(remaining);
    // Removing an already-merged contract re-merges the rest so its parties
    // drop out of the table (unless the user has hand-edited the rows).
    if (removed?.status === "done" && !rowsDirty && splitRows.some((r) => !r.isYou)) {
      const done = remaining.filter((q) => q.status === "done");
      if (done.length > 0) {
        applyMerge(done);
      } else {
        setSplitRows([]);
        setSplitsSource(null);
        setMergeConflicts([]);
        setMainArtistInContract(true);
      }
    }
  };

  const handleParseAll = async () => {
    if (
      rowsDirty &&
      splitRows.some((r) => !r.isYou) &&
      !window.confirm("Parsing will rebuild the split table and replace your edits. Continue?")
    ) {
      return;
    }
    const done: QueuedContract[] = [];
    for (const qc of queuedContracts) {
      if (qc.status === "done" && qc.parties) {
        // already parsed — reuse the result, don't re-run the AI
        done.push(qc);
        continue;
      }
      patchQueued(qc.id, { status: "parsing", error: undefined });
      try {
        const result = await parseSplits.mutateAsync(
          qc.kind === "upload"
            ? { file: qc.file!, mainArtistName: artistName }
            : { contractFileId: qc.contractFileId!, mainArtistName: artistName }
        );
        const parsed: QueuedContract = {
          ...qc,
          status: "done",
          parties: result.parties,
          mainArtistFound: result.main_artist_found,
        };
        patchQueued(qc.id, {
          status: "done",
          parties: result.parties,
          mainArtistFound: result.main_artist_found,
        });
        done.push(parsed);
      } catch (e) {
        patchQueued(qc.id, { status: "error", error: (e as Error).message || "Parse failed" });
      }
    }
    if (done.length === 0) {
      // every contract failed to parse — stay in AI mode so the per-file
      // errors remain visible and the user can retry or remove files
      toast.error("We couldn't read these contracts — see the errors below");
      return;
    }
    applyMerge(done);
  };

  const handleRowsEdit = (rows: SplitRow[]) => {
    setSplitRows(rows);
    setRowsDirty(true);
  };

  // Attach the contracts whose splits were read to the new work's Related
  // Documents. Best-effort: a failure here never fails work creation.
  // Project-picked contracts already have a project_files id; fresh uploads
  // are persisted first (same pipeline as the work page's "Add documents").
  const linkUsedContracts = async (workId: string) => {
    const used = queuedContracts.filter((q) => q.status === "done");
    if (used.length === 0) return;
    const linkedFileIds = new Set<string>();
    let anyFailed = false;
    for (const q of used) {
      try {
        let fileId = q.kind === "project" ? q.contractFileId ?? null : null;
        if (!fileId && q.kind === "upload" && q.file) {
          const file = q.file;
          // Soft storage-cap check. Skipped while entitlements are loading or
          // errored (cap reads 0 then) — a real overage still fails server-side
          // and is caught below.
          if (
            !storageStatus.loading &&
            !storageStatus.error &&
            storageStatus.cap !== -1 &&
            storageStatus.used + file.size > storageStatus.cap
          ) {
            anyFailed = true;
            continue;
          }
          const hashBuffer = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
          const contentHash = Array.from(new Uint8Array(hashBuffer))
            .map((b) => b.toString(16).padStart(2, "0"))
            .join("");
          // Dedup: if this exact file already exists in the project, link the
          // existing row instead of re-uploading.
          const { data: existing } = await supabase
            .from("project_files")
            .select("id")
            .eq("project_id", projectIdInUse)
            .eq("content_hash", contentHash)
            .limit(1);
          if (existing && existing.length > 0) {
            fileId = existing[0].id;
          } else {
            const filePath = `${projectIdInUse}/contract/${Date.now()}_${file.name}`;
            const { error: uploadError } = await supabase.storage
              .from("project-files")
              .upload(filePath, file);
            if (uploadError) throw uploadError;
            const { data: urlData } = supabase.storage
              .from("project-files")
              .getPublicUrl(filePath);
            const { data: inserted, error: dbError } = await supabase
              .from("project_files")
              .insert({
                project_id: projectIdInUse,
                file_name: file.name,
                file_url: urlData.publicUrl,
                file_path: filePath,
                folder_category: "contract",
                file_size: file.size,
                file_type: file.type,
                content_hash: contentHash,
              })
              .select("id")
              .single();
            if (dbError) {
              await supabase.storage.from("project-files").remove([filePath]);
              throw dbError;
            }
            fileId = inserted.id;
          }
        }
        if (!fileId || linkedFileIds.has(fileId)) continue;
        await apiFetch(`${API_URL}/registry/works/${workId}/files?file_id=${fileId}`, {
          method: "POST",
        });
        linkedFileIds.add(fileId);
      } catch {
        anyFailed = true;
      }
    }
    if (linkedFileIds.size > 0) {
      queryClient.invalidateQueries({ queryKey: ["work-files", workId] });
      queryClient.invalidateQueries({ queryKey: ["project-files-tab", projectIdInUse] });
    }
    if (anyFailed) {
      toast.warning(
        "Work created, but a contract couldn't be attached — you can link it from the work page."
      );
    }
  };

  // The rows the user actually sees on the splits step: the manual branch
  // renders a default "You" row when splitRows is empty, so mirror it here —
  // what's shown in the confirm dialog is exactly what gets submitted.
  const effectiveRows: SplitRow[] =
    splitRows.length > 0
      ? splitRows
      : royMode === "manual"
        ? [{ key: "you", name: artistName, role: "Primary Artist", isYou: true, master: 0, publishing: 0 }]
        : [];

  const finish = async (overrideRows?: SplitRow[]) => {
    const rows = overrideRows ?? splitRows;
    if (!projectIdInUse || !artistIdInUse) {
      toast.error("Pick a project first");
      return;
    }
    setSubmitting(true);
    // Credited artists come from Spotify on the released path, or from the
    // user's manual entries on the unreleased path. Drop blank-name rows.
    const creditedArtists: CreditedArtist[] =
      chosen?.artists && chosen.artists.length > 0
        ? chosen.artists.map((a) => ({ name: a.name, role: a.role, spotify_url: a.spotify_url ?? null }))
        : manualArtists
            .map((a) => ({ name: a.name.trim(), role: a.role, spotify_url: null }))
            .filter((a) => a.name);
    try {
      const created = await createWork.mutateAsync({
        artist_id: artistIdInUse,
        project_id: projectIdInUse,
        title: (chosen?.title || title).trim(),
        work_type: workType,
        is_released: released ?? true,
        ...(meta.isrc ? { isrc: meta.isrc } : {}),
        ...(meta.upc ? { upc: meta.upc } : {}),
        ...(meta.releaseDate ? { release_date: meta.releaseDate } : {}),
        ...(meta.spotifyUrl ? { notes: meta.spotifyUrl } : {}),
        ...(meta.genre ? { genre: meta.genre } : {}),
        ...(meta.label ? { label: meta.label } : {}),
        ...(creditedArtists.length > 0 ? { featured_artists: creditedArtists } : {}),
      });
      const workId = (created as { id?: string })?.id;

      // Persist splits as ownership_stakes rows
      let stakesSaved = 0;
      if (workId) {
        for (const row of rows) {
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
            stakesSaved += 1;
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
            stakesSaved += 1;
          }
          if ((row.soundexchange ?? 0) > 0) {
            await createStake.mutateAsync({
              work_id: workId,
              stake_type: "soundexchange",
              holder_name: name,
              holder_role: row.role || (row.isYou ? "Primary Artist" : "Collaborator"),
              percentage: row.soundexchange,
              is_owner_stake: row.isYou === true,
            });
            stakesSaved += 1;
          }
        }
      }
      // useCreateStake intentionally doesn't toast per stake — show one
      // consolidated toast here (useCreateWork already toasts "Work created").
      if (stakesSaved > 0) {
        toast.success(`${stakesSaved} royalty stake${stakesSaved === 1 ? "" : "s"} saved`);
      }

      // Attach the contracts the splits came from as Related Documents.
      if (workId) {
        await linkUsedContracts(workId);
      }

      setConfirmOpen(false);
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
          <Button
            variant="outline"
            onClick={() => {
              if (royMode !== null) {
                // Return to the "pull from contract vs by hand" choice, not the
                // metadata step. Queued contracts are cleared (a fresh choice
                // starts clean) but parsed/edited rows survive.
                setRoyMode(null);
                clearContracts();
              } else {
                setStep(released ? 2 : 1);
              }
            }}
          >
            <ArrowLeft className="w-4 h-4 mr-1" /> Back
          </Button>
          <Button
            onClick={() => {
              if (!projectIdInUse || !artistIdInUse) {
                toast.error("Pick a project first");
                return;
              }
              setConfirmOpen(true);
            }}
            disabled={submitting}
          >
            <CheckCircle2 className="w-4 h-4 mr-1" />
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
    <>
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-w-4xl">
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
        <div className="max-h-[78vh] overflow-y-auto px-2 -mx-2 py-1 -my-1">
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
              onAddNewArtist={() => setNewArtistOpen(true)}
              onAddNewProject={() => setNewProjectOpen(true)}
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
              manualArtists={manualArtists}
              setManualArtists={setManualArtists}
            />
          ) : onRoyalty ? (
            <RoyaltyStep
              royMode={royMode}
              setRoyMode={setRoyMode}
              queuedContracts={queuedContracts}
              onAddContracts={addContracts}
              onRemoveContract={removeContract}
              fileInputRef={fileInputRef}
              parsing={queuedContracts.some((q) => q.status === "parsing")}
              splitRows={splitRows}
              setSplitRows={handleRowsEdit}
              splitsSource={splitsSource}
              onParseAll={handleParseAll}
              mergeConflicts={mergeConflicts}
              artistName={artistName}
              mainArtistInContract={mainArtistInContract}
              projectId={projectIdInUse}
            />
          ) : null}
        </div>

        <DialogFooter className="gap-2 mt-6 pt-4 border-t">{footer}</DialogFooter>
      </DialogContent>
    </Dialog>

    {/* Sibling dialogs so they layer above the wizard cleanly. */}
    <NewArtistDialog
      open={newArtistOpen}
      onOpenChange={setNewArtistOpen}
      onCreated={handleArtistCreated}
    />
    <ProjectFormDialog
      open={newProjectOpen}
      onOpenChange={setNewProjectOpen}
      artists={artistsQuery.data || []}
      defaultArtistId={selectedArtistId}
      onSave={handleProjectSave}
    />
    <AddWorkConfirmDialog
      open={confirmOpen}
      onOpenChange={setConfirmOpen}
      workTitle={(chosen?.title || title).trim() || "Untitled work"}
      artistName={artistName}
      projectName={projectName}
      rows={effectiveRows}
      submitting={submitting}
      onConfirm={() => finish(effectiveRows)}
    />
    </>
  );
}

// ============================================================
// Steps
// ============================================================

// Sentinel value used inside the Artist Select to route the user into the
// "create a new artist" dialog instead of selecting an existing one.
const ADD_NEW_ARTIST_VALUE = "__add_new_artist__";

function DestinationStep({
  artists,
  projects,
  selectedArtistId,
  onSelectArtist,
  onSelectProject,
  onAddNewArtist,
  onAddNewProject,
}: {
  artists: ArtistOption[];
  projects: ProjectOption[];
  selectedArtistId: string;
  onSelectArtist: (id: string) => void;
  onSelectProject: (id: string) => void;
  onAddNewArtist: () => void;
  onAddNewProject: () => void;
}) {
  const handleArtistChange = (value: string) => {
    if (value === ADD_NEW_ARTIST_VALUE) {
      onAddNewArtist();
      return;
    }
    onSelectArtist(value);
  };
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
        <Select value={selectedArtistId} onValueChange={handleArtistChange}>
          <SelectTrigger className="mt-1">
            <SelectValue placeholder="Pick an artist…" />
          </SelectTrigger>
          <SelectContent>
            {artists.map((a) => (
              <SelectItem key={a.id} value={a.id}>
                {a.name}
              </SelectItem>
            ))}
            {artists.length > 0 && <div className="my-1 h-px bg-border" role="separator" />}
            <SelectItem value={ADD_NEW_ARTIST_VALUE} className="text-primary font-medium">
              <span className="flex items-center gap-2">
                <Plus className="w-4 h-4" />
                Add new artist
              </span>
            </SelectItem>
          </SelectContent>
        </Select>
      </div>
      {selectedArtistId && (
        <div>
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Project
            </label>
            <Button type="button" variant="ghost" size="sm" onClick={onAddNewProject} className="h-7 px-2 text-xs text-primary">
              <Plus className="w-3.5 h-3.5 mr-1" /> New project
            </Button>
          </div>
          {projects.length === 0 ? (
            <p className="text-sm text-muted-foreground mt-2">
              No projects yet for this artist. Click <b>New project</b> above to create one.
            </p>
          ) : (
            <div className="mt-2 grid grid-cols-1 gap-1.5 max-h-[28rem] overflow-y-auto">
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
            placeholder="e.g. Burna Boy Last Last"
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
      <p className="text-xs text-muted-foreground -mt-2">
        Tip: include the <b>artist name</b> with the song title for better matches.
      </p>

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
      {chosen.artists && chosen.artists.length > 0 && (
        <div className="rounded-lg border p-3 space-y-2">
          <p className="text-xs font-semibold text-foreground">
            Credited artists
          </p>
          <div className="flex flex-wrap gap-1.5">
            {chosen.artists.map((a, i) => {
              const isMain = /main/i.test(a.role);
              return (
                <span
                  key={`${a.name}-${i}`}
                  className="inline-flex items-center gap-1.5 rounded-full border bg-muted/40 pl-1 pr-2.5 py-1 text-xs"
                >
                  <RegistryAvatar name={a.name || "?"} size={18} />
                  <span className="font-medium">{a.name}</span>
                  <Badge
                    variant="secondary"
                    className={cn(
                      "text-[9px] font-semibold uppercase tracking-wide px-1.5 py-0",
                      isMain
                        ? "bg-primary/15 text-primary"
                        : "bg-muted text-muted-foreground"
                    )}
                  >
                    {isMain ? "Main" : "Featured"}
                  </Badge>
                </span>
              );
            })}
          </div>
        </div>
      )}
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

const CREDIT_ROLES = ["Featured artist", "Main artist"] as const;

function UnreleasedStep({
  title,
  setTitle,
  workType,
  setWorkType,
  artistName,
  meta,
  setMeta,
  manualArtists,
  setManualArtists,
}: {
  title: string;
  setTitle: (v: string) => void;
  workType: string;
  setWorkType: (v: string) => void;
  artistName: string;
  meta: { isrc: string; upc: string; releaseDate: string; label: string; genre: string; spotifyUrl: string };
  setMeta: React.Dispatch<React.SetStateAction<typeof meta>>;
  manualArtists: CreditedArtist[];
  setManualArtists: React.Dispatch<React.SetStateAction<CreditedArtist[]>>;
}) {
  const set = (patch: Partial<typeof meta>) => setMeta((m) => ({ ...m, ...patch }));
  const addArtist = () =>
    setManualArtists((list) => [...list, { name: "", role: "Featured artist", spotify_url: null }]);
  const updateArtist = (idx: number, patch: Partial<CreditedArtist>) =>
    setManualArtists((list) => list.map((a, i) => (i === idx ? { ...a, ...patch } : a)));
  const removeArtist = (idx: number) =>
    setManualArtists((list) => list.filter((_, i) => i !== idx));
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

      {/* Credited artists — manual entry, since there's no Spotify data to pull
          for an unreleased track. Saved as display-only credits, separate from
          the royalty splits set on the next step. */}
      <div className="pt-1">
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs font-semibold text-foreground">Featured artists</label>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={addArtist}
            className="h-7 px-2 text-xs text-primary"
          >
            <Plus className="w-3.5 h-3.5 mr-1" /> Add artist
          </Button>
        </div>
        {manualArtists.length === 0 ? (
          <p className="text-[11px] text-muted-foreground">
            Optional — add any featured or co-credited artists. These are display-only
            credits and don't affect royalty splits.
          </p>
        ) : (
          <div className="space-y-2">
            {manualArtists.map((a, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="flex-1">
                  <Input
                    value={a.name}
                    placeholder="Artist name"
                    onChange={(e) => updateArtist(i, { name: e.target.value })}
                  />
                </div>
                <Select value={a.role} onValueChange={(v) => updateArtist(i, { role: v })}>
                  <SelectTrigger className="w-40 shrink-0">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {CREDIT_ROLES.map((r) => (
                      <SelectItem key={r} value={r}>
                        {r}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-9 w-9 shrink-0 text-muted-foreground hover:text-destructive"
                  onClick={() => removeArtist(i)}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
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
  queuedContracts,
  onAddContracts,
  onRemoveContract,
  fileInputRef,
  parsing,
  splitRows,
  setSplitRows,
  splitsSource,
  onParseAll,
  mergeConflicts,
  artistName,
  mainArtistInContract,
  projectId,
}: {
  royMode: null | "ai" | "manual";
  setRoyMode: (v: null | "ai" | "manual") => void;
  queuedContracts: QueuedContract[];
  onAddContracts: (
    items: Array<
      | { kind: "upload"; file: File; displayName: string }
      | { kind: "project"; contractFileId: string; displayName: string }
    >
  ) => void;
  onRemoveContract: (id: string) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  parsing: boolean;
  splitRows: SplitRow[];
  setSplitRows: (r: SplitRow[]) => void;
  splitsSource: null | { type: "ai" | "manual"; file?: string; date: string };
  onParseAll: () => void;
  mergeConflicts: MergeConflict[];
  artistName: string;
  mainArtistInContract: boolean;
  projectId: string;
}) {
  const [aiSource, setAiSource] = useState<AiSource>("project");
  // Re-opens the contract picker after a merge (the "Add another contract" button).
  const [showPicker, setShowPicker] = useState(false);

  // The footer Back button resets royMode to return to the splits-source
  // choice; the picker state lives here, so reset it alongside.
  useEffect(() => {
    if (royMode === null) setShowPicker(false);
  }, [royMode]);

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

  const queuedProjectIds = new Set(
    queuedContracts.filter((q) => q.kind === "project").map((q) => q.contractFileId)
  );

  const toggleProjectContract = (c: ProjectContract) => {
    const existing = queuedContracts.find(
      (q) => q.kind === "project" && q.contractFileId === c.id
    );
    if (existing) onRemoveContract(existing.id);
    else onAddContracts([{ kind: "project", contractFileId: c.id, displayName: c.file_name }]);
  };

  const canParse =
    !parsing && queuedContracts.some((q) => q.status === "pending" || q.status === "error");
  const doneContracts = queuedContracts.filter((q) => q.status === "done");

  return (
    <div className="space-y-4">
      <div>
        <p className="text-base font-semibold">Royalty splits</p>
        <p className="text-sm text-muted-foreground">
          Let AI read the splits from the related contract, or set them by hand.
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
            <div className="text-sm font-semibold">Get splits from the contract</div>
            <div className="text-xs text-muted-foreground">
              Pick the related contract — AI reads it and fills in each party's split.
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

      {royMode === "ai" && (splitRows.filter((r) => !r.isYou).length === 0 || showPicker) && (
        <>
          <div className="text-xs font-semibold text-muted-foreground -mb-2">
            Step 1 · Pick the contract(s) that cover this work
          </div>
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
                <>
                <div className="max-h-60 overflow-y-auto rounded-lg border divide-y">
                  {(projectContractsQuery.data || []).map((c) => {
                    const picked = queuedProjectIds.has(c.id);
                    return (
                      <button
                        key={c.id}
                        type="button"
                        onClick={() => toggleProjectContract(c)}
                        disabled={parsing}
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
                <p className="mt-1.5 text-[11px] text-muted-foreground">
                  Select every contract that covers this work's splits — you can pick more than one.
                </p>
                </>
              )}
            </div>
          )}

          {aiSource === "upload" && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf,.pdf"
                multiple
                className="hidden"
                onChange={(e) => {
                  const files = Array.from(e.target.files || []);
                  if (files.length > 0) {
                    onAddContracts(
                      files.map((f) => ({ kind: "upload" as const, file: f, displayName: f.name }))
                    );
                  }
                  // allow re-picking the same file after removing it
                  e.target.value = "";
                }}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="w-full rounded-xl border-2 border-dashed p-6 flex flex-col items-center gap-2 hover:border-primary/40 hover:bg-muted/40"
              >
                <Upload className="w-6 h-6 text-muted-foreground" />
                <div className="text-sm font-semibold">
                  Click to add the related contract{queuedContracts.length > 0 ? "s" : ""}
                </div>
                <div className="text-xs text-muted-foreground">
                  PDFs up to ~25 MB · you can add several · text is read by AI
                </div>
              </button>
            </>
          )}

          {queuedContracts.length > 0 && (
            <div className="rounded-lg border divide-y">
              {queuedContracts.map((q) => (
                <div key={q.id} className="flex items-center gap-3 px-3 py-2">
                  {q.status === "parsing" ? (
                    <Loader2 className="w-4 h-4 shrink-0 animate-spin text-primary" />
                  ) : q.status === "done" ? (
                    <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-500" />
                  ) : q.status === "error" ? (
                    <AlertTriangle className="w-4 h-4 shrink-0 text-destructive" />
                  ) : (
                    <Clock className="w-4 h-4 shrink-0 text-muted-foreground" />
                  )}
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">{q.displayName}</div>
                    <div
                      className={cn(
                        "text-[11px]",
                        q.status === "error" ? "text-destructive" : "text-muted-foreground"
                      )}
                    >
                      {q.status === "pending" && "Ready to parse"}
                      {q.status === "parsing" && "Reading with AI…"}
                      {q.status === "done" &&
                        `Parsed · ${q.parties?.length ?? 0} ${(q.parties?.length ?? 0) === 1 ? "party" : "parties"} found`}
                      {q.status === "error" && (q.error || "Parse failed")}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => onRemoveContract(q.id)}
                    disabled={q.status === "parsing"}
                    aria-label={`Remove ${q.displayName}`}
                    className="shrink-0 text-muted-foreground hover:text-destructive disabled:opacity-40"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div>
            <div className="text-xs font-semibold text-muted-foreground mb-2">
              Step 2 · Read the contract{queuedContracts.length > 1 ? "s" : ""}
            </div>
            <div className="flex items-center gap-3">
              <Button
                onClick={() => {
                  setShowPicker(false);
                  onParseAll();
                }}
                disabled={!canParse}
              >
                {parsing ? (
                  <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4 mr-1" />
                )}
                {parsing
                  ? `Reading contract${queuedContracts.length > 1 ? "s" : ""}…`
                  : queuedContracts.length <= 1
                    ? "Read contract"
                    : `Read ${queuedContracts.length} contracts`}
              </Button>
            </div>
            {queuedContracts.length === 0 && (
              <p className="mt-1.5 text-[11px] text-muted-foreground">
                Select or upload at least one contract first.
              </p>
            )}
          </div>
        </>
      )}

      {royMode === "ai" && splitRows.length > 0 && (
        <>
          {!mainArtistInContract && (
            <div className="flex items-start gap-2 text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-lg px-3 py-2">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              <span>
                We didn't see <b>{artistName}</b> in the contract
                {doneContracts.length > 1 ? "s" : ""}.
                {splitRows.some((r) => r.isYou) ? (
                  <> If you hold a share, enter it in your row below.</>
                ) : (
                  <>
                    {" "}
                    If you hold a share of this work,{" "}
                    <button
                      type="button"
                      className="font-semibold underline underline-offset-2 hover:text-amber-300"
                      onClick={() =>
                        setSplitRows([
                          {
                            key: "you",
                            name: artistName,
                            role: "Primary Artist",
                            isYou: true,
                            master: 0,
                            publishing: 0,
                          },
                          ...splitRows,
                        ])
                      }
                    >
                      add your own row
                    </button>{" "}
                    and set your split.
                  </>
                )}
              </span>
            </div>
          )}
          {mergeConflicts.length > 0 && (
            <div className="flex items-start gap-2 text-xs bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-lg px-3 py-2">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              <div className="space-y-1">
                <p className="font-medium">
                  The contracts disagree on some splits — we kept the first value. Edit the rows
                  below to the correct totals.
                </p>
                {mergeConflicts.map((conflict) => (
                  <p key={conflict.name}>
                    <b>{conflict.name}</b>:{" "}
                    {conflict.values
                      .map(
                        (v) =>
                          `Master ${v.master}% / Publishing ${v.publishing}%` +
                          (v.soundexchange > 0 ? ` / SoundExchange ${v.soundexchange}%` : "") +
                          ` (${v.file})`
                      )
                      .join(" vs ")}
                  </p>
                ))}
              </div>
            </div>
          )}
          <div className="text-xs font-semibold text-muted-foreground -mb-2">
            Step 3 · Review and edit the splits
          </div>
          <RoyaltySplitsTable
            rows={splitRows}
            onChange={setSplitRows}
            editable
            allowAddRow
            source={splitsSource}
            warnOnImbalance={false}
          />
          {doneContracts.length > 1 && (
            <div className="text-[11px] text-muted-foreground space-y-0.5">
              <p className="font-medium text-muted-foreground">Sources</p>
              {doneContracts.map((q) => (
                <p key={q.id} className="truncate">
                  {q.displayName}: {(q.parties || []).map((p) => p.name).join(", ") || "no parties"}
                </p>
              ))}
            </div>
          )}
          {!showPicker && (
            <Button type="button" variant="outline" size="sm" onClick={() => setShowPicker(true)}>
              <Plus className="w-4 h-4 mr-1" /> Add another contract
            </Button>
          )}
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
                      name: artistName,
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
