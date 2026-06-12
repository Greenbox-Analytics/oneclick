import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  ChevronRight,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Music,
  FileText,
  Shield,
  Eye,
  Lock,
  Loader2,
  RefreshCw,
  Download,
  ExternalLink,
  Tag,
  Link as LinkIcon,
  Trash2,
  Search,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import {
  useUpdateWork,
  useCreateStake,
  useUpdateStake,
  useDeleteStake,
  useExportProof,
  type WorkFull,
  type OwnershipStake,
} from "@/hooks/useRegistry";
import {
  useWorkFiles,
  useLinkFileToWork,
  useUnlinkFileFromWork,
} from "@/hooks/useWorkFiles";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { useSpotifyTrack, spotifyTrackIdFromUrl } from "@/hooks/useSpotifySearch";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";
import { ReleaseTag } from "./ReleaseTag";
import { RegistryStatusBadge } from "./RegistryStatusBadge";
import { RegistryAvatar } from "./RegistryAvatar";
import { RoyaltySplitsTable, type SplitRow } from "./RoyaltySplitsTable";

interface WorkEditorProps {
  work: WorkFull;
}

function fmtDuration(ms: number | null): string {
  if (!ms) return "—";
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

interface JoinedProject {
  id: string;
  name: string;
  artist_id: string;
}

interface JoinedArtist {
  id: string;
  name: string;
}

/** Single-work editor — two-column grid (main + sticky sidebar). */
export function WorkEditor({ work }: WorkEditorProps) {
  const navigate = useNavigate();
  const { user } = useAuth();
  const updateWork = useUpdateWork();
  const exportProof = useExportProof();

  // Join project + artist for breadcrumb + sidebar
  const projectQuery = useQuery<JoinedProject | null>({
    queryKey: ["registry-editor-project", work.project_id],
    queryFn: async () => {
      if (!work.project_id) return null;
      const { data, error } = await supabase
        .from("projects")
        .select("id, name, artist_id")
        .eq("id", work.project_id)
        .maybeSingle();
      if (error) throw error;
      return (data as JoinedProject) || null;
    },
    enabled: !!work.project_id,
  });
  const project = projectQuery.data;

  const artistQuery = useQuery<JoinedArtist | null>({
    queryKey: ["registry-editor-artist", project?.artist_id || work.artist_id],
    queryFn: async () => {
      const artistId = project?.artist_id || work.artist_id;
      if (!artistId) return null;
      const { data, error } = await supabase
        .from("artists")
        .select("id, name")
        .eq("id", artistId)
        .maybeSingle();
      if (error) throw error;
      return (data as JoinedArtist) || null;
    },
    enabled: !!(project?.artist_id || work.artist_id),
  });
  const artist = artistQuery.data;

  // "Released" is derived from release_date: a track is released iff
  // release_date is set. Toggling on seeds today's date as a default; toggling
  // off clears it. Either direction is editable from the date field after.
  const released = !!work.release_date;
  const isOwner = work.user_id === user?.id;
  const canEdit = isOwner; // TODO: also derive from project membership when wired

  const set = (patch: Partial<WorkFull>) =>
    updateWork.mutate({ workId: work.id, ...patch });

  const toggleReleased = (val: boolean) => {
    if (val) {
      // Going Unreleased → Released: seed today's date unless one already exists.
      if (!work.release_date) {
        set({ release_date: new Date().toISOString().slice(0, 10) });
      }
    } else {
      // Going Released → Unreleased: clear the date.
      set({ release_date: null });
    }
  };

  // Spotify metadata pull (only when stored Spotify URL exists)
  const trackId = spotifyTrackIdFromUrl(work.notes ? null : null); // placeholder; we don't have spotifyUrl on Work
  // The DB has no spotify_url column today; we store it under `notes` is a stretch.
  // For now we attempt a pull when `work.notes` contains a Spotify URL.
  const inferredSpotifyUrl = useMemo(() => {
    const sources: (string | null | undefined)[] = [work.notes];
    for (const s of sources) {
      if (s && /open\.spotify\.com\/track\/[a-zA-Z0-9]+/.test(s)) return s;
    }
    return null;
  }, [work.notes]);
  const spotifyId = spotifyTrackIdFromUrl(inferredSpotifyUrl);
  const spotifyTrack = useSpotifyTrack(spotifyId);
  const [pulling, setPulling] = useState(false);
  const pullMetadata = async () => {
    if (!spotifyId) {
      toast.error("Add a Spotify track link first");
      return;
    }
    setPulling(true);
    try {
      const data = await spotifyTrack.refetch();
      if (data.data) {
        set({
          isrc: work.isrc || data.data.isrc || null,
          upc: work.upc || data.data.upc || null,
          release_date: work.release_date || data.data.release_date || null,
        });
        toast.success("Metadata pulled from Spotify");
      }
    } finally {
      setPulling(false);
    }
  };

  // Splits — built from stakes
  const stakeRows: SplitRow[] = useMemo(() => {
    // Group stakes by holder_name, pivot master + publishing into the same row
    const byHolder: Record<
      string,
      {
        masterStake?: OwnershipStake;
        publishingStake?: OwnershipStake;
        role: string;
      }
    > = {};
    for (const s of work.stakes || []) {
      if (!byHolder[s.holder_name])
        byHolder[s.holder_name] = { role: s.holder_role };
      if (s.stake_type === "master") byHolder[s.holder_name].masterStake = s;
      if (s.stake_type === "publishing") byHolder[s.holder_name].publishingStake = s;
    }
    return Object.entries(byHolder).map(([name, info]) => {
      const isYou = !!(
        artist &&
        (name === artist.name || name.toLowerCase() === artist.name.toLowerCase())
      );
      return {
        key: name,
        name,
        role: info.role || "",
        isYou,
        master: info.masterStake?.percentage || 0,
        publishing: info.publishingStake?.percentage || 0,
      };
    });
  }, [work.stakes, artist]);

  const issues = useMemo(() => {
    const out: string[] = [];
    if (work.release_date && !work.isrc) out.push("Missing ISRC");
    const masterSum = (work.stakes || [])
      .filter((s) => s.stake_type === "master")
      .reduce((n, s) => n + (s.percentage || 0), 0);
    const pubSum = (work.stakes || [])
      .filter((s) => s.stake_type === "publishing")
      .reduce((n, s) => n + (s.percentage || 0), 0);
    if ((work.stakes || []).length > 0 && (masterSum !== 100 || pubSum !== 100)) {
      out.push(`Splits don't total 100% (master ${masterSum}%, publishing ${pubSum}%)`);
    }
    return out;
  }, [work]);

  // Documents (project files linked to this work via work_files)
  const workFilesQuery = useWorkFiles(work.id);
  const workFiles = workFilesQuery.data || [];
  const [linkOpen, setLinkOpen] = useState(false);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_312px] gap-6 items-start">
      {/* main column */}
      <div className="flex flex-col gap-4">
        {/* breadcrumb */}
        <div>
          <Button
            variant="ghost"
            size="sm"
            className="pl-0 text-muted-foreground"
            onClick={() => navigate("/tools/registry")}
          >
            <ArrowLeft className="w-4 h-4 mr-1" /> Rights Registry
          </Button>
          {artist && project && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground mt-2">
              <RegistryAvatar name={artist.name} size={20} />
              {artist.name} ·
              <button
                type="button"
                className="hover:text-foreground"
                onClick={() => navigate(`/projects/${project.id}`)}
              >
                {project.name}
              </button>
              <ChevronRight className="w-3 h-3" />
              <span>Works Registry</span>
            </div>
          )}
          <div className="flex items-center gap-3 flex-wrap mt-2">
            <h2 className="text-2xl font-bold tracking-tight">
              {work.title || "Untitled work"}
            </h2>
            <ReleaseTag released={released} size="lg" />
            <RegistryStatusBadge status={work.status} />
          </div>
        </div>

        {!canEdit && (
          <Card className="p-3 flex items-center gap-3 border-amber-500/30 bg-amber-500/5">
            <Lock className="w-4 h-4 text-amber-400 shrink-0" />
            <div className="text-sm">
              <strong className="font-semibold">View only.</strong>{" "}
              <span className="text-muted-foreground">
                Ask the owner of {project?.name || "this project"} for edit access.
              </span>
            </div>
          </Card>
        )}

        {canEdit && issues.length > 0 && (
          <Card className="p-3 flex items-start gap-3 border-amber-500/30 bg-amber-500/5">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-semibold mb-1">This work needs attention</p>
              <ul className="list-disc list-inside text-xs text-muted-foreground space-y-0.5">
                {issues.map((it, i) => (
                  <li key={i}>{it}</li>
                ))}
              </ul>
            </div>
          </Card>
        )}

        {/* Release status */}
        <CardBlock
          icon={released ? <CheckCircle2 className="w-4 h-4" /> : <Clock className="w-4 h-4" />}
          title="Release status"
          desc={
            released
              ? "This track is live. ISRC, release date, and Spotify link are required to stay registry-complete."
              : "Not yet released. Release fields below stay disabled until you flip this on."
          }
          right={
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "text-xs font-semibold",
                  released ? "text-emerald-400" : "text-muted-foreground"
                )}
              >
                {released ? "Released" : "Unreleased"}
              </span>
              <Switch
                checked={released}
                disabled={!canEdit}
                onCheckedChange={toggleReleased}
              />
            </div>
          }
        />

        {/* Track details */}
        <CardBlock
          icon={<Music className="w-4 h-4" />}
          title="Track details"
          desc="Core identifiers and metadata for this work."
          right={
            released &&
            spotifyId && (
              <Button
                size="sm"
                variant="outline"
                onClick={pullMetadata}
                disabled={!canEdit || pulling}
              >
                {pulling ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin mr-1.5" />
                ) : (
                  <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
                )}
                Pull from Spotify
              </Button>
            )
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="Track name" required>
              <Input
                value={work.title}
                placeholder="Song title"
                disabled={!canEdit}
                onChange={(e) => set({ title: e.target.value })}
              />
            </Field>
            <Field label="Work type" hint="single, EP track, composition…">
              <Input
                value={work.work_type || ""}
                disabled={!canEdit}
                onChange={(e) => set({ work_type: e.target.value })}
              />
            </Field>
            <Field
              label="ISRC"
              required={released}
              missing={released && !work.isrc}
              hint="International Standard Recording Code"
            >
              <Input
                className="font-mono"
                value={work.isrc || ""}
                placeholder={released ? "e.g. USRC17607831" : "—"}
                disabled={!canEdit || !released}
                onChange={(e) => set({ isrc: e.target.value })}
              />
            </Field>
            <Field label="UPC" hint="Universal Product Code">
              <Input
                className="font-mono"
                value={work.upc || ""}
                placeholder={released ? "0000000000000" : "—"}
                disabled={!canEdit || !released}
                onChange={(e) => set({ upc: e.target.value })}
              />
            </Field>
            <Field
              label="Release date"
              required={released}
              missing={released && !work.release_date}
            >
              <Input
                type="date"
                value={work.release_date || ""}
                disabled={!canEdit || !released}
                onChange={(e) => set({ release_date: e.target.value })}
              />
            </Field>
            <Field label="ISWC" hint="Composition identifier">
              <Input
                className="font-mono"
                value={work.iswc || ""}
                disabled={!canEdit}
                onChange={(e) => set({ iswc: e.target.value })}
              />
            </Field>
            <Field label="Duration" hint={spotifyTrack.data ? "From Spotify" : "Pulls when synced"}>
              <div className="h-10 rounded-md border bg-muted/40 px-3 flex items-center text-sm text-muted-foreground">
                {fmtDuration(spotifyTrack.data?.duration_ms || null)}
              </div>
            </Field>
            <Field
              label="Spotify link"
              required={released}
              missing={released && !inferredSpotifyUrl}
              hint="Used to pull DSP metadata"
            >
              <div className="flex gap-2">
                <Input
                  placeholder={released ? "https://open.spotify.com/track/…" : "—"}
                  value={work.notes || ""}
                  disabled={!canEdit || !released}
                  onChange={(e) => set({ notes: e.target.value })}
                />
                {inferredSpotifyUrl && (
                  <Button asChild variant="outline">
                    <a href={inferredSpotifyUrl} target="_blank" rel="noreferrer">
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </Button>
                )}
              </div>
            </Field>
          </div>
        </CardBlock>

        {/* Documents */}
        <CardBlock
          icon={<FileText className="w-4 h-4" />}
          title="Related documents"
          desc="Contracts, split sheets, and project files tied to this work."
          right={
            canEdit && (
              <Button size="sm" variant="outline" onClick={() => setLinkOpen(true)}>
                <LinkIcon className="w-3.5 h-3.5 mr-1.5" /> Link from project
              </Button>
            )
          }
        >
          {workFilesQuery.isLoading ? (
            <div className="py-6 text-center text-sm text-muted-foreground">
              <Loader2 className="w-4 h-4 inline mr-2 animate-spin" /> Loading…
            </div>
          ) : workFiles.length === 0 ? (
            <div className="py-6 text-center text-sm text-muted-foreground border border-dashed rounded-lg">
              No documents attached yet.
              {canEdit && (
                <>
                  {" "}
                  <button
                    type="button"
                    className="text-primary hover:underline"
                    onClick={() => setLinkOpen(true)}
                  >
                    Link one from the project.
                  </button>
                </>
              )}
            </div>
          ) : (
            <ul className="divide-y">
              {workFiles.map((wf) => (
                <WorkFileRow
                  key={wf.id}
                  link={wf}
                  workId={work.id}
                  canEdit={canEdit}
                />
              ))}
            </ul>
          )}
        </CardBlock>
      </div>

      {/* Sidebar */}
      <div className="flex flex-col gap-4 lg:sticky lg:top-24">
        {/* Access */}
        <Card className="p-4">
          <div className="text-xs font-semibold tracking-wider uppercase text-muted-foreground mb-3">
            Your access
          </div>
          <div className="flex items-center gap-3 mb-3">
            <div
              className={cn(
                "w-9 h-9 rounded-lg flex items-center justify-center",
                isOwner
                  ? "bg-primary/15 text-primary"
                  : "bg-blue-500/15 text-blue-400"
              )}
            >
              {isOwner ? <Shield className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-semibold">{isOwner ? "Owner" : "Collaborator"}</div>
              <div className="text-xs text-muted-foreground">
                {isOwner ? "Full control" : canEdit ? "Can edit" : "View only"}
              </div>
            </div>
          </div>
        </Card>

        {/* Royalty splits */}
        <Card className="p-4">
          <SplitsSidebar work={work} canEdit={canEdit} initialRows={stakeRows} />
        </Card>

        {/* Traceability */}
        <Card className="p-4">
          <div className="text-xs font-semibold tracking-wider uppercase text-muted-foreground mb-3">
            Traceability
          </div>
          <div className="space-y-2.5">
            <TraceRow
              icon={<FileText className="w-3.5 h-3.5" />}
              label="Documents linked"
              value={`${workFiles.length} document${workFiles.length !== 1 ? "s" : ""}`}
              ok={workFiles.length > 0}
            />
            <TraceRow
              icon={<Tag className="w-3.5 h-3.5" />}
              label="ISRC assigned"
              value={work.isrc || "Not set"}
              ok={!!work.isrc}
            />
            <TraceRow
              icon={<CheckCircle2 className="w-3.5 h-3.5" />}
              label="Stakes recorded"
              value={`${(work.stakes || []).length} stake${(work.stakes || []).length !== 1 ? "s" : ""}`}
              ok={(work.stakes || []).length > 0}
            />
          </div>

          <Button
            variant="outline"
            size="sm"
            className="w-full mt-4"
            onClick={() => exportProof.mutate(work.id)}
            disabled={exportProof.isPending}
          >
            {exportProof.isPending ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Download className="w-4 h-4 mr-2" />
            )}
            Export Proof of Ownership
          </Button>
        </Card>
      </div>

      {linkOpen && (
        <LinkProjectFilesDialog
          open={linkOpen}
          onClose={() => setLinkOpen(false)}
          workId={work.id}
          projectId={work.project_id || ""}
          alreadyLinkedFileIds={new Set(workFiles.map((wf) => wf.file_id))}
        />
      )}
    </div>
  );
}

interface CardBlockProps {
  icon: React.ReactNode;
  title: string;
  desc?: string;
  right?: React.ReactNode;
  children?: React.ReactNode;
}

function CardBlock({ icon, title, desc, right, children }: CardBlockProps) {
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-lg bg-muted text-muted-foreground flex items-center justify-center shrink-0">
            {icon}
          </div>
          <div>
            <h3 className="text-sm font-bold tracking-tight">{title}</h3>
            {desc && <p className="text-xs text-muted-foreground mt-0.5">{desc}</p>}
          </div>
        </div>
        {right}
      </div>
      {children}
    </Card>
  );
}

interface FieldProps {
  label: string;
  required?: boolean;
  missing?: boolean;
  hint?: string;
  children: React.ReactNode;
}

function Field({ label, required, missing, hint, children }: FieldProps) {
  return (
    <div>
      <label className="flex items-center gap-1.5 text-xs font-semibold text-foreground mb-1.5">
        {label}
        {required && <span className="text-destructive">*</span>}
        {missing && (
          <Badge variant="outline" className="text-[9px] border-amber-500/40 text-amber-400 bg-amber-500/5">
            Required to register
          </Badge>
        )}
      </label>
      {children}
      {hint && <p className="text-[11px] text-muted-foreground mt-1">{hint}</p>}
    </div>
  );
}

interface TraceRowProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  ok: boolean;
}

function TraceRow({ icon, label, value, ok }: TraceRowProps) {
  return (
    <div className="flex items-center gap-2.5">
      <div
        className={cn(
          "w-7 h-7 rounded-md flex items-center justify-center shrink-0",
          ok
            ? "bg-emerald-500/15 text-emerald-400"
            : "bg-muted text-muted-foreground"
        )}
      >
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[11px] text-muted-foreground">{label}</div>
        <div className="text-xs font-medium truncate">{value}</div>
      </div>
      {ok && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
    </div>
  );
}

/** Sidebar splits card — local editable copy, persists on "Done" via stake CRUD. */
function SplitsSidebar({
  work,
  canEdit,
  initialRows,
}: {
  work: WorkFull;
  canEdit: boolean;
  initialRows: SplitRow[];
}) {
  const [isEditing, setEditing] = useState(false);
  const [draft, setDraft] = useState<SplitRow[]>(initialRows);
  const createStake = useCreateStake();
  const updateStake = useUpdateStake();
  const deleteStake = useDeleteStake();

  useEffect(() => {
    if (!isEditing) setDraft(initialRows);
  }, [initialRows, isEditing]);

  const save = async () => {
    // Diff against work.stakes, emit minimal create/update/delete calls.
    const existing = work.stakes || [];
    const byHolder = new Map<string, { master?: OwnershipStake; publishing?: OwnershipStake }>();
    for (const s of existing) {
      if (!byHolder.has(s.holder_name)) byHolder.set(s.holder_name, {});
      const entry = byHolder.get(s.holder_name)!;
      if (s.stake_type === "master") entry.master = s;
      if (s.stake_type === "publishing") entry.publishing = s;
    }

    const promises: Promise<unknown>[] = [];
    const seenNames = new Set<string>();
    for (const row of draft) {
      const name = (row.name || "").trim();
      if (!name) continue;
      seenNames.add(name);
      const entry = byHolder.get(name) || {};
      // master
      if (row.master > 0) {
        if (entry.master) {
          if (entry.master.percentage !== row.master)
            promises.push(
              updateStake.mutateAsync({ stakeId: entry.master.id, percentage: row.master })
            );
        } else {
          promises.push(
            createStake.mutateAsync({
              work_id: work.id,
              stake_type: "master",
              holder_name: name,
              holder_role: row.role || "Collaborator",
              percentage: row.master,
            })
          );
        }
      } else if (entry.master) {
        promises.push(deleteStake.mutateAsync(entry.master.id));
      }
      // publishing
      if (row.publishing > 0) {
        if (entry.publishing) {
          if (entry.publishing.percentage !== row.publishing)
            promises.push(
              updateStake.mutateAsync({
                stakeId: entry.publishing.id,
                percentage: row.publishing,
              })
            );
        } else {
          promises.push(
            createStake.mutateAsync({
              work_id: work.id,
              stake_type: "publishing",
              holder_name: name,
              holder_role: row.role || "Collaborator",
              percentage: row.publishing,
            })
          );
        }
      } else if (entry.publishing) {
        promises.push(deleteStake.mutateAsync(entry.publishing.id));
      }
    }

    // Delete stakes whose holders were removed from the table
    for (const [name, entry] of byHolder.entries()) {
      if (seenNames.has(name)) continue;
      if (entry.master) promises.push(deleteStake.mutateAsync(entry.master.id));
      if (entry.publishing) promises.push(deleteStake.mutateAsync(entry.publishing.id));
    }

    try {
      await Promise.all(promises);
      toast.success("Royalty splits saved");
    } catch {
      // individual mutations already toast their errors
    }
  };

  return (
    <RoyaltySplitsTable
      rows={isEditing ? draft : initialRows}
      onChange={isEditing ? setDraft : undefined}
      editable={isEditing}
      showEditToggle={canEdit}
      isEditing={isEditing}
      onToggleEdit={async () => {
        if (isEditing) {
          await save();
        }
        setEditing((v) => !v);
      }}
      allowAddRow={isEditing}
      warnOnImbalance
    />
  );
}

// ============================================================
// Related-documents row + link-from-project dialog
// ============================================================

interface WorkFileLinkData {
  id: string;
  work_id: string;
  file_id: string;
  created_at: string;
  project_files?: {
    id: string;
    file_name: string;
    file_url: string;
    file_type: string | null;
    folder_category: string;
    created_at: string;
  };
}

const FOLDER_LABEL: Record<string, string> = {
  contract: "Contract",
  split_sheet: "Split sheet",
  royalty_statement: "Royalty statement",
  other: "Other",
};

function WorkFileRow({
  link,
  workId,
  canEdit,
}: {
  link: WorkFileLinkData;
  workId: string;
  canEdit: boolean;
}) {
  const unlink = useUnlinkFileFromWork();
  const file = link.project_files;
  if (!file) return null;
  return (
    <li className="flex items-center gap-3 py-2">
      <FileText className="w-4 h-4 text-muted-foreground shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium truncate">{file.file_name}</div>
        <div className="text-[11px] text-muted-foreground">
          {FOLDER_LABEL[file.folder_category] || file.folder_category}
        </div>
      </div>
      {file.file_url && (
        <a
          href={file.file_url}
          target="_blank"
          rel="noreferrer"
          className="text-muted-foreground hover:text-foreground"
          title="Open"
        >
          <ExternalLink className="w-4 h-4" />
        </a>
      )}
      {canEdit && (
        <button
          type="button"
          onClick={() => unlink.mutate({ workId, linkId: link.id })}
          disabled={unlink.isPending}
          className="text-muted-foreground hover:text-destructive p-1"
          title="Unlink from this work"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      )}
    </li>
  );
}

interface ProjectFileOption {
  id: string;
  file_name: string;
  folder_category: string;
  created_at: string;
}

function LinkProjectFilesDialog({
  open,
  onClose,
  workId,
  projectId,
  alreadyLinkedFileIds,
}: {
  open: boolean;
  onClose: () => void;
  workId: string;
  projectId: string;
  alreadyLinkedFileIds: Set<string>;
}) {
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const linkFile = useLinkFileToWork();

  const filesQuery = useQuery<ProjectFileOption[]>({
    queryKey: ["link-project-files", projectId],
    queryFn: async () => {
      if (!projectId) return [];
      const { data, error } = await supabase
        .from("project_files")
        .select("id, file_name, folder_category, created_at")
        .eq("project_id", projectId)
        .order("created_at", { ascending: false });
      if (error) throw error;
      return (data || []) as ProjectFileOption[];
    },
    enabled: open && !!projectId,
  });

  const available = (filesQuery.data || []).filter(
    (f) => !alreadyLinkedFileIds.has(f.id)
  );
  const q = search.trim().toLowerCase();
  const filtered = q
    ? available.filter(
        (f) =>
          f.file_name.toLowerCase().includes(q) ||
          (FOLDER_LABEL[f.folder_category] || "").toLowerCase().includes(q)
      )
    : available;

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const linkSelected = async () => {
    for (const fileId of selected) {
      await linkFile.mutateAsync({ workId, fileId });
    }
    setSelected(new Set());
    onClose();
  };

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Link documents from project</DialogTitle>
        </DialogHeader>
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search files…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
        <div className="max-h-[50vh] overflow-y-auto rounded-lg border divide-y">
          {filesQuery.isLoading ? (
            <div className="py-8 text-center text-sm text-muted-foreground">
              <Loader2 className="w-4 h-4 inline mr-2 animate-spin" /> Loading…
            </div>
          ) : filtered.length === 0 ? (
            <div className="py-8 text-center text-sm text-muted-foreground">
              {q
                ? "No matching files."
                : available.length === 0
                  ? "All project files are already linked to this work."
                  : "No project files yet — upload from the project Files tab."}
            </div>
          ) : (
            filtered.map((f) => {
              const picked = selected.has(f.id);
              return (
                <button
                  key={f.id}
                  type="button"
                  onClick={() => toggle(f.id)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-muted/40",
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
                    <div className="text-sm font-medium truncate">{f.file_name}</div>
                    <div className="text-[11px] text-muted-foreground">
                      {FOLDER_LABEL[f.folder_category] || f.folder_category}
                    </div>
                  </div>
                  {picked && <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />}
                </button>
              );
            })
          )}
        </div>
        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            disabled={selected.size === 0 || linkFile.isPending}
            onClick={linkSelected}
          >
            {linkFile.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Link {selected.size > 0 ? `(${selected.size})` : ""}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
