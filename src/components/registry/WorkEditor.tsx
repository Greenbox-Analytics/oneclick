import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  ChevronRight,
  ChevronDown,
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
  Trash2,
  Upload,
  Plus,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import {
  useUpdateWork,
  useCreateStake,
  useUpdateStake,
  useDeleteStake,
  useExportProof,
  useFileDownloadUrl,
  useWorkAccess,
  type WorkFull,
  type OwnershipStake,
} from "@/hooks/useRegistry";
import { PermissionsPanel } from "./PermissionsPanel";
import {
  useWorkFiles,
  useUnlinkFileFromWork,
} from "@/hooks/useWorkFiles";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useStorageStatus } from "@/hooks/useEntitlements";
import { useGatedAction } from "@/hooks/useGatedAction";
import { useSpotifyTrack, spotifyTrackIdFromUrl } from "@/hooks/useSpotifySearch";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { supabase } from "@/integrations/supabase/client";
import { cn } from "@/lib/utils";
import { ReleaseTag } from "./ReleaseTag";
import { RegistryStatusBadge } from "./RegistryStatusBadge";
import { RegistryAvatar } from "./RegistryAvatar";
import { RoyaltySplitsTable, type SplitRow } from "./RoyaltySplitsTable";
import { DeleteWorkConfirmModal } from "./DeleteWorkConfirmModal";
import FetchSpotifyMetadataDialog from "./FetchSpotifyMetadataDialog";

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
  const updateWork = useUpdateWork();
  const exportProof = useExportProof();
  const { data: access } = useWorkAccess(work.id);

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

  // "Released" is an explicit flag on the work. We fall back to release_date
  // for any row predating the is_released column (the backfill defaults that
  // value to true, so this branch is effectively never hit on fresh data).
  const released = work.is_released ?? !!work.release_date;
  // Edit affordances are driven by the resolved WorkAccess (project membership +
  // work role), not by raw ownership. While access is still loading, default to
  // non-editable to avoid a flash of editable controls for viewers.
  const canEdit = !!access?.can_edit;
  // Work-only viewers only receive their own slice of the splits, so a summed
  // "total" (and the "should total 100%" nudge) would be meaningless for them.
  // This is only ever true for people who see every stake (owner/admin/editor or
  // a project member), so it's the right gate for the totals + imbalance warning.
  const canSeeFullOwnership = !!access?.can_see_full_ownership;

  // "Your access" card label/description, derived from the resolved access.
  const accessRole = useMemo(() => {
    if (access?.work_role === "owner")
      return { isOwner: true, label: "Owner", desc: "Full control" };
    if (access?.can_manage)
      return { isOwner: false, label: "Admin", desc: "Can manage & edit" };
    if (access?.can_edit)
      return { isOwner: false, label: "Editor", desc: "Can edit" };
    return { isOwner: false, label: "Viewer", desc: "View only" };
  }, [access?.work_role, access?.can_manage, access?.can_edit]);

  // The owner's identity (name + contact email), surfaced to non-owners so they
  // know who to reach for edit access or a fuller view. The backend always
  // includes an owner row (status "owner") in the collaborators list for
  // work-only viewers; `name` falls back to the literal "Owner" when the profile
  // has no name, which we treat as "unknown" so we never render a placeholder as
  // a real name. `email` may be null if the auth lookup fails.
  const owner = useMemo(() => {
    const row = (work.collaborators || []).find((c) => c.status === "owner");
    if (!row) return null;
    const rawName = row.name?.trim();
    const name = rawName && rawName.toLowerCase() !== "owner" ? rawName : null;
    const email = row.email?.trim() || null;
    if (!name && !email) return null;
    return { name, email };
  }, [work.collaborators]);

  const set = (patch: Partial<WorkFull>) =>
    updateWork.mutate({ workId: work.id, ...patch });

  const [spotifyFetchOpen, setSpotifyFetchOpen] = useState(false);

  const toggleReleased = (val: boolean) => {
    // Going Unreleased → Released: seed today's date unless one already exists.
    // We bundle release_date with is_released so the user doesn't have to fill
    // both fields in two steps. Going off keeps release_date as-is — clearing
    // it would discard a date the user may want to keep on record.
    const wasReleased = work.is_released ?? !!work.release_date;
    if (val && !work.release_date) {
      set({ is_released: true, release_date: new Date().toISOString().slice(0, 10) });
    } else {
      set({ is_released: val });
    }
    // First flip into "released" → offer to pull the newly-public Spotify
    // metadata. Skipped when the artist has no name to search with.
    if (val && !wasReleased && artist?.name) {
      setSpotifyFetchOpen(true);
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
        const credited = (data.data.artists || []).map((a) => ({
          name: a.name,
          role: a.role,
        }));
        set({
          isrc: work.isrc || data.data.isrc || null,
          upc: work.upc || data.data.upc || null,
          release_date: work.release_date || data.data.release_date || null,
          genre: work.genre || data.data.genre || null,
          label: work.label || data.data.label || null,
          featured_artists:
            work.featured_artists && work.featured_artists.length > 0
              ? work.featured_artists
              : credited.length > 0
                ? credited
                : null,
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
  const [deleteOpen, setDeleteOpen] = useState(false);

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
            {access?.can_delete && (
              <Button
                variant="ghost"
                size="sm"
                className="ml-auto text-muted-foreground hover:text-destructive"
                onClick={() => setDeleteOpen(true)}
              >
                <Trash2 className="w-3.5 h-3.5 mr-1.5" /> Delete work
              </Button>
            )}
          </div>
        </div>

        {!canEdit && (
          <Card className="p-3 flex items-center gap-3 border-amber-500/30 bg-amber-500/5">
            <Lock className="w-4 h-4 text-amber-400 shrink-0" />
            <div className="text-sm">
              <strong className="font-semibold">View only.</strong>{" "}
              <span className="text-muted-foreground">
                {owner ? (
                  <>
                    Contact{" "}
                    {owner.email ? (
                      <a
                        href={`mailto:${owner.email}`}
                        className="font-medium text-foreground hover:underline"
                      >
                        {owner.name || owner.email}
                      </a>
                    ) : (
                      <span className="font-medium text-foreground">{owner.name}</span>
                    )}{" "}
                    (the owner) for edit access or a fuller view of this work.
                  </>
                ) : (
                  `Ask the owner of ${project?.name || "this project"} for edit access.`
                )}
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
            <Field label="Genre" hint="Pulls from Spotify">
              <Input
                value={work.genre || ""}
                placeholder="e.g. Afrobeats"
                disabled={!canEdit}
                onChange={(e) => set({ genre: e.target.value })}
              />
            </Field>
            <Field label="Label" hint="Record label / distributor">
              <Input
                value={work.label || ""}
                placeholder="e.g. Sony Music"
                disabled={!canEdit}
                onChange={(e) => set({ label: e.target.value })}
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

          {/* Credited artists — display-only metadata pulled from Spotify,
              separate from the ownership / royalty splits below. */}
          <div className="mt-4 pt-4 border-t border-border/60">
            <div className="text-sm font-medium mb-2">Featured artists &amp; credits</div>
            {work.featured_artists && work.featured_artists.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {work.featured_artists.map((a, i) => (
                  <span
                    key={`${a.name}-${i}`}
                    className="inline-flex items-center gap-1.5 rounded-full border bg-muted/40 px-2.5 py-1 text-xs"
                  >
                    <span className="font-medium">{a.name}</span>
                    <span className="text-muted-foreground">· {a.role}</span>
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                No credited artists yet — pull from Spotify to populate them.
              </p>
            )}
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
                <Plus className="w-3.5 h-3.5 mr-1.5" /> Add documents
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
                    Add a document.
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

        {/* People & access — owners/admins manage roles + per-resource visibility */}
        {access?.can_manage && (
          <PermissionsPanel workId={work.id} projectName={project?.name} />
        )}
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
                accessRole.isOwner
                  ? "bg-primary/15 text-primary"
                  : "bg-blue-500/15 text-blue-400"
              )}
            >
              {accessRole.isOwner ? (
                <Shield className="w-4 h-4" />
              ) : (
                <Eye className="w-4 h-4" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-semibold">{accessRole.label}</div>
              <div className="text-xs text-muted-foreground">{accessRole.desc}</div>
            </div>
          </div>
          {!accessRole.isOwner && owner && (
            <div className="pt-3 border-t border-border/60">
              <div className="text-[11px] font-medium text-muted-foreground mb-2">
                Owner
              </div>
              <div className="flex items-center gap-2 min-w-0">
                <RegistryAvatar name={owner.name || owner.email || "Owner"} size={26} />
                <div className="min-w-0">
                  <div className="text-xs font-semibold truncate">
                    {owner.name || "Owner"}
                  </div>
                  {owner.email ? (
                    <a
                      href={`mailto:${owner.email}`}
                      className="block text-[11px] text-primary hover:underline truncate"
                      title={`Email ${owner.name || owner.email}`}
                    >
                      {owner.email}
                    </a>
                  ) : (
                    <div className="text-[11px] text-muted-foreground">
                      Email unavailable
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Royalty splits */}
        <Card className="p-4">
          <SplitsSidebar
            work={work}
            canEdit={canEdit}
            canSeeFullOwnership={canSeeFullOwnership}
            initialRows={stakeRows}
          />
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

      {access?.can_delete && (
        <DeleteWorkConfirmModal
          workId={work.id}
          workTitle={work.title || "Untitled work"}
          open={deleteOpen}
          onOpenChange={setDeleteOpen}
          onDeleted={() => navigate("/tools/registry")}
        />
      )}

      <FetchSpotifyMetadataDialog
        open={spotifyFetchOpen}
        onOpenChange={setSpotifyFetchOpen}
        artistName={artist?.name || ""}
        workTitle={work.title}
        currentMeta={{
          isrc: work.isrc,
          upc: work.upc,
          release_date: work.release_date,
          notes: work.notes,
          genre: work.genre,
          label: work.label,
        }}
        onApply={(patch) => updateWork.mutate({ workId: work.id, ...patch })}
      />
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
  canSeeFullOwnership,
  initialRows,
}: {
  work: WorkFull;
  canEdit: boolean;
  canSeeFullOwnership: boolean;
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
              ...(row.isYou === true ? { is_owner_stake: true } : {}),
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
              ...(row.isYou === true ? { is_owner_stake: true } : {}),
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
      warnOnImbalance={canEdit}
      showTotals={canSeeFullOwnership}
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
  const downloadUrl = useFileDownloadUrl();
  const file = link.project_files;
  if (!file) return null;

  const handleOpen = async () => {
    try {
      const { url } = await downloadUrl.mutateAsync({
        workId,
        fileId: file.id,
      });
      window.open(url, "_blank", "noopener,noreferrer");
    } catch (e) {
      toast.error(
        e instanceof Error ? e.message : "Could not open this document"
      );
    }
  };

  return (
    <li className="flex items-center gap-3 py-2">
      <FileText className="w-4 h-4 text-muted-foreground shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium truncate">{file.file_name}</div>
        <div className="text-[11px] text-muted-foreground">
          {FOLDER_LABEL[file.folder_category] || file.folder_category}
        </div>
      </div>
      {file.id && (
        <button
          type="button"
          onClick={handleOpen}
          disabled={downloadUrl.isPending}
          className="text-muted-foreground hover:text-foreground disabled:opacity-50"
          title="Open"
        >
          <ExternalLink className="w-4 h-4" />
        </button>
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

// Categories shown in the Add Documents dialog. Mirrors the Project FilesTab
// (same keys, labels, and accent colors) so users see a consistent grouping
// across the work and project surfaces.
const DOC_CATEGORIES = [
  { key: "contract", label: "Contracts", accent: "border-l-blue-600 dark:border-l-blue-500", iconColor: "text-blue-600 dark:text-blue-400" },
  { key: "split_sheet", label: "Split Sheets", accent: "border-l-purple-600 dark:border-l-purple-500", iconColor: "text-purple-600 dark:text-purple-400" },
  { key: "royalty_statement", label: "Royalty Statements", accent: "border-l-orange-600 dark:border-l-orange-500", iconColor: "text-orange-600 dark:text-orange-400" },
  { key: "other", label: "Other", accent: "border-l-slate-500 dark:border-l-slate-400", iconColor: "text-slate-600 dark:text-slate-400" },
] as const;

function fmtBytes(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)} GB`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)} MB`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)} KB`;
  return `${n} B`;
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
  const queryClient = useQueryClient();
  const storageStatus = useStorageStatus();
  const fileInputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    contract: true,
    split_sheet: true,
    royalty_statement: false,
    other: false,
  });
  const [uploadingCategory, setUploadingCategory] = useState<string | null>(null);
  const [linkingInProgress, setLinkingInProgress] = useState(false);

  // Load all of the project's files so we can group them by category and let
  // the user pick which ones to link. Re-fetch on every open so freshly-uploaded
  // files (from any surface) appear right away.
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

  // Files not yet linked to this work, grouped by category. Files outside the
  // known categories fall into "other".
  const filesByCategory = useMemo(() => {
    const groups: Record<string, ProjectFileOption[]> = {
      contract: [],
      split_sheet: [],
      royalty_statement: [],
      other: [],
    };
    for (const f of filesQuery.data || []) {
      if (alreadyLinkedFileIds.has(f.id)) continue;
      const key = f.folder_category in groups ? f.folder_category : "other";
      groups[key].push(f);
    }
    return groups;
  }, [filesQuery.data, alreadyLinkedFileIds]);

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSection = (key: string) =>
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));

  // Upload pipeline: SHA-256 dedup → storage upload → project_files insert →
  // immediate work-link. Mirrors FilesTab's upload behavior, but auto-links
  // (no work-picker round-trip) since we already know the target work.
  const { mutate: gatedUpload, paywallElement } = useGatedAction<
    void,
    { file: File; category: string }
  >({
    mutationFn: async ({ file, category }) => {
      const hashBuffer = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
      const contentHash = Array.from(new Uint8Array(hashBuffer))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");

      // Dedup: if this exact file is already in the project, just link the
      // existing row instead of re-uploading. Silent — no confirm dialog.
      const { data: existing } = await supabase
        .from("project_files")
        .select("id")
        .eq("project_id", projectId)
        .eq("content_hash", contentHash)
        .limit(1);

      let fileIdToLink = existing && existing.length > 0 ? existing[0].id : null;

      if (!fileIdToLink) {
        const filePath = `${projectId}/${category}/${Date.now()}_${file.name}`;
        const { error: uploadError } = await supabase.storage
          .from("project-files")
          .upload(filePath, file);
        if (uploadError) throw uploadError;
        const { data: urlData } = supabase.storage
          .from("project-files")
          .getPublicUrl(filePath);
        const { data: insertedData, error: dbError } = await supabase
          .from("project_files")
          .insert({
            project_id: projectId,
            file_name: file.name,
            file_url: urlData.publicUrl,
            file_path: filePath,
            folder_category: category,
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
        fileIdToLink = insertedData.id;
      }

      if (!alreadyLinkedFileIds.has(fileIdToLink)) {
        await apiFetch(
          `${API_URL}/registry/works/${workId}/files?file_id=${fileIdToLink}`,
          { method: "POST" }
        );
      }

      queryClient.invalidateQueries({ queryKey: ["link-project-files", projectId] });
      queryClient.invalidateQueries({ queryKey: ["project-files-tab", projectId] });
      queryClient.invalidateQueries({ queryKey: ["work-files", workId] });
      toast.success(existing && existing.length > 0 ? "Linked existing file" : "Uploaded and linked");
      setUploadingCategory(null);
    },
    onError: (err) => {
      setUploadingCategory(null);
      toast.error(err instanceof Error ? err.message : "Upload failed");
    },
  });

  const handleUpload = (category: string, event: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = Array.from(event.target.files || []);
    event.target.value = "";
    if (fileList.length === 0) return;

    const totalSize = fileList.reduce((sum, f) => sum + f.size, 0);
    if (storageStatus.cap !== -1 && storageStatus.used + totalSize > storageStatus.cap) {
      toast.error(
        `Uploading ${fileList.length} file(s) (${fmtBytes(totalSize)}) would exceed your storage cap.`
      );
      return;
    }

    setUploadingCategory(category);
    for (const file of fileList) {
      gatedUpload({ file, category });
    }
  };

  const linkSelected = async () => {
    if (selected.size === 0) return;
    setLinkingInProgress(true);
    try {
      // Link in parallel — these are independent inserts; no need to serialize.
      await Promise.all(
        Array.from(selected).map((fileId) =>
          apiFetch(
            `${API_URL}/registry/works/${workId}/files?file_id=${fileId}`,
            { method: "POST" }
          )
        )
      );
      queryClient.invalidateQueries({ queryKey: ["work-files", workId] });
      queryClient.invalidateQueries({ queryKey: ["link-project-files", projectId] });
      toast.success(`Linked ${selected.size} document${selected.size === 1 ? "" : "s"}`);
      setSelected(new Set());
      onClose();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to link documents");
    } finally {
      setLinkingInProgress(false);
    }
  };

  const isLoading = filesQuery.isLoading;
  const totalAvailable = Object.values(filesByCategory).reduce((n, list) => n + list.length, 0);

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-w-xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Add documents</DialogTitle>
          <p className="text-xs text-muted-foreground mt-1">
            Upload new files from your computer or pick existing project files. Files are organized by category.
          </p>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-2 -mx-1 px-1">
          {isLoading ? (
            <div className="py-12 text-center text-sm text-muted-foreground">
              <Loader2 className="w-4 h-4 inline mr-2 animate-spin" /> Loading…
            </div>
          ) : (
            DOC_CATEGORIES.map((cat) => {
              const items = filesByCategory[cat.key] || [];
              const isOpen = openSections[cat.key];
              const isUploading = uploadingCategory === cat.key;
              return (
                <Card key={cat.key} className={cn("border-l-4 overflow-hidden", cat.accent)}>
                  <Collapsible open={isOpen} onOpenChange={() => toggleSection(cat.key)}>
                    <CollapsibleTrigger className="w-full flex items-center gap-2.5 px-3 py-2.5 hover:bg-muted/30 text-left">
                      {isOpen ? (
                        <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
                      )}
                      <FileText className={cn("w-4 h-4 shrink-0", cat.iconColor)} />
                      <span className="text-sm font-semibold">{cat.label}</span>
                      <Badge variant="outline" className="text-[10px] font-medium">
                        {items.length}
                      </Badge>
                      <span
                        role="button"
                        tabIndex={0}
                        onClick={(e) => {
                          e.stopPropagation();
                          fileInputRefs.current[cat.key]?.click();
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            e.stopPropagation();
                            fileInputRefs.current[cat.key]?.click();
                          }
                        }}
                        className="ml-auto inline-flex items-center gap-1.5 text-xs font-medium text-primary hover:underline cursor-pointer"
                      >
                        {isUploading ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Upload className="w-3.5 h-3.5" />
                        )}
                        Upload
                      </span>
                      <input
                        ref={(el) => {
                          fileInputRefs.current[cat.key] = el;
                        }}
                        type="file"
                        multiple
                        className="hidden"
                        onChange={(e) => handleUpload(cat.key, e)}
                      />
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      {items.length === 0 ? (
                        <div className="px-4 py-3 text-xs text-muted-foreground border-t">
                          No unlinked {cat.label.toLowerCase()} in this project — upload one from your computer.
                        </div>
                      ) : (
                        <ul className="divide-y border-t">
                          {items.map((f) => {
                            const picked = selected.has(f.id);
                            return (
                              <li key={f.id}>
                                <button
                                  type="button"
                                  onClick={() => toggle(f.id)}
                                  className={cn(
                                    "w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-muted/40",
                                    picked && "bg-muted/40"
                                  )}
                                >
                                  <Checkbox checked={picked} className="pointer-events-none" />
                                  <FileText className="w-4 h-4 text-muted-foreground shrink-0" />
                                  <div className="flex-1 min-w-0">
                                    <div className="text-sm font-medium truncate">{f.file_name}</div>
                                  </div>
                                </button>
                              </li>
                            );
                          })}
                        </ul>
                      )}
                    </CollapsibleContent>
                  </Collapsible>
                </Card>
              );
            })
          )}

          {!isLoading && totalAvailable === 0 && (
            <div className="text-xs text-muted-foreground text-center py-2">
              No unlinked project files. Upload from above to add documents.
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 pt-2">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button disabled={selected.size === 0 || linkingInProgress} onClick={linkSelected}>
            {linkingInProgress && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Link selected{selected.size > 0 ? ` (${selected.size})` : ""}
          </Button>
        </DialogFooter>
        {paywallElement}
      </DialogContent>
    </Dialog>
  );
}
