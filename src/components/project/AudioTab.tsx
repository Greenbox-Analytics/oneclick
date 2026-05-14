import { useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
import { Loader2, Music, Upload, Volume2, Trash2, Send, Download } from "lucide-react";
import { toast } from "sonner";
import type { AudioFile, ProjectAudioLink } from "@/types/audio";
import ShareViaEmailDialog from "./ShareViaEmailDialog";
import BulkActionReviewDialog, { type BulkAction } from "./BulkActionReviewDialog";

interface AudioTabProps {
  projectId: string;
  userRole: string | null;
  artistId: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sb = supabase as any;

const MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024; // 50 MB

const canEdit = (role: string | null) => role === "owner" || role === "admin" || role === "editor";

function formatFileSize(bytes: number | null): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function computeSha256(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export default function AudioTab({ projectId, userRole, artistId }: AudioTabProps) {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<AudioFile | null>(null);
  const [deleteInFlight, setDeleteInFlight] = useState(false);
  const [shareAudioIds, setShareAudioIds] = useState<string[] | null>(null);
  const [shareSubject, setShareSubject] = useState<string>("");

  // Multi-select state
  const [selectedAudioIds, setSelectedAudioIds] = useState<Set<string>>(new Set());
  const [bulkAction, setBulkAction] = useState<BulkAction | null>(null);
  const [bulkInFlight, setBulkInFlight] = useState(false);

  const { data: audioLinks, isLoading: linksLoading, isError: linksError } = useQuery<
    (ProjectAudioLink & { audio_files: AudioFile })[]
  >({
    queryKey: ["project-audio-tab", projectId],
    queryFn: async () => {
      const { data, error } = await sb
        .from("project_audio_links")
        .select("*, audio_files(*)")
        .eq("project_id", projectId);
      if (error) throw error;
      return data || [];
    },
    enabled: !!projectId,
  });

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("audio/")) {
      toast.error("Only audio files are supported");
      event.target.value = "";
      return;
    }

    if (file.size > MAX_AUDIO_SIZE_BYTES) {
      toast.error(`File too large. Max ${Math.floor(MAX_AUDIO_SIZE_BYTES / (1024 * 1024))} MB.`);
      event.target.value = "";
      return;
    }

    setUploading(true);
    let uploadedPath: string | null = null;
    let insertedAudioId: string | null = null;
    try {
      const contentHash = await computeSha256(file);

      const { data: folders } = await sb
        .from("audio_folders")
        .select("id")
        .eq("artist_id", artistId)
        .is("parent_id", null)
        .limit(1);

      let folderId: string;
      if (folders && folders.length > 0) {
        folderId = folders[0].id;
      } else {
        const { data: newFolder, error: folderErr } = await sb
          .from("audio_folders")
          .insert({ artist_id: artistId, name: "Default" })
          .select("id")
          .single();
        if (folderErr) throw folderErr;
        folderId = newFolder.id;
      }

      const { data: existing } = await sb
        .from("audio_files")
        .select("id")
        .eq("folder_id", folderId)
        .eq("content_hash", contentHash)
        .maybeSingle();

      let audioFileId: string;
      if (existing?.id) {
        audioFileId = existing.id;
      } else {
        const filePath = `${artistId}/${folderId}/${Date.now()}_${file.name}`;
        const { error: uploadError } = await supabase.storage
          .from("audio-files")
          .upload(filePath, file);
        if (uploadError) throw uploadError;
        uploadedPath = filePath;

        const { data: urlData } = supabase.storage.from("audio-files").getPublicUrl(filePath);

        const { data: audioFile, error: dbError } = await sb
          .from("audio_files")
          .insert({
            folder_id: folderId,
            file_name: file.name,
            file_url: urlData.publicUrl,
            file_path: filePath,
            file_size: file.size,
            file_type: file.type,
            content_hash: contentHash,
          })
          .select("id")
          .single();
        if (dbError) throw dbError;
        audioFileId = audioFile.id;
        insertedAudioId = audioFile.id;
      }

      const { error: linkError } = await sb
        .from("project_audio_links")
        .upsert(
          { audio_file_id: audioFileId, project_id: projectId },
          { onConflict: "audio_file_id,project_id" }
        );
      if (linkError) throw linkError;

      queryClient.invalidateQueries({ queryKey: ["project-audio-tab", projectId] });
      toast.success(existing?.id ? "Audio linked (duplicate reused)" : "Audio uploaded and linked");
    } catch (error: unknown) {
      if (insertedAudioId) {
        await sb.from("audio_files").delete().eq("id", insertedAudioId);
      }
      if (uploadedPath) {
        await supabase.storage.from("audio-files").remove([uploadedPath]);
      }
      toast.error(error instanceof Error ? error.message : "Failed to upload audio");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  const deriveAudioFilePath = (audio: { file_path?: string | null; file_url?: string | null }): string | null => {
    if (audio.file_path) return audio.file_path;
    if (!audio.file_url) return null;
    const marker = "/storage/v1/object/public/audio-files/";
    const idx = audio.file_url.indexOf(marker);
    if (idx < 0) return null;
    return decodeURIComponent(audio.file_url.slice(idx + marker.length));
  };

  const handleDownloadAudio = async (audio: AudioFile) => {
    try {
      const path = deriveAudioFilePath(audio);
      if (!path) {
        toast.error("File not available for download");
        return;
      }
      const { data, error } = await supabase.storage
        .from("audio-files")
        .createSignedUrl(path, 60, { download: audio.file_name || true });
      if (error || !data?.signedUrl) {
        toast.error(error?.message || "Failed to generate audio URL");
        return;
      }
      const a = document.createElement("a");
      a.href = data.signedUrl;
      a.download = audio.file_name || "audio";
      a.click();
    } catch {
      toast.error("Failed to download audio");
    }
  };

  const handleDelete = async (audio: AudioFile) => {
    setDeleteInFlight(true);
    try {
      await sb.from("project_audio_links").delete().eq("audio_file_id", audio.id);
      await sb.from("work_audio").delete().eq("audio_file_id", audio.id);
      const { error: rowErr } = await sb.from("audio_files").delete().eq("id", audio.id);
      if (rowErr) throw rowErr;
      await supabase.storage.from("audio-files").remove([audio.file_path]);

      queryClient.invalidateQueries({ queryKey: ["project-audio-tab", projectId] });
      toast.success("Audio deleted");
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : "Failed to delete audio");
    } finally {
      setDeleteInFlight(false);
      setPendingDelete(null);
    }
  };

  const toggleAudioSelected = (id: string) => {
    setSelectedAudioIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const clearAudioSelected = () => setSelectedAudioIds(new Set());

  const getSelectedAudios = (audios: AudioFile[]) =>
    audios.filter((a) => selectedAudioIds.has(a.id));

  const bulkDownload = async (audios: AudioFile[]) => {
    setBulkInFlight(true);
    try {
      for (const audio of audios) {
        const path = deriveAudioFilePath(audio);
        if (!path) continue;
        const { data } = await supabase.storage
          .from("audio-files")
          .createSignedUrl(path, 60, { download: audio.file_name || true });
        if (data?.signedUrl) {
          const a = document.createElement("a");
          a.href = data.signedUrl;
          a.download = audio.file_name || "audio";
          a.click();
          await new Promise((r) => setTimeout(r, 150));
        }
      }
      toast.success(`Downloaded ${audios.length} file${audios.length === 1 ? "" : "s"}`);
      clearAudioSelected();
    } catch {
      toast.error("Failed to download some files");
    } finally {
      setBulkInFlight(false);
      setBulkAction(null);
    }
  };

  const bulkDelete = async (audios: AudioFile[]) => {
    setBulkInFlight(true);
    try {
      const ids = audios.map((a) => a.id);
      const paths = audios.map((a) => a.file_path).filter(Boolean) as string[];
      await sb.from("project_audio_links").delete().in("audio_file_id", ids);
      await sb.from("work_audio").delete().in("audio_file_id", ids);
      const { error } = await sb.from("audio_files").delete().in("id", ids);
      if (error) throw error;
      if (paths.length > 0) {
        await supabase.storage.from("audio-files").remove(paths);
      }
      queryClient.invalidateQueries({ queryKey: ["project-audio-tab", projectId] });
      toast.success(`Deleted ${ids.length} file${ids.length === 1 ? "" : "s"}`);
      clearAudioSelected();
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : "Failed to delete some files");
    } finally {
      setBulkInFlight(false);
      setBulkAction(null);
    }
  };

  if (linksLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (linksError) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <Volume2 className="w-10 h-10 text-destructive/40 mb-3" />
        <p className="text-sm text-muted-foreground">Failed to load audio files</p>
        <p className="text-xs text-muted-foreground/60 mt-1">Please try refreshing the page</p>
      </div>
    );
  }

  const audioFiles = (audioLinks || [])
    .map((l) => l.audio_files)
    .filter(Boolean) as AudioFile[];

  const selectedAudios = getSelectedAudios(audioFiles);
  const hasSelection = selectedAudios.length > 0;

  return (
    <div className="space-y-4">
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={handleUpload}
      />

      {canEdit(userRole) && (
        <div className="flex justify-end">
          <Button
            variant="outline"
            size="sm"
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
          >
            {uploading ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Upload className="w-4 h-4 mr-2" />
            )}
            Upload Audio
          </Button>
        </div>
      )}

      {hasSelection && (
        <div className="flex items-center gap-2 p-2 rounded-lg border border-border bg-muted/40">
          <span className="text-sm font-medium text-foreground">
            {selectedAudios.length} selected
          </span>
          <div className="flex-1" />
          <Button
            size="sm"
            variant="outline"
            onClick={() => setBulkAction("download")}
            title="Download selected"
          >
            <Download className="w-3.5 h-3.5 mr-1.5" />
            Download
          </Button>
          {canEdit(userRole) && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                setShareAudioIds(selectedAudios.map((a) => a.id));
                setShareSubject(`${selectedAudios.length} audio file${selectedAudios.length === 1 ? "" : "s"}`);
              }}
              title="Send selected"
            >
              <Send className="w-3.5 h-3.5 mr-1.5" />
              Send
            </Button>
          )}
          {canEdit(userRole) && (
            <Button
              size="sm"
              variant="outline"
              className="text-destructive hover:text-destructive"
              onClick={() => setBulkAction("delete")}
              title="Delete selected"
            >
              <Trash2 className="w-3.5 h-3.5 mr-1.5" />
              Delete
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={clearAudioSelected}>
            Clear
          </Button>
        </div>
      )}

      {audioFiles.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <Volume2 className="w-10 h-10 text-muted-foreground/40 mb-3" />
          <p className="text-sm text-muted-foreground">No audio files yet</p>
          <p className="text-xs text-muted-foreground/60 mt-1">Upload audio files to use them in this project</p>
        </div>
      ) : (
        <div className="grid gap-2">
          {audioFiles.map((audio) => {
            const isSelected = selectedAudioIds.has(audio.id);
            return (
              <Card key={audio.id} className="p-3">
                <div className="flex items-center gap-3">
                  <Checkbox
                    checked={isSelected}
                    onCheckedChange={() => toggleAudioSelected(audio.id)}
                    aria-label={`Select ${audio.file_name}`}
                  />
                  <div className="w-8 h-8 rounded bg-primary/10 flex items-center justify-center shrink-0">
                    <Music className="w-4 h-4 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">
                      {audio.file_name}
                    </p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      {audio.file_type && <span>{audio.file_type}</span>}
                      {audio.file_size && <span>{formatFileSize(audio.file_size)}</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-2 text-xs shrink-0"
                      onClick={() => handleDownloadAudio(audio)}
                      title="Download"
                    >
                      <Download className="w-3 h-3" />
                    </Button>
                    {canEdit(userRole) && (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2 text-xs shrink-0"
                        onClick={() => {
                          setShareAudioIds([audio.id]);
                          setShareSubject(`Audio: ${audio.file_name}`);
                        }}
                        title="Share via email"
                      >
                        <Send className="w-3 h-3" />
                      </Button>
                    )}
                    {canEdit(userRole) && (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2 text-xs shrink-0 text-destructive hover:text-destructive"
                        onClick={() => setPendingDelete(audio)}
                        title="Delete audio"
                      >
                        <Trash2 className="w-3 h-3" />
                      </Button>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}

      <AlertDialog open={!!pendingDelete} onOpenChange={(o) => { if (!o) setPendingDelete(null); }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this audio file?</AlertDialogTitle>
            <AlertDialogDescription>
              This permanently removes &ldquo;{pendingDelete?.file_name}&rdquo; and any work links pointing to it. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleteInFlight}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              disabled={deleteInFlight}
              onClick={(e) => {
                e.preventDefault();
                if (pendingDelete) handleDelete(pendingDelete);
              }}
            >
              {deleteInFlight ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <ShareViaEmailDialog
        open={!!shareAudioIds}
        onClose={() => setShareAudioIds(null)}
        projectId={projectId}
        audioFileIds={shareAudioIds || []}
        defaultSubject={shareSubject}
      />

      <BulkActionReviewDialog
        open={bulkAction !== null}
        onOpenChange={(open) => { if (!open) setBulkAction(null); }}
        action={bulkAction ?? "download"}
        files={selectedAudios.map((a) => ({ id: a.id, name: a.file_name }))}
        onConfirm={() => {
          if (bulkAction === "download") bulkDownload(selectedAudios);
          else if (bulkAction === "delete") bulkDelete(selectedAudios);
        }}
        isWorking={bulkInFlight}
      />
    </div>
  );
}
