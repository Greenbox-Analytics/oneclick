import { useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
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
import { Loader2, Music, Upload, Volume2, Link as LinkIcon, Trash2, Mail } from "lucide-react";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import type { AudioFile, ProjectAudioLink } from "@/types/audio";
import ShareViaEmailDialog from "./ShareViaEmailDialog";

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
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  const [linkingAudioId, setLinkingAudioId] = useState<string | null>(null);
  const [linkingInProgress, setLinkingInProgress] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<AudioFile | null>(null);
  const [deleteInFlight, setDeleteInFlight] = useState(false);
  const [shareAudio, setShareAudio] = useState<AudioFile | null>(null);

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

  const { data: projectWorks } = useQuery({
    queryKey: ["project-works-for-audio-linking", projectId],
    queryFn: async () => {
      const { data, error } = await sb
        .from("works_registry")
        .select("id, title")
        .eq("project_id", projectId)
        .order("title");
      if (error) return [];
      return data || [];
    },
    enabled: !!projectId,
  });

  const audioFileIds = (audioLinks || [])
    .map((l) => l.audio_files?.id)
    .filter(Boolean);

  const { data: workAudioLinks } = useQuery({
    queryKey: ["work-audio-links-for-project", projectId, audioFileIds],
    queryFn: async () => {
      if (audioFileIds.length === 0) return [];
      const { data, error } = await sb
        .from("work_audio")
        .select("audio_file_id, work_id, works_registry(id, title)")
        .in("audio_file_id", audioFileIds);
      if (error) return [];
      return data || [];
    },
    enabled: audioFileIds.length > 0,
  });

  const audioWorksMap = new Map<string, { workId: string; title: string }[]>();
  if (workAudioLinks) {
    for (const link of workAudioLinks) {
      const w = link.works_registry;
      if (!w) continue;
      if (!audioWorksMap.has(link.audio_file_id)) audioWorksMap.set(link.audio_file_id, []);
      audioWorksMap.get(link.audio_file_id)!.push({ workId: w.id, title: w.title });
    }
  }

  const handleLinkToWork = async (audioFileId: string, workId: string) => {
    if (!user?.id) return;
    setLinkingInProgress(true);
    try {
      await apiFetch(
        `${API_URL}/registry/works/${workId}/audio?audio_file_id=${audioFileId}`,
        { method: "POST" }
      );
      queryClient.invalidateQueries({ queryKey: ["work-audio-links-for-project", projectId] });
      toast.success("Audio linked to work");
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : "Failed to link audio");
    } finally {
      setLinkingInProgress(false);
      setLinkingAudioId(null);
    }
  };

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

      // Dedup: reuse an existing audio_files row with the same hash in the folder.
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
      // Roll back any partial work to avoid orphans.
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

  const handleOpen = async (audio: AudioFile) => {
    const { data, error } = await supabase.storage
      .from("audio-files")
      .createSignedUrl(audio.file_path, 60 * 60);
    if (error || !data?.signedUrl) {
      toast.error(error?.message || "Failed to generate audio URL");
      return;
    }
    window.open(data.signedUrl, "_blank");
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
      queryClient.invalidateQueries({ queryKey: ["work-audio-links-for-project", projectId] });
      toast.success("Audio deleted");
    } catch (error: unknown) {
      toast.error(error instanceof Error ? error.message : "Failed to delete audio");
    } finally {
      setDeleteInFlight(false);
      setPendingDelete(null);
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

      {audioFiles.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <Volume2 className="w-10 h-10 text-muted-foreground/40 mb-3" />
          <p className="text-sm text-muted-foreground">No audio files yet</p>
          <p className="text-xs text-muted-foreground/60 mt-1">Upload audio files to link them to works in this project</p>
        </div>
      ) : (
        <div className="grid gap-2">
          {audioFiles.map((audio) => {
            const linkedWorks = audioWorksMap.get(audio.id);
            const hasNoWorkLinks = !linkedWorks || linkedWorks.length === 0;
            const isLinkingThis = linkingAudioId === audio.id;
            return (
              <Card key={audio.id} className="p-3">
                <div className="flex items-center gap-3">
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
                      {linkedWorks && linkedWorks.length > 0 ? (
                        <span className="flex items-center gap-1">
                          Relevant works:{" "}
                          {linkedWorks.map((w) => (
                            <Badge key={w.workId} variant="outline" className="text-[10px] px-1.5 py-0">
                              {w.title}
                            </Badge>
                          ))}
                        </span>
                      ) : (
                        <span className="italic text-muted-foreground">Not linked to any work</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    {hasNoWorkLinks && canEdit(userRole) && projectWorks && projectWorks.length > 0 && (
                      isLinkingThis ? (
                        <Select
                          onValueChange={(workId) => handleLinkToWork(audio.id, workId)}
                          disabled={linkingInProgress}
                        >
                          <SelectTrigger className="h-7 w-32 text-xs">
                            <SelectValue placeholder="Select work..." />
                          </SelectTrigger>
                          <SelectContent>
                            {projectWorks.map((work: { id: string; title: string }) => (
                              <SelectItem key={work.id} value={work.id}>
                                {work.title}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      ) : (
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 px-2 text-xs"
                          disabled={linkingInProgress}
                          onClick={() => setLinkingAudioId(audio.id)}
                        >
                          <LinkIcon className="w-3 h-3 mr-1" />
                          Link to work
                        </Button>
                      )
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-2 text-xs shrink-0"
                      onClick={() => handleOpen(audio)}
                    >
                      Open
                    </Button>
                    {canEdit(userRole) && (
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2 text-xs shrink-0"
                        onClick={() => setShareAudio(audio)}
                        title="Share via email"
                      >
                        <Mail className="w-3 h-3" />
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
        open={!!shareAudio}
        onClose={() => setShareAudio(null)}
        projectId={projectId}
        audioFileIds={shareAudio ? [shareAudio.id] : []}
        defaultSubject={shareAudio ? `Audio: ${shareAudio.file_name}` : ""}
      />
    </div>
  );
}
