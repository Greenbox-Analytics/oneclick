import { useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, Music, Upload, Volume2 } from "lucide-react";
import { toast } from "sonner";
import type { AudioFile, ProjectAudioLink } from "@/types/audio";

interface AudioTabProps {
  projectId: string;
  userRole: string | null;
  artistId: string;
}

const sb = supabase as any;

const canEdit = (role: string | null) => role === "owner" || role === "admin" || role === "editor";

function formatFileSize(bytes: number | null): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function AudioTab({ projectId, userRole, artistId }: AudioTabProps) {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);

  // Fetch audio files linked to this project via project_audio_links
  const { data: audioLinks, isLoading: linksLoading } = useQuery<
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

  // Fetch work_audio links for these audio files to show relevant works
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

  // Build audio file -> works map
  const audioWorksMap = new Map<string, { workId: string; title: string }[]>();
  if (workAudioLinks) {
    for (const link of workAudioLinks) {
      const w = link.works_registry;
      if (!w) continue;
      if (!audioWorksMap.has(link.audio_file_id)) audioWorksMap.set(link.audio_file_id, []);
      audioWorksMap.get(link.audio_file_id)!.push({ workId: w.id, title: w.title });
    }
  }

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      // First, ensure the artist has an audio folder (create one if needed)
      let { data: folders } = await sb
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

      // Upload to storage
      const filePath = `${artistId}/${folderId}/${Date.now()}_${file.name}`;
      const { error: uploadError } = await supabase.storage
        .from("audio-files")
        .upload(filePath, file);
      if (uploadError) throw uploadError;

      const { data: urlData } = supabase.storage.from("audio-files").getPublicUrl(filePath);

      // Insert audio file record
      const { data: audioFile, error: dbError } = await sb
        .from("audio_files")
        .insert({
          folder_id: folderId,
          file_name: file.name,
          file_url: urlData.publicUrl,
          file_path: filePath,
          file_size: file.size,
          file_type: file.type,
        })
        .select("id")
        .single();

      if (dbError) {
        await supabase.storage.from("audio-files").remove([filePath]);
        throw dbError;
      }

      // Link to this project
      const { error: linkError } = await sb
        .from("project_audio_links")
        .insert({ audio_file_id: audioFile.id, project_id: projectId });
      if (linkError) throw linkError;

      queryClient.invalidateQueries({ queryKey: ["project-audio-tab", projectId] });
      toast.success("Audio uploaded and linked");
    } catch (error: any) {
      toast.error(error.message || "Failed to upload audio");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  if (linksLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const audioFiles = (audioLinks || [])
    .map((l) => l.audio_files)
    .filter(Boolean) as AudioFile[];

  return (
    <div className="space-y-4">
      {canEdit(userRole) && (
        <div className="flex justify-end">
          <Button size="sm" disabled={uploading} onClick={() => fileInputRef.current?.click()}>
            {uploading ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Upload className="w-4 h-4 mr-2" />
            )}
            Upload Audio
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            className="hidden"
            onChange={handleUpload}
          />
        </div>
      )}

      {audioFiles.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 gap-2">
          <Volume2 className="w-12 h-12 text-muted-foreground/40" />
          <p className="text-muted-foreground">No audio files linked to this project.</p>
          {canEdit(userRole) && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="w-4 h-4 mr-2" /> Upload Audio
            </Button>
          )}
        </div>
      ) : (
        <div className="grid gap-2">
          {audioFiles.map((audio) => {
            const linkedWorks = audioWorksMap.get(audio.id);
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
                      {linkedWorks && linkedWorks.length > 0 && (
                        <span className="flex items-center gap-1">
                          Works:{" "}
                          {linkedWorks.map((w) => (
                            <Badge key={w.workId} variant="outline" className="text-[10px] px-1.5 py-0">
                              {w.title}
                            </Badge>
                          ))}
                        </span>
                      )}
                    </div>
                  </div>
                  {audio.file_url && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 px-2 text-xs shrink-0"
                      onClick={() => window.open(audio.file_url, "_blank")}
                    >
                      Open
                    </Button>
                  )}
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
