import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import { supabase } from "@/integrations/supabase/client";
import type { AudioFolder, AudioFile, ProjectAudioLink } from "@/types/audio";

// Supabase types.ts doesn't include audio tables yet, so we cast through `any`
const sb = supabase as any;

export function useAudioData(artistIds: string[], projectIds: string[]) {
  const queryClient = useQueryClient();

  // Query 1: Fetch all audio folders for user's artists (root only for MVP)
  const foldersQuery = useQuery<AudioFolder[]>({
    queryKey: ["audio-folders", artistIds],
    queryFn: async () => {
      if (artistIds.length === 0) return [];
      const { data, error } = await sb
        .from("audio_folders")
        .select("*")
        .in("artist_id", artistIds)
        .is("parent_id", null)
        .order("name");
      if (error) { console.error("Error fetching audio folders:", error); return []; }
      return data || [];
    },
    enabled: artistIds.length > 0,
  });

  const folderIds = useMemo(
    () => (foldersQuery.data || []).map((f) => f.id),
    [foldersQuery.data]
  );

  // Query 2: Fetch all audio files for those folders
  const filesQuery = useQuery<AudioFile[]>({
    queryKey: ["audio-files", folderIds],
    queryFn: async () => {
      if (folderIds.length === 0) return [];
      const batchSize = 100;
      const allFiles: AudioFile[] = [];
      for (let i = 0; i < folderIds.length; i += batchSize) {
        const batch = folderIds.slice(i, i + batchSize);
        const { data, error } = await sb
          .from("audio_files")
          .select("*")
          .in("folder_id", batch)
          .order("file_name");
        if (error) { console.error("Error fetching audio files:", error); continue; }
        allFiles.push(...(data || []));
      }
      return allFiles;
    },
    enabled: folderIds.length > 0,
  });

  // Query 3: Fetch all project_audio_links for user's projects
  const linksQuery = useQuery<ProjectAudioLink[]>({
    queryKey: ["project-audio-links", projectIds],
    queryFn: async () => {
      if (projectIds.length === 0) return [];
      const batchSize = 100;
      const allLinks: ProjectAudioLink[] = [];
      for (let i = 0; i < projectIds.length; i += batchSize) {
        const batch = projectIds.slice(i, i + batchSize);
        const { data, error } = await sb
          .from("project_audio_links")
          .select("*")
          .in("project_id", batch);
        if (error) { console.error("Error fetching audio links:", error); continue; }
        allLinks.push(...(data || []));
      }
      return allLinks;
    },
    enabled: projectIds.length > 0,
  });

  // Build lookup maps
  const foldersByArtist = useMemo(() => {
    const map = new Map<string, AudioFolder[]>();
    for (const f of foldersQuery.data || []) {
      if (!map.has(f.artist_id)) map.set(f.artist_id, []);
      map.get(f.artist_id)!.push(f);
    }
    return map;
  }, [foldersQuery.data]);

  const filesByFolder = useMemo(() => {
    const map = new Map<string, AudioFile[]>();
    for (const f of filesQuery.data || []) {
      if (!map.has(f.folder_id)) map.set(f.folder_id, []);
      map.get(f.folder_id)!.push(f);
    }
    return map;
  }, [filesQuery.data]);

  const audioFileMap = useMemo(() => {
    const map = new Map<string, AudioFile>();
    for (const f of filesQuery.data || []) map.set(f.id, f);
    return map;
  }, [filesQuery.data]);

  const audioLinksByProject = useMemo(() => {
    const map = new Map<string, AudioFile[]>();
    for (const link of linksQuery.data || []) {
      const file = audioFileMap.get(link.audio_file_id);
      if (!file) continue;
      if (!map.has(link.project_id)) map.set(link.project_id, []);
      map.get(link.project_id)!.push(file);
    }
    return map;
  }, [linksQuery.data, audioFileMap]);

  const projectsByAudioFile = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const link of linksQuery.data || []) {
      if (!map.has(link.audio_file_id)) map.set(link.audio_file_id, []);
      map.get(link.audio_file_id)!.push(link.project_id);
    }
    return map;
  }, [linksQuery.data]);

  // Invalidate all audio queries
  const refetchAll = () => {
    queryClient.invalidateQueries({ queryKey: ["audio-folders"] });
    queryClient.invalidateQueries({ queryKey: ["audio-files"] });
    queryClient.invalidateQueries({ queryKey: ["project-audio-links"] });
  };

  // --- Mutation helpers ---

  const createFolder = async (artistId: string, name: string) => {
    // Check for duplicate folder name for this artist
    const { data: existing } = await sb
      .from("audio_folders")
      .select("id")
      .eq("artist_id", artistId)
      .ilike("name", name.trim())
      .limit(1);
    if (existing && existing.length > 0) {
      throw new Error(`DUPLICATE:An audio folder named "${name.trim()}" already exists for this artist.`);
    }
    const { error } = await sb
      .from("audio_folders")
      .insert({ artist_id: artistId, name: name.trim() });
    if (error) throw error;
    refetchAll();
  };

  const renameFolder = async (folderId: string, newName: string) => {
    const { error } = await sb
      .from("audio_folders")
      .update({ name: newName.trim() })
      .eq("id", folderId);
    if (error) throw error;
    refetchAll();
  };

  const deleteFolder = async (folderId: string) => {
    // Files cascade-delete via FK, but we should also remove from storage
    const folderFiles = filesByFolder.get(folderId) || [];
    if (folderFiles.length > 0) {
      const paths = folderFiles.map((f) => f.file_path).filter(Boolean);
      if (paths.length > 0) {
        await supabase.storage.from("audio-files").remove(paths);
      }
    }
    const { error } = await sb.from("audio_folders").delete().eq("id", folderId);
    if (error) throw error;
    refetchAll();
  };

  const uploadAudioFile = async (folderId: string, artistId: string, file: File) => {
    // Check for duplicate file name in this folder
    const { data: existingFiles } = await sb
      .from("audio_files")
      .select("id, file_name")
      .eq("folder_id", folderId);
    if (existingFiles) {
      const hasDuplicate = existingFiles.some(
        (f: any) => f.file_name.trim().toLowerCase() === file.name.trim().toLowerCase()
      );
      if (hasDuplicate) {
        throw new Error(`DUPLICATE:An audio file named "${file.name}" already exists in this folder.`);
      }
    }
    const filePath = `${artistId}/${folderId}/${Date.now()}_${file.name}`;
    const { error: uploadError } = await supabase.storage
      .from("audio-files")
      .upload(filePath, file);
    if (uploadError) throw uploadError;

    const { data: urlData } = supabase.storage
      .from("audio-files")
      .getPublicUrl(filePath);

    const { error: dbError } = await sb.from("audio_files").insert({
      folder_id: folderId,
      file_name: file.name,
      file_url: urlData.publicUrl,
      file_path: filePath,
      file_size: file.size,
      file_type: file.type,
    });
    if (dbError) {
      await supabase.storage.from("audio-files").remove([filePath]);
      throw dbError;
    }
    refetchAll();
  };

  const deleteAudioFile = async (fileId: string, filePath: string) => {
    await supabase.storage.from("audio-files").remove([filePath]);
    const { error } = await sb.from("audio_files").delete().eq("id", fileId);
    if (error) throw error;
    refetchAll();
  };

  const linkAudioToProject = async (audioFileId: string, projectId: string) => {
    const { error } = await sb
      .from("project_audio_links")
      .insert({ audio_file_id: audioFileId, project_id: projectId });
    if (error) throw error;
    refetchAll();
  };

  const unlinkAudioFromProject = async (audioFileId: string, projectId: string) => {
    const { error } = await sb
      .from("project_audio_links")
      .delete()
      .eq("audio_file_id", audioFileId)
      .eq("project_id", projectId);
    if (error) throw error;
    refetchAll();
  };

  return {
    foldersByArtist,
    filesByFolder,
    audioLinksByProject,
    projectsByAudioFile,
    isLoading: foldersQuery.isLoading || filesQuery.isLoading || linksQuery.isLoading,
    createFolder,
    renameFolder,
    deleteFolder,
    uploadAudioFile,
    deleteAudioFile,
    linkAudioToProject,
    unlinkAudioFromProject,
  };
}
