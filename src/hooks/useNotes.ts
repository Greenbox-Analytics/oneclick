import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface Note {
  id: string;
  user_id: string;
  folder_id: string | null;
  artist_id: string | null;
  project_id: string | null;
  title: string;
  content: unknown[];
  pinned: boolean;
  created_at: string;
  updated_at: string;
}

export interface NoteFolder {
  id: string;
  user_id: string;
  artist_id: string | null;
  project_id: string | null;
  name: string;
  parent_folder_id: string | null;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export function useNotes(scope: { artistId?: string; projectId?: string; folderId?: string }) {
  const { user } = useAuth();
  return useQuery<Note[]>({
    queryKey: ["notes", user?.id, scope.artistId, scope.projectId, scope.folderId],
    queryFn: async () => {
      if (!user?.id) return [];
      const params = new URLSearchParams();
      if (scope.artistId) params.set("artist_id", scope.artistId);
      if (scope.projectId) params.set("project_id", scope.projectId);
      if (scope.folderId) params.set("folder_id", scope.folderId);
      const qs = params.toString();
      const data = await apiFetch<{ notes: Note[] }>(`${API_URL}/registry/notes${qs ? `?${qs}` : ""}`);
      return data.notes;
    },
    enabled: !!user?.id,
  });
}

export function useNote(noteId: string | undefined) {
  const { user } = useAuth();
  return useQuery<Note | null>({
    queryKey: ["note", user?.id, noteId],
    queryFn: async () => {
      if (!user?.id || !noteId) return null;
      return apiFetch<Note>(`${API_URL}/registry/notes/${noteId}`);
    },
    enabled: !!user?.id && !!noteId,
  });
}

export function useCreateNote() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      title?: string; content?: unknown[]; artist_id?: string;
      project_id?: string; folder_id?: string; pinned?: boolean;
    }) =>
      apiFetch<Note>(`${API_URL}/registry/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notes"] });
      toast.success("Note created");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateNote() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ noteId, ...body }: {
      noteId: string; title?: string; content?: unknown[];
      folder_id?: string | null; pinned?: boolean;
    }) =>
      apiFetch<Note>(`${API_URL}/registry/notes/${noteId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notes"] });
      qc.invalidateQueries({ queryKey: ["note"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteNote() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (noteId: string) =>
      apiFetch(`${API_URL}/registry/notes/${noteId}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notes"] });
      toast.success("Note deleted");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useFolders(scope: { artistId?: string; projectId?: string }) {
  const { user } = useAuth();
  return useQuery<NoteFolder[]>({
    queryKey: ["note-folders", user?.id, scope.artistId, scope.projectId],
    queryFn: async () => {
      if (!user?.id) return [];
      const params = new URLSearchParams();
      if (scope.artistId) params.set("artist_id", scope.artistId);
      if (scope.projectId) params.set("project_id", scope.projectId);
      const qs = params.toString();
      const data = await apiFetch<{ folders: NoteFolder[] }>(`${API_URL}/registry/folders${qs ? `?${qs}` : ""}`);
      return data.folders;
    },
    enabled: !!user?.id,
  });
}

export function useCreateFolder() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      name: string; artist_id?: string; project_id?: string;
      parent_folder_id?: string; sort_order?: number;
    }) =>
      apiFetch<NoteFolder>(`${API_URL}/registry/folders`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["note-folders"] });
      toast.success("Folder created");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateFolder() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ folderId, ...body }: {
      folderId: string; name?: string; parent_folder_id?: string | null; sort_order?: number;
    }) =>
      apiFetch<NoteFolder>(`${API_URL}/registry/folders/${folderId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["note-folders"] }),
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteFolder() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (folderId: string) =>
      apiFetch(`${API_URL}/registry/folders/${folderId}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["note-folders"] });
      qc.invalidateQueries({ queryKey: ["notes"] });
      toast.success("Folder deleted");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useProjectAbout(projectId: string | undefined) {
  const { user } = useAuth();
  return useQuery<unknown[]>({
    queryKey: ["project-about", user?.id, projectId],
    queryFn: async () => {
      if (!user?.id || !projectId) return [];
      const data = await apiFetch<{ about_content: unknown[] }>(
        `${API_URL}/registry/projects/${projectId}/about`
      );
      return data.about_content;
    },
    enabled: !!user?.id && !!projectId,
  });
}

export function useUpdateProjectAbout() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ projectId, about_content }: { projectId: string; about_content: unknown[] }) =>
      apiFetch(`${API_URL}/registry/projects/${projectId}/about`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ about_content }),
      }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ["project-about", vars.projectId] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
}
