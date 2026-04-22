import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { toast } from "sonner";
import type { DriveFile } from "@/types/integrations";

export function useDriveBrowse(folderId: string = "root", enabled: boolean = true, search: string = "") {
  const { user } = useAuth();

  return useQuery<DriveFile[]>({
    queryKey: ["drive-files", user?.id, folderId, search],
    queryFn: async () => {
      const params = new URLSearchParams({ folder_id: folderId });
      if (search.trim()) params.set("search", search.trim());
      const data = await apiFetch<{ files: DriveFile[] }>(
        `${API_URL}/integrations/google-drive/browse?${params}`
      );
      return data.files;
    },
    enabled: !!user?.id && enabled,
  });
}

export function useDriveImport() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: { drive_file_id: string; project_id: string; file_type?: string }) => {
      return apiFetch<{ file: Record<string, unknown>; source: string }>(
        `${API_URL}/integrations/google-drive/import`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["project-files"] });
      queryClient.invalidateQueries({ queryKey: ["project-files-tab"] });
    },
    onError: (error: Error) => {
      // Don't toast here — let the caller handle per-file errors for multi-select
      if (!error.message.includes("already been imported")) {
        toast.error(`Import failed: ${error.message}`);
      }
    },
  });
}

export function useDriveExport() {
  return useMutation({
    mutationFn: async (params: { project_file_id: string; drive_folder_id?: string }) => {
      return apiFetch<{ drive_file: Record<string, unknown>; source: string }>(
        `${API_URL}/integrations/google-drive/export`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        }
      );
    },
    onSuccess: () => {
      toast.success("File exported to Google Drive");
    },
    onError: (error: Error) => {
      toast.error(`Export failed: ${error.message}`);
    },
  });
}
