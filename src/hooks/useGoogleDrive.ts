import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { toast } from "sonner";
import type { DriveFile, IntegrationProvider } from "@/types/integrations";
import { PROVIDER_URL_SEGMENT } from "@/hooks/useIntegrations";

export function useDriveBrowse(
  folderId: string = "root",
  enabled: boolean = true,
  search: string = "",
  provider: IntegrationProvider = "google_drive"
) {
  const { user } = useAuth();

  return useQuery<DriveFile[]>({
    queryKey: ["drive-files", user?.id, provider, folderId, search],
    queryFn: async () => {
      const params = new URLSearchParams({ folder_id: folderId });
      if (search.trim()) params.set("search", search.trim());
      const data = await apiFetch<{ files: DriveFile[] }>(
        `${API_URL}/integrations/${PROVIDER_URL_SEGMENT[provider]}/browse?${params}`
      );
      return data.files;
    },
    enabled: !!user?.id && enabled,
  });
}

export function useDriveImport(provider: IntegrationProvider = "google_drive") {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: { file_id: string; project_id: string; file_type?: string }) => {
      const body =
        provider === "dropbox"
          ? { dropbox_file_id: params.file_id, project_id: params.project_id, file_type: params.file_type }
          : { drive_file_id: params.file_id, project_id: params.project_id, file_type: params.file_type };
      return apiFetch<{ file: Record<string, unknown>; source: string }>(
        `${API_URL}/integrations/${PROVIDER_URL_SEGMENT[provider]}/import`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
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
