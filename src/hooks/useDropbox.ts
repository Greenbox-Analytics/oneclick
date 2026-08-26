import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import { toast } from "sonner";

export function useDropboxExportStatus(projectFileId: string | null, enabled: boolean) {
  const { user } = useAuth();

  return useQuery<{ saved: boolean; share_url: string | null }>({
    queryKey: ["dropbox-export-status", user?.id, projectFileId],
    queryFn: async () =>
      apiFetch<{ saved: boolean; share_url: string | null }>(
        `${API_URL}/integrations/dropbox/export-status?project_file_id=${projectFileId}`
      ),
    enabled: !!user?.id && !!projectFileId && enabled,
  });
}

export function useDropboxExport() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: { project_file_id: string; dropbox_folder_id?: string }) => {
      return apiFetch<{ dropbox_file: Record<string, unknown>; share_url: string | null; already_saved: boolean }>(
        `${API_URL}/integrations/dropbox/export`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        }
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dropbox-export-status"] });
    },
    onError: (error: Error) => {
      // apiFetch's ApiError carries the backend's `detail` string as `message`
      // (see src/lib/apiFetch.ts apiErrorFromBody) — this is how the 413
      // "file too large" message reaches the user instead of a generic one.
      toast.error(error.message || "Could not save to Dropbox");
    },
  });
}
