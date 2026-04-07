import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface WorkFileLink {
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

export function useWorkFiles(workId?: string) {
  const { user } = useAuth();
  return useQuery<WorkFileLink[]>({
    queryKey: ["work-files", workId],
    queryFn: async () => {
      if (!user?.id || !workId) return [];
      const data = await apiFetch<{ files: WorkFileLink[] }>(`${API_URL}/registry/works/${workId}/files`);
      return data.files;
    },
    enabled: !!user?.id && !!workId,
  });
}

export function useLinkFileToWork() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ workId, fileId }: { workId: string; fileId: string }) =>
      apiFetch(
        `${API_URL}/registry/works/${workId}/files?file_id=${fileId}`,
        { method: "POST" }
      ),
    onSuccess: (_, { workId }) => {
      queryClient.invalidateQueries({ queryKey: ["work-files", workId] });
      toast.success("File linked to work");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useUnlinkFileFromWork() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ workId, linkId }: { workId: string; linkId: string }) =>
      apiFetch(
        `${API_URL}/registry/works/${workId}/files/${linkId}`,
        { method: "DELETE" }
      ),
    onSuccess: (_, { workId }) => {
      queryClient.invalidateQueries({ queryKey: ["work-files", workId] });
      toast.success("File unlinked");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}
