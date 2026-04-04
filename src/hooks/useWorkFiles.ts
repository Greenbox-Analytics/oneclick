import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

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
      const res = await fetch(`${API_URL}/registry/works/${workId}/files?user_id=${user.id}`);
      if (!res.ok) throw new Error("Failed to fetch work files");
      const data = await res.json();
      return data.files;
    },
    enabled: !!user?.id && !!workId,
  });
}

export function useLinkFileToWork() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ workId, fileId }: { workId: string; fileId: string }) => {
      const res = await fetch(
        `${API_URL}/registry/works/${workId}/files?file_id=${fileId}&user_id=${user!.id}`,
        { method: "POST" }
      );
      if (!res.ok) throw new Error("Failed to link file");
      return res.json();
    },
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
    mutationFn: async ({ workId, linkId }: { workId: string; linkId: string }) => {
      const res = await fetch(
        `${API_URL}/registry/works/${workId}/files/${linkId}?user_id=${user!.id}`,
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error("Failed to unlink file");
      return res.json();
    },
    onSuccess: (_, { workId }) => {
      queryClient.invalidateQueries({ queryKey: ["work-files", workId] });
      toast.success("File unlinked");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}
