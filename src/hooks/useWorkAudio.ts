import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface WorkAudioLink {
  id: string;
  work_id: string;
  audio_file_id: string;
  created_at: string;
  audio_files?: {
    id: string;
    file_name: string;
    file_url: string;
    file_type: string | null;
    file_size: number | null;
    created_at: string;
  };
}

export function useWorkAudio(workId?: string) {
  const { user } = useAuth();
  return useQuery<WorkAudioLink[]>({
    queryKey: ["work-audio", workId],
    queryFn: async () => {
      if (!user?.id || !workId) return [];
      const data = await apiFetch<{ audio: WorkAudioLink[] }>(`${API_URL}/registry/works/${workId}/audio`);
      return data.audio;
    },
    enabled: !!user?.id && !!workId,
  });
}

export function useLinkAudioToWork() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ workId, audioFileId }: { workId: string; audioFileId: string }) =>
      apiFetch(
        `${API_URL}/registry/works/${workId}/audio?audio_file_id=${audioFileId}`,
        { method: "POST" }
      ),
    onSuccess: (_, { workId }) => {
      queryClient.invalidateQueries({ queryKey: ["work-audio", workId] });
      toast.success("Audio linked to work");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useUnlinkAudioFromWork() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ workId, linkId }: { workId: string; linkId: string }) =>
      apiFetch(
        `${API_URL}/registry/works/${workId}/audio/${linkId}`,
        { method: "DELETE" }
      ),
    onSuccess: (_, { workId }) => {
      queryClient.invalidateQueries({ queryKey: ["work-audio", workId] });
      toast.success("Audio unlinked");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}
