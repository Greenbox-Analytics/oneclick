import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

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
      const res = await fetch(`${API_URL}/registry/works/${workId}/audio?user_id=${user.id}`);
      if (!res.ok) throw new Error("Failed to fetch work audio");
      const data = await res.json();
      return data.audio;
    },
    enabled: !!user?.id && !!workId,
  });
}

export function useLinkAudioToWork() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ workId, audioFileId }: { workId: string; audioFileId: string }) => {
      const res = await fetch(
        `${API_URL}/registry/works/${workId}/audio?audio_file_id=${audioFileId}&user_id=${user!.id}`,
        { method: "POST" }
      );
      if (!res.ok) throw new Error("Failed to link audio");
      return res.json();
    },
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
    mutationFn: async ({ workId, linkId }: { workId: string; linkId: string }) => {
      const res = await fetch(
        `${API_URL}/registry/works/${workId}/audio/${linkId}?user_id=${user!.id}`,
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error("Failed to unlink audio");
      return res.json();
    },
    onSuccess: (_, { workId }) => {
      queryClient.invalidateQueries({ queryKey: ["work-audio", workId] });
      toast.success("Audio unlinked");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}
