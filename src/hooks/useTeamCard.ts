import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface TeamCard {
  id: string;
  user_id: string;
  display_name: string;
  first_name: string;
  last_name: string;
  email: string;
  avatar_url: string | null;
  bio: string | null;
  phone: string | null;
  website: string | null;
  company: string | null;
  role: string | null;
  social_links: Record<string, string>;
  dsp_links: Record<string, string>;
  custom_links: Array<{ label: string; url: string }>;
  visible_fields: string[];
  created_at: string;
  updated_at: string;
}

export function useMyTeamCard() {
  const { user } = useAuth();
  return useQuery<TeamCard | null>({
    queryKey: ["team-card", user?.id],
    queryFn: async () => {
      if (!user?.id) return null;
      return apiFetch<TeamCard>(`${API_URL}/registry/teamcard`);
    },
    enabled: !!user?.id,
  });
}

export function useUpdateTeamCard() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: Partial<Omit<TeamCard, "id" | "user_id" | "email" | "created_at" | "updated_at">>) =>
      apiFetch<TeamCard>(`${API_URL}/registry/teamcard`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["team-card"] });
      toast.success("TeamCard updated");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useCollaboratorTeamCard(collaboratorUserId: string | undefined) {
  const { user } = useAuth();
  return useQuery<TeamCard | null>({
    queryKey: ["collaborator-team-card", collaboratorUserId],
    queryFn: async () => {
      if (!user?.id || !collaboratorUserId) return null;
      return apiFetch<TeamCard>(
        `${API_URL}/registry/teamcard/${collaboratorUserId}`
      );
    },
    enabled: !!user?.id && !!collaboratorUserId,
  });
}
