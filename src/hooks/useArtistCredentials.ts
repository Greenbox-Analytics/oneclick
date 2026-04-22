import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export interface ArtistCredential {
  id: string;
  artist_id: string;
  platform_name: string;
  login_identifier: string;
  url: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface CredentialInput {
  platform_name: string;
  login_identifier: string;
  password: string;
  url?: string | null;
  notes?: string | null;
}

export interface CredentialPatch {
  platform_name?: string;
  login_identifier?: string;
  password?: string;
  url?: string | null;
  notes?: string | null;
}

export function useArtistCredentials(artistId?: string) {
  const { user } = useAuth();
  return useQuery<ArtistCredential[]>({
    queryKey: ["artist-credentials", artistId],
    queryFn: async () => {
      if (!user?.id || !artistId) return [];
      return apiFetch<ArtistCredential[]>(`${API_URL}/credentials?artist_id=${artistId}`);
    },
    enabled: !!user?.id && !!artistId,
  });
}

export function useCreateCredential(artistId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (input: CredentialInput) =>
      apiFetch<ArtistCredential>(`${API_URL}/credentials`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ artist_id: artistId, ...input }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["artist-credentials", artistId] });
      toast.success("Credential saved");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useUpdateCredential(artistId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ credentialId, patch }: { credentialId: string; patch: CredentialPatch }) =>
      apiFetch<ArtistCredential>(`${API_URL}/credentials/${credentialId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["artist-credentials", artistId] });
      toast.success("Credential updated");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useDeleteCredential(artistId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (credentialId: string) =>
      apiFetch(`${API_URL}/credentials/${credentialId}`, { method: "DELETE" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["artist-credentials", artistId] });
      toast.success("Credential deleted");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useRevealCredential() {
  return useMutation({
    mutationFn: async ({
      credentialId,
      msaniiPassword,
    }: {
      credentialId: string;
      msaniiPassword: string;
    }) =>
      apiFetch<{ password: string }>(`${API_URL}/credentials/${credentialId}/reveal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ msanii_password: msaniiPassword }),
      }),
  });
}
