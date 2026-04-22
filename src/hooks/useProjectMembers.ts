import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch, getAuthHeaders } from "@/lib/apiFetch";

export interface ProjectMember {
  id: string;
  project_id: string;
  user_id: string;
  role: "owner" | "admin" | "editor" | "viewer";
  invited_by: string | null;
  created_at: string;
  updated_at: string;
}

export interface PendingInvite {
  id: string;
  project_id: string;
  email: string;
  role: string;
  invited_by: string;
  created_at: string;
  expires_at: string;
  last_email_error: string | null;
  last_email_attempt_at: string | null;
}

export function useProjectMembers(projectId?: string) {
  const { user } = useAuth();
  return useQuery<ProjectMember[]>({
    queryKey: ["project-members", projectId],
    queryFn: async () => {
      if (!user?.id || !projectId) return [];
      const data = await apiFetch<{ members: ProjectMember[] }>(`${API_URL}/projects/${projectId}/members`);
      return data.members;
    },
    enabled: !!user?.id && !!projectId,
  });
}

export function useMyRole(projectId?: string) {
  const { data: members } = useProjectMembers(projectId);
  const { user } = useAuth();
  if (!members || !user) return null;
  const me = members.find((m) => m.user_id === user.id);
  return me?.role ?? null;
}

export function useAddProjectMember() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ projectId, email, role }: { projectId: string; email: string; role: string }) =>
      apiFetch<{ type: "added" | "pending" }>(`${API_URL}/projects/${projectId}/members`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, role }),
      }),
    onSuccess: (data, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-members", projectId] });
      queryClient.invalidateQueries({ queryKey: ["project-pending-invites", projectId] });
      toast.success(data?.type === "pending" ? "Invite email queued" : "Member added");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useResendPendingInvite() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ projectId, inviteId }: { projectId: string; inviteId: string }) =>
      apiFetch(`${API_URL}/projects/${projectId}/pending-invites/${inviteId}/resend`, {
        method: "POST",
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-pending-invites", projectId] });
      toast.success("Invite email resent");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useUpdateMemberRole() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ projectId, memberId, role }: { projectId: string; memberId: string; role: string }) =>
      apiFetch(`${API_URL}/projects/${projectId}/members/${memberId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-members", projectId] });
      toast.success("Role updated");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useRemoveProjectMember() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ projectId, memberId }: { projectId: string; memberId: string }) =>
      apiFetch(`${API_URL}/projects/${projectId}/members/${memberId}`, {
        method: "DELETE",
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-members", projectId] });
      toast.success("Member removed");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function usePendingInvites(projectId?: string) {
  const { user } = useAuth();
  return useQuery<PendingInvite[]>({
    queryKey: ["project-pending-invites", projectId],
    queryFn: async () => {
      if (!user?.id || !projectId) return [];
      const data = await apiFetch<{ invites: PendingInvite[] }>(`${API_URL}/projects/${projectId}/pending-invites`);
      return data.invites;
    },
    enabled: !!user?.id && !!projectId,
  });
}

export function useCancelPendingInvite() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ projectId, inviteId }: { projectId: string; inviteId: string }) =>
      apiFetch(`${API_URL}/projects/${projectId}/pending-invites/${inviteId}`, {
        method: "DELETE",
      }),
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-pending-invites", projectId] });
      toast.success("Invite cancelled");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}
