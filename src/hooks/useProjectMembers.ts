import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";

const API_URL = import.meta.env.VITE_BACKEND_API_URL || "http://localhost:8000";

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
}

export function useProjectMembers(projectId?: string) {
  const { user } = useAuth();
  return useQuery<ProjectMember[]>({
    queryKey: ["project-members", projectId],
    queryFn: async () => {
      if (!user?.id || !projectId) return [];
      const res = await fetch(`${API_URL}/projects/${projectId}/members?user_id=${user.id}`);
      if (!res.ok) throw new Error("Failed to fetch members");
      const data = await res.json();
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
    mutationFn: async ({ projectId, email, role }: { projectId: string; email: string; role: string }) => {
      const res = await fetch(`${API_URL}/projects/${projectId}/members?user_id=${user!.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, role }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to add member");
      }
      return res.json();
    },
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-members", projectId] });
      toast.success("Member added");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useUpdateMemberRole() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ projectId, memberId, role }: { projectId: string; memberId: string; role: string }) => {
      const res = await fetch(`${API_URL}/projects/${projectId}/members/${memberId}?user_id=${user!.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to update role");
      }
      return res.json();
    },
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
    mutationFn: async ({ projectId, memberId }: { projectId: string; memberId: string }) => {
      const res = await fetch(`${API_URL}/projects/${projectId}/members/${memberId}?user_id=${user!.id}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to remove member");
      }
      return res.json();
    },
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
      const res = await fetch(`${API_URL}/projects/${projectId}/pending-invites?user_id=${user.id}`);
      if (!res.ok) throw new Error("Failed to fetch pending invites");
      const data = await res.json();
      return data.invites;
    },
    enabled: !!user?.id && !!projectId,
  });
}

export function useCancelPendingInvite() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async ({ projectId, inviteId }: { projectId: string; inviteId: string }) => {
      const res = await fetch(`${API_URL}/projects/${projectId}/pending-invites/${inviteId}?user_id=${user!.id}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to cancel invite");
      return res.json();
    },
    onSuccess: (_, { projectId }) => {
      queryClient.invalidateQueries({ queryKey: ["project-pending-invites", projectId] });
      toast.success("Invite cancelled");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}
