import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import type { Team, TeamInvite, TeamMember } from "@/types/teams";

export function useTeams() {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["teams", user?.id],
    queryFn: async () => (await apiFetch<{ teams: Team[] }>(`${API_URL}/teams`)).teams,
    enabled: !!user?.id,
  });
}

export function useCreateTeam() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { name: string; description?: string }) =>
      apiFetch<Team>(`${API_URL}/teams`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["teams"] });
      toast.success("Team created");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useArchiveTeam() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (teamId: string) =>
      apiFetch<{ archived: string; boards: number; tasks: number; members: number }>(
        `${API_URL}/teams/${teamId}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["teams"] });
      // Archiving a team removes access to its boards — drop the cached board lists.
      qc.invalidateQueries({ queryKey: ["boards-list"] });
      toast.success("Team archived");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteTeam() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ teamId, confirmName }: { teamId: string; confirmName: string }) =>
      apiFetch<{ deleted: string; boards: number; tasks: number; members: number }>(
        `${API_URL}/teams/${teamId}/delete`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confirm_name: confirmName }),
        },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["teams"] });
      qc.invalidateQueries({ queryKey: ["archived-teams"] });
      toast.success("Team permanently deleted");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRestoreTeam() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (teamId: string) =>
      apiFetch(`${API_URL}/teams/${teamId}/restore`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["teams"] });
      qc.invalidateQueries({ queryKey: ["archived-teams"] });
      qc.invalidateQueries({ queryKey: ["boards-list"] });
      toast.success("Team restored");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useArchivedTeams() {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["archived-teams", user?.id],
    queryFn: async () => (await apiFetch<{ teams: Team[] }>(`${API_URL}/teams/archived`)).teams,
    enabled: !!user?.id,
  });
}

export function useTeamMembers(teamId?: string) {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["team-members", teamId],
    queryFn: async () =>
      (await apiFetch<{ members: TeamMember[] }>(`${API_URL}/teams/${teamId}/members`)).members,
    enabled: !!user?.id && !!teamId,
  });
}

export function useTeamInvites(teamId?: string) {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["team-invites", teamId],
    queryFn: async () =>
      (await apiFetch<{ invites: TeamInvite[] }>(`${API_URL}/teams/${teamId}/invites`)).invites,
    enabled: !!user?.id && !!teamId,
  });
}

export function useInviteTeamMember() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ teamId, email, role }: { teamId: string; email: string; role: string }) =>
      apiFetch<{ type: string; notify_user_id: string | null }>(`${API_URL}/teams/${teamId}/invites`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, role }),
      }),
    onSuccess: (data, { teamId }) => {
      qc.invalidateQueries({ queryKey: ["team-invites", teamId] });
      qc.invalidateQueries({ queryKey: ["team-members", teamId] });
      // Existing user → both channels (dual-channel invites); new user → email only.
      toast.success(data?.notify_user_id ? "Invitation sent (email + in-app)" : "Invitation email sent");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useCancelTeamInvite() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ teamId, inviteId }: { teamId: string; inviteId: string }) =>
      apiFetch(`${API_URL}/teams/${teamId}/invites/${inviteId}`, { method: "DELETE" }),
    onSuccess: (_d, { teamId }) => {
      qc.invalidateQueries({ queryKey: ["team-invites", teamId] });
      toast.success("Invitation canceled");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateTeamMemberRole() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ teamId, memberId, role }: { teamId: string; memberId: string; role: string }) =>
      apiFetch(`${API_URL}/teams/${teamId}/members/${memberId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      }),
    onSuccess: (_d, { teamId }) => {
      qc.invalidateQueries({ queryKey: ["team-members", teamId] });
      toast.success("Role updated");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRemoveTeamMember() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      teamId,
      memberId,
    }: {
      teamId: string;
      memberId: string;
      /** Overrides the default success toast (e.g. "You left the team" for self-leave). */
      successMessage?: string;
    }) => apiFetch(`${API_URL}/teams/${teamId}/members/${memberId}`, { method: "DELETE" }),
    onSuccess: (_d, { teamId, successMessage }) => {
      qc.invalidateQueries({ queryKey: ["team-members", teamId] });
      qc.invalidateQueries({ queryKey: ["teams"] });
      // Leaving a team (self-remove) drops access to its boards — refresh board lists.
      qc.invalidateQueries({ queryKey: ["boards-list"] });
      toast.success(successMessage ?? "Member removed");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useAcceptTeamInvite() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (token: string) =>
      apiFetch<{ type: string; team_id: string }>(`${API_URL}/teams/invites/${token}/accept`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["teams"] });
      qc.invalidateQueries({ queryKey: ["registry-notifications"] });
      toast.success("Joined team");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeclineTeamInvite() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (token: string) =>
      apiFetch<{ type: string }>(`${API_URL}/teams/invites/${token}/decline`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-notifications"] });
      toast.success("Invitation declined");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}
