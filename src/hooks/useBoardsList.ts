import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import type { Board } from "@/types/teams";

export function useBoardsList(teamId?: string | null) {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["boards-list", user?.id, teamId ?? "personal"],
    queryFn: async () => {
      const qs = teamId ? `?team_id=${teamId}` : "";
      return (await apiFetch<{ boards: Board[] }>(`${API_URL}/boards/boards${qs}`)).boards;
    },
    enabled: !!user?.id,
  });
}

export function useCreateBoard() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { name: string; team_id?: string | null; description?: string }) =>
      apiFetch<Board>(`${API_URL}/boards/boards`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["boards-list"] });
      toast.success("Board created");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRenameBoard() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ boardId, name }: { boardId: string; name: string }) =>
      apiFetch<Board>(`${API_URL}/boards/boards/${boardId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["boards-list"] }),
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useArchiveBoard() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (boardId: string) =>
      apiFetch(`${API_URL}/boards/boards/${boardId}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["boards-list"] });
      toast.success("Board archived");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteBoard() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ boardId, confirmName }: { boardId: string; confirmName: string }) =>
      apiFetch<{ deleted: string; tasks: number }>(`${API_URL}/boards/boards/${boardId}/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm_name: confirmName }),
      }),
    onSuccess: (d) => {
      qc.invalidateQueries({ queryKey: ["boards-list"] });
      qc.invalidateQueries({ queryKey: ["archived-boards"] });
      toast.success(`Board deleted (${d?.tasks ?? 0} tasks removed)`);
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRestoreBoard() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (boardId: string) =>
      apiFetch(`${API_URL}/boards/boards/${boardId}/restore`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["boards-list"] });
      qc.invalidateQueries({ queryKey: ["archived-boards"] });
      toast.success("Board restored");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useArchivedBoards(teamId?: string | null) {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["archived-boards", user?.id, teamId ?? "personal"],
    queryFn: async () => {
      const qs = teamId ? `?team_id=${teamId}` : "";
      return (await apiFetch<{ boards: Board[] }>(`${API_URL}/boards/archived${qs}`)).boards;
    },
    enabled: !!user?.id,
  });
}
