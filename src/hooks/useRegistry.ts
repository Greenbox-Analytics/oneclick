import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch } from "@/lib/apiFetch";

// --- Types ---

export interface Work {
  id: string; user_id: string; artist_id: string; project_id: string;
  title: string; work_type: string; isrc: string | null; iswc: string | null;
  upc: string | null; release_date: string | null; status: string;
  notes: string | null; created_at: string; updated_at: string;
}

export interface OwnershipStake {
  id: string; work_id: string; user_id: string; stake_type: string;
  holder_name: string; holder_role: string; percentage: number;
  holder_email: string | null; holder_ipi: string | null;
  publisher_or_label: string | null; notes: string | null;
  created_at: string; updated_at: string;
}

export interface LicensingRight {
  id: string; work_id: string; user_id: string; license_type: string;
  licensee_name: string; licensee_email: string | null; territory: string;
  start_date: string; end_date: string | null; terms: string | null;
  status: string; created_at: string; updated_at: string;
}

export interface Agreement {
  id: string; work_id: string; user_id: string; agreement_type: string;
  title: string; description: string | null; effective_date: string;
  parties: Array<{ name: string; role: string; email?: string }>;
  file_id: string | null; document_hash: string | null; created_at: string;
}

export interface Collaborator {
  id: string; work_id: string; stake_id: string | null; invited_by: string;
  collaborator_user_id: string | null; email: string; name: string; role: string;
  status: string; invite_token: string; dispute_reason: string | null;
  expires_at: string; invited_at: string; responded_at: string | null;
}

export interface WorkFull extends Work {
  stakes: OwnershipStake[]; licenses: LicensingRight[];
  agreements: Agreement[]; collaborators: Collaborator[];
}

// --- Works ---

export function useWorks(artistId?: string) {
  const { user } = useAuth();
  return useQuery<Work[]>({
    queryKey: ["registry-works", user?.id, artistId],
    queryFn: async () => {
      if (!user?.id) return [];
      let url = `${API_URL}/registry/works?user_id=${user.id}`;
      if (artistId) url += `&artist_id=${artistId}`;
      const data = await apiFetch<{ works: Work[] }>(url);
      return data.works;
    },
    enabled: !!user?.id,
  });
}

export function useMyCollaborations() {
  const { user } = useAuth();
  return useQuery<Work[]>({
    queryKey: ["registry-my-collaborations", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const data = await apiFetch<{ works: Work[] }>(
        `${API_URL}/registry/works/my-collaborations?user_id=${user.id}`
      );
      return data.works;
    },
    enabled: !!user?.id,
  });
}

export function useWorksByProject(projectId: string | undefined) {
  const { user } = useAuth();
  return useQuery<Work[]>({
    queryKey: ["registry-works-by-project", user?.id, projectId],
    queryFn: async () => {
      if (!user?.id || !projectId) return [];
      const data = await apiFetch<{ works: Work[] }>(
        `${API_URL}/registry/works/by-project/${projectId}?user_id=${user.id}`
      );
      return data.works;
    },
    enabled: !!user?.id && !!projectId,
  });
}

export function useWorkFull(workId: string | undefined) {
  const { user } = useAuth();
  return useQuery<WorkFull | null>({
    queryKey: ["registry-work-full", user?.id, workId],
    queryFn: async () => {
      if (!user?.id || !workId) return null;
      return apiFetch<WorkFull>(
        `${API_URL}/registry/works/${workId}/full?user_id=${user.id}`
      );
    },
    enabled: !!user?.id && !!workId,
  });
}

export function useCreateWork() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      artist_id: string; project_id: string; title: string; work_type?: string;
      isrc?: string; iswc?: string; upc?: string; release_date?: string; notes?: string;
    }) =>
      apiFetch(`${API_URL}/registry/works?user_id=${user!.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-works"] }); toast.success("Work registered"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateWork() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ workId, ...body }: { workId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/works/${workId}?user_id=${user!.id}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-works"] });
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Work updated");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteWork() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (workId: string) =>
      apiFetch(`${API_URL}/registry/works/${workId}?user_id=${user!.id}`, { method: "DELETE" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-works"] }); toast.success("Work deleted"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Stakes ---

export function useCreateStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string; stake_type: string; holder_name: string; holder_role: string;
      percentage: number; holder_email?: string; holder_ipi?: string;
      publisher_or_label?: string; notes?: string;
    }) =>
      apiFetch(`${API_URL}/registry/stakes?user_id=${user!.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("Ownership stake added"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ stakeId, ...body }: { stakeId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/stakes/${stakeId}?user_id=${user!.id}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("Ownership stake updated"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (stakeId: string) =>
      apiFetch(`${API_URL}/registry/stakes/${stakeId}?user_id=${user!.id}`, { method: "DELETE" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("Ownership stake removed"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Licenses ---

export function useCreateLicense() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string; license_type: string; licensee_name: string;
      licensee_email?: string; territory?: string; start_date: string;
      end_date?: string; terms?: string;
    }) =>
      apiFetch(`${API_URL}/registry/licenses?user_id=${user!.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("License added"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateLicense() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ licenseId, ...body }: { licenseId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/licenses/${licenseId}?user_id=${user!.id}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("License updated"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteLicense() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (licenseId: string) =>
      apiFetch(`${API_URL}/registry/licenses/${licenseId}?user_id=${user!.id}`, { method: "DELETE" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("License removed"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Agreements ---

export function useCreateAgreement() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string; agreement_type: string; title: string; description?: string;
      effective_date: string; parties: Array<{ name: string; role: string; email?: string }>;
      file_id?: string; document_hash?: string;
    }) =>
      apiFetch(`${API_URL}/registry/agreements?user_id=${user!.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("Agreement recorded"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Collaboration ---

export function useInviteCollaborator() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      work_id: string; email: string; name: string; role: string; stake_id?: string;
    }) =>
      apiFetch<Collaborator>(`${API_URL}/registry/collaborators/invite?user_id=${user!.id}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); toast.success("Invitation sent"); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useClaimInvitation() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (inviteToken: string) =>
      apiFetch<Collaborator>(
        `${API_URL}/registry/collaborators/claim?invite_token=${inviteToken}&user_id=${user!.id}`,
        { method: "POST" }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-pending-review"] });
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useConfirmStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (collaboratorId: string) =>
      apiFetch<Collaborator>(
        `${API_URL}/registry/collaborators/${collaboratorId}/confirm?user_id=${user!.id}`,
        { method: "POST" }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      qc.invalidateQueries({ queryKey: ["registry-pending-review"] });
      toast.success("Stake confirmed");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDisputeStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ collaboratorId, reason }: { collaboratorId: string; reason: string }) =>
      apiFetch<Collaborator>(
        `${API_URL}/registry/collaborators/${collaboratorId}/dispute?user_id=${user!.id}`,
        {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason }),
        }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      qc.invalidateQueries({ queryKey: ["registry-pending-review"] });
      toast.success("Dispute submitted");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useSubmitForApproval() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (workId: string) =>
      apiFetch(`${API_URL}/registry/works/${workId}/submit-for-approval?user_id=${user!.id}`, {
        method: "POST",
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-works"] });
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Submitted for approval — invitations sent");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRevokeCollaborator() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (collaboratorId: string) =>
      apiFetch<Collaborator>(
        `${API_URL}/registry/collaborators/${collaboratorId}/revoke?user_id=${user!.id}`,
        { method: "POST" }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Collaborator revoked");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useResendInvitation() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (collaboratorId: string) =>
      apiFetch<Collaborator>(
        `${API_URL}/registry/collaborators/${collaboratorId}/resend?user_id=${user!.id}`,
        { method: "POST" }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Invitation resent");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Export ---

export function useExportProof() {
  const { user } = useAuth();
  return useMutation({
    mutationFn: async (workId: string) => {
      const res = await fetch(`${API_URL}/registry/works/${workId}/export?user_id=${user!.id}`);
      if (!res.ok) throw new Error("Failed to generate proof of ownership");
      const blob = await res.blob();
      const disposition = res.headers.get("Content-Disposition") || "";
      const match = disposition.match(/filename="?(.+?)"?$/);
      const filename = match ? match[1] : "Proof_of_Ownership.pdf";
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename;
      document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
    },
    onSuccess: () => toast.success("Proof of ownership downloaded"),
    onError: (e: Error) => toast.error(e.message),
  });
}
