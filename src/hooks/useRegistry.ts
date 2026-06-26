import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { API_URL, apiFetch, getAuthHeaders } from "@/lib/apiFetch";

// --- Types ---

export interface CreditedArtist {
  name: string;
  role: string; // "Main artist" | "Featured artist" — display-only, from Spotify
}

export interface Work {
  id: string; user_id: string; artist_id: string; project_id: string;
  title: string; work_type: string; isrc: string | null; iswc: string | null;
  upc: string | null; release_date: string | null; is_released: boolean;
  status: string; notes: string | null;
  genre: string | null; label: string | null;
  featured_artists: CreditedArtist[] | null;
  created_at: string; updated_at: string;
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
  status: string; invite_token: string;
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
      let url = `${API_URL}/registry/works`;
      if (artistId) url += `?artist_id=${artistId}`;
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
        `${API_URL}/registry/works/my-collaborations`
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
        `${API_URL}/registry/works/by-project/${projectId}`
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
        `${API_URL}/registry/works/${workId}/full`
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
      custom_work_type?: string; isrc?: string; iswc?: string; upc?: string;
      release_date?: string; is_released?: boolean; notes?: string;
      genre?: string; label?: string; featured_artists?: CreditedArtist[];
    }) =>
      apiFetch(`${API_URL}/registry/works`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onMutate: async (body) => {
      // Optimistic update — show the work immediately in the project list
      const projectKey = ["registry-works-by-project", user!.id, body.project_id];
      await qc.cancelQueries({ queryKey: projectKey });
      const previous = qc.getQueryData<Work[]>(projectKey);
      const optimistic: Work = {
        id: `temp-${Date.now()}`,
        user_id: user!.id,
        artist_id: body.artist_id,
        project_id: body.project_id,
        title: body.title,
        work_type: body.work_type || "single",
        isrc: body.isrc || null,
        iswc: body.iswc || null,
        upc: body.upc || null,
        release_date: body.release_date || null,
        is_released: body.is_released ?? true,
        status: "draft",
        notes: body.notes || null,
        genre: body.genre || null,
        label: body.label || null,
        featured_artists: body.featured_artists || null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      qc.setQueryData<Work[]>(projectKey, (old) => [...(old || []), optimistic]);
      return { previous, projectKey };
    },
    onError: (e: Error, _body, context) => {
      // Roll back optimistic update on error
      if (context?.previous !== undefined) {
        qc.setQueryData(context.projectKey, context.previous);
      }
      toast.error(e.message);
    },
    onSuccess: (_data, body) => {
      toast.success("Work created");
      // Refetch to replace optimistic entry with real server data
      qc.invalidateQueries({ queryKey: ["registry-works"] });
      qc.invalidateQueries({ queryKey: ["registry-works-by-project", user!.id, body.project_id] });
    },
  });
}

export function useUpdateWork() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ workId, ...body }: { workId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/works/${workId}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onMutate: async ({ workId, ...body }) => {
      // Optimistic update — reflect changes instantly in the work detail view
      const fullKey = ["registry-work-full", user!.id, workId];
      await qc.cancelQueries({ queryKey: fullKey });
      const previous = qc.getQueryData<WorkFull>(fullKey);
      if (previous) {
        qc.setQueryData<WorkFull>(fullKey, { ...previous, ...body, updated_at: new Date().toISOString() });
      }
      return { previous, fullKey };
    },
    onError: (e: Error, _vars, context) => {
      if (context?.previous) qc.setQueryData(context.fullKey, context.previous);
      toast.error(e.message);
    },
    onSuccess: () => {
      toast.success("Work updated");
      qc.invalidateQueries({ queryKey: ["registry-works"] });
      qc.invalidateQueries({ queryKey: ["registry-works-by-project"] });
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
    },
  });
}

export function useDeleteWork() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (workId: string) =>
      apiFetch(`${API_URL}/registry/works/${workId}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-works"] });
      qc.invalidateQueries({ queryKey: ["registry-works-by-project"] });
      toast.success("Work deleted");
    },
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
      publisher_or_label?: string; notes?: string; is_owner_stake?: boolean;
    }) =>
      apiFetch(`${API_URL}/registry/stakes`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    // No per-stake success toast: stake mutations are batched (SplitsSidebar/wizard), so the
    // batching caller shows ONE consolidated toast ("Royalty splits saved").
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpdateStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ stakeId, ...body }: { stakeId: string; [key: string]: unknown }) =>
      apiFetch(`${API_URL}/registry/stakes/${stakeId}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteStake() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (stakeId: string) =>
      apiFetch(`${API_URL}/registry/stakes/${stakeId}`, { method: "DELETE" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-full"] }); },
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
      apiFetch(`${API_URL}/registry/licenses`, {
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
      apiFetch(`${API_URL}/registry/licenses/${licenseId}`, {
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
      apiFetch(`${API_URL}/registry/licenses/${licenseId}`, { method: "DELETE" }),
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
      apiFetch(`${API_URL}/registry/agreements`, {
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
      apiFetch<Collaborator>(`${API_URL}/registry/collaborators/invite`, {
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
        `${API_URL}/registry/collaborators/claim?invite_token=${inviteToken}`,
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
        `${API_URL}/registry/collaborators/${collaboratorId}/confirm`,
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

export function useSubmitForApproval() {
  const { user } = useAuth();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (workId: string) =>
      apiFetch(`${API_URL}/registry/works/${workId}/submit-for-approval`, {
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
        `${API_URL}/registry/collaborators/${collaboratorId}/revoke`,
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
        `${API_URL}/registry/collaborators/${collaboratorId}/resend`,
        { method: "POST" }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["registry-work-full"] });
      toast.success("Invitation resent");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

// --- Dashboard Invites ---

export interface DashboardInvite {
  id: string;
  work_id: string;
  stake_id: string | null;
  invited_by: string;
  collaborator_user_id: string | null;
  email: string;
  name: string;
  role: string;
  status: string;
  invite_token: string;
  expires_at: string;
  invited_at: string;
  responded_at: string | null;
  works_registry: {
    id: string;
    title: string;
    project_id: string;
    status: string;
  } | null;
}

export function useMyInvites() {
  const { user } = useAuth();
  return useQuery<DashboardInvite[]>({
    queryKey: ["registry-my-invites", user?.id],
    queryFn: async () => {
      if (!user?.id) return [];
      const data = await apiFetch<{ invites: DashboardInvite[] }>(`${API_URL}/registry/collaborators/my-invites`);
      if (!data) return [];
      return data.invites;
    },
    enabled: !!user?.id,
  });
}

export function useAcceptFromDashboard() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async (collaboratorId: string) =>
      apiFetch(
        `${API_URL}/registry/collaborators/${collaboratorId}/accept-from-dashboard`,
        { method: "POST" }
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry-my-invites"] });
      queryClient.invalidateQueries({ queryKey: ["registry-works"] });
      queryClient.invalidateQueries({ queryKey: ["registry-my-collaborations"] });
      toast.success("Invitation accepted");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

export function useDeclineInvitation() {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  return useMutation({
    mutationFn: async (collaboratorId: string) =>
      apiFetch(
        `${API_URL}/registry/collaborators/${collaboratorId}/decline`,
        { method: "POST" }
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["registry-my-invites"] });
      toast.success("Invitation declined");
    },
    onError: (error: Error) => toast.error(error.message),
  });
}

// --- Export ---

export function useExportProof() {
  const { user } = useAuth();
  return useMutation({
    mutationFn: async (workId: string) => {
      const authHeaders = await getAuthHeaders();
      const res = await fetch(`${API_URL}/registry/works/${workId}/export`, {
        headers: authHeaders,
      });
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

// --- File downloads (access-checked signed URLs) ---

// Fetches a short-lived signed URL on demand so registry file access is gated
// by the same WorkAccess checks as the rest of the work's data, instead of
// relying on direct (public) bucket URLs.
export function useFileDownloadUrl() {
  return useMutation({
    mutationFn: async ({ workId, fileId }: { workId: string; fileId: string }) =>
      apiFetch<{ url: string }>(
        `${API_URL}/registry/works/${workId}/files/${fileId}/download-url`
      ),
  });
}

// --- Access & visibility grants (permission UI) ---

export interface WorkAccess {
  work_role: "owner" | "admin" | "viewer" | "none";
  project_role: "owner" | "admin" | "editor" | "viewer" | "none";
  can_view: boolean; can_edit: boolean; can_manage: boolean; can_delete: boolean;
  can_see_full_ownership: boolean; is_project_member: boolean;
  my_collaborator_id: string | null; all_visible: boolean;
  visible_stake_ids: string[]; visible_file_ids: string[]; visible_audio_ids: string[];
  visible_license_ids: string[]; visible_agreement_ids: string[];
}

export interface GrantItem { resource_type: "project_file" | "audio_file" | "license" | "agreement"; resource_id: string; }

// Shape returned by the grant-matrix endpoint. The matrix collaborator is a
// slim projection (id, name, email, role, access_level, status) — not the full
// Collaborator record — so it gets its own interface.
export interface GrantRow { resource_type: string; resource_id: string | null; }
export interface GrantMatrixCollaborator {
  id: string; name: string; email: string; role: string;
  access_level: string; status: string;
}
export interface GrantMatrix {
  collaborators: GrantMatrixCollaborator[];
  grants_by_collaborator: Record<string, GrantRow[]>;
}

export function useWorkAccess(workId: string | undefined) {
  const { user } = useAuth();
  return useQuery<WorkAccess | null>({
    queryKey: ["registry-work-access", user?.id, workId],
    queryFn: async () => {
      if (!user?.id || !workId) return null;
      return apiFetch<WorkAccess>(`${API_URL}/registry/works/${workId}/access`);
    },
    enabled: !!user?.id && !!workId,
  });
}

export function useWorkGrants(workId: string | undefined) {
  const { user } = useAuth();
  return useQuery<GrantMatrix | null>({
    queryKey: ["registry-work-grants", user?.id, workId],
    queryFn: async () => {
      if (!user?.id || !workId) return null;
      return apiFetch<GrantMatrix>(
        `${API_URL}/registry/works/${workId}/grants`);
    },
    enabled: !!user?.id && !!workId,
  });
}

export function useAddGrants() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ collaboratorId, grants, ownershipBreakdown }:
      { collaboratorId: string; grants?: GrantItem[]; ownershipBreakdown?: boolean }) =>
      apiFetch(`${API_URL}/registry/collaborators/${collaboratorId}/grants`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ grants: grants || [], ownership_breakdown: ownershipBreakdown }),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-grants"] }); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRemoveGrants() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ collaboratorId, grants }: { collaboratorId: string; grants: GrantItem[] }) =>
      apiFetch(`${API_URL}/registry/collaborators/${collaboratorId}/grants`, {
        method: "DELETE", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ grants }),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-grants"] }); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useSetAccessLevel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ collaboratorId, accessLevel }: { collaboratorId: string; accessLevel: "viewer" | "admin" }) =>
      apiFetch(`${API_URL}/registry/collaborators/${collaboratorId}/access-level`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ access_level: accessLevel }),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-grants"] }); qc.invalidateQueries({ queryKey: ["registry-work-full"] }); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useSetWorkRole() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ collaboratorId, role }: { collaboratorId: string; role: string }) =>
      apiFetch(`${API_URL}/registry/collaborators/${collaboratorId}/role`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role }),
      }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["registry-work-grants"] }); qc.invalidateQueries({ queryKey: ["registry-work-full"] }); },
    onError: (e: Error) => toast.error(e.message),
  });
}

export interface DeriveResult {
  found: boolean; confidence: "high" | "low";
  master_pct: number | null; publishing_pct: number | null;
  terms: Array<{ label: string; value: string }>; matched_file_ids: string[];
}
export function useDeriveFromContracts() {
  return useMutation({
    mutationFn: async (body: { work_id: string; name: string; email?: string; contract_file_ids?: string[] }) =>
      apiFetch<DeriveResult>(`${API_URL}/registry/collaborators/derive-from-contracts`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onError: (e: Error) => toast.error(e.message),
  });
}

export interface InvitePreview {
  expired?: boolean;
  email_mismatch?: boolean;
  invite_email?: string;
  work_title?: string;
  collaborator?: { name: string; role: string; terms: Array<{ label: string; value: string }> };
  stakes?: Array<{ stake_type: string; percentage: number; holder_role?: string }>;
  work?: { title?: string; project_id?: string; artist_id?: string } | null;
}

export function useInvitePreview(token: string | undefined) {
  const { user } = useAuth();
  return useQuery<InvitePreview | null>({
    queryKey: ["registry-invite-preview", user?.id, token],
    queryFn: async () => {
      if (!token) return null;
      return apiFetch<InvitePreview>(`${API_URL}/registry/collaborators/invite/${token}/preview`);
    },
    enabled: !!token,
  });
}
