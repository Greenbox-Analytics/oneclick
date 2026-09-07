// src/hooks/usePartnerKeys.ts
// Phase-2 portal (spec 2026-09-04): an org admin's partner API keys. Per-key
// spend is NOT here — it rides on useOrgUsage (get_org_usage.byKey). Same
// idioms as useOrgs.ts; query key ["orgs", orgId, "partner-keys"].
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

/** One row of GET /orgs/{id}/partner-keys. `status` is what is STORED —
 * "expired" is derived client-side (lib/partnerKeys.keyStatus).
 * `created_by_label` is the creator's resolved email. */
export interface PartnerKey {
  id: string;
  label: string;
  key_prefix: string;
  status: "active" | "revoked";
  expires_at: string | null;
  created_at: string;
  created_by: string | null;
  created_by_label: string | null;
  last_used_at: string | null;
  /** When the key was revoked; the backend drops the row 30 days after this. */
  revoked_at: string | null;
  folder_id: string | null;
}

/** A team's grouping of keys — a use case or project. Spend is attributed by
 * the key's CURRENT folder, so moving a key moves its history with it. */
export interface PartnerKeyFolder {
  id: string;
  org_id: string;
  name: string;
  created_at: string;
}

/** POST response: the stored row plus the plaintext `secret`, shown ONCE. */
export interface MintedPartnerKey {
  id: string;
  secret: string;
  label?: string;
  key_prefix?: string;
}

export interface CreatePartnerKeyInput {
  orgId: string;
  label: string;
  /** UTC ISO string, end of the chosen local day (lib/partnerKeys.endOfLocalDayIso). */
  expires_at?: string;
  folder_id?: string | null;
}

const keysKey = (orgId?: string) => ["orgs", orgId, "partner-keys"] as const;

export interface PartnerKeysPayload {
  keys: PartnerKey[];
  folders: PartnerKeyFolder[];
}

export function usePartnerKeys(orgId?: string) {
  const { user } = useAuth();
  return useQuery<PartnerKeysPayload>({
    queryKey: keysKey(orgId),
    queryFn: () => apiFetch<PartnerKeysPayload>(`${API_URL}/orgs/${orgId}/partner-keys`),
    enabled: !!user?.id && !!orgId,
    staleTime: 15_000,
  });
}

/** No success toast: the dialog owns the success state (it has a secret to
 * show). No onError toast either — the dialog renders the ApiError message
 * inline next to the form, like OrgInvitesPanel does for the seat wall. */
export function useCreatePartnerKey() {
  const qc = useQueryClient();
  return useMutation<MintedPartnerKey, Error, CreatePartnerKeyInput>({
    mutationFn: ({ orgId, ...body }) =>
      apiFetch<MintedPartnerKey>(`${API_URL}/orgs/${orgId}/partner-keys`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: (_data, { orgId }) => {
      qc.invalidateQueries({ queryKey: keysKey(orgId) });
    },
  });
}

export function useRevokePartnerKey() {
  const qc = useQueryClient();
  return useMutation<{ status: string }, Error, { orgId: string; keyId: string }>({
    mutationFn: ({ orgId, keyId }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/partner-keys/${keyId}`, { method: "DELETE" }),
    onSuccess: (_data, { orgId }) => {
      qc.invalidateQueries({ queryKey: keysKey(orgId) });
      toast.success("Key revoked");
    },
    onError: (e) => toast.error(e.message),
  });
}

/** Idempotent on an existing name, so "New folder…" can be submitted twice. */
export function useCreatePartnerKeyFolder() {
  const qc = useQueryClient();
  return useMutation<PartnerKeyFolder, Error, { orgId: string; name: string }>({
    mutationFn: ({ orgId, name }) =>
      apiFetch<PartnerKeyFolder>(`${API_URL}/orgs/${orgId}/partner-key-folders`, {
        method: "POST",
        body: JSON.stringify({ name }),
      }),
    onSuccess: (_data, { orgId }) => {
      qc.invalidateQueries({ queryKey: keysKey(orgId) });
    },
  });
}

/** Spend is attributed by the key's CURRENT folder, so usage moves too. */
export function useSetPartnerKeyFolder() {
  const qc = useQueryClient();
  return useMutation<{ ok: boolean }, Error, { orgId: string; keyId: string; folderId: string | null }>({
    mutationFn: ({ orgId, keyId, folderId }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/partner-keys/${keyId}/folder`, {
        method: "PUT",
        body: JSON.stringify({ folder_id: folderId }),
      }),
    onSuccess: (_data, { orgId }) => {
      qc.invalidateQueries({ queryKey: keysKey(orgId) });
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "usage"] });
      toast.success("Key moved");
    },
    onError: (e) => toast.error(e.message),
  });
}
