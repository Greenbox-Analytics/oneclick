// src/hooks/useOrgs.ts
// Licensing Phase B (spec §7, plan Task 12) — typed hooks for every /orgs/*
// endpoint the `/organization` admin console consumes. Mirrors useTeams.ts's
// idioms: query keys namespaced ["orgs", ...], mutations invalidate the
// relevant keys, and hook-level toasts surface backend errors (409s carry
// human-written copy — apiFetch's ApiError.message already IS that string
// verbatim for plain-string `detail` bodies, so `toast.error(e.message)`
// needs no re-wording).
//
// The whole /orgs surface 404s when LICENSING_ENABLED is off (router-level
// gate in orgs/router.py) — callers probe that via `useMyOrgs()`'s error:
// `error instanceof ApiError && error.status === 404` ⇒ flag off (or, for
// every OTHER endpoint here, "not a member of that org" — same
// no-existence-oracle 404 orgs/authz.py uses throughout).
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, ApiError, apiFetch } from "@/lib/apiFetch";
import { supabase } from "@/integrations/supabase/client";

export type OrgStatus = "pending" | "active" | "suspended";
export type OrgRole = "admin" | "member";
export type OrgMemberStatus = "active" | "suspended" | "removed";
export type CreditRequestStatus = "pending" | "approved" | "denied";

/** Row shape from GET /orgs (list_my_orgs) — annotated with the caller's own
 * membership. No pool/activation fields here; fetch a single org (useOrg)
 * for those. */
export interface OrgSummary {
  id: string;
  name: string;
  created_by?: string | null;
  min_initial_purchase_credits?: number | null;
  default_seat_allowance?: number | null;
  status: OrgStatus;
  archived_at?: string | null;
  created_at?: string;
  updated_at?: string;
  my_role?: OrgRole | null;
  my_status?: OrgMemberStatus | null;
}

/** GET /orgs/{id} (get_org) — org row + computed pool/activation fields.
 * Member-only (404s for non-members); a suspended/removed seat also 404s
 * (require_member only counts ACTIVE rows). */
export interface OrgDetail extends OrgSummary {
  pool_balance: number;
  cumulative_purchased: number;
  remaining_to_activate: number;
  member_count: number;
}

/** One row of GET /orgs/{id}/usage's `seats` array (admin-only). */
export interface OrgSeatUsage {
  orgMemberId: string;
  userId: string;
  email: string | null;
  role: OrgRole;
  status: OrgMemberStatus;
  seatBalance: number;
  spentAllTime: number;
  storageBytes: number;
  storageCapBytes: number;
}

/** GET /orgs/{id}/usage — admin-only per-seat rollup. */
export interface OrgUsage {
  poolBalance: number;
  cumulativePurchased: number;
  seats: OrgSeatUsage[];
}

export interface OrgInvite {
  id: string;
  org_id: string;
  email: string;
  role: OrgRole;
  token?: string;
  status: "pending" | "accepted" | "declined";
  invited_by?: string | null;
  created_at?: string;
  expires_at?: string;
}

export interface OrgCreditRequest {
  id: string;
  org_id: string;
  org_member_id: string;
  requested_credits: number | null;
  note: string | null;
  status: CreditRequestStatus;
  resolved_by?: string | null;
  resolved_credits?: number | null;
  created_at?: string;
  resolved_at?: string | null;
}

/** transfer_credits RPC's JSONB return, surfaced verbatim by allocate/reclaim.
 * `removed` is present only on reclaim's amount=null no-op (`{"removed": 0}`,
 * never calls the RPC). */
export interface TransferResult {
  duplicate?: boolean;
  from_balance?: number;
  to_balance?: number;
  removed?: number;
}

const errMessage = (e: unknown, fallback: string): string => (e instanceof Error ? e.message : fallback);

/** POST /orgs/invites/{token}/{accept,decline} JSON body. */
export interface OrgInviteActionResult {
  type: "accepted" | "already_accepted" | "declined";
  org_id: string;
}

// ---------------------------------------------------------------------------
// Invite claim (by token) — src/pages/OrgInviteClaim.tsx. Unlike every other
// hook in this file, these intentionally do NOT auto-toast on error: the
// claim page owns the success/error UI (expired vs. wrong-email vs. not-found
// all need distinct copy — plan Task 13), so raw errors are handed back to
// the caller instead of being swallowed into a generic toast.
// ---------------------------------------------------------------------------

export function useAcceptOrgInvite() {
  const qc = useQueryClient();
  return useMutation<OrgInviteActionResult, Error, string>({
    mutationFn: (token) =>
      apiFetch<OrgInviteActionResult>(`${API_URL}/orgs/invites/${token}/accept`, { method: "POST" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["orgs", "list"] });
      qc.invalidateQueries({ queryKey: ["entitlements"] });
    },
  });
}

export function useDeclineOrgInvite() {
  return useMutation<OrgInviteActionResult, Error, string>({
    mutationFn: (token) =>
      apiFetch<OrgInviteActionResult>(`${API_URL}/orgs/invites/${token}/decline`, { method: "POST" }),
  });
}

// ---------------------------------------------------------------------------
// Orgs: list / get / create / update / archive
// ---------------------------------------------------------------------------

/** GET /orgs — every org the caller holds a non-removed seat in. 404 (via
 * ApiError.status) means LICENSING_ENABLED is off — `retry: false` so that
 * probe doesn't hammer the backend. */
export function useMyOrgs() {
  const { user } = useAuth();
  return useQuery<OrgSummary[]>({
    queryKey: ["orgs", "list"],
    queryFn: async () => (await apiFetch<{ organizations: OrgSummary[] }>(`${API_URL}/orgs`)).organizations,
    enabled: !!user?.id,
    retry: false,
    staleTime: 30_000,
  });
}

export function useOrg(orgId?: string) {
  const { user } = useAuth();
  return useQuery<OrgDetail>({
    queryKey: ["orgs", orgId, "detail"],
    queryFn: () => apiFetch<OrgDetail>(`${API_URL}/orgs/${orgId}`),
    enabled: !!user?.id && !!orgId,
    staleTime: 15_000,
  });
}

export function useCreateOrg() {
  const qc = useQueryClient();
  return useMutation<OrgSummary, Error, { name: string }>({
    mutationFn: ({ name }) =>
      apiFetch<OrgSummary>(`${API_URL}/orgs`, { method: "POST", body: JSON.stringify({ name }) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["orgs", "list"] });
      toast.success("Organization created");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't create organization.")),
  });
}

export interface UpdateOrgArgs {
  orgId: string;
  name?: string;
  /** `undefined` = leave untouched; `null` clears back to manual-only. */
  default_seat_allowance?: number | null;
}

export function useUpdateOrg() {
  const qc = useQueryClient();
  return useMutation<OrgSummary, Error, UpdateOrgArgs>({
    mutationFn: ({ orgId, ...fields }) =>
      apiFetch<OrgSummary>(`${API_URL}/orgs/${orgId}`, { method: "PUT", body: JSON.stringify(fields) }),
    onSuccess: (_d, { orgId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", "list"] });
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "detail"] });
      toast.success("Organization updated");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't update organization.")),
  });
}

export function useArchiveOrg() {
  const qc = useQueryClient();
  return useMutation<unknown, Error, { orgId: string }>({
    mutationFn: ({ orgId }) => apiFetch(`${API_URL}/orgs/${orgId}/archive`, { method: "POST" }),
    onSuccess: (_d, { orgId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", "list"] });
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "detail"] });
      toast.success("Organization archived");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't archive organization.")),
  });
}

/** GET /orgs/{id}/usage — admin-only. */
export function useOrgUsage(orgId?: string) {
  const { user } = useAuth();
  return useQuery<OrgUsage>({
    queryKey: ["orgs", orgId, "usage"],
    queryFn: () => apiFetch<OrgUsage>(`${API_URL}/orgs/${orgId}/usage`),
    enabled: !!user?.id && !!orgId,
    staleTime: 15_000,
  });
}

// ---------------------------------------------------------------------------
// Members: role, suspend, reactivate, remove
// ---------------------------------------------------------------------------

function invalidateOrgUsage(qc: ReturnType<typeof useQueryClient>, orgId: string) {
  qc.invalidateQueries({ queryKey: ["orgs", orgId, "usage"] });
  qc.invalidateQueries({ queryKey: ["orgs", orgId, "detail"] });
}

export function useUpdateOrgMemberRole() {
  const qc = useQueryClient();
  return useMutation<unknown, Error, { orgId: string; memberId: string; role: OrgRole }>({
    mutationFn: ({ orgId, memberId, role }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/members/${memberId}/role`, {
        method: "PUT",
        body: JSON.stringify({ role }),
      }),
    onSuccess: (_d, { orgId }) => {
      invalidateOrgUsage(qc, orgId);
      toast.success("Role updated");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't update role.")),
  });
}

export function useSuspendOrgMember() {
  const qc = useQueryClient();
  return useMutation<unknown, Error, { orgId: string; memberId: string }>({
    mutationFn: ({ orgId, memberId }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/members/${memberId}/suspend`, { method: "POST" }),
    onSuccess: (_d, { orgId }) => {
      invalidateOrgUsage(qc, orgId);
      toast.success("Member suspended — their credits were reclaimed to the pool");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't suspend member.")),
  });
}

export function useReactivateOrgMember() {
  const qc = useQueryClient();
  return useMutation<unknown, Error, { orgId: string; memberId: string }>({
    mutationFn: ({ orgId, memberId }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/members/${memberId}/reactivate`, { method: "POST" }),
    onSuccess: (_d, { orgId }) => {
      invalidateOrgUsage(qc, orgId);
      toast.success("Member reactivated");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't reactivate member.")),
  });
}

export function useRemoveOrgMember() {
  const qc = useQueryClient();
  return useMutation<unknown, Error, { orgId: string; memberId: string }>({
    mutationFn: ({ orgId, memberId }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/members/${memberId}`, { method: "DELETE" }),
    onSuccess: (_d, { orgId }) => {
      invalidateOrgUsage(qc, orgId);
      toast.success("Member removed — their credits were reclaimed to the pool");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't remove member.")),
  });
}

// ---------------------------------------------------------------------------
// Allocate / reclaim — pool <-> seat money movement via transfer_credits.
// Each submission gets its OWN fresh idempotency key (generated inside
// mutationFn, so a fresh key is minted per .mutate() call): a STABLE/reused
// key across distinct admin actions would make the second, unrelated
// allocation silently no-op as a "duplicate" of the first.
// ---------------------------------------------------------------------------

export function useAllocateCredits() {
  const qc = useQueryClient();
  const { user } = useAuth();
  return useMutation<TransferResult, Error, { orgId: string; memberId: string; amount: number }>({
    mutationFn: ({ orgId, memberId, amount }) =>
      apiFetch<TransferResult>(`${API_URL}/orgs/${orgId}/members/${memberId}/allocate`, {
        method: "POST",
        body: JSON.stringify({ amount, idempotency_key: crypto.randomUUID() }),
      }),
    onSuccess: (_d, { orgId }) => {
      invalidateOrgUsage(qc, orgId);
      qc.invalidateQueries({ queryKey: ["entitlements", user?.id] });
      toast.success("Credits allocated");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't allocate credits.")),
  });
}

export function useReclaimCredits() {
  const qc = useQueryClient();
  const { user } = useAuth();
  return useMutation<TransferResult, Error, { orgId: string; memberId: string; amount: number | null }>({
    mutationFn: ({ orgId, memberId, amount }) =>
      apiFetch<TransferResult>(`${API_URL}/orgs/${orgId}/members/${memberId}/reclaim`, {
        method: "POST",
        body: JSON.stringify({ amount, idempotency_key: crypto.randomUUID() }),
      }),
    onSuccess: (_d, { orgId }) => {
      invalidateOrgUsage(qc, orgId);
      qc.invalidateQueries({ queryKey: ["entitlements", user?.id] });
      toast.success("Credits reclaimed to the pool");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't reclaim credits.")),
  });
}

// ---------------------------------------------------------------------------
// Invites
// ---------------------------------------------------------------------------

export function useOrgInvites(orgId?: string) {
  const { user } = useAuth();
  return useQuery<OrgInvite[]>({
    queryKey: ["orgs", orgId, "invites"],
    queryFn: async () => (await apiFetch<{ invites: OrgInvite[] }>(`${API_URL}/orgs/${orgId}/invites`)).invites,
    enabled: !!user?.id && !!orgId,
    staleTime: 15_000,
  });
}

export function useInviteOrgMember() {
  const qc = useQueryClient();
  return useMutation<{ type: string; notify_user_id?: string | null }, Error, { orgId: string; email: string; role: OrgRole }>({
    mutationFn: ({ orgId, email, role }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/invites`, {
        method: "POST",
        body: JSON.stringify({ email, role }),
      }),
    onSuccess: (data, { orgId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "invites"] });
      toast.success(data?.notify_user_id ? "Invitation sent (email + in-app)" : "Invitation email sent");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't send invite.")),
  });
}

export function useCancelOrgInvite() {
  const qc = useQueryClient();
  return useMutation<unknown, Error, { orgId: string; inviteId: string }>({
    mutationFn: ({ orgId, inviteId }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/invites/${inviteId}`, { method: "DELETE" }),
    onSuccess: (_d, { orgId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "invites"] });
      toast.success("Invitation canceled");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't cancel invite.")),
  });
}

// ---------------------------------------------------------------------------
// Credit requests — member ask -> admin approve/deny
// ---------------------------------------------------------------------------

export function useOrgCreditRequests(orgId?: string) {
  const { user } = useAuth();
  return useQuery<OrgCreditRequest[]>({
    queryKey: ["orgs", orgId, "credit-requests"],
    queryFn: async () =>
      (await apiFetch<{ requests: OrgCreditRequest[] }>(`${API_URL}/orgs/${orgId}/credit-requests`)).requests,
    enabled: !!user?.id && !!orgId,
    staleTime: 15_000,
  });
}

function invalidateCreditRequests(qc: ReturnType<typeof useQueryClient>, orgId: string) {
  qc.invalidateQueries({ queryKey: ["orgs", orgId, "credit-requests"] });
  invalidateOrgUsage(qc, orgId);
}

/** POST /orgs/{id}/credit-requests — any ACTIVE member (src/pages/Organization.tsx's
 * member view, plan Task 13). `requestedCredits` omitted = "more, admin decides"
 * (matches the nullable `requested_credits` column). The DB's one-open-request-
 * per-seat index turns a second submit into a 409 — surfaced with dedicated
 * copy rather than the generic error toast. */
export function useSubmitCreditRequest() {
  const qc = useQueryClient();
  return useMutation<OrgCreditRequest, Error, { orgId: string; requestedCredits?: number; note?: string }>({
    mutationFn: ({ orgId, requestedCredits, note }) =>
      apiFetch<OrgCreditRequest>(`${API_URL}/orgs/${orgId}/credit-requests`, {
        method: "POST",
        body: JSON.stringify({ requested_credits: requestedCredits ?? null, note: note?.trim() || null }),
      }),
    onSuccess: (_d, { orgId }) => {
      invalidateCreditRequests(qc, orgId);
      toast.success("Request sent to your admin");
    },
    onError: (e) => {
      if (e instanceof ApiError && e.status === 409) {
        toast.error("You already have a request waiting for your admin.");
        return;
      }
      toast.error(errMessage(e, "Couldn't send request."));
    },
  });
}

export function useApproveCreditRequest() {
  const qc = useQueryClient();
  const { user } = useAuth();
  return useMutation<OrgCreditRequest, Error, { orgId: string; requestId: string; credits: number }>({
    mutationFn: ({ orgId, requestId, credits }) =>
      apiFetch<OrgCreditRequest>(`${API_URL}/orgs/${orgId}/credit-requests/${requestId}/approve`, {
        method: "POST",
        body: JSON.stringify({ credits }),
      }),
    onSuccess: (_d, { orgId }) => {
      invalidateCreditRequests(qc, orgId);
      qc.invalidateQueries({ queryKey: ["entitlements", user?.id] });
      toast.success("Request approved");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't approve request.")),
  });
}

export function useDenyCreditRequest() {
  const qc = useQueryClient();
  return useMutation<OrgCreditRequest, Error, { orgId: string; requestId: string; note?: string }>({
    mutationFn: ({ orgId, requestId, note }) =>
      apiFetch<OrgCreditRequest>(`${API_URL}/orgs/${orgId}/credit-requests/${requestId}/deny`, {
        method: "POST",
        body: JSON.stringify({ note: note || undefined }),
      }),
    onSuccess: (_d, { orgId }) => {
      invalidateCreditRequests(qc, orgId);
      toast.success("Request denied");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't deny request.")),
  });
}

// ---------------------------------------------------------------------------
// Project links (Licensing Phase C, spec §6, plan Task 8) — the project
// OWNER links/unlinks their project to an org they hold an active seat in
// (rule 1: linking = consent, never an org admin's call); org admins can
// only VIEW linked projects and manage SEAT ACCESS on them (Task 3).
// ---------------------------------------------------------------------------

export type OrgProjectRole = "viewer" | "editor" | "admin";

/** The ONE org (if any) a project is linked to, from the OWNER's point of
 * view. There is no backend GET-by-project endpoint for this (only the org-
 * admin-scoped `GET /orgs/{id}/projects` list, which the owner may not be
 * authorized to call at all if they're a plain member) — the owner-facing
 * check reads `org_project_links` directly instead, which its RLS policy
 * (20260723000001 migration) explicitly grants the project OWNER SELECT on.
 * That table (and `organizations`) isn't in the generated Supabase types yet
 * (the migration is written, not run), hence the `as any` cast below — same
 * pattern as `src/components/project/AudioTab.tsx`'s `sb`. */
export interface OrgProjectLinkInfo {
  orgId: string;
  orgName: string | null;
  linkedAt: string | null;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const sbAny = supabase as any;

export function useProjectOrgLink(projectId?: string) {
  const { user } = useAuth();
  return useQuery<OrgProjectLinkInfo | null>({
    queryKey: ["orgs", "project-link", projectId],
    queryFn: async () => {
      const { data, error } = await sbAny
        .from("org_project_links")
        .select("org_id, created_at, organizations(name)")
        .eq("project_id", projectId)
        .maybeSingle();
      if (error) throw error;
      if (!data) return null;
      return {
        orgId: data.org_id as string,
        // Null when the owner's own seat in that org no longer exists (e.g.
        // offboarded after linking) — `organizations` RLS is member-scoped,
        // so the embed silently drops rather than erroring. The UI falls
        // back to generic "your organization" wording in that case.
        orgName: (data.organizations?.name as string | undefined) ?? null,
        linkedAt: (data.created_at as string | undefined) ?? null,
      };
    },
    enabled: !!user?.id && !!projectId,
    staleTime: 10_000,
  });
}

export interface LinkProjectResult {
  id?: string;
  org_id: string;
  project_id: string;
  linked_by?: string | null;
  created_at?: string;
}

/** POST /orgs/{org_id}/projects/{project_id}/link. 409 (already linked — to
 * this org or a DIFFERENT one, rule 8) carries exact backend copy, surfaced
 * as-is per this file's header note. */
export function useLinkProjectToOrg() {
  const qc = useQueryClient();
  return useMutation<LinkProjectResult, Error, { orgId: string; projectId: string }>({
    mutationFn: ({ orgId, projectId }) =>
      apiFetch<LinkProjectResult>(`${API_URL}/orgs/${orgId}/projects/${projectId}/link`, { method: "POST" }),
    onSuccess: (_d, { projectId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", "project-link", projectId] });
      qc.invalidateQueries({ queryKey: ["entitlements"] });
      toast.success("Project linked to your organization");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't link this project.")),
  });
}

export interface UnlinkProjectResult {
  revoked: number;
}

/** DELETE /orgs/{org_id}/projects/{project_id}/link — owner-only, works even
 * if the owner's own seat has since lapsed (unlinking never re-checks seat
 * status, only ownership). Surfaces the exact revocation count from
 * `orgs/projects.py`'s `{"revoked": n}` response (rule 3). */
export function useUnlinkProjectFromOrg() {
  const qc = useQueryClient();
  return useMutation<UnlinkProjectResult, Error, { orgId: string; projectId: string }>({
    mutationFn: ({ orgId, projectId }) =>
      apiFetch<UnlinkProjectResult>(`${API_URL}/orgs/${orgId}/projects/${projectId}/link`, { method: "DELETE" }),
    onSuccess: (result, { projectId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", "project-link", projectId] });
      qc.invalidateQueries({ queryKey: ["project-members", projectId] });
      qc.invalidateQueries({ queryKey: ["entitlements"] });
      const revoked = result?.revoked ?? 0;
      toast.success(
        revoked > 0
          ? `Project unlinked — ${revoked} teammate${revoked === 1 ? "" : "s"} lost the access the organization granted`
          : "Project unlinked from your organization",
      );
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't unlink this project.")),
  });
}

/** One row of GET /orgs/{id}/projects (org ADMIN console — Task 2 AC 3). */
export interface OrgLinkedProject {
  projectId: string;
  name: string | null;
  ownerEmail: string | null;
  linkedAt: string | null;
  orgGrantedMemberCount: number;
}

export function useOrgLinkedProjects(orgId?: string) {
  const { user } = useAuth();
  return useQuery<OrgLinkedProject[]>({
    queryKey: ["orgs", orgId, "linked-projects"],
    queryFn: async () =>
      (await apiFetch<{ projects: OrgLinkedProject[] }>(`${API_URL}/orgs/${orgId}/projects`)).projects,
    enabled: !!user?.id && !!orgId,
    staleTime: 15_000,
  });
}

/** Shared response shape of both Task 3 membership endpoints — `"organic"`
 * is a NO-OP (rule 2: never overwrite independent access), never an error;
 * `member`/`revoked` are present only on `"granted"`/`"revoked"` respectively. */
export interface OrgProjectMemberActionResult {
  status: "granted" | "organic" | "revoked";
  detail?: string;
  member?: { id: string; project_id: string; user_id: string; role: OrgProjectRole; org_id: string | null } | null;
  revoked?: number;
}

/** PUT /orgs/{org_id}/projects/{project_id}/members/{member_id} — org ADMIN.
 * `onSuccess` deliberately does NOT toast: the caller renders the returned
 * `status` inline (granted/organic) rather than as a toast, since "organic"
 * is informational, not a success/failure binary. Actual errors (404s, the
 * 409 owner-target case) DO get a generic toast — there's no bespoke per-
 * error UI here the way the invite-claim page needs, so the standard
 * pattern this file uses everywhere else applies. */
export function useSetOrgProjectMemberRole() {
  const qc = useQueryClient();
  return useMutation<
    OrgProjectMemberActionResult,
    Error,
    { orgId: string; projectId: string; memberId: string; role: OrgProjectRole }
  >({
    mutationFn: ({ orgId, projectId, memberId, role }) =>
      apiFetch<OrgProjectMemberActionResult>(`${API_URL}/orgs/${orgId}/projects/${projectId}/members/${memberId}`, {
        method: "PUT",
        body: JSON.stringify({ role }),
      }),
    onSuccess: (_d, { orgId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "linked-projects"] });
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't update this member's access.")),
  });
}

/** DELETE /orgs/{org_id}/projects/{project_id}/members/{member_id} — org
 * ADMIN. Same no-toast-on-success rationale as `useSetOrgProjectMemberRole`. */
export function useRemoveOrgProjectMember() {
  const qc = useQueryClient();
  return useMutation<OrgProjectMemberActionResult, Error, { orgId: string; projectId: string; memberId: string }>({
    mutationFn: ({ orgId, projectId, memberId }) =>
      apiFetch<OrgProjectMemberActionResult>(`${API_URL}/orgs/${orgId}/projects/${projectId}/members/${memberId}`, {
        method: "DELETE",
      }),
    onSuccess: (_d, { orgId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "linked-projects"] });
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't remove this member's access.")),
  });
}
