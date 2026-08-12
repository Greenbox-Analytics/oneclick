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
  default_member_cap?: number | null;
  monthly_dispersal_credits?: number;
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
  /** Purchases AND monthly dispersals — everything the org has paid us. */
  cumulative_paid_in: number;
  remaining_to_activate: number;
  member_count: number;
  /** Active admins, visible to every member — their only remedy for a reached
   * cap or a dry pool is "ask an admin", which needs a name to ask. The rest of
   * the roster (and every cap/spend figure) stays admin-only in `/usage`. */
  admins?: OrgAdminContact[];
}

export interface OrgAdminContact {
  userId: string;
  email: string | null;
  fullName: string | null;
}

/** One row of GET /orgs/{id}/usage's `seats` array (admin-only).
 * Members hold no balance — they spend from the pool against a monthly cap, so
 * what matters per member is their ceiling and what they've used of it. */
export interface OrgSeatUsage {
  orgMemberId: string;
  userId: string;
  email: string | null;
  role: OrgRole;
  status: OrgMemberStatus;
  /** This member's own cap; null = inherits the org default. */
  monthlyCap: number | null;
  /** Cap actually in force after the org-default fallback; null = uncapped. */
  effectiveCap: number | null;
  /** Counter maintained by debit_credits, reset each period. */
  capUsed: number;
  /** Ledger-derived spend for the pool's current period. */
  spentThisPeriod: number;
}

/** GET /orgs/{id}/usage — admin-only per-member rollup. */
export interface OrgUsage {
  poolBalance: number;
  cumulativePaidIn: number;
  monthlyDispersalCredits: number;
  defaultMemberCap: number | null;
  periodStart: string | null;
  periodEnd: string | null;
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
  /** The cap the member asked for; null = "raise it, admin decides". */
  requested_cap: number | null;
  note: string | null;
  status: CreditRequestStatus;
  resolved_by?: string | null;
  resolved_cap?: number | null;
  created_at?: string;
  resolved_at?: string | null;
}

/** An org_members row echoed back by the cap endpoints. */
export interface OrgMemberRow {
  id: string;
  org_id: string;
  user_id: string;
  role: OrgRole;
  status: OrgMemberStatus;
  monthly_cap?: number | null;
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
  /** `undefined` = leave untouched; `null` clears members back to uncapped.
   * The monthly dispersal has its own endpoint (useSetOrgDispersal) — it's the
   * contract, not a display preference. */
  default_member_cap?: number | null;
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
      toast.success("Member suspended");
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
      toast.success("Member removed");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't remove member.")),
  });
}

// ---------------------------------------------------------------------------
// Caps — the enforcement mechanism. Nothing moves, so there is no idempotency
// key to mint and no duplicate-transfer case to handle: writing a ceiling twice
// lands the same ceiling.
// ---------------------------------------------------------------------------

export function useSetMemberCap() {
  const qc = useQueryClient();
  const { user } = useAuth();
  return useMutation<OrgMemberRow, Error, { orgId: string; memberId: string; cap: number | null }>({
    mutationFn: ({ orgId, memberId, cap }) =>
      apiFetch<OrgMemberRow>(`${API_URL}/orgs/${orgId}/members/${memberId}/cap`, {
        method: "PUT",
        body: JSON.stringify({ cap }),
      }),
    onSuccess: (_d, { orgId, cap }) => {
      invalidateOrgUsage(qc, orgId);
      qc.invalidateQueries({ queryKey: ["entitlements", user?.id] });
      toast.success(cap === null ? "Using the organization default" : `Limit set to ${cap.toLocaleString()} credits`);
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't update the limit.")),
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
 * member view). `requestedCap` omitted = "raise it, admin decides" (matches the
 * nullable `requested_cap` column). The DB's one-open-request-per-member index
 * turns a second submit into a 409 — surfaced with dedicated copy rather than
 * the generic error toast. */
export function useSubmitCreditRequest() {
  const qc = useQueryClient();
  return useMutation<OrgCreditRequest, Error, { orgId: string; requestedCap?: number; note?: string }>({
    mutationFn: ({ orgId, requestedCap, note }) =>
      apiFetch<OrgCreditRequest>(`${API_URL}/orgs/${orgId}/credit-requests`, {
        method: "POST",
        body: JSON.stringify({ requested_cap: requestedCap ?? null, note: note?.trim() || null }),
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
  return useMutation<OrgCreditRequest, Error, { orgId: string; requestId: string; cap: number }>({
    mutationFn: ({ orgId, requestId, cap }) =>
      apiFetch<OrgCreditRequest>(`${API_URL}/orgs/${orgId}/credit-requests/${requestId}/approve`, {
        method: "POST",
        body: JSON.stringify({ cap }),
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
// Org admins VIEW the projects their org owns (ownership comes from the
// project's ARTIST) and manage SEAT ACCESS on them (Task 3). There is no
// per-project link to create or remove any more — see 20260804000001.
// ---------------------------------------------------------------------------

export type OrgProjectRole = "viewer" | "editor" | "admin";

/** One row of GET /orgs/{id}/projects — the org ADMIN console list of the
 * projects this org owns. Ownership comes from the project's ARTIST
 * (`artists.team_id`); the per-project `org_project_links` edge, and the
 * owner-facing link/unlink controls that went with it, were retired in
 * 20260804000001. An owner hands a whole artist to a team from the artist's
 * profile instead (`useTransferArtistToTeam` in `useArtistTeam.ts`). */
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
