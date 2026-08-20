// src/hooks/useOrgs.ts
// Licensing Phase B (spec §7, plan Task 12) — typed hooks for every /orgs/*
// endpoint the `/teams` admin console consumes. House idioms:
// query keys namespaced ["orgs", ...], mutations invalidate the
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

export type OrgStatus = "pending" | "active" | "suspended" | "lapsed";
export type OrgRole = "admin" | "member";
export type OrgMemberStatus = "active" | "suspended" | "removed";
export type CreditRequestStatus = "pending" | "approved" | "denied";
/** 'enterprise' = today's negotiated/admin-managed org (default, incl. every
 * pre-2026-08-15 row). 'self_serve' = paid-tier team creation (2026-08-15
 * pricing/teams spec) — only these get coverage/grace/lapse/dissolve. */
export type OrgKind = "self_serve" | "enterprise";

/** Row shape from GET /orgs (list_my_orgs) — annotated with the caller's own
 * membership. No pool/activation fields here; fetch a single org (useOrg)
 * for those.
 *
 * `kind`/`covered_by`/`covered_at`/`dissolved_at` are NOT in the backend's
 * `_ADMIN_ONLY_ORG_FIELDS` redaction set, so every member (not just admins)
 * sees them — they're optional here only for payloads served before the
 * backend shipped these columns (pre-migration rows), not for role reasons.
 * `undefined` kind should be treated as enterprise (today's behavior). */
export interface OrgSummary {
  id: string;
  name: string;
  created_by?: string | null;
  min_initial_purchase_credits?: number | null;
  default_member_cap?: number | null;
  monthly_dispersal_credits?: number;
  status: OrgStatus;
  archived_at?: string | null;
  kind?: OrgKind;
  /** Admin currently on the hook for this org's slot/storage/billing. Stays
   * set (last coverer) even when released — see release_coverage. */
  covered_by?: string | null;
  /** Null = released/never claimed, even if covered_by is set. */
  covered_at?: string | null;
  dissolved_at?: string | null;
  created_at?: string;
  updated_at?: string;
  my_role?: OrgRole | null;
  my_status?: OrgMemberStatus | null;
}

/** GET /orgs/{id} (get_org) — org row + computed pool/activation fields.
 * Member-only (404s for non-members); a suspended/removed seat also 404s
 * (require_member only counts ACTIVE rows).
 *
 * The pool/dispersal/activation fields are ADMIN-ONLY and the backend omits
 * them entirely for a plain member — hence the `?`. Never render them without
 * an `org.my_role === "admin"` guard, or a member sees `undefined`. */
export interface OrgDetail extends OrgSummary {
  pool_balance?: number;
  /** Purchases AND monthly dispersals — everything the org has paid us. */
  cumulative_paid_in?: number;
  remaining_to_activate?: number;
  member_count: number;
  /** Active admins, visible to every member — their only remedy for a reached
   * cap or a dry pool is "ask an admin", which needs a name to ask. The rest of
   * the roster (and every cap/spend figure) stays admin-only in `/usage`. */
  admins?: OrgAdminContact[];
  /** ADMIN-ONLY (in `_ADMIN_ONLY_ORG_FIELDS`) — when grace started, if the
   * org is uncovered. Absent for members, and absent when not in grace. */
  grace_started_at?: string | null;
  /**
   * Configured grace window length in days, for the banner's "loses access
   * on {date}" copy. REAL as of Task 15 (was a stub before) — a global
   * constant, so present for every org/role, not admin-gated. Still keep
   * rendering defensively: when absent, say "soon" instead of computing a date.
   */
  graceDays?: number;
  /** ADMIN-ONLY, self_serve-only (Task 15, spec §6): the covering admin's
   * team storage pool, sized against their plan's `team_storage_bytes`.
   * Absent for enterprise orgs, for a released org (no covered_by), and for
   * any non-admin — same redaction posture as `pool_balance` etc. */
  teamStorage?: {
    usedBytes: number;
    poolBytes: number;
    /** ceil((usedBytes - poolBytes) / 1 GiB), floored at 0. */
    overageGb: number;
    /** USD per GB over the pool, billed to the covering admin's Stripe account. */
    ratePerGb: number;
  };
  /** ADMIN-ONLY (Task 15) — set once an admin has started this org's
   * recurring monthly credit top-up (POST /billing/org-topup-checkout).
   * Null/absent = no active top-up. */
  topup_stripe_subscription_id?: string | null;
  /** ADMIN-ONLY (Task 15) — the admin whose card the recurring top-up bills
   * to. Only that admin's own Stripe portal manages/cancels the underlying
   * card, but ANY active admin may cancel the top-up itself
   * (POST .../cancel-topup). */
  topup_admin_id?: string | null;
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
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["orgs", "list"] });
      qc.invalidateQueries({ queryKey: ["entitlements"] });
      // A new seat changes the roster the board pickers read.
      qc.invalidateQueries({ queryKey: ["orgs", data.org_id, "roster"] });
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

/** The live orgs the caller actually holds a seat in — an archived / lapsed
 * org (or a suspended seat) can't own a board you may act on. Order is stable
 * per input, so callers keying colours on index stay consistent. */
export function liveOrgs(orgs: OrgSummary[] | undefined): OrgSummary[] {
  return (orgs ?? []).filter((o) => o.my_status === "active" && !o.archived_at && o.status !== "lapsed");
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

type Qc = ReturnType<typeof useQueryClient>;

function invalidateOrgSummary(qc: Qc, orgId: string) {
  qc.invalidateQueries({ queryKey: ["orgs", "list"] });
  qc.invalidateQueries({ queryKey: ["orgs", orgId, "detail"] });
}

/** Body-less org mutation: hit `path(args)`, invalidate, toast. */
function useOrgPost<TArgs extends { orgId: string }, TData = unknown>(
  path: (args: TArgs) => string,
  okMsg: string,
  failMsg: string,
  invalidate: (qc: Qc, orgId: string) => void = invalidateOrgSummary,
  method: "POST" | "DELETE" = "POST",
) {
  const qc = useQueryClient();
  return useMutation<TData, Error, TArgs>({
    mutationFn: (args) => apiFetch<TData>(path(args), { method }),
    onSuccess: (_d, { orgId }) => {
      invalidate(qc, orgId);
      toast.success(okMsg);
    },
    onError: (e) => toast.error(errMessage(e, failMsg)),
  });
}

export function useArchiveOrg() {
  return useOrgPost<{ orgId: string }>(
    ({ orgId }) => `${API_URL}/orgs/${orgId}/archive`,
    "Organization archived",
    "Couldn't archive organization.",
  );
}

// ---------------------------------------------------------------------------
// Self-serve lifecycle (2026-08-15 pricing/teams spec §3): coverage
// claim/release, unarchive, and dissolve. Self_serve-only on the backend —
// enterprise orgs 409 ("This organization is managed by Msanii"), which the
// default onError below surfaces verbatim via ApiError.message (a plain-string
// `detail`), same as every other error this file renders. No request body for
// claim/release/unarchive.
// ---------------------------------------------------------------------------

export function useClaimCoverage() {
  return useOrgPost<{ orgId: string }, OrgSummary>(
    ({ orgId }) => `${API_URL}/orgs/${orgId}/coverage/claim`,
    "Coverage claimed",
    "Couldn't claim coverage.",
  );
}

export function useReleaseCoverage() {
  return useOrgPost<{ orgId: string }, OrgSummary>(
    ({ orgId }) => `${API_URL}/orgs/${orgId}/coverage/release`,
    "Coverage released",
    "Couldn't release coverage.",
  );
}

/** POST /orgs/{id}/unarchive — self-serve reactivation. 402 on no free slot
 * (router wraps `NoSlotError` in a structured `{reason, upgradeRequired: true}`
 * dict) or on the storage guard (a plain-string reason instead — see
 * orgs/router.py); either shape is handled by `apiErrorFromBody`'s fallback,
 * so `e.message` is the human copy to show either way. */
export function useUnarchiveOrg() {
  return useOrgPost<{ orgId: string }, OrgSummary>(
    ({ orgId }) => `${API_URL}/orgs/${orgId}/unarchive`,
    "Organization unarchived",
    "Couldn't unarchive organization.",
  );
}

/** One row of GET /orgs/{id}/dissolve-preview's `recipients` array. */
export interface DissolvePreviewRecipient {
  artistId: string;
  artistName: string | null;
  userId: string;
  email: string | null;
  /** True when the artist's creator no longer holds an active seat, so it
   * reverts to the dissolving admin instead ("returned to you" copy). */
  fallback: boolean;
}

/** GET /orgs/{id}/dissolve-preview — admin, self_serve only. What POST
 * .../dissolve is about to do, for the confirm dialog. */
export interface DissolvePreview {
  recipients: DissolvePreviewRecipient[];
  /** Purchased/comped reserve credits the pool forfeits (clawback). */
  forfeitReserve: number;
  /** Expiring monthly-bundle credits left inert on the dissolved org's wallet. */
  inertBundle: number;
  memberCount: number;
}

/** Fetched only while the dissolve dialog is open — `enabled` keeps this from
 * firing every time the lifecycle panel mounts. */
export function useDissolvePreview(orgId: string, enabled: boolean) {
  const { user } = useAuth();
  return useQuery<DissolvePreview>({
    queryKey: ["orgs", orgId, "dissolve-preview"],
    queryFn: () => apiFetch<DissolvePreview>(`${API_URL}/orgs/${orgId}/dissolve-preview`),
    enabled: !!user?.id && !!orgId && enabled,
  });
}

/** POST /orgs/{id}/dissolve — terminal, name-confirmed. `{already: true}` on
 * a replayed call (idempotent, nothing written twice); a mismatched name 400s
 * with "Type the team name exactly as it appears to confirm" before this
 * should ever be reachable, since the dialog gates its confirm button on an
 * exact match client-side too. */
export function useDissolveOrg() {
  const qc = useQueryClient();
  return useMutation<{ already?: boolean } & Record<string, unknown>, Error, { orgId: string; confirmName: string }>({
    mutationFn: ({ orgId, confirmName }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/dissolve`, {
        method: "POST",
        body: JSON.stringify({ confirm_name: confirmName }),
      }),
    onSuccess: (_d, { orgId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", "list"] });
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "detail"] });
      toast.success("Organization dissolved");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't dissolve organization.")),
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
// Org billing panel (Task 15, spec §6): pool ledger, personal-reserve
// transfer, and the recurring monthly top-up. Mirrors this file's idioms —
// namespaced query keys, invalidate-on-success — except the two hooks whose
// callers own bespoke error UI (useTransferCredits, useStartOrgTopup) skip
// the auto-toast onError this file uses elsewhere, same rationale as the
// invite-claim hooks up top: a generic toast would swallow the structured
// copy (409 reason, checkout failure) the dialog/panel needs to render itself.
// ---------------------------------------------------------------------------

/** One row of GET /orgs/{id}/ledger — the pool's activity feed. `action` is
 * the metered action for a `kind: "debit"` row (e.g. "oneclick_run"); null
 * for every other kind. `metadata.org_member_id` (debit rows) / `metadata.admin_user_id`
 * (transfer_in rows) are NOT resolved to an email client-side — the panel
 * renders simple kind-based labels ("member spend", "transfer from admin")
 * rather than doing its own identity lookup. */
export interface OrgLedgerEntry {
  kind: string;
  action: string | null;
  delta: number;
  metadata: Record<string, unknown>;
  created_at: string;
}

/** GET /orgs/{id}/ledger — admin-only, newest 50 pool ledger rows. */
export function useOrgLedger(orgId?: string) {
  const { user } = useAuth();
  return useQuery<OrgLedgerEntry[]>({
    queryKey: ["orgs", orgId, "ledger"],
    queryFn: async () => (await apiFetch<{ ledger: OrgLedgerEntry[] }>(`${API_URL}/orgs/${orgId}/ledger`)).ledger,
    enabled: !!user?.id && !!orgId,
    staleTime: 15_000,
  });
}

/** POST /orgs/{id}/transfer-credits — an active admin moves credits from
 * their OWN personal reserve into this org's pool. `duplicate: true` on a
 * retried request is a normal success (idempotent), not an error.
 *
 * No onError toast: a 409 (insufficient reserve) carries a structured
 * `{reason, reserveBalance}` — TransferCreditsDialog renders `error.message`
 * (== `reason`, per apiErrorFromBody's precedence) verbatim next to the
 * amount field instead of a toast, same as every other credit-wall dialog
 * in this codebase. */
export function useTransferCredits() {
  const qc = useQueryClient();
  const { user } = useAuth();
  return useMutation<{ duplicate?: boolean }, Error, { orgId: string; amount: number }>({
    mutationFn: ({ orgId, amount }) =>
      apiFetch(`${API_URL}/orgs/${orgId}/transfer-credits`, {
        method: "POST",
        body: JSON.stringify({ amount }),
      }),
    onSuccess: (data, { orgId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", "list"] });
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "detail"] });
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "ledger"] });
      qc.invalidateQueries({ queryKey: ["entitlements", user?.id] });
      if (!data?.duplicate) toast.success("Credits transferred to the pool");
    },
  });
}

/** POST /billing/org-topup-checkout — starts this org's recurring monthly
 * credit top-up (the SAME pack, sold as a Stripe subscription instead of a
 * one-time purchase) and redirects to Stripe Checkout. Catalog: filter
 * `useCreditPacks()`'s rows to `recurringPriceId != null` — no separate
 * endpoint. Mirrors `useCreateTopupSession` in useCreditPacks.ts (same
 * redirect-on-success shape, same no-onError-toast — the panel renders the
 * raw error inline like TopUpCreditsDialog does). */
export function useStartOrgTopup() {
  return useMutation<void, Error, { orgId: string; key: string }>({
    mutationFn: async ({ orgId, key }) => {
      const res = await apiFetch<{ url: string }>(`${API_URL}/billing/org-topup-checkout`, {
        method: "POST",
        body: JSON.stringify({ org_id: orgId, key }),
      });
      window.location.href = res.url;
    },
  });
}

/** POST /orgs/{id}/cancel-topup — ANY active admin may cancel, not just the
 * purchaser. `{canceled: false}` is a no-op (nothing was running); only a
 * real cancel toasts. */
export function useCancelOrgTopup() {
  const qc = useQueryClient();
  return useMutation<{ canceled: boolean }, Error, { orgId: string }>({
    mutationFn: ({ orgId }) => apiFetch(`${API_URL}/orgs/${orgId}/cancel-topup`, { method: "POST" }),
    onSuccess: (data, { orgId }) => {
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "detail"] });
      if (data?.canceled) toast.success("Monthly top-up canceled");
    },
    onError: (e) => toast.error(errMessage(e, "Couldn't cancel the monthly top-up.")),
  });
}

// ---------------------------------------------------------------------------
// Members: role, suspend, reactivate, remove
// ---------------------------------------------------------------------------

// Every caller writes org_members (role / suspend / reactivate / remove / cap),
// so the roster the board pickers read goes stale with them — invalidate it here
// rather than in each hook, or the assignee/filter/member pickers keep showing
// people who no longer hold a seat.
function invalidateOrgUsage(qc: ReturnType<typeof useQueryClient>, orgId: string) {
  qc.invalidateQueries({ queryKey: ["orgs", orgId, "usage"] });
  qc.invalidateQueries({ queryKey: ["orgs", orgId, "detail"] });
  qc.invalidateQueries({ queryKey: ["orgs", orgId, "roster"] });
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

type MemberArgs = { orgId: string; memberId: string };

export function useSuspendOrgMember() {
  return useOrgPost<MemberArgs>(
    ({ orgId, memberId }) => `${API_URL}/orgs/${orgId}/members/${memberId}/suspend`,
    "Member suspended",
    "Couldn't suspend member.",
    invalidateOrgUsage,
  );
}

export function useReactivateOrgMember() {
  return useOrgPost<MemberArgs>(
    ({ orgId, memberId }) => `${API_URL}/orgs/${orgId}/members/${memberId}/reactivate`,
    "Member reactivated",
    "Couldn't reactivate member.",
    invalidateOrgUsage,
  );
}

export function useRemoveOrgMember() {
  return useOrgPost<MemberArgs>(
    ({ orgId, memberId }) => `${API_URL}/orgs/${orgId}/members/${memberId}`,
    "Member removed",
    "Couldn't remove member.",
    invalidateOrgUsage,
    "DELETE",
  );
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
      qc.invalidateQueries({ queryKey: ["orgs", orgId, "roster"] });
      toast.success(data?.notify_user_id ? "Invitation sent (email + in-app)" : "Invitation email sent");
    },
    onError: (e) => {
      // Seat-wall 402s carry a {reason, limit, nextStep} detail — the caller
      // (OrgInvitesPanel) renders that inline with an upgrade/contact CTA
      // instead, so skip the generic toast to avoid saying it twice.
      if (e instanceof ApiError && e.detail && typeof e.detail === "object" && "nextStep" in (e.detail as object)) {
        return;
      }
      toast.error(errMessage(e, "Couldn't send invite."));
    },
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

// ---------------------------------------------------------------------------
// Roster — member-visible names/avatars (no emails), feeds the board assignee /
// filter / board-member pickers. Distinct from the admin console's seat table,
// which needs emails and caps.
// ---------------------------------------------------------------------------

export interface OrgRosterMember {
  user_id: string;
  role: OrgRole;
  full_name?: string | null;
  avatar_url?: string | null;
}

/**
 * GET /orgs/{id}/members — the member-visible roster: ACTIVE seats only
 * (suspended and removed seats are excluded), with name/avatar and no emails.
 * Requires a LIVE org too — a lapsed or archived org 404s, matching every other
 * predicate in this feature.
 *
 * The active-only filter is why any picker seeded from a stored id list (e.g. a
 * board's `member_user_ids`) must render a synthetic option for ids missing
 * here — see BoardSettingsDialog. Otherwise a suspended teammate becomes an
 * invisible, unremovable entry.
 */
export function useOrgRoster(orgId?: string | null) {
  const { user } = useAuth();
  return useQuery<OrgRosterMember[]>({
    queryKey: ["orgs", orgId, "roster"],
    queryFn: async () =>
      (await apiFetch<{ members: OrgRosterMember[] }>(`${API_URL}/orgs/${orgId}/members`)).members,
    enabled: !!user?.id && !!orgId,
    staleTime: 30_000,
  });
}
