import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";
import type { TierKey } from "@/lib/tiers";

/** Canonical tier keys live in @/lib/tiers (single source of truth — keys are
 * permanent, display labels come from tierLabel()). Re-exported here so the
 * many existing `import { Tier } from useEntitlements` sites keep working. */
export type Tier = TierKey;
export type SubscriptionStatus = "active" | "canceled" | "past_due" | "trialing";

export interface EntitlementCaps {
  maxArtists: number;
  maxProjects: number;
  maxTasks: number;
  maxStorageBytes: number;
  maxSplitSheetsPerMonth: number;
  maxOneclickRunsPerMonth: number;
  /** Credits system (present when the backend has the columns). */
  maxWorks?: number;
  includedStorageBytes?: number;
  monthlyCredits?: number;
  /**
   * Self-serve teams (2026-08-15 pricing/teams spec §1) — team SLOTS this
   * tier can own/cover, seats per owned team EXCLUDING the covering owner,
   * and the per-owner storage pool shared across all owned teams. 0 on Free.
   * Optional for payloads served before the backend shipped these columns.
   */
  maxTeams?: number;
  maxTeamMembers?: number;
  teamStorageBytes?: number;
}

/**
 * Per-action credit BASE RATES (backend `credit_prices`, camelCase).
 *
 * This IS the price, not an estimate: the charge is max(base, metered), and the
 * metered half only wins on a pathological run (spec 2026-08-17 §2). Safe to
 * render as a firm number — "Runs for 30 credits" — rather than hedged copy.
 */
export interface CreditPrices {
  zoeMessage: number;
  oneclickRun: number;
  registryParse: number;
  splitSheet: number;
}

/**
 * The org whose seat is paying, when the caller is in ORG billing context
 * (Licensing Phase B, spec §5). Mirrors backend `ManagedByOrg.to_dict()`.
 */
export interface OrgBillingContext {
  orgId: string;
  orgName: string;
  role: string;
  /** "self_serve" | "enterprise" | null (pre-migration org row). Drives
   * orgNoun()/orgNounCap() in @/lib/tiers for team-vs-organization copy. */
  kind?: "self_serve" | "enterprise" | null;
}

/**
 * One entry in `availableContexts` — every billing context the caller could
 * switch to via `useSetBillingContext` (Licensing Phase B, spec §5). Present
 * only when `LICENSING_ENABLED` is on.
 */
export type BillingContextOption =
  | { type: "personal" }
  | (OrgBillingContext & { type: "org"; pending: boolean });

/**
 * The caller's CURRENT billing-context identity (Licensing follow-ups Task 3).
 * Present whenever `LICENSING_ENABLED` is on, REGARDLESS of `CREDITS_ENABLED` —
 * unlike `credits.managedByOrg`, which only exists when the credits block
 * itself is built. Prefer this field for org/personal rendering; components
 * written before this field existed fall back to `credits?.managedByOrg`.
 */
export type BillingContext = { type: "personal" } | (OrgBillingContext & { type: "org" });

/**
 * Wallet state — present only when CREDITS_ENABLED is on (else `credits` is null).
 * Shape mirrors the backend Entitlements.to_dict()["credits"] block.
 */
export interface EntitlementCredits {
  /**
   * Spendable = bundleBalance + reserveBalance.
   *
   * `null` means REDACTED, never "zero": the caller is a plain member of an
   * org, and the shared pool is the org's money — admins only. Since this
   * whole block is absent when credits are off, `null` here has exactly one
   * meaning. Render the member's own `memberCap` instead, or — with no cap —
   * say they draw on the org pool, with no number.
   */
  balance: number | null;
  /** Monthly grant remainder; expires at period rollover. null = redacted. */
  bundleBalance: number | null;
  /** Admin grants; never expire. null = redacted. */
  reserveBalance: number | null;
  /** This tier's monthly credit grant. */
  monthlyGrant: number;
  overageThisPeriod: number;
  overageEnabled: boolean;
  /** USD per overage credit — quote this, never a hardcoded rate. */
  overageUsdPerCredit: number;
  overageCapCredits: number | null;
  /** Org context only: this member's monthly ceiling on the shared pool, and
   * what they've spent against it this period. null/0 in personal context,
   * where the wallet balance IS the limit. */
  memberCap?: number | null;
  memberCapUsed?: number;
  /** ISO timestamp when the credit period resets. */
  periodEnd: string | null;
  prices: CreditPrices;
  /** Present only in ORG billing context — the seat's org (Licensing Phase B, spec §5). */
  managedByOrg?: OrgBillingContext | null;
}

export interface EntitlementFeatures {
  zoeEnabled: boolean;
  oneclickEnabled: boolean;
  registryEnabled: boolean;
  integrationsAllowed: string[];
}

export interface EntitlementUsage {
  totalStorageBytes: number;
  splitSheetsThisPeriod: number;
  zoeQueriesThisPeriod: number;
  oneclickRunsThisPeriod: number;
  /** ISO timestamp when the current split-sheet period ends. */
  periodEnd: string;
}

export interface EntitlementSubscription {
  /** Stripe subscription ID — null for free or admin-grant-only Pro users. */
  stripeSubscriptionId: string | null;
  /** Stripe price ID — null when no Stripe subscription. */
  stripePriceId: string | null;
  /** ISO timestamp of when the current billing period ends. */
  currentPeriodEnd: string | null;
  /** True if the subscription is set to cancel at the end of the period. */
  cancelAtPeriodEnd: boolean;
  /** "monthly" | "annual" | null — derived from price_id on the backend. */
  planPeriod: "monthly" | "annual" | null;
}

export interface Entitlements {
  tier: Tier;
  status: SubscriptionStatus;
  caps: EntitlementCaps;
  features: EntitlementFeatures;
  usage: EntitlementUsage;
  hasOverrides: boolean;
  /** True when the backend served safe defaults due to an internal error. */
  degraded: boolean;
  /** Stripe billing details — always present; fields are null for free users. */
  subscription: EntitlementSubscription;
  /** Credit wallet — null unless CREDITS_ENABLED is on. */
  credits?: EntitlementCredits | null;
  /**
   * Every context the caller could switch billing to — personal + active
   * seats (Licensing Phase B, spec §5). Present only when `LICENSING_ENABLED`
   * is on; `undefined`/`null` otherwise so pre-licensing payloads are unaffected.
   */
  availableContexts?: BillingContextOption[] | null;
  /**
   * The caller's current billing-context identity (Licensing follow-ups
   * Task 3). Present whenever `LICENSING_ENABLED` is on, independent of
   * `CREDITS_ENABLED` — `undefined`/`null` when licensing is off.
   */
  billingContext?: BillingContext | null;
}

// ---------------------------------------------------------------------------
// Silent payer-switch detection: when the billing context flips org → personal
// WITHOUT the user choosing it (seat suspended/removed, org lapsed), tell them
// once that their personal wallet is now paying. A deliberate switch goes
// through useSetBillingContext, which calls clearRememberedBillingContext()
// first so this never false-fires. Runs inside the queryFn (once per actual
// fetch, deduped across every component using this hook).
// ---------------------------------------------------------------------------

const lastBillingContextKey = (userId: string) => `msanii:lastBillingContext:${userId}`;

/** Forget the remembered context — called on a DELIBERATE switch so the next
 * entitlements fetch records the new context without treating it as a
 * surprise. */
export function clearRememberedBillingContext(userId?: string): void {
  if (!userId) return;
  try {
    localStorage.removeItem(lastBillingContextKey(userId));
  } catch {
    // storage unavailable — nothing to clear
  }
}

function noticePayerSwitch(userId: string, ents: Entitlements): void {
  try {
    const key = lastBillingContextKey(userId);
    const current =
      ents.billingContext?.type === "org"
        ? { type: "org", orgId: ents.billingContext.orgId, orgName: ents.billingContext.orgName }
        : { type: "personal" as const };
    const raw = localStorage.getItem(key);
    const prev = raw ? (JSON.parse(raw) as { type?: string; orgName?: string }) : null;
    if (prev?.type === "org" && current.type === "personal") {
      toast(
        `You're no longer billing to ${prev.orgName ?? "your organization"} — your usage now comes out of your personal plan.`,
      );
    }
    localStorage.setItem(key, JSON.stringify(current));
  } catch {
    // storage unavailable / bad JSON — skip silently, never block the fetch
  }
}

/**
 * Primary hook — returns the merged entitlements for the current user.
 * Cached for 60s via React Query staleTime.
 */
export function useEntitlements(): UseQueryResult<Entitlements> {
  const { user } = useAuth();
  return useQuery<Entitlements>({
    queryKey: ["entitlements", user?.id],
    queryFn: async () => {
      const ents = await apiFetch<Entitlements>(`${API_URL}/me/entitlements`);
      if (user?.id) noticePayerSwitch(user.id, ents);
      return ents;
    },
    enabled: !!user?.id,
    staleTime: 60_000,
    gcTime: 5 * 60_000,
  });
}

// ---------------------------------------------------------------------------
// Convenience hooks — ALL return { value/allowed, loading, error }.
// Consumers render spinners on loading, polite error toasts on error,
// and only paywall on a confirmed denied result.
// ---------------------------------------------------------------------------

export type CountableResource = "artist" | "project" | "task";

const CAP_FIELD_BY_RESOURCE: Record<CountableResource, keyof EntitlementCaps> = {
  artist: "maxArtists",
  project: "maxProjects",
  task: "maxTasks",
};

export function useCanCreate(
  resource: CountableResource,
  currentCount: number,
): { allowed: boolean; current: number; cap: number; loading: boolean; error: Error | null } {
  const { data, isLoading, error } = useEntitlements();
  if (error) {
    return { allowed: false, current: currentCount, cap: 0, loading: false, error: error as Error };
  }
  if (isLoading || !data) {
    return { allowed: false, current: currentCount, cap: 0, loading: true, error: null };
  }
  const cap = data.caps[CAP_FIELD_BY_RESOURCE[resource]];
  if (cap === -1) return { allowed: true, current: currentCount, cap, loading: false, error: null };
  return { allowed: currentCount < cap, current: currentCount, cap, loading: false, error: null };
}

export type GatedFeature = "zoe" | "oneclick" | "registry";

const FEATURE_FIELD: Record<GatedFeature, keyof EntitlementFeatures> = {
  zoe: "zoeEnabled",
  oneclick: "oneclickEnabled",
  registry: "registryEnabled",
};

export function useCanUseFeature(
  feature: GatedFeature,
): { allowed: boolean; loading: boolean; error: Error | null } {
  const { data, isLoading, error } = useEntitlements();
  if (error) return { allowed: false, loading: false, error: error as Error };
  if (isLoading || !data) return { allowed: false, loading: true, error: null };
  return { allowed: data.features[FEATURE_FIELD[feature]] === true, loading: false, error: null };
}

export function useStorageStatus(): {
  used: number;
  cap: number;
  pct: number;
  nearLimit: boolean;
  loading: boolean;
  error: Error | null;
} {
  const { data, isLoading, error } = useEntitlements();
  if (error) {
    return { used: 0, cap: 0, pct: 0, nearLimit: false, loading: false, error: error as Error };
  }
  if (isLoading || !data) {
    return { used: 0, cap: 0, pct: 0, nearLimit: false, loading: true, error: null };
  }
  const used = data.usage.totalStorageBytes;
  const cap = data.caps.maxStorageBytes;
  if (cap === -1) return { used, cap, pct: 0, nearLimit: false, loading: false, error: null };
  const pct = cap > 0 ? used / cap : 0;
  return { used, cap, pct, nearLimit: pct >= 0.8, loading: false, error: null };
}

/** Raw tier_overrides row shape (returned by GET /admin/users/{id}). */
export interface RawOverride {
  user_id: string;
  max_artists: number | null;
  max_projects: number | null;
  max_tasks: number | null;
  max_storage_bytes: number | null;
  max_split_sheets_per_month: number | null;
  zoe_enabled: boolean | null;
  oneclick_enabled: boolean | null;
  registry_enabled: boolean | null;
  integrations_allowed: string[] | null;
  reason: string | null;
  granted_at: string;
  expires_at: string | null;
}

export interface AdminUserDetailUser {
  id: string;
  email: string | null;
  /** profiles.full_name — null until the user finishes onboarding. */
  name: string | null;
  created_at: string | null;
  is_admin: boolean;
  is_env_admin: boolean;
}

export interface AdminUserDetail {
  user: AdminUserDetailUser;
  entitlements: Entitlements;
  override: RawOverride | null;
}

/**
 * Admin-only hook to fetch any user's full entitlements + identity + raw override.
 * Calls GET /admin/users/{id}; only enabled when userId is set.
 *
 * The raw `override` row is returned alongside merged entitlements so the
 * override-editor can pre-fill with current values (vs starting empty and
 * accidentally clearing existing overrides via incomplete re-submit).
 */
export function useEntitlementsForUser(userId: string | null): UseQueryResult<AdminUserDetail> {
  return useQuery({
    queryKey: ["admin", "users", userId, "detail"],
    queryFn: async () =>
      apiFetch<AdminUserDetail>(`${API_URL}/admin/users/${userId}`),
    enabled: !!userId,
    staleTime: 30_000,
  });
}
