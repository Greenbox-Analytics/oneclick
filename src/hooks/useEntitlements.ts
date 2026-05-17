import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { useAuth } from "@/contexts/AuthContext";
import { API_URL, apiFetch } from "@/lib/apiFetch";

export type Tier = "free" | "pro";
export type SubscriptionStatus = "active" | "canceled" | "past_due" | "trialing";

export interface EntitlementCaps {
  maxArtists: number;
  maxProjects: number;
  maxTasks: number;
  maxStorageBytes: number;
  maxSplitSheetsPerMonth: number;
  maxOneclickRunsPerMonth: number;
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
}

/**
 * Primary hook — returns the merged entitlements for the current user.
 * Cached for 60s via React Query staleTime.
 */
export function useEntitlements(): UseQueryResult<Entitlements> {
  const { user } = useAuth();
  return useQuery<Entitlements>({
    queryKey: ["entitlements", user?.id],
    queryFn: async () => apiFetch<Entitlements>(`${API_URL}/me/entitlements`),
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

export type IntegrationName = "google_drive" | "slack" | "notion";

export function useIntegrationAllowed(
  name: IntegrationName,
): { allowed: boolean; loading: boolean; error: Error | null } {
  const { data, isLoading, error } = useEntitlements();
  if (error) return { allowed: false, loading: false, error: error as Error };
  if (isLoading || !data) return { allowed: false, loading: true, error: null };
  return { allowed: data.features.integrationsAllowed.includes(name), loading: false, error: null };
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
