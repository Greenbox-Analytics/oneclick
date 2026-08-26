/**
 * Canonical tier keys and their display labels. Keys match the labels since
 * 20260728000001_rename_tier_keys.sql ("pro" was the $25 plan labeled "Basic",
 * "pro_max" the $50 plan labeled "Pro" — a translation table nobody could keep
 * straight). Still go through tierLabel() rather than printing a raw key.
 * Enterprise has no DB key (org seats, Phase B).
 */
export type TierKey = "free" | "basic" | "pro";

export const TIER_LABELS: Record<TierKey, string> = {
  free: "Free",
  basic: "Basic",
  pro: "Pro",
};

export const ENTERPRISE_LABEL = "Enterprise";

/** UI noun for an org row: self-serve rows are "team", enterprise (and
 * pre-migration undefined kind) stay "organization" (2026-08-16 rebrand —
 * backend names/routes are unchanged, this is copy only). */
export const orgNoun = (kind?: string | null) => (kind === "self_serve" ? "team" : "organization");
export const orgNounCap = (kind?: string | null) => (kind === "self_serve" ? "Team" : "Organization");

export function tierLabel(tier: string | null | undefined): string {
  if (!tier) return "Free";
  return TIER_LABELS[tier as TierKey] ?? tier;
}

export function isPaidTier(tier: string | null | undefined): boolean {
  return tier === "basic" || tier === "pro";
}

/**
 * List prices in USD — the ONLY place the frontend states them. Must match the
 * Stripe prices behind STRIPE_PRICE_* / STRIPE_PRICE_PRO_MAX_*; change together.
 */
export const TIER_PRICES: Record<TierKey, { monthly: number; annual: number }> = {
  free: { monthly: 0, annual: 0 },
  basic: { monthly: 30, annual: 300 },
  pro: { monthly: 50, annual: 500 },
};

/**
 * Pro team-storage pay-as-you-go rate, USD per GB per month past the tier's
 * team_storage_bytes. Must match the backend env TEAM_STORAGE_OVERAGE_USD_PER_GB
 * (orgs/storage_guard.py, default 0.025) — change together.
 */
export const TEAM_STORAGE_OVERAGE_USD_PER_GB = 0.025;

/** "US$30" / "US$41.67" / "US$0.025" — no decimals for whole dollars, otherwise as many as the amount needs (min 2, max 3). */
export function usd(amount: number): string {
  if (Number.isInteger(amount)) return `US$${amount}`;
  return `US$${amount.toLocaleString("en-US", { useGrouping: false, minimumFractionDigits: 2, maximumFractionDigits: 3 })}`;
}

/** An annual price restated per month, e.g. "US$41.67" for pro. */
export function annualPerMonth(tier: TierKey): string {
  return usd(Math.round((TIER_PRICES[tier].annual / 12) * 100) / 100);
}
