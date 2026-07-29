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
  basic: { monthly: 25, annual: 250 },
  pro: { monthly: 50, annual: 500 },
};

/** "US$25" / "US$20.83" — cents only when the amount has them. */
export function usd(amount: number): string {
  return `US$${Number.isInteger(amount) ? amount : amount.toFixed(2)}`;
}

/** An annual price restated per month, e.g. "US$20.83" for pro. */
export function annualPerMonth(tier: TierKey): string {
  return usd(Math.round((TIER_PRICES[tier].annual / 12) * 100) / 100);
}
