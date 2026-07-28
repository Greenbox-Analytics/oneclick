/**
 * Canonical tier keys vs display labels (spec 2026-07-19 §2).
 * DB keys are PERMANENT: "pro" = the $25 plan LABELED "Basic";
 * "pro_max" = the $50 plan LABELED "Pro". Never compare a label,
 * never display a raw key. Enterprise has no DB key (org seats, Phase B).
 */
export type TierKey = "free" | "pro" | "pro_max";

export const TIER_LABELS: Record<TierKey, string> = {
  free: "Free",
  pro: "Basic",
  pro_max: "Pro",
};

export const ENTERPRISE_LABEL = "Enterprise";

export function tierLabel(tier: string | null | undefined): string {
  if (!tier) return "Free";
  return TIER_LABELS[tier as TierKey] ?? tier;
}

export function isPaidTier(tier: string | null | undefined): boolean {
  return tier === "pro" || tier === "pro_max";
}

/**
 * List prices in USD — the ONLY place the frontend states them. Must match the
 * Stripe prices behind STRIPE_PRICE_* / STRIPE_PRICE_PRO_MAX_*; change together.
 */
export const TIER_PRICES: Record<TierKey, { monthly: number; annual: number }> = {
  free: { monthly: 0, annual: 0 },
  pro: { monthly: 25, annual: 250 },
  pro_max: { monthly: 50, annual: 500 },
};

/** "US$25" / "US$20.83" — cents only when the amount has them. */
export function usd(amount: number): string {
  return `US$${Number.isInteger(amount) ? amount : amount.toFixed(2)}`;
}

/** An annual price restated per month, e.g. "US$20.83" for pro. */
export function annualPerMonth(tier: TierKey): string {
  return usd(Math.round((TIER_PRICES[tier].annual / 12) * 100) / 100);
}
