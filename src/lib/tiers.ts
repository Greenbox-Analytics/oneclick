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
