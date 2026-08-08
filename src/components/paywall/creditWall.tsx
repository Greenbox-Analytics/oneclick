// src/components/paywall/creditWall.tsx
// Shared helper for the org credit-wall (402) UX: derive the structured
// fields off an ApiError's `detail`. Co-located so the derivation stays in
// one place across the paywall card, the OneClick error alert, and the
// AddWork parse queue.

export interface CreditWallInfo {
  /** Denial came from an org billing context (the shared pool). */
  managedByOrg: boolean;
  /** True when the MEMBER hit their own monthly limit (remedy: ask for a
   * raise). False on a dry pool, where only an admin buying credits helps —
   * two different walls that must not offer the same CTA. */
  capReached: boolean;
  /** Where "Ask for a higher limit" navigates (member request form) when present. */
  requestUrl?: string;
}

/**
 * Derive the org credit-wall fields from an ApiError's structured
 * `detail`. Mirrors the 402 shape from subscriptions/enforcement.py — every
 * field is presence-checked so a legacy plain-string detail (or any non-object)
 * yields an all-false/undefined result.
 */
export function parseCreditWallDetail(detail: unknown): CreditWallInfo {
  const d = (detail && typeof detail === "object" ? detail : {}) as Record<string, unknown>;
  const managedByOrg = d.managedByOrg === true;
  return {
    managedByOrg,
    capReached: managedByOrg && d.capReached === true,
    requestUrl: managedByOrg && typeof d.requestUrl === "string" ? d.requestUrl : undefined,
  };
}
