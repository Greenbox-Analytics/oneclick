import type { Entitlements, EntitlementCredits, OrgBillingContext } from "@/hooks/useEntitlements";
import type { CreditAction } from "@/hooks/useCreditUsage";

/**
 * The org whose pool is paying for the caller, or nullish in personal context.
 * `billingContext` is the canonical signal (present whenever LICENSING_ENABLED
 * is on, even with credits off); `credits.managedByOrg` is the back-compat
 * fallback for payloads that predate it.
 */
export const orgContext = (ent: Entitlements | null | undefined): OrgBillingContext | null | undefined =>
  ent?.billingContext?.type === "org" ? ent.billingContext : ent?.credits?.managedByOrg;

/** Per-action display label + ring/bar colour (usage card + /teams member view). */
export const TOOL_META: Record<CreditAction, { label: string; color: string }> = {
  oneclick_run: { label: "OneClick run", color: "var(--t-oneclick)" },
  registry_parse: { label: "Registry parse", color: "var(--t-registry)" },
  zoe_message: { label: "Zoe message", color: "var(--t-zoe)" },
  split_sheet: { label: "Split sheet", color: "var(--t-split)" },
};

/**
 * What a user has left, and out of what.
 *
 * `null` means there is no number to show — the caller is a plain member of an
 * org with no monthly cap, so they draw straight from a pool whose balance is
 * admin-only. Callers must render prose ("pulling from the org credits pool")
 * rather than a figure. It is NOT the same as zero, and treating it as zero
 * shows "0 credits left" to someone who has plenty.
 *
 * This lives here because the header ticker, the inline chip and the billing
 * usage card all need the same answer, and three copies of the fallback chain
 * is three chances to leak the pool or to render `0` for a redacted balance.
 */
export interface CreditStanding {
  remaining: number;
  /** Denominator for meters — the cap in org context, else grant + reserve. */
  total: number;
}

export function creditStanding(credits: EntitlementCredits | null | undefined): CreditStanding | null {
  if (!credits) return null;

  // An org member's ceiling is their cap, whatever the pool holds.
  if (credits.memberCap != null) {
    return {
      remaining: Math.max(0, credits.memberCap - (credits.memberCapUsed ?? 0)),
      total: credits.memberCap,
    };
  }

  // No cap: fall back to the wallet. In org context that balance is redacted
  // for non-admins (null) — no cap AND no visible pool means no number exists.
  if (credits.balance == null) return null;

  return {
    remaining: credits.balance,
    total: credits.monthlyGrant + (credits.reserveBalance ?? 0),
  };
}

/** Copy for the no-number case, so every surface words it identically. */
export const POOL_ONLY_LABEL = "Pulling from org credits pool";
