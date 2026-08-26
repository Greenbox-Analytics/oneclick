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

/**
 * Per-action display label + ring/bar colour (usage card + /teams member view).
 *
 * `plural` exists so the credit-purchase card can say what a bundle buys
 * ("≈40 OneClick runs") in the SAME words the usage card names the tool. Two
 * label sets would drift the moment a tool is renamed.
 */
export const TOOL_META: Record<CreditAction, { label: string; plural: string; color: string }> = {
  oneclick_run: { label: "OneClick run", plural: "OneClick runs", color: "var(--t-oneclick)" },
  registry_parse: { label: "Registry parse", plural: "Registry parses", color: "var(--t-registry)" },
  zoe_message: { label: "Zoe message", plural: "Zoe messages", color: "var(--t-zoe)" },
  split_sheet: { label: "Split sheet", plural: "split sheets", color: "var(--t-split)" },
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

/**
 * Per-action credit prices as the API serves them (camelCase). Structurally
 * what `EntitlementCredits["prices"]` and GET /billing/credit-packs both
 * return — one type, because the backend builds both blocks from the same
 * `credit_prices` rows.
 */
export interface ToolCreditPrices {
  zoeMessage?: number | null;
  oneclickRun?: number | null;
  registryParse?: number | null;
  splitSheet?: number | null;
}

/** API key -> CreditAction. Order is the order the summary reads in: the
 * headline tool first, then by how much work a credit buys. */
const PRICE_KEY_TO_ACTION: [keyof ToolCreditPrices, CreditAction][] = [
  ["oneclickRun", "oneclick_run"],
  ["zoeMessage", "zoe_message"],
  ["splitSheet", "split_sheet"],
  ["registryParse", "registry_parse"],
];

/**
 * What a given number of credits typically buys, e.g.
 * "≈40 OneClick runs · ≈240 Zoe messages · ≈60 split sheets · ≈40 Registry parses".
 *
 * APPROXIMATE, and deliberately so. Charges are METERED off real token spend;
 * `credit_prices` is the base rate, and a run that costs less than its base
 * still charges the base — so these counts are a floor the product can only
 * overdeliver on, never a promise it can miss.
 *
 * Derived from live prices rather than hardcoded for the same reason
 * `creditStanding` exists: a copy of the numbers is a copy that goes stale the
 * next time base rates move (they moved on 2026-08-19).
 *
 * Returns `null` when nothing can be quoted — no prices, or an amount too
 * small to buy a single run of anything — so callers render no subtitle at all
 * rather than an empty or zeroed one.
 */
export function usageSummary(credits: number, prices?: ToolCreditPrices | null): string | null {
  if (!prices || !Number.isFinite(credits) || credits <= 0) return null;

  const parts = PRICE_KEY_TO_ACTION.flatMap(([priceKey, action]) => {
    const price = prices[priceKey];
    if (price == null || price <= 0) return [];
    const runs = Math.floor(credits / price);
    if (runs < 1) return [];
    const meta = TOOL_META[action];
    return [`≈${runs.toLocaleString()} ${runs === 1 ? meta.label : meta.plural}`];
  });

  return parts.length > 0 ? parts.join(" · ") : null;
}
