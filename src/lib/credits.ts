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

// ---------------------------------------------------------------------------
// Cost estimator (docs page)
// ---------------------------------------------------------------------------

/**
 * Mirror of the backend charge rule, for the "what will this cost me?" widget.
 *
 * BACKEND SOURCE: subscriptions/service.py::debit_for_action computes
 *   max(base, metered, base + tail)
 * with `metered`/`tail` from subscriptions/ai_pricing.py. The constants below
 * are that module's dials plus the measured cost-per-token from the 2026-08-27
 * calibration against ai_usage_log.
 *
 * This is an ESTIMATE and says so in the UI: real token counts vary with a
 * document's density, and a cache hit costs the base flat. It exists so a user
 * can answer "is a 90-page deal going to blow my month?" without running one.
 * `src/lib/__tests__/credits-estimate.test.ts` pins it to the same page/credit
 * pairs as the Python tests, so the two can't silently drift.
 */
const TOKENS_PER_PAGE = 650;
/** Measured: 4,824 tokens per OneClick run cost $0.0149 across the pipeline. */
const USD_PER_TOKEN = 0.0149 / 4824;
/** CREDIT_MARKUP (3.0) / CREDIT_OVERAGE_USD (0.02). */
const CREDITS_PER_USD = 150;
/** ai_pricing.TAIL_FREE_TOKENS — tokens included before size starts to cost. */
const FREE_TOKENS: Record<CreditAction, number> = {
  oneclick_run: 6_500,
  registry_parse: 6_500,
  zoe_message: 30_000,
  split_sheet: 6_500,
};

/** Actions whose price moves with how much there is to read. */
export const SIZED_ACTIONS: CreditAction[] = ["oneclick_run", "registry_parse"];

/** Canonical display order — usage-card rows and ring segments, docs estimator.
 *  One list so a new action appears everywhere or nowhere. */
export const ACTION_ORDER: CreditAction[] = ["oneclick_run", "registry_parse", "zoe_message", "split_sheet"];

/** Estimated credits for one run of `action` over `pages` pages of documents. */
export function estimateCredits(action: CreditAction, pages: number, base: number): number {
  if (!Number.isFinite(base) || base <= 0) return 0;
  // Split sheets make no LLM call and Zoe isn't measured in pages, so neither
  // has a size input in the UI — they're their base rate, flat.
  if (!SIZED_ACTIONS.includes(action)) return base;

  const tokens = Math.max(0, pages) * TOKENS_PER_PAGE;
  const cost = tokens * USD_PER_TOKEN;
  const metered = cost > 0 ? Math.ceil(cost * CREDITS_PER_USD) : 0;

  const free = FREE_TOKENS[action] ?? 6_500;
  const tail = tokens > free ? Math.ceil(cost * (1 - free / tokens) * CREDITS_PER_USD) : 0;

  return Math.max(base, metered, base + tail);
}

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
