/**
 * Remembers which paid plan a user started buying, so an abandoned Stripe
 * Checkout can be resumed from wherever they land next.
 *
 * Both checkout entry points (the onboarding plan step and /pricing) write it
 * right before handing the browser to Stripe. Cleared by:
 *   - the success return (useCheckoutReturn) ON ARRIVAL — reaching the
 *     success URL means Checkout completed, and a late webhook must never
 *     leave a "Finish upgrading" button around that would start a SECOND
 *     paid checkout;
 *   - an explicit Free choice (onboarding, /pricing);
 *   - the resume banner's dismiss;
 *   - any reader that sees a paid tier in entitlements.
 *
 * localStorage, unlike pendingInvite's sessionStorage: closing the Stripe tab
 * is precisely the case this exists for, so it has to outlive the tab. Keyed
 * per user (like onboardingCache) so a shared browser never shows one
 * person's intent to another. Self-expires after PENDING_PLAN_TTL_MS.
 *
 * This is memory of INTENT only. Nothing reads it to decide what a user may
 * do — entitlements always come from /me/entitlements — and its only side
 * effect is re-creating a Checkout through the existing endpoint.
 */
import type { CheckoutPlan } from "@/hooks/useBilling";
import { tierLabel } from "@/lib/tiers";

const PREFIX = "msanii_pending_plan.";

/** How long an abandoned checkout stays resumable. Long enough for "after
 * payday", short enough not to nag someone who decided against it. */
export const PENDING_PLAN_TTL_MS = 7 * 24 * 60 * 60 * 1000;

const PLANS: readonly string[] = ["basic_monthly", "basic_annual", "pro_monthly", "pro_annual"];

export interface PendingPlan {
  plan: CheckoutPlan;
  /** ISO timestamp of when the checkout was started (or last resumed). */
  startedAt: string;
}

const keyFor = (userId: string) => `${PREFIX}${userId}`;

const isPlan = (value: unknown): value is CheckoutPlan => typeof value === "string" && PLANS.includes(value);

export function readPendingPlan(userId: string | null | undefined): PendingPlan | null {
  if (!userId) return null;
  try {
    const raw = localStorage.getItem(keyFor(userId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PendingPlan>;
    if (!isPlan(parsed.plan)) return null;
    const started = typeof parsed.startedAt === "string" ? Date.parse(parsed.startedAt) : NaN;
    if (Number.isNaN(started) || Date.now() - started > PENDING_PLAN_TTL_MS) {
      // Expired (or unreadable): drop it so the key doesn't linger forever.
      localStorage.removeItem(keyFor(userId));
      return null;
    }
    return { plan: parsed.plan, startedAt: parsed.startedAt as string };
  } catch {
    return null;
  }
}

export function stashPendingPlan(userId: string, plan: CheckoutPlan): void {
  try {
    const entry: PendingPlan = { plan, startedAt: new Date().toISOString() };
    localStorage.setItem(keyFor(userId), JSON.stringify(entry));
  } catch {
    /* localStorage unavailable — the checkout still proceeds, just without a resume memory */
  }
}

export function clearPendingPlan(userId: string | null | undefined): void {
  if (!userId) return;
  try {
    localStorage.removeItem(keyFor(userId));
  } catch {
    /* ignore */
  }
}

/** "Basic (monthly)" — how banner and button copy name the plan. */
export function planLabel(plan: CheckoutPlan): string {
  const [tier, period] = plan.split("_");
  return `${tierLabel(tier)} (${period})`;
}
