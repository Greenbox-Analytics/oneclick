import { describe, expect, it } from "vitest";
import { creditStanding } from "@/lib/credits";
import type { EntitlementCredits } from "@/hooks/useEntitlements";

/** A personal wallet — nothing redacted. */
function personal(over: Partial<EntitlementCredits> = {}): EntitlementCredits {
  return {
    balance: 120,
    bundleBalance: 100,
    reserveBalance: 20,
    monthlyGrant: 150,
    overageThisPeriod: 0,
    overageEnabled: false,
    overageUsdPerCredit: 0.02,
    overageCapCredits: null,
    periodEnd: null,
    prices: { zoeMessage: 3, oneclickRun: 21, registryParse: 12, splitSheet: 20 },
    ...over,
  };
}

/** An org member: the backend redacts the shared pool to null for non-admins. */
function orgMember(over: Partial<EntitlementCredits> = {}): EntitlementCredits {
  return personal({
    balance: null,
    bundleBalance: null,
    reserveBalance: null,
    monthlyGrant: 0,
    ...over,
  });
}

describe("creditStanding", () => {
  it("uses the wallet balance in personal context", () => {
    expect(creditStanding(personal())).toEqual({ remaining: 120, total: 170 });
  });

  it("uses the member's cap in org context, not the pool", () => {
    // Admin: pool visible at 5000, but their own ceiling is what bounds them.
    const admin = personal({ balance: 5000, memberCap: 300, memberCapUsed: 120 });
    expect(creditStanding(admin)).toEqual({ remaining: 180, total: 300 });
  });

  it("gives a capped member their limit even with the pool hidden", () => {
    const m = orgMember({ memberCap: 300, memberCapUsed: 275 });
    expect(creditStanding(m)).toEqual({ remaining: 25, total: 300 });
  });

  it("never reports negative remaining when a member overshoots their cap", () => {
    // Over-cap debits are recorded, never rejected (charge-on-success).
    const m = orgMember({ memberCap: 300, memberCapUsed: 410 });
    expect(creditStanding(m)?.remaining).toBe(0);
  });

  it("returns null for an uncapped member whose pool is redacted", () => {
    // THE case this helper exists for: no cap and no visible pool means there
    // is no number. Callers must render prose, not a figure.
    expect(creditStanding(orgMember({ memberCap: null }))).toBeNull();
  });

  it("does not confuse a redacted pool with an empty one", () => {
    // A member with a hidden pool must not render as "0 credits left" — that
    // would send them chasing an admin who has plenty.
    expect(creditStanding(orgMember())).toBeNull();
    expect(creditStanding(personal({ balance: 0, monthlyGrant: 0, reserveBalance: 0 }))).toEqual({
      remaining: 0,
      total: 0,
    });
  });

  it("shows an uncapped ADMIN the pool, since they may see it", () => {
    const admin = personal({ balance: 5000, reserveBalance: 5000, monthlyGrant: 0, memberCap: null });
    expect(creditStanding(admin)).toEqual({ remaining: 5000, total: 5000 });
  });

  it("returns null while entitlements are still loading", () => {
    expect(creditStanding(undefined)).toBeNull();
    expect(creditStanding(null)).toBeNull();
  });
});
