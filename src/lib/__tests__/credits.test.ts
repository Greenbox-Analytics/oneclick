import { describe, expect, it } from "vitest";
import { creditStanding, usageSummary } from "@/lib/credits";
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

describe("usageSummary", () => {
  // Base rates as of 20260819000001_credit_base_rates.sql.
  const PRICES = { zoeMessage: 5, oneclickRun: 30, registryParse: 30, splitSheet: 20 };

  it("quotes what a bundle typically buys, headline tool first", () => {
    // pack_1200 — the ladder was sized so these land on round numbers.
    expect(usageSummary(1200, PRICES)).toBe(
      "≈40 OneClick runs · ≈240 Zoe messages · ≈60 split sheets · ≈40 Registry parses"
    );
  });

  it("floors — never promises a run the credits can't cover", () => {
    // pack_4000: 4000 / 30 = 133.33 runs. Quoting 134 would oversell.
    expect(usageSummary(4000, PRICES)).toContain("≈133 OneClick runs");
  });

  it("drops a tool the amount can't afford even once", () => {
    // 25 credits buys Zoe messages and one split sheet, but no 30-credit run.
    const summary = usageSummary(25, PRICES);
    expect(summary).not.toContain("OneClick");
    expect(summary).not.toContain("Registry");
    expect(summary).toContain("≈5 Zoe messages");
    expect(summary).toContain("≈1 Split sheet");
  });

  it("singularises using the same label the usage card shows", () => {
    expect(usageSummary(30, { oneclickRun: 30 })).toBe("≈1 OneClick run");
  });

  it("returns null rather than quoting zeros", () => {
    // Nothing to render beats "≈0 OneClick runs" on a purchase card.
    expect(usageSummary(4, PRICES)).toBeNull();
    expect(usageSummary(0, PRICES)).toBeNull();
    expect(usageSummary(-100, PRICES)).toBeNull();
  });

  it("returns null when prices are missing or unusable", () => {
    // The endpoint omits `prices` entirely when credit_prices reads empty.
    expect(usageSummary(1200, undefined)).toBeNull();
    expect(usageSummary(1200, null)).toBeNull();
    expect(usageSummary(1200, {})).toBeNull();
    expect(usageSummary(1200, { oneclickRun: 0, zoeMessage: null })).toBeNull();
  });

  it("skips only the tool whose price is unusable", () => {
    expect(usageSummary(1200, { oneclickRun: 30, zoeMessage: null })).toBe("≈40 OneClick runs");
  });
});
