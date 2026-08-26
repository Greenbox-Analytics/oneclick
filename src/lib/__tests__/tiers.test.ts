import { describe, expect, it } from "vitest";
import { orgNoun, orgNounCap } from "@/lib/tiers";

describe("orgNoun / orgNounCap", () => {
  it("calls a self-serve org a team", () => {
    expect(orgNoun("self_serve")).toBe("team");
    expect(orgNounCap("self_serve")).toBe("Team");
  });

  it("calls an enterprise org, and a pre-migration undefined kind, an organization", () => {
    expect(orgNoun("enterprise")).toBe("organization");
    expect(orgNoun(undefined)).toBe("organization");
    expect(orgNoun(null)).toBe("organization");
    expect(orgNounCap(undefined)).toBe("Organization");
  });
});

describe("usd", () => {
  it("formats whole dollars, cents, and sub-cent rates", async () => {
    const { usd } = await import("@/lib/tiers");
    expect(usd(30)).toBe("US$30");
    expect(usd(41.67)).toBe("US$41.67");
    expect(usd(0.5)).toBe("US$0.50");
    expect(usd(0.025)).toBe("US$0.025");
  });
});
