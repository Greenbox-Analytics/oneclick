import { describe, it, expect } from "vitest";
import { currencySymbol, formatCurrency } from "../currency";

describe("formatCurrency", () => {
  it("renders USD as US$", () => {
    expect(formatCurrency(1234.5, "USD")).toBe("US$1,234.50");
    expect(formatCurrency(1234.5)).toBe("US$1,234.50");
  });

  it("accepts lowercase codes (Stripe style)", () => {
    expect(formatCurrency(25, "usd")).toBe("US$25.00");
    expect(formatCurrency(25, "eur")).toBe("€25.00");
  });

  it("renders other known currencies with their symbols", () => {
    expect(formatCurrency(10, "CAD")).toBe("CA$10.00");
    expect(formatCurrency(10, "AUD")).toBe("A$10.00");
    expect(formatCurrency(10, "GBP")).toBe("£10.00");
  });

  it("falls back to a code suffix for unknown currencies", () => {
    expect(formatCurrency(10, "KES")).toBe("10.00 KES");
  });

  it("treats nullish amounts as zero", () => {
    expect(formatCurrency(undefined as unknown as number)).toBe("US$0.00");
  });
});

describe("currencySymbol", () => {
  it("returns the display symbol for known codes", () => {
    expect(currencySymbol("USD")).toBe("US$");
    expect(currencySymbol("cad")).toBe("CA$");
    expect(currencySymbol("XYZ")).toBeUndefined();
  });
});
