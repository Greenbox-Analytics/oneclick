import { describe, it, expect } from "vitest";
import { clampPct, splitTotals } from "@/components/registry/splitsShared";

describe("clampPct", () => {
  it("parses a plain number string", () => {
    expect(clampPct("30")).toBe(30);
  });
  it("strips a trailing percent sign and spaces", () => {
    expect(clampPct("30 %")).toBe(30);
  });
  it("returns 0 for empty or non-numeric input", () => {
    expect(clampPct("")).toBe(0);
    expect(clampPct("abc")).toBe(0);
  });
  it("clamps above 100 down to 100", () => {
    expect(clampPct("150")).toBe(100);
  });
  it("accepts a numeric argument", () => {
    expect(clampPct(42)).toBe(42);
  });
});

describe("splitTotals", () => {
  it("sums master and publishing across rows", () => {
    expect(
      splitTotals([
        { master: 30, publishing: 25 },
        { master: 70, publishing: 75 },
      ])
    ).toEqual({ master: 100, publishing: 100 });
  });
  it("treats missing percentages as zero", () => {
    expect(splitTotals([{ master: 40 }, { publishing: 60 }])).toEqual({
      master: 40,
      publishing: 60,
    });
  });
  it("returns zeros for an empty list", () => {
    expect(splitTotals([])).toEqual({ master: 0, publishing: 0 });
  });
});
