import { describe, it, expect } from "vitest";
import {
  clampPct,
  splitTotals,
  splitsOverCap,
  SPLIT_TOTAL_MARGIN,
} from "@/components/registry/splitsShared";

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

describe("splitsOverCap", () => {
  it("is not over cap when totals are exactly 100%", () => {
    expect(splitsOverCap([{ master: 100, publishing: 100 }])).toEqual({
      master: false,
      publishing: false,
      over: false,
    });
  });

  it("allows totals within the 0.5% rounding margin (100.5%)", () => {
    expect(splitsOverCap([{ master: 100.5, publishing: 100.5 }])).toEqual({
      master: false,
      publishing: false,
      over: false,
    });
  });

  it("flags totals just past the margin (100.6%)", () => {
    expect(splitsOverCap([{ master: 100.6, publishing: 100.6 }])).toEqual({
      master: true,
      publishing: true,
      over: true,
    });
  });

  it("flags master and publishing independently", () => {
    expect(splitsOverCap([{ master: 101, publishing: 100 }])).toEqual({
      master: true,
      publishing: false,
      over: true,
    });
    expect(splitsOverCap([{ master: 100, publishing: 101 }])).toEqual({
      master: false,
      publishing: true,
      over: true,
    });
  });

  it("sums across rows before comparing to the cap", () => {
    expect(
      splitsOverCap([
        { master: 60, publishing: 50 },
        { master: 45, publishing: 45 },
      ])
    ).toEqual({ master: true, publishing: false, over: true });
  });

  it("honors a custom margin argument", () => {
    // 102% is under cap only when a 2-point margin is passed explicitly.
    expect(splitsOverCap([{ master: 102, publishing: 102 }], 2).over).toBe(false);
    expect(splitsOverCap([{ master: 102, publishing: 102 }]).over).toBe(true);
  });

  it("defaults the margin to SPLIT_TOTAL_MARGIN", () => {
    expect(SPLIT_TOTAL_MARGIN).toBe(0.5);
    const rows = [{ master: 100.4, publishing: 100 }];
    expect(splitsOverCap(rows)).toEqual(splitsOverCap(rows, SPLIT_TOTAL_MARGIN));
    expect(splitsOverCap(rows).over).toBe(false);
  });
});
