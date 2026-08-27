import { describe, it, expect } from "vitest";
import { estimateCredits } from "@/lib/credits";

/**
 * The docs estimator re-implements the backend charge rule in TS, so the only
 * thing keeping it honest is this file. Every pair below is also asserted in
 * `src/backend/tests/test_credits_service.py::TestSizeTail` — if a backend dial
 * moves (base rate, CREDIT_MARKUP, the token allowance) and the frontend copy
 * isn't updated with it, these fail and the docs stop lying before a user sees
 * a number that doesn't match their invoice.
 */
describe("estimateCredits", () => {
  describe("OneClick — the curve published in /docs", () => {
    // [pages, credits] — must match the table in Documentation.tsx exactly.
    const CURVE: [number, number][] = [
      [10, 30],
      [15, 32],
      [30, 37],
      [60, 46],
      [100, 58],
    ];

    it.each(CURVE)("%i pages costs %i credits", (pages, expected) => {
      expect(estimateCredits("oneclick_run", pages, 30)).toBe(expected);
    });

    it("charges exactly the base up to the ~10-page allowance", () => {
      for (const pages of [0, 1, 5, 9, 10]) {
        expect(estimateCredits("oneclick_run", pages, 30)).toBe(30);
      }
    });

    it("never decreases as the document grows", () => {
      const charges = [1, 10, 11, 25, 50, 120, 300].map((p) => estimateCredits("oneclick_run", p, 30));
      expect(charges).toEqual([...charges].sort((a, b) => a - b));
    });

    it("actually distinguishes sizes past the allowance", () => {
      // The whole point of the size tail: a big run must not cost what a small
      // one does. This is the assertion that fails if the tail is removed.
      const small = estimateCredits("oneclick_run", 10, 30);
      const large = estimateCredits("oneclick_run", 60, 30);
      expect(large).toBeGreaterThan(small);
    });
  });

  describe("flat-priced actions", () => {
    it("prices split sheets at the base regardless of pages", () => {
      expect(estimateCredits("split_sheet", 1, 20)).toBe(20);
      expect(estimateCredits("split_sheet", 500, 20)).toBe(20);
    });

    it("prices a Zoe message at the base — it isn't measured in pages", () => {
      expect(estimateCredits("zoe_message", 200, 5)).toBe(5);
    });
  });

  describe("guards", () => {
    it("returns 0 when the price hasn't loaded yet", () => {
      // prices arrive from the API, so the component renders before they land.
      expect(estimateCredits("oneclick_run", 30, 0)).toBe(0);
      expect(estimateCredits("oneclick_run", 30, NaN)).toBe(0);
    });

    it("treats negative pages as zero rather than crediting the user", () => {
      expect(estimateCredits("oneclick_run", -50, 30)).toBe(30);
    });
  });
});
