import { describe, it, expect } from "vitest";
import { ApiError, apiErrorFromBody } from "@/lib/apiFetch";
import { parseCreditWallDetail } from "@/components/paywall/creditWall";

describe("split-sheet 402 handling", () => {
  it("apiErrorFromBody keeps a structured detail readable", () => {
    const body = {
      detail: {
        reason: "Not enough credits. A split sheet costs 20.",
        price: 20,
        upgradeRequired: false,
        overageAvailable: true,
      },
    };
    const err = apiErrorFromBody(body, 402, "Upgrade required");
    expect(err.message).toBe("Not enough credits. A split sheet costs 20.");
    expect(err.message).not.toContain("[object Object]");
    expect(err.detail).toEqual(body.detail);
  });

  it("hand-built ApiError loses the detail — the bug this task fixes", () => {
    const detail = { reason: "Not enough credits.", price: 20 };
    // Reproduces the old SplitSheet.tsx line verbatim.
    const err = new ApiError(detail as unknown as string, 402);
    expect(String(err.message)).toContain("[object Object]");
    expect(err.detail).toBeUndefined();
  });

  it("parseCreditWallDetail flags a credit wall and preserves org fields", () => {
    const cw = parseCreditWallDetail({
      reason: "Your team is out of credits.",
      price: 20,
      managedByOrg: true,
      capReached: true,
      requestUrl: "/teams",
    });
    expect(cw.isCreditWall).toBe(true);
    expect(cw.managedByOrg).toBe(true);
    expect(cw.capReached).toBe(true);
    expect(cw.requestUrl).toBe("/teams");
  });

  it("a plain-string cap detail is not a credit wall", () => {
    const cw = parseCreditWallDetail("You've used your 5 split sheet(s) for this period.");
    expect(cw.isCreditWall).toBe(false);
    expect(cw.managedByOrg).toBe(false);
  });

  it("a price:0 outage/config deny is NOT a credit wall", () => {
    // check_credits emits a structured detail with price 0 for the unseeded-action
    // config error and for the degraded deny. Neither is fixed by buying credits.
    for (const reason of [
      "This action isn't set up for credits yet. Please contact support.",
      "We couldn't check your plan just now — please try again in a moment.",
    ]) {
      expect(parseCreditWallDetail({ reason, price: 0 }).isCreditWall).toBe(false);
    }
  });
});
