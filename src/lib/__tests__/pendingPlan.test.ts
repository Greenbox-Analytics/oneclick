import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  PENDING_PLAN_TTL_MS,
  clearPendingPlan,
  planLabel,
  readPendingPlan,
  stashPendingPlan,
} from "../pendingPlan";

const NOW = new Date("2026-09-09T12:00:00Z").getTime();
const KEY = "msanii_pending_plan.u1";

beforeEach(() => {
  localStorage.clear();
  vi.useFakeTimers({ now: NOW, toFake: ["Date"] });
});

afterEach(() => {
  vi.useRealTimers();
});

describe("pending plan stash", () => {
  it("is empty by default", () => {
    expect(readPendingPlan("u1")).toBeNull();
    expect(readPendingPlan(null)).toBeNull();
    expect(readPendingPlan(undefined)).toBeNull();
  });

  it("stores and reads the plan the user started buying", () => {
    stashPendingPlan("u1", "basic_monthly");
    expect(readPendingPlan("u1")).toEqual({
      plan: "basic_monthly",
      startedAt: new Date(NOW).toISOString(),
    });
  });

  it("is scoped per user so a shared browser never leaks one user's intent to another", () => {
    stashPendingPlan("u1", "pro_annual");
    expect(readPendingPlan("u2")).toBeNull();
    expect(readPendingPlan("u1")?.plan).toBe("pro_annual");
  });

  it("ignores garbage, unknown plans, and rows with no start time", () => {
    localStorage.setItem(KEY, "{not json");
    expect(readPendingPlan("u1")).toBeNull();

    localStorage.setItem(KEY, JSON.stringify({ plan: "enterprise", startedAt: new Date(NOW).toISOString() }));
    expect(readPendingPlan("u1")).toBeNull();

    localStorage.setItem(KEY, JSON.stringify({ plan: "basic_monthly" }));
    expect(readPendingPlan("u1")).toBeNull();
  });

  it("survives up to the TTL, then expires and removes its own key", () => {
    stashPendingPlan("u1", "basic_annual");

    vi.setSystemTime(NOW + PENDING_PLAN_TTL_MS - 1);
    expect(readPendingPlan("u1")?.plan).toBe("basic_annual");

    vi.setSystemTime(NOW + PENDING_PLAN_TTL_MS + 1);
    expect(readPendingPlan("u1")).toBeNull();
    expect(localStorage.getItem(KEY)).toBeNull();
  });

  it("clears", () => {
    stashPendingPlan("u1", "basic_monthly");
    clearPendingPlan("u1");
    expect(readPendingPlan("u1")).toBeNull();
    // A missing user id is a no-op, never a throw.
    expect(() => clearPendingPlan(null)).not.toThrow();
  });

  it("re-stashing refreshes the start time", () => {
    stashPendingPlan("u1", "basic_monthly");
    vi.setSystemTime(NOW + 1000);
    stashPendingPlan("u1", "basic_monthly");
    expect(readPendingPlan("u1")?.startedAt).toBe(new Date(NOW + 1000).toISOString());
  });
});

describe("planLabel", () => {
  it("names the tier and period for banner copy", () => {
    expect(planLabel("basic_monthly")).toBe("Basic (monthly)");
    expect(planLabel("pro_annual")).toBe("Pro (annual)");
  });
});
