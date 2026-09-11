import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fmtDaysLeft } from "@/lib/utils";

// Local-time constructors throughout: fmtDaysLeft counts CALENDAR days in the
// viewer's zone, the same zone fmtDate renders the date in beside it.
describe("fmtDaysLeft", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 8, 28, 12, 0, 0)); // Sep 28, 2026, noon
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("counts calendar days, not 24-hour blocks", () => {
    // Oct 10 at 08:00 is 11 days 20 hours away — still 12 days on the calendar.
    expect(fmtDaysLeft(new Date(2026, 9, 10, 8, 0, 0).toISOString())).toBe("12 days left");
  });

  it("is singular on the last full day", () => {
    expect(fmtDaysLeft(new Date(2026, 8, 29, 1, 0, 0).toISOString())).toBe("1 day left");
  });

  it("says today on the end date itself", () => {
    expect(fmtDaysLeft(new Date(2026, 8, 28, 23, 59, 0).toISOString())).toBe("today");
  });

  it("is empty once the date has passed", () => {
    expect(fmtDaysLeft(new Date(2026, 8, 27, 23, 59, 0).toISOString())).toBe("");
  });

  it("is empty for nothing and for garbage", () => {
    expect(fmtDaysLeft(null)).toBe("");
    expect(fmtDaysLeft(undefined)).toBe("");
    expect(fmtDaysLeft("")).toBe("");
    expect(fmtDaysLeft("not a date")).toBe("");
  });
});
