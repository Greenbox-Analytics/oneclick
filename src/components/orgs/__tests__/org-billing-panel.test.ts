import { describe, it, expect } from "vitest";
import { adminName, memberEmail, ledgerLabel } from "../OrgBillingPanel";
import type { OrgDetail, OrgLedgerEntry, OrgSeatUsage } from "@/hooks/useOrgs";

// Pure-function tests only — no rendering, no hook mocking. Covers the
// coordinator's spec-review fix: pool activity / monthly top-up rows must
// resolve to an identity when possible, and fall back to the generic label
// (never a raw UUID) when they can't.

const org: OrgDetail = {
  id: "org-1",
  name: "Greenbox Analytics",
  status: "active",
  member_count: 2,
  admins: [{ userId: "admin-1", email: "admin@example.com", fullName: "Ada Admin" }],
};

const seats: OrgSeatUsage[] = [
  {
    orgMemberId: "member-1",
    userId: "u2",
    email: "mo@example.com",
    role: "member",
    status: "active",
    monthlyCap: null,
    effectiveCap: null,
    capUsed: 0,
    spentThisPeriod: 10,
  },
];

describe("adminName", () => {
  it("resolves fullName, falling back to email, then null", () => {
    expect(adminName(org, "admin-1")).toBe("Ada Admin");
    expect(adminName({ ...org, admins: [{ userId: "admin-1", email: "a@x.com", fullName: null }] }, "admin-1")).toBe(
      "a@x.com",
    );
    expect(adminName(org, "departed-admin")).toBeNull();
    expect(adminName(org, undefined)).toBeNull();
  });
});

describe("memberEmail", () => {
  it("resolves the spending member's email, or null when unresolvable", () => {
    expect(memberEmail(seats, "member-1")).toBe("mo@example.com");
    expect(memberEmail(seats, "removed-member")).toBeNull();
    expect(memberEmail(undefined, "member-1")).toBeNull();
  });
});

describe("ledgerLabel", () => {
  it("names the admin on a resolvable transfer_in row", () => {
    const entry: OrgLedgerEntry = {
      kind: "transfer_in",
      action: "org_transfer",
      delta: 500,
      metadata: { admin_user_id: "admin-1" },
      created_at: "2026-08-10T00:00:00Z",
    };
    expect(ledgerLabel(entry, org, seats).label).toBe("Transfer from Ada Admin");
  });

  it("falls back to the generic label — never a raw UUID — when the admin is gone", () => {
    const entry: OrgLedgerEntry = {
      kind: "transfer_in",
      action: "org_transfer",
      delta: 500,
      metadata: { admin_user_id: "departed-admin" },
      created_at: "2026-08-10T00:00:00Z",
    };
    const label = ledgerLabel(entry, org, seats).label;
    expect(label).toBe("Transfer from admin");
    expect(label).not.toContain("departed-admin");
  });

  it("names the member on a resolvable debit row", () => {
    const entry: OrgLedgerEntry = {
      kind: "debit",
      action: "oneclick_run",
      delta: -12,
      metadata: { org_member_id: "member-1" },
      created_at: "2026-08-10T00:00:00Z",
    };
    expect(ledgerLabel(entry, org, seats).label).toBe("Member spend by mo@example.com");
  });

  it("falls back to 'Member spend' when the member is gone", () => {
    const entry: OrgLedgerEntry = {
      kind: "debit",
      action: "oneclick_run",
      delta: -12,
      metadata: { org_member_id: "removed-member" },
      created_at: "2026-08-10T00:00:00Z",
    };
    expect(ledgerLabel(entry, org, seats).label).toBe("Member spend");
  });

  it("renders an unknown kind as underscore-to-space, never a raw UUID", () => {
    const entry: OrgLedgerEntry = {
      kind: "org_migration",
      action: null,
      delta: 100,
      metadata: {},
      created_at: "2026-08-10T00:00:00Z",
    };
    expect(ledgerLabel(entry, org, seats).label).toBe("org migration");
  });
});
