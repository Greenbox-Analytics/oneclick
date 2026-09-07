import { describe, it, expect } from "vitest";
import { endOfLocalDayIso, expiryFromPreset, expiryLabel, keyRows, keyStatus, tomorrowInputValue } from "@/lib/partnerKeys";
import type { PartnerKey } from "@/hooks/usePartnerKeys";
import type { PartnerKeyUsageRow } from "@/hooks/useOrgs";

const base: PartnerKey = {
  id: "prod",
  label: "Prod",
  key_prefix: "mk_live_abcd",
  status: "active",
  expires_at: null,
  created_at: "2026-09-01T00:00:00+00:00",
  created_by: "u1",
  created_by_label: "admin@label.test",
  last_used_at: null,
  revoked_at: null,
  folder_id: null,
};
const NOW = new Date("2026-09-04T12:00:00Z");

describe("keyStatus", () => {
  it("is the stored status when there is no expiry", () => {
    expect(keyStatus(base, NOW)).toBe("active");
    expect(keyStatus({ ...base, status: "revoked" }, NOW)).toBe("revoked");
  });
  it("derives expired from a past expires_at on an active key", () => {
    expect(keyStatus({ ...base, expires_at: "2026-09-04T11:59:59+00:00" }, NOW)).toBe("expired");
    expect(keyStatus({ ...base, expires_at: "2026-09-04T12:00:01+00:00" }, NOW)).toBe("active");
  });
  it("revoked wins over expired", () => {
    expect(keyStatus({ ...base, status: "revoked", expires_at: "2020-01-01T00:00:00+00:00" }, NOW)).toBe("revoked");
  });
});

describe("endOfLocalDayIso", () => {
  it("is the last second of the chosen LOCAL day, serialized in UTC", () => {
    const iso = endOfLocalDayIso("2026-09-30");
    const d = new Date(iso);
    expect(iso.endsWith("Z")).toBe(true);
    expect(d.getFullYear()).toBe(2026);
    expect(d.getMonth()).toBe(8);
    expect(d.getDate()).toBe(30);
    expect([d.getHours(), d.getMinutes(), d.getSeconds()]).toEqual([23, 59, 59]);
  });
  it("tomorrowInputValue is yyyy-mm-dd of the next local day", () => {
    expect(tomorrowInputValue(new Date(2026, 8, 30, 15))).toBe("2026-10-01");
  });
});

describe("expiryFromPreset", () => {
  // Jan 31 + 1/3/6 months clamps to the month's last day (Feb 28, Apr 30, Jul 31).
  const NOW = new Date(2026, 0, 31, 10);

  it("computes each duration preset in local time", () => {
    expect(expiryFromPreset("7d", NOW)).toBe("2026-02-07");
    expect(expiryFromPreset("1m", NOW)).toBe("2026-02-28");
    expect(expiryFromPreset("3m", NOW)).toBe("2026-04-30");
    expect(expiryFromPreset("6m", NOW)).toBe("2026-07-31");
    expect(expiryFromPreset("1y", NOW)).toBe("2027-01-31");
  });

  it("never and custom have no computed date", () => {
    expect(expiryFromPreset("never", NOW)).toBeNull();
    expect(expiryFromPreset("custom", NOW)).toBeNull();
  });
});

describe("expiryLabel", () => {
  it("says never expires when there is no date", () => {
    expect(expiryLabel(null)).toBe("Never expires. You can revoke it any time.");
  });
  it("names the local day it stops working", () => {
    expect(expiryLabel("2026-10-06")).toBe("Stops working at the end of Oct 6, 2026.");
  });
});

describe("keyRows", () => {
  const older: PartnerKey = { ...base, id: "staging", label: "Staging", created_at: "2026-08-01T00:00:00+00:00" };
  const dead: PartnerKey = { ...base, id: "dead", status: "revoked", label: "Old", created_at: "2026-09-03T00:00:00+00:00" };
  const spend = (keyId: string, credits: number, runs: number, lastUsedAt: string | null): PartnerKeyUsageRow => ({
    keyId, label: keyId, keyPrefix: `mk_live_${keyId}`, status: "active", folderId: null, folderName: null,
    credits, runs, lastUsedAt, byAction: [],
  });
  const usage: PartnerKeyUsageRow[] = [
    spend("staging", 67, 2, "2026-09-03T00:00:00+00:00"),
    spend("prod", 30, 1, null),
  ];

  it("joins this period's usage and orders live first, then newest", () => {
    // `dead` is the newest row but inactive — liveness must beat recency.
    const rows = keyRows([dead, older, base], usage, NOW);
    expect(rows.map((r) => r.key.id)).toEqual(["prod", "staging", "dead"]);
    expect(rows[0].usage?.credits).toBe(30);
    expect(rows[1].usage?.credits).toBe(67);
    expect(rows[2].usage).toBeUndefined();
    expect(rows[2].status).toBe("revoked");
  });
});
