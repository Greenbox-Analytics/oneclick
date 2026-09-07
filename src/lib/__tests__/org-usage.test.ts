import { describe, it, expect } from "vitest";
import {
  bucketSeries, byToolTotals, delta, foldTools, folderSubject, folderUsageRows, groupKeysByFolder, keySubject,
  keyUsageRows, memberRows, memberSubject, topTool, totals, weekStart, windowLabel,
  mergeSeries,
} from "../orgUsage";
import { toolOf } from "../usageTools";
import type { FolderUsageRow, OrgSeatUsage, PartnerKeyUsageRow, SeriesDay } from "@/hooks/useOrgs";
import type { PartnerKey } from "@/hooks/usePartnerKeys";

const SERIES: SeriesDay[] = [
  { day: "2026-09-02", actions: [{ action: "oneclick_run", credits: 30, runs: 1 }, { action: "zoe_message", credits: 5, runs: 1 }] },
  { day: "2026-09-03", actions: [{ action: "partner_oneclick_run", credits: 60, runs: 2 }, { action: "partner_split_sheet", credits: 20, runs: 1 }] },
  { day: "2026-09-09", actions: [{ action: "registry_parse", credits: 30, runs: 1 }, { action: "made_up", credits: 999, runs: 9 }] },
];

describe("tools", () => {
  it("folds product and partner actions into one tool each, dropping unknown actions", () => {
    expect(toolOf("oneclick_run")).toBe("oneclick");
    expect(toolOf("partner_oneclick_run")).toBe("oneclick");
    expect(toolOf("partner_registry_parse")).toBe("registry");
    expect(toolOf("split_sheet")).toBe("splitsheet");
    expect(toolOf("partner_zoe_message")).toBe("zoe");
    expect(toolOf("made_up")).toBeNull();
    expect(toolOf("constructor")).toBeNull();
    expect(foldTools(SERIES[1].actions)).toEqual({ oneclick: 60, registry: 0, splitsheet: 20, zoe: 0 });
  });

  it("totals the series and names the top tool with its share", () => {
    expect(totals(SERIES)).toEqual({ credits: 1144, runs: 15 });
    expect(byToolTotals(SERIES)).toEqual({ oneclick: 90, registry: 30, splitsheet: 20, zoe: 5 });
    expect(topTool(byToolTotals(SERIES))).toEqual({ id: "oneclick", label: "OneClick", share: 90 / 145 });
    expect(topTool({ oneclick: 0, registry: 0, splitsheet: 0, zoe: 0 })).toBeNull();
  });

  it("computes the delta against the previous window, or none", () => {
    expect(delta(115, 50)).toBe(130);
    expect(delta(40, 50)).toBe(-20);
    expect(delta(0, 0)).toBe(0);
    expect(delta(10, 0)).toBeNull();
    expect(delta(10, null)).toBeNull();
  });

  it("never returns -0", () => {
    const d = delta(999, 1000);
    expect(d).toBe(0);
    expect(Object.is(d, -0)).toBe(false);
  });

  it("topTool ties break toward the earlier tool in TOOLS order", () => {
    expect(topTool({ oneclick: 10, registry: 10, splitsheet: 0, zoe: 0 })).toEqual({
      id: "oneclick", label: "OneClick", share: 0.5,
    });
  });
});

describe("bucketSeries", () => {
  it("fills every day of a day range from since to today, zero where nothing was spent", () => {
    const b = bucketSeries(SERIES.slice(0, 2), "7d", "2026-09-01T00:00:00+00:00", "2026-09-05");
    expect(b.map((x) => x.bucket)).toEqual(["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05"]);
    expect(b[1]).toMatchObject({ label: "Sep 2", credits: 35, tools: { oneclick: 30, zoe: 5 } });
    expect(b[3].credits).toBe(0);
  });

  it("buckets a year by ISO week (Mondays) and all time by month", () => {
    expect(weekStart("2026-09-03")).toBe("2026-08-31"); // Thursday -> Monday
    expect(weekStart("2026-08-31")).toBe("2026-08-31");
    const weeks = bucketSeries(SERIES, "1y", "2026-08-31T00:00:00+00:00", "2026-09-13");
    expect(weeks.map((w) => [w.bucket, w.credits])).toEqual([["2026-08-31", 115], ["2026-09-07", 30]]);
    const months = bucketSeries([...SERIES, { day: "2026-07-14", actions: [{ action: "zoe_message", credits: 5, runs: 1 }] }], "all", null);
    expect(months.map((m) => [m.bucket, m.label, m.credits])).toEqual([["2026-07-01", "Jul 2026", 5], ["2026-09-01", "Sep 2026", 145]]);
  });

  it("with no floor (mtd, since null), lists only days with spend and does not fill gaps", () => {
    const b = bucketSeries(SERIES, "mtd", null);
    expect(b.map((x) => x.bucket)).toEqual(["2026-09-02", "2026-09-03", "2026-09-09"]);
    expect(b.map((x) => x.credits)).toEqual([35, 80, 30]);
  });

  it("does not fill or crash when since is later than today", () => {
    expect(bucketSeries([], "7d", "2026-09-10T00:00:00+00:00", "2026-09-05")).toEqual([]);
  });

  it("keeps a series day before `since` as its own bucket, sorted first", () => {
    const b = bucketSeries(SERIES.slice(0, 1), "7d", "2026-09-05T00:00:00+00:00", "2026-09-06");
    expect(b.map((x) => x.bucket)).toEqual(["2026-09-02", "2026-09-05", "2026-09-06"]);
    expect(b[0].credits).toBe(35);
  });

  it("tolerates a payload without series", () => {
    expect(totals(undefined as never)).toEqual({ credits: 0, runs: 0 });
    expect(byToolTotals(undefined as never)).toEqual({ oneclick: 0, registry: 0, splitsheet: 0, zoe: 0 });
    expect(bucketSeries(undefined as never, "all", null)).toEqual([]);
  });
});

describe("tables", () => {
  const key = (id: string, created_by: string | null): PartnerKey => ({
    id, label: `Key ${id}`, key_prefix: "mk_live_" + id, status: "active", expires_at: null,
    created_at: "2026-09-01T00:00:00+00:00", created_by, created_by_label: null, last_used_at: null,
    revoked_at: null, folder_id: null,
  });
  const usageRow = (over: Partial<PartnerKeyUsageRow> & { keyId: string }): PartnerKeyUsageRow => ({
    label: `Key ${over.keyId}`, keyPrefix: "mk_live_" + over.keyId, status: "active",
    folderId: null, folderName: null, credits: 0, runs: 0, lastUsedAt: null, byAction: [], ...over,
  });
  const seat = (id: string, userId: string, spent: number, byAction: OrgSeatUsage["byAction"]): OrgSeatUsage => ({
    orgMemberId: id, userId, email: `${id}@x.test`, role: "member", status: "active", monthlyCap: null,
    effectiveCap: null, capUsed: 0, spentThisPeriod: spent, byAction,
  });

  it("joins the keys a member created and sorts members by spend", () => {
    const rows = memberRows(
      [seat("a", "u1", 5, [{ action: "zoe_message", credits: 5, runs: 1 }]), seat("b", "u2", 60, [{ action: "oneclick_run", credits: 60, runs: 2 }])],
      [key("k1", "u1"), key("k2", "u1"), key("k3", null)]
    );
    expect(rows.map((r) => r.seat.orgMemberId)).toEqual(["b", "a"]);
    expect(rows[1].keys.map((k) => k.id)).toEqual(["k1", "k2"]);
    expect(rows[0].keys).toEqual([]);
    expect(rows[0].tools).toEqual({ oneclick: 60, registry: 0, splitsheet: 0, zoe: 0 });
    // Runs come off byAction; a payload with no per-row series reads as [].
    expect(rows.map((r) => [r.runs, r.series])).toEqual([[2, []], [1, []]]);
  });

  it("carries the per-row series through for the detail dialog", () => {
    const withSeries: OrgSeatUsage = {
      ...seat("a", "u1", 5, [{ action: "zoe_message", credits: 5, runs: 1 }]),
      series: [{ day: "2026-09-02", actions: [{ action: "zoe_message", credits: 5, runs: 1 }] }],
    };
    expect(memberRows([withSeries], [])[0].series).toEqual(withSeries.series);
    const k = keyUsageRows([usageRow({ keyId: "k1", runs: 3, series: [{ day: "2026-09-02", actions: [] }] })]);
    expect(k[0]).toMatchObject({ runs: 3, series: [{ day: "2026-09-02", actions: [] }] });
    const f = folderUsageRows([
      { folderId: "f1", name: "Ingest", keys: 1, credits: 0, runs: 4, byAction: [], series: [{ day: "2026-09-03", actions: [] }] },
    ]);
    expect(f[0]).toMatchObject({ runs: 4, series: [{ day: "2026-09-03", actions: [] }] });
  });

  it("adds the member's API spend to their total and folds partner actions into tools", () => {
    // spentThisPeriod stays product-only (it is what the cap is measured against).
    const withApi: OrgSeatUsage = {
      ...seat("a", "u1", 5, [{ action: "zoe_message", credits: 5, runs: 1 }, { action: "partner_oneclick_run", credits: 60, runs: 2 }]),
      apiCredits: 60, apiRuns: 2,
    };
    const rows = memberRows([withApi, seat("b", "u2", 50, [{ action: "oneclick_run", credits: 50, runs: 1 }])], []);
    // 65 > 50, so API spend also reorders the table.
    expect(rows.map((r) => [r.seat.orgMemberId, r.total])).toEqual([["a", 65], ["b", 50]]);
    expect(rows[0].tools).toEqual({ oneclick: 60, registry: 0, splitsheet: 0, zoe: 5 });
  });

  it("treats a payload with no apiCredits as product-only spend", () => {
    expect(memberRows([seat("a", "u1", 5, [])], [])[0].total).toBe(5);
  });

  it("reads label, status and folder straight off the payload, highest spend first", () => {
    const rows = keyUsageRows([
      usageRow({ keyId: "k1" }),
      usageRow({
        keyId: "k2", label: "Production", status: "revoked", folderId: "f1", folderName: "Ingest",
        credits: 80, runs: 3,
        byAction: [{ action: "partner_oneclick_run", credits: 60, runs: 2 }, { action: "partner_split_sheet", credits: 20, runs: 1 }],
      }),
    ]);
    expect(rows.map((r) => [r.row.keyId, r.total])).toEqual([["k2", 80], ["k1", 0]]);
    expect(rows[0].row).toMatchObject({ label: "Production", status: "revoked", folderName: "Ingest" });
    expect(rows[0].tools).toEqual({ oneclick: 60, registry: 0, splitsheet: 20, zoe: 0 });
  });

  it("breaks a spend tie on the label, and tolerates a payload without byKey", () => {
    const rows = keyUsageRows([usageRow({ keyId: "b", label: "Beta" }), usageRow({ keyId: "a", label: "Alpha" })]);
    expect(rows.map((r) => r.row.label)).toEqual(["Alpha", "Beta"]);
    expect(keyUsageRows(undefined as never)).toEqual([]);
  });

  it("folds folder spend into tools, credits desc, unfiled included", () => {
    const byFolder: FolderUsageRow[] = [
      { folderId: null, name: "No folder", keys: 1, credits: 20, runs: 1, byAction: [{ action: "partner_split_sheet", credits: 20, runs: 1 }] },
      { folderId: "f1", name: "Ingest", keys: 2, credits: 60, runs: 2, byAction: [{ action: "partner_oneclick_run", credits: 60, runs: 2 }] },
    ];
    const rows = folderUsageRows(byFolder);
    expect(rows.map((r) => [r.row.name, r.total, r.row.keys])).toEqual([["Ingest", 60, 2], ["No folder", 20, 1]]);
    expect(rows[0].tools).toEqual({ oneclick: 60, registry: 0, splitsheet: 0, zoe: 0 });
    expect(folderUsageRows(undefined as never)).toEqual([]);
  });

  it("describes a member, a key and a folder for the detail dialog", () => {
    const member = memberSubject(memberRows([seat("a", "u1", 5, [{ action: "zoe_message", credits: 5, runs: 1 }])], [key("k1", "u1")])[0]);
    expect(member).toMatchObject({ kind: "member", id: "a", title: "a@x.test", subtitle: "Member · Member", total: 5, runs: 1 });
    expect(member.facts).toEqual([{ label: "Role", value: "Member" }, { label: "Keys created", value: "1" }]);

    const noKeys = memberSubject(memberRows([{ ...seat("a", "u1", 0, []), status: "suspended" }], [])[0]);
    expect(noKeys.facts).toEqual([
      { label: "Role", value: "Member" },
      { label: "Status", value: "Suspended" },
      { label: "Keys created", value: "None" },
    ]);

    const k = keySubject(keyUsageRows([usageRow({
      keyId: "k1", label: "Production", status: "revoked", folderId: "f1", folderName: "Ingest",
      credits: 80, runs: 3, byAction: [{ action: "partner_oneclick_run", credits: 80, runs: 3 }],
    })])[0]);
    expect(k).toMatchObject({ kind: "key", id: "k1", title: "Production", subtitle: "API key · Ingest", total: 80, runs: 3 });
    expect(k.facts).toEqual([
      { label: "Key", value: "mk_live_k1…" },
      { label: "Status", value: "Revoked" },
      { label: "Folder", value: "Ingest" },
      // A key that has never been used says so, rather than showing a dash.
      { label: "Last used", value: "Never" },
    ]);

    const unfiled = keySubject(keyUsageRows([usageRow({ keyId: "k2", lastUsedAt: "2026-09-03T12:00:00+00:00" })])[0]);
    expect(unfiled.subtitle).toBe("API key · No folder");
    expect(unfiled.facts).toContainEqual({ label: "Folder", value: "No folder" });
    expect(unfiled.facts.find((f) => f.label === "Last used")!.value).not.toBe("Never");

    const f = folderSubject(folderUsageRows([{ folderId: "f1", name: "Ingest", keys: 2, credits: 60, runs: 2, byAction: [] }])[0]);
    expect(f).toMatchObject({ kind: "folder", id: "f1", title: "Ingest", subtitle: "Folder", total: 60, runs: 2 });
    expect(f.facts).toEqual([{ label: "Keys", value: "2" }]);
  });
});

describe("groupKeysByFolder", () => {
  const usageRow = (over: Partial<PartnerKeyUsageRow> & { keyId: string }): PartnerKeyUsageRow => ({
    label: `Key ${over.keyId}`, keyPrefix: "mk_live_" + over.keyId, status: "active",
    folderId: null, folderName: null, credits: 0, runs: 0, lastUsedAt: null, byAction: [], ...over,
  });
  const folder = (folderId: string | null, name: string, credits: number, keys = 1): FolderUsageRow =>
    ({ folderId, name, keys, credits, runs: 0, byAction: [] });

  it("nests keys under their folder, credits desc, with No folder last", () => {
    const groups = groupKeysByFolder(
      [
        usageRow({ keyId: "a", folderId: "f1", folderName: "Ingest", credits: 50 }),
        usageRow({ keyId: "b", folderId: null, credits: 10 }),
        usageRow({ keyId: "c", folderId: "f2", folderName: "Batch", credits: 80 }),
      ],
      [folder("f1", "Ingest", 50), folder(null, "No folder", 10), folder("f2", "Batch", 80)],
    );
    expect(groups.map((g) => [g.folder.row.name, g.keys.map((k) => k.row.keyId)])).toEqual([
      ["Batch", ["c"]],
      ["Ingest", ["a"]],
      ["No folder", ["b"]],
    ]);
  });

  it("keeps a folder with no keys, and files a key whose folder is not listed under No folder", () => {
    const groups = groupKeysByFolder(
      [usageRow({ keyId: "a", folderId: "gone", folderName: "Deleted", credits: 5 })],
      [folder("f1", "Empty", 0, 0)],
    );
    expect(groups.map((g) => [g.folder.row.name, g.keys.length])).toEqual([["Empty", 0], ["No folder", 1]]);
    // The synthetic group subtotals the keys that landed in it.
    expect(groups[1].folder.total).toBe(5);
  });

  it("folds the synthetic No-folder group's mix and series from its keys", () => {
    const day = (d: string, action: string, credits: number): SeriesDay => ({ day: d, actions: [{ action, credits, runs: 1 }] });
    const groups = groupKeysByFolder(
      [
        usageRow({ keyId: "a", folderId: "gone", credits: 5, byAction: [{ action: "partner_zoe_message", credits: 5, runs: 1 }], series: [day("2026-09-02", "partner_zoe_message", 5)] }),
        usageRow({ keyId: "b", folderId: null, credits: 30, byAction: [{ action: "partner_oneclick_run", credits: 30, runs: 1 }], series: [day("2026-09-01", "partner_oneclick_run", 30)] }),
      ],
      [],
    );
    const [{ folder }] = groups;
    expect(folder.total).toBe(35);
    expect(folder.tools).toMatchObject({ zoe: 5, oneclick: 30 });
    expect(folder.series.map((d) => d.day)).toEqual(["2026-09-01", "2026-09-02"]);
  });

  it("mergeSeries sums per day and per action", () => {
    const merged = mergeSeries([
      [{ day: "2026-09-01", actions: [{ action: "x", credits: 1, runs: 1 }] }],
      [{ day: "2026-09-01", actions: [{ action: "x", credits: 2, runs: 1 }, { action: "y", credits: 9, runs: 1 }] }, { day: "2026-08-30", actions: [{ action: "x", credits: 4, runs: 2 }] }],
    ]);
    expect(merged).toEqual([
      { day: "2026-08-30", actions: [{ action: "x", credits: 4, runs: 2 }] },
      { day: "2026-09-01", actions: [{ action: "y", credits: 9, runs: 1 }, { action: "x", credits: 3, runs: 2 }] },
    ]);
  });

  it("adds no No-folder group when every key is filed", () => {
    const groups = groupKeysByFolder([usageRow({ keyId: "a", folderId: "f1", folderName: "Ingest", credits: 5 })], [folder("f1", "Ingest", 5)]);
    expect(groups.map((g) => g.folder.row.name)).toEqual(["Ingest"]);
  });
});

describe("windowLabel", () => {
  it("labels the window", () => {
    expect(windowLabel("all", null)).toBe("All time");
    expect(windowLabel("7d", "2026-09-01T00:00:00+00:00", new Date("2026-09-08T12:00:00Z"))).toMatch(/Sep 1, 2026 – Sep 8, 2026/);
  });
});
