import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { OrgUsageAnalysis } from "../OrgUsageAnalysis";
import type { OrgUsage, UsageRange } from "@/hooks/useOrgs";
import type { PartnerKey, PartnerKeyFolder } from "@/hooks/usePartnerKeys";

// Recharts measures its container with ResizeObserver, which jsdom lacks.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

// The Download PDF button goes through downloadPdf → fetch → object URL.
// The stub is re-applied per test: the config sets unstubGlobals.
const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => new Response(new Blob(["%PDF"]), { status: 200 }));
vi.mock("@/lib/apiFetch", async (orig) => ({
  ...(await orig<typeof import("@/lib/apiFetch")>()),
  API_URL: "https://api.test",
  getAuthHeaders: async () => ({ Authorization: "Bearer t" }),
}));

const KEY: PartnerKey = {
  id: "k1", label: "Production", key_prefix: "mk_live_abcd", status: "active", expires_at: null,
  created_at: "2026-09-01T00:00:00+00:00", created_by: "u-admin", created_by_label: "admin@label.test", last_used_at: null,
  revoked_at: null, folder_id: "f1",
};
const FOLDER: PartnerKeyFolder = { id: "f1", org_id: "org-1", name: "Ingest", created_at: "2026-09-01T00:00:00+00:00" };

const USAGE: OrgUsage = {
  poolBalance: 500, cumulativePaidIn: 0, monthlyDispersalCredits: 0, defaultMemberCap: 2000,
  periodStart: "2026-09-01T00:00:00+00:00", periodEnd: "2026-10-01T00:00:00+00:00",
  range: "mtd", since: "2026-09-01T00:00:00+00:00", previous: { credits: 50, runs: 2 },
  series: [
    { day: "2026-09-02", actions: [{ action: "oneclick_run", credits: 30, runs: 1 }, { action: "zoe_message", credits: 5, runs: 1 }] },
    { day: "2026-09-03", actions: [{ action: "partner_oneclick_run", credits: 60, runs: 2 }, { action: "partner_split_sheet", credits: 20, runs: 1 }] },
  ],
  seats: [
    { orgMemberId: "m1", userId: "u-admin", email: "admin@label.test", role: "admin", status: "active", monthlyCap: null, effectiveCap: 2000, capUsed: 35, spentThisPeriod: 35,
      apiCredits: 80, apiRuns: 3,
      byAction: [{ action: "oneclick_run", credits: 30, runs: 1 }, { action: "zoe_message", credits: 5, runs: 1 },
        { action: "partner_oneclick_run", credits: 60, runs: 2 }, { action: "partner_split_sheet", credits: 20, runs: 1 }] },
    { orgMemberId: "m2", userId: "u-member", email: "sam@label.test", role: "member", status: "active", monthlyCap: null, effectiveCap: 2000, capUsed: 0, spentThisPeriod: 0, byAction: [] },
  ],
  byKey: [{
    keyId: "k1", label: "Production", keyPrefix: "mk_live_abcd", status: "revoked", folderId: "f1", folderName: "Ingest",
    credits: 80, runs: 3, lastUsedAt: "2026-09-03T12:00:00+00:00",
    byAction: [{ action: "partner_oneclick_run", credits: 60, runs: 2 }, { action: "partner_split_sheet", credits: 20, runs: 1 }],
  }],
  byFolder: [
    { folderId: "f1", name: "Ingest", keys: 1, credits: 80, runs: 3,
      byAction: [{ action: "partner_oneclick_run", credits: 60, runs: 2 }, { action: "partner_split_sheet", credits: 20, runs: 1 }] },
    { folderId: null, name: "No folder", keys: 2, credits: 5, runs: 1, byAction: [{ action: "partner_zoe_message", credits: 5, runs: 1 }] },
  ],
};

let usageData: OrgUsage | undefined;
let usageSuccess = true;
let usageError = false;
const ranges: UsageRange[] = [];
vi.mock("@/hooks/useOrgs", () => ({
  useOrgUsage: (_orgId: string, range: UsageRange) => {
    ranges.push(range);
    return { data: usageData, isSuccess: usageSuccess, isError: usageError };
  },
}));
vi.mock("@/hooks/usePartnerKeys", () => ({
  usePartnerKeys: (orgId?: string) => ({ data: orgId ? { keys: [KEY], folders: [FOLDER] } : undefined }),
}));

beforeEach(() => {
  usageData = USAGE;
  usageSuccess = true;
  usageError = false;
  ranges.length = 0;
  fetchMock.mockClear();
  vi.stubGlobal("fetch", fetchMock);
  URL.createObjectURL = vi.fn(() => "blob:usage");
  URL.revokeObjectURL = vi.fn();
});
afterEach(cleanup);

const row = (text: string) => screen.getByText(text).closest("tr")!;
// "Runs" is now both a tile and a column header — match the tile's own label element.
const tile = (label: string) =>
  screen.getAllByText(label).find((el) => el.className.includes("uppercase"))!.parentElement!;

describe("OrgUsageAnalysis", () => {
  it("shows the window's totals, the tool mix and the previous-window delta", () => {
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    expect(tile("Credits used")).toHaveTextContent("115");
    expect(tile("Credits used")).toHaveTextContent("+130% vs the previous period");
    expect(tile("Runs")).toHaveTextContent("5");
    expect(tile("Most used tool")).toHaveTextContent("OneClick");
    expect(tile("Most used tool")).toHaveTextContent("78% of credits");
    expect(tile("Active members")).toHaveTextContent("1");
    expect(screen.getByText("Credits over time")).toBeInTheDocument();
    expect(screen.getByText("By tool")).toBeInTheDocument();
  });

  it("lists members with their total, runs and tool mix, highest spend first", () => {
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    const admin = within(row("admin@label.test"));
    // Member | Total | Runs | Mix | Keys created. The total is product spend (35)
    // plus spend through this member's keys (80).
    expect(admin.getAllByRole("cell").slice(1, 3).map((c) => c.textContent)).toEqual(["115", "5"]);
    // The per-tool split rides in one bar now, not one column per tool.
    expect(admin.getByRole("img")).toHaveAttribute("aria-label", "OneClick 90 · Split sheet 20 · Zoe 5");
    expect(admin.getByText("Production")).toBeInTheDocument();
    expect(admin.getByText(/mk_live_abcd/)).toBeInTheDocument();
    expect(within(row("sam@label.test")).getAllByRole("cell")[4]).toHaveTextContent("—");
    const members = screen.getAllByRole("row").filter((r) => r.textContent?.includes("@label.test"));
    expect(members[0]).toHaveTextContent("admin@label.test");
  });

  it("has no per-tool columns — a fifth tool must not widen the table", () => {
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    for (const label of ["OneClick", "Registry", "Split sheet", "Zoe"]) {
      expect(screen.queryByRole("columnheader", { name: label })).not.toBeInTheDocument();
    }
    expect(screen.getAllByRole("columnheader").map((h) => h.textContent))
      .toEqual(["Member", "Total", "Runs", "Mix", "Keys created"]);
  });

  it("opens the breakdown dialog for the member whose row was clicked", () => {
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    fireEvent.click(screen.getByRole("button", { name: "admin@label.test" }));
    const dialog = within(screen.getByRole("dialog"));
    expect(dialog.getByRole("heading", { name: "admin@label.test" })).toBeInTheDocument();
    expect(dialog.getByText("Credits used").parentElement).toHaveTextContent("115");
    expect(dialog.getByText("Keys created")).toBeInTheDocument();
  });

  it("builds the per-key table from the payload — label, folder and status, not the key list", () => {
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "By key" }));
    const key = within(row("Production"));
    // Key | Folder | Total | Runs | Mix | Last used
    const cells = key.getAllByRole("cell").map((c) => c.textContent);
    expect(cells[0]).toContain("revoked"); // the payload's status, not the stored one
    expect(cells[1]).toBe("Ingest");
    expect(cells.slice(2, 4)).toEqual(["80", "3"]);
    expect(key.getByRole("img")).toHaveAttribute("aria-label", "OneClick 60 · Split sheet 20");
  });

  it("rolls spend up by folder, unfiled included", () => {
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "By folder" }));
    // Folder | Keys | Total | Runs | Mix
    expect(within(row("Ingest")).getAllByRole("cell").slice(0, 4).map((c) => c.textContent))
      .toEqual(["Ingest", "1", "80", "3"]);
    const unfiled = within(row("No folder")).getAllByRole("cell").map((c) => c.textContent);
    expect(unfiled.slice(0, 3)).toEqual(["No folder", "2", "5"]);
    // Credits desc: the filed folder outranks the unfiled row.
    const names = screen.getAllByRole("row").map((r) => r.textContent);
    expect(names.findIndex((t) => t?.startsWith("Ingest"))).toBeLessThan(names.findIndex((t) => t?.startsWith("No folder")));
  });

  it("says so when no key has a folder yet", () => {
    usageData = { ...USAGE, byFolder: [] };
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "By folder" }));
    expect(screen.getByText("No folders yet.")).toBeInTheDocument();
  });

  it("uses the hooks it is handed, so the admin drawer can point it at /admin", () => {
    const fakeUsage = vi.fn(() => ({ data: { ...USAGE, byKey: [], byFolder: [] }, isSuccess: true, isError: false }));
    const fakeKeys = vi.fn(() => ({ data: { keys: [], folders: [] } }));
    render(
      <OrgUsageAnalysis
        orgId="org-9"
        partnerApiEnabled
        useUsage={fakeUsage as never}
        useKeys={fakeKeys as never}
      />,
    );
    expect(fakeUsage).toHaveBeenCalledWith("org-9", "mtd");
    expect(fakeKeys).toHaveBeenCalledWith("org-9");
    // The card's own hooks stayed out of it.
    expect(ranges).toEqual([]);
  });

  it("refetches with the chosen range, and labels All time without a delta", () => {
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    expect(ranges[0]).toBe("mtd");
    usageData = { ...USAGE, range: "all", since: null, previous: null };
    fireEvent.click(screen.getByRole("radio", { name: "All" }));
    expect(ranges[ranges.length - 1]).toBe("all");
    expect(screen.getByText("All time")).toBeInTheDocument();
    expect(screen.queryByText(/vs the previous/)).not.toBeInTheDocument();
  });

  it("hides the key tab without API access, and shows dashes until the range has loaded", () => {
    usageSuccess = false;
    usageData = undefined;
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled={false} />);
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    expect(tile("Credits used")).toHaveTextContent("—");
  });

  it("downloads the report for the shown range, from the org route by default", async () => {
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    fireEvent.click(screen.getByRole("button", { name: /download pdf/i }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(fetchMock.mock.calls[0][0]).toBe("https://api.test/orgs/org-1/usage/report.pdf?range=mtd");
  });

  it("uses the report path it is handed, so the admin drawer downloads from /admin", async () => {
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled reportPath="/admin/orgs/org-1/usage/report.pdf" />);
    fireEvent.click(screen.getByRole("button", { name: /download pdf/i }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(fetchMock.mock.calls[0][0]).toBe("https://api.test/admin/orgs/org-1/usage/report.pdf?range=mtd");
  });

  it("says when the usage could not be loaded", () => {
    usageError = true;
    usageSuccess = false;
    usageData = undefined;
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    expect(screen.getByText(/Couldn't load usage/)).toBeInTheDocument();
    expect(document.querySelector(".animate-spin")).toBeNull();
    expect(screen.queryByText("Credits over time")).not.toBeInTheDocument();
  });

  it("says so when the window has no usage", () => {
    usageData = { ...USAGE, series: [], previous: { credits: 0, runs: 0 } };
    render(<OrgUsageAnalysis orgId="org-1" partnerApiEnabled />);
    expect(screen.getByText("No usage in this period.")).toBeInTheDocument();
    expect(screen.queryByText("Credits over time")).not.toBeInTheDocument();
  });
});
