import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { AdminOrgsPanel } from "../AdminOrgsPanel";
import type { AdminOrgRow } from "@/hooks/useAdminOrgs";

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

// The card's Download PDF button goes through downloadPdf → fetch → object URL.
const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => new Response(new Blob(["%PDF"]), { status: 200 }));
vi.mock("@/lib/apiFetch", async (orig) => ({
  ...(await orig<typeof import("@/lib/apiFetch")>()),
  API_URL: "https://api.test",
  getAuthHeaders: async () => ({ Authorization: "Bearer t" }),
}));

const ORG: AdminOrgRow = {
  id: "org-1", name: "Label Co", status: "active", archivedAt: null, kind: "enterprise",
  partnerApiEnabled: true, memberCount: 3, bundleBalance: 100, reserveBalance: 0,
  monthlyDispersalCredits: 500, activationFloor: 0, cumulativePaidIn: 500,
};

const adminUsage = vi.fn(() => ({
  data: {
    poolBalance: 100, cumulativePaidIn: 500, monthlyDispersalCredits: 500, defaultMemberCap: null,
    periodStart: null, periodEnd: null, range: "mtd", since: null, previous: null, seats: [],
    series: [{ day: "2026-09-02", actions: [{ action: "partner_oneclick_run", credits: 60, runs: 2 }] }],
    byKey: [], byFolder: [{ folderId: "f1", name: "Ingest", keys: 1, credits: 60, runs: 2, byAction: [] }],
  },
  isSuccess: true,
  isError: false,
}));
const adminKeys = vi.fn(() => ({
  data: {
    keys: [{
      id: "k1", label: "Production", key_prefix: "mk_live_abcd", status: "active", expires_at: null,
      created_at: "2026-09-01T00:00:00+00:00", created_by: "u1", created_by_label: "admin@label.test",
      last_used_at: "2026-09-03T00:00:00+00:00", revoked_at: null, folder_id: "f1",
    }],
    folders: [{ id: "f1", org_id: "org-1", name: "Ingest", created_at: "2026-09-01T00:00:00+00:00" }],
  },
  isLoading: false,
  isError: false,
}));

vi.mock("@/hooks/useAdminOrgs", () => ({
  useAdminOrgs: () => ({ data: [ORG], isLoading: false, error: null }),
  useAdminOrgPool: () => ({ data: undefined }),
  useAdminOrgMutations: () => ({
    grantCredits: { mutate: vi.fn(), isPending: false },
    setDispersal: { mutate: vi.fn(), isPending: false },
    setStatus: { mutate: vi.fn(), isPending: false },
    setPartnerApi: { mutate: vi.fn(), isPending: false },
  }),
  useAdminOrgUsage: (...a: unknown[]) => adminUsage(...(a as [])),
  useAdminPartnerKeys: (...a: unknown[]) => adminKeys(...(a as [])),
}));

beforeEach(() => {
  fetchMock.mockClear();
  vi.stubGlobal("fetch", fetchMock);
  URL.createObjectURL = vi.fn(() => "blob:usage");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => { cleanup(); adminUsage.mockClear(); adminKeys.mockClear(); });

describe("Admin → Organizations drawer, Usage tab", () => {
  const openUsage = () => {
    render(<AdminOrgsPanel selectedOrgId="org-1" onSelectOrg={() => {}} />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Usage" }));
  };

  it("renders the org's Usage card against the admin routes", () => {
    openUsage();
    expect(screen.getByRole("heading", { name: /Usage/ })).toBeInTheDocument();
    expect(adminUsage).toHaveBeenCalledWith("org-1", "mtd");
    // The card's folder rollup comes from the admin payload.
    fireEvent.mouseDown(screen.getByRole("tab", { name: "By folder" }));
    const folderRow = screen.getAllByText("Ingest").map((el) => el.closest("tr")!).find((tr) => tr.textContent?.includes("60"))!;
    expect(within(folderRow).getByText("60")).toBeInTheDocument();
  });

  it("downloads the usage report from the admin route, not the org's own", async () => {
    openUsage();
    fireEvent.click(screen.getByRole("button", { name: /download pdf/i }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(fetchMock.mock.calls[0][0]).toBe("https://api.test/admin/orgs/org-1/usage/report.pdf?range=mtd");
  });

  it("lists the org's keys read-only, with their folder", () => {
    openUsage();
    expect(adminKeys).toHaveBeenCalledWith("org-1");
    const keyRow = screen.getByText("Production").closest("tr")!;
    expect(within(keyRow).getByText("Ingest")).toBeInTheDocument();
    expect(within(keyRow).getByText("admin@label.test")).toBeInTheDocument();
    // Read-only: no minting or revoking from the admin console.
    expect(screen.queryByRole("button", { name: /new api key/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /revoke/i })).not.toBeInTheDocument();
  });
});
