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

// The key-trace box sits above the org list; idle unless a test arms it.
// mockClear does NOT reset a return value, so afterEach restores IDLE — without
// that, a test that arms the lookup silently arms every test after it.
const traceMutate = vi.fn();
const revokeMutate = vi.fn();
const IDLE = { mutate: traceMutate, data: undefined as { keys: unknown[] } | undefined, isPending: false, isError: false };
const adminLookup = vi.fn(() => IDLE);

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
  useAdminKeyLookup: () => adminLookup(),
  useAdminRevokePartnerKey: () => ({ mutate: revokeMutate, isPending: false }),
}));

beforeEach(() => {
  fetchMock.mockClear();
  vi.stubGlobal("fetch", fetchMock);
  URL.createObjectURL = vi.fn(() => "blob:usage");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => {
  cleanup();
  adminUsage.mockClear();
  adminKeys.mockClear();
  adminLookup.mockClear();
  adminLookup.mockReturnValue(IDLE);
  traceMutate.mockClear();
  revokeMutate.mockClear();
});

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
    // The drawer's table stays read-only — no minting, and no per-row revoke.
    // Revoking a key lives in the trace box, behind a confirm.
    expect(screen.queryByRole("button", { name: /new api key/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^revoke/i })).not.toBeInTheDocument();
  });
});

describe("Admin → Organizations, trace an API key", () => {
  const HIT = {
    id: "k1", org_id: "org-1", org_name: "Label Co", label: "Production",
    key_prefix: "mk_live_abcd", status: "active", expires_at: null,
    created_at: "2026-09-01T00:00:00+00:00", last_used_at: "2026-09-07T00:00:00+00:00",
    recent_ips: [
      { ip: "203.0.113.9", requests: 41, last_seen: "2026-09-07T10:00:00+00:00" },
      { ip: "198.51.100.4", requests: 2, last_seen: "2026-09-07T09:00:00+00:00" },
    ],
  };
  const armTrace = (overrides: Partial<typeof HIT> = {}) =>
    adminLookup.mockReturnValue({
      mutate: traceMutate,
      isPending: false,
      isError: false,
      data: { keys: [{ ...HIT, ...overrides }] },
    } as typeof IDLE);

  const paste = (value: string) => {
    render(<AdminOrgsPanel selectedOrgId={null} onSelectOrg={() => {}} />);
    fireEvent.change(screen.getByPlaceholderText("mk_live_…"), { target: { value } });
    fireEvent.click(screen.getByRole("button", { name: /look up/i }));
  };

  it("sends the pasted key in a POST body, never a URL", () => {
    paste("  mk_live_abcdEFGH  ");
    // Trimmed and handed to the mutation — the hook posts it as a body, so the
    // secret never reaches a query string, an access log or the query cache.
    expect(traceMutate).toHaveBeenCalledWith("mk_live_abcdEFGH");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("names the owning org and the source IPs a leak would show", () => {
    armTrace();
    paste("mk_live_abcd");
    expect(screen.getByText("Production")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Label Co" })).toBeInTheDocument();
    // Two IPs on one key is the whole signal; the count has to survive to the UI.
    expect(screen.getByText(/203\.0\.113\.9 ×41.*198\.51\.100\.4 ×2/)).toBeInTheDocument();
  });

  it("says so plainly when nothing matches", () => {
    adminLookup.mockReturnValue({
      mutate: traceMutate, isPending: false, isError: false, data: { keys: [] },
    } as typeof IDLE);
    paste("mk_live_nope");
    expect(screen.getByText(/no key matches that prefix/i)).toBeInTheDocument();
  });
});

describe("Admin → Organizations, revoking a traced key", () => {
  const HIT = {
    id: "k1", org_id: "org-1", org_name: "Label Co", label: "Production",
    key_prefix: "mk_live_abcd", status: "active", expires_at: null,
    created_at: "2026-09-01T00:00:00+00:00", last_used_at: "2026-09-07T00:00:00+00:00",
    recent_ips: [],
  };
  const trace = (overrides: Partial<typeof HIT> = {}) => {
    adminLookup.mockReturnValue({
      mutate: traceMutate, isPending: false, isError: false,
      data: { keys: [{ ...HIT, ...overrides }] },
    } as typeof IDLE);
    render(<AdminOrgsPanel selectedOrgId={null} onSelectOrg={() => {}} />);
    fireEvent.change(screen.getByPlaceholderText("mk_live_…"), { target: { value: "mk_live_abcd" } });
    fireEvent.click(screen.getByRole("button", { name: /look up/i }));
  };

  it("never revokes on the first click — killing a partner's key is confirmed", () => {
    trace();
    fireEvent.click(screen.getByRole("button", { name: "Revoke Production" }));
    expect(revokeMutate).not.toHaveBeenCalled();
    // The confirm has to name the team, or an admin can revoke into the wrong one.
    expect(screen.getByText(/Revoke "Production" for Label Co\?/)).toBeInTheDocument();
  });

  it("revokes against the key's OWN org, not the drawer's selection", () => {
    trace({ org_id: "org-9" });
    fireEvent.click(screen.getByRole("button", { name: "Revoke Production" }));
    fireEvent.click(screen.getByRole("button", { name: /revoke key/i }));
    expect(revokeMutate.mock.calls[0][0]).toEqual({ orgId: "org-9", keyId: "k1" });
  });

  it("offers no revoke on a key that is already dead", () => {
    trace({ status: "revoked" });
    expect(screen.queryByRole("button", { name: /^revoke/i })).not.toBeInTheDocument();
  });
});
