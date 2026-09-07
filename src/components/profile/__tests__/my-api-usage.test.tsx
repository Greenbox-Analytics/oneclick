import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { MyApiUsage } from "../MyApiUsage";
import type { MyApiUsage as Payload } from "@/hooks/useCreditUsage";
import type { UsageRange } from "@/hooks/useOrgs";

// The card renders a Recharts chart inside its detail dialog.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

// Download PDF goes through downloadPdf → fetch → object URL. Re-stubbed per
// test: the config sets unstubGlobals.
const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => new Response(new Blob(["%PDF"]), { status: 200 }));
vi.mock("@/lib/apiFetch", async (orig) => ({
  ...(await orig<typeof import("@/lib/apiFetch")>()),
  API_URL: "https://api.test",
  getAuthHeaders: async () => ({ Authorization: "Bearer t" }),
}));

const key = (keyId: string, label: string, folderName: string | null, credits: number) => ({
  keyId, label, keyPrefix: `mk_live_${keyId}`, status: "active" as const,
  folderId: folderName ? "f1" : null, folderName, credits, runs: 1, lastUsedAt: null,
  byAction: [{ action: "partner_oneclick_run", credits, runs: 1 }],
});

const PAYLOAD: Payload = {
  range: "mtd",
  orgs: [
    {
      orgId: "org-1", orgName: "Label Co", since: null, credits: 60, runs: 2,
      byKey: [key("k1", "Production", "Ingest", 60), key("k3", "Scratch", null, 0)],
      byFolder: [
        { folderId: "f1", name: "Ingest", keys: 1, credits: 60, runs: 2, byAction: [{ action: "partner_oneclick_run", credits: 60, runs: 2 }] },
        { folderId: null, name: "No folder", keys: 1, credits: 0, runs: 0, byAction: [] },
      ],
    },
    {
      orgId: "org-2", orgName: "Side Project", since: null, credits: 10, runs: 2,
      byKey: [key("k2", "Sandbox", null, 10)],
      byFolder: [{ folderId: null, name: "No folder", keys: 1, credits: 10, runs: 2, byAction: [] }],
    },
  ],
};

let payload: Payload | undefined;
let success = true;
let creditsEnabled = true;
const ranges: UsageRange[] = [];

vi.mock("@/hooks/useCreditUsage", () => ({
  useMyApiUsage: (range: UsageRange) => {
    ranges.push(range);
    return { data: payload, isSuccess: success, isError: false };
  },
  useCreditUsage: () => ({ data: { enabled: creditsEnabled } }),
}));

beforeEach(() => {
  payload = PAYLOAD;
  success = true;
  creditsEnabled = true;
  ranges.length = 0;
  fetchMock.mockClear();
  vi.stubGlobal("fetch", fetchMock);
  URL.createObjectURL = vi.fn(() => "blob:usage");
  URL.revokeObjectURL = vi.fn();
});
afterEach(cleanup);

describe("MyApiUsage", () => {
  it("totals every team's API spend and nests each key under its folder", () => {
    render(<MyApiUsage />);
    expect(screen.getByText("70 credits · 4 runs in this window")).toBeInTheDocument();
    expect(screen.getByText("Label Co")).toBeInTheDocument();
    expect(screen.getByText("Side Project")).toBeInTheDocument();
    // Key | Total | Runs | Mix | Last used — no per-tool columns.
    const prod = within(screen.getByText("Production").closest("tr")!).getAllByRole("cell");
    expect(prod.slice(1, 3).map((c) => c.textContent)).toEqual(["60", "1"]);
    expect(within(prod[3]).getByRole("img")).toHaveAttribute("aria-label", "OneClick 60");
    // Label Co's rows: the Ingest folder heading, its key, then the unfiled group.
    const rows = screen.getAllByRole("row").map((r) => r.textContent ?? "");
    const order = ["Ingest", "Production", "No folder", "Scratch"].map((t) => rows.findIndex((x) => x.startsWith(t)));
    expect(order).toEqual([...order].sort((a, b) => a - b));
    expect(order.every((i) => i >= 0)).toBe(true);
  });

  it("opens the breakdown dialog for the key that was clicked", () => {
    render(<MyApiUsage />);
    fireEvent.click(screen.getByRole("button", { name: "Production" }));
    const dialog = within(screen.getByRole("dialog"));
    expect(dialog.getByRole("heading", { name: "Production" })).toBeInTheDocument();
    expect(dialog.getByText(/API key · Ingest/)).toBeInTheDocument();
    // The share is measured against that team's window total, not every team's.
    expect(dialog.getByText("Share of total").parentElement).toHaveTextContent("100%");
  });

  it("downloads the report for the chosen range", async () => {
    render(<MyApiUsage />);
    fireEvent.click(screen.getByRole("radio", { name: "7D" }));
    fireEvent.click(screen.getByRole("button", { name: /download pdf/i }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(fetchMock.mock.calls[0][0]).toBe("https://api.test/me/api-usage/report.pdf?range=7d");
  });

  it("says so when the person has no keys", () => {
    payload = { range: "mtd", orgs: [] };
    render(<MyApiUsage />);
    expect(screen.getByText("You haven't created any API keys yet.")).toBeInTheDocument();
  });

  it("refetches with the chosen range", () => {
    render(<MyApiUsage />);
    expect(ranges[0]).toBe("mtd");
    fireEvent.click(screen.getByRole("radio", { name: "1Y" }));
    expect(ranges[ranges.length - 1]).toBe("1y");
  });

  it("shows dashes until the window has loaded", () => {
    payload = undefined;
    success = false;
    render(<MyApiUsage />);
    expect(screen.getByText("— credits · — runs in this window")).toBeInTheDocument();
  });

  it("renders nothing when credits are off", () => {
    creditsEnabled = false;
    const { container } = render(<MyApiUsage />);
    expect(container).toBeEmptyDOMElement();
  });
});
