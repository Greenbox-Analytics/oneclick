import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, within } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { MemoryRouter } from "react-router-dom";
import { OrgApiKeysPanel } from "../OrgApiKeysPanel";
import type { PartnerKey, PartnerKeyFolder } from "@/hooks/usePartnerKeys";
import type { OrgUsage } from "@/hooks/useOrgs";
import { endOfLocalDayIso, expiryFromPreset, EXPIRY_PRESETS } from "@/lib/partnerKeys";

const ORG = "org-1";
const NOW = new Date("2026-09-04T12:00:00Z");

const root: PartnerKey = {
  id: "root", label: "Production backend", key_prefix: "mk_live_abcd",
  status: "active", expires_at: null, created_at: "2026-09-01T00:00:00+00:00",
  created_by: "u1", created_by_label: "admin@label.test", last_used_at: "2026-09-03T00:00:00+00:00",
  revoked_at: null, folder_id: "f1",
};
const FOLDERS: PartnerKeyFolder[] = [
  { id: "f1", org_id: ORG, name: "Ingest", created_at: "2026-09-01T00:00:00+00:00" },
  { id: "f2", org_id: ORG, name: "Reporting", created_at: "2026-09-02T00:00:00+00:00" },
];
// Older, and its creator can't be resolved — the Created-by cell must not guess.
const staging: PartnerKey = { ...root, id: "staging", label: "Staging", key_prefix: "mk_live_efgh",
  created_at: "2026-08-01T00:00:00+00:00", created_by: null, created_by_label: null, folder_id: null };
const expired: PartnerKey = { ...root, id: "exp", label: "Old integration", key_prefix: "mk_live_ijkl",
  expires_at: "2026-09-01T00:00:00+00:00" };
const revoked: PartnerKey = { ...root, id: "rev", label: "Leaked key", key_prefix: "mk_live_mnop", status: "revoked" };

// Column order of a key row: Name | Folder | Key | Created by | Status | ...
const COL = { folder: 1, createdBy: 3, credits: 5, runs: 6, lastUsed: 7 };

const spend = (keyId: string, credits: number, runs: number, lastUsedAt: string | null) => ({
  keyId, label: keyId, keyPrefix: `mk_live_${keyId}`, status: "active" as const,
  folderId: null, folderName: null, credits, runs, lastUsedAt, byAction: [],
});

let keysData: { keys: PartnerKey[]; folders: PartnerKeyFolder[] } | undefined;
let keysError = false;
let usageData: Pick<OrgUsage, "byKey"> | undefined;
let usageSuccess = true;
const createMutate = vi.fn();
const revokeMutate = vi.fn();
let createState: { isPending: boolean; error: Error | null; data?: { id: string; secret: string } } = { isPending: false, error: null };
// The real hook's reset() drops the minted key — that is what makes the secret unrecoverable.
const createReset = vi.fn(() => { createState = { isPending: false, error: null }; });
const setFolderMutate = vi.fn();
const createFolderAsync = vi.fn(async ({ name }: { name: string }) => ({ id: "f-new", org_id: ORG, name, created_at: "" }));

vi.mock("@/hooks/usePartnerKeys", () => ({
  usePartnerKeys: () => ({ data: keysData, isLoading: false, isError: keysError }),
  useCreatePartnerKey: () => ({ mutate: createMutate, reset: createReset, ...createState }),
  useRevokePartnerKey: () => ({ mutate: revokeMutate, isPending: false }),
  useSetPartnerKeyFolder: () => ({ mutate: setFolderMutate, isPending: false }),
  useCreatePartnerKeyFolder: () => ({ mutateAsync: createFolderAsync, isPending: false }),
}));
// Per-key spend comes from the usage hook the console already loads.
vi.mock("@/hooks/useOrgs", () => ({
  useOrgUsage: () => ({ data: usageData, isSuccess: usageSuccess }),
}));

beforeEach(() => {
  vi.useFakeTimers({ now: NOW, toFake: ["Date"] });
  keysData = { keys: [root, staging, expired, revoked], folders: FOLDERS };
  keysError = false;
  usageData = { byKey: [spend("root", 30, 1, null), spend("staging", 67, 2, null)] };
  usageSuccess = true;
  createState = { isPending: false, error: null };
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  createMutate.mockReset();
  revokeMutate.mockReset();
  createReset.mockClear();
  setFolderMutate.mockReset();
  createFolderAsync.mockClear();
});

describe("OrgApiKeysPanel table", () => {
  it("hides expired and revoked keys until Show inactive, and derives the Expired pill", () => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    expect(screen.getByText("Production backend")).toBeInTheDocument();
    expect(screen.queryByText("Old integration")).not.toBeInTheDocument();
    expect(screen.queryByText("Leaked key")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /show inactive/i }));
    const oldRow = screen.getByText("Old integration").closest("tr")!;
    expect(within(oldRow).getByText("Expired")).toBeInTheDocument();
    const leakedRow = screen.getByText("Leaked key").closest("tr")!;
    expect(within(leakedRow).getByText("Revoked")).toBeInTheDocument();
  });

  it("lists keys newest first with Created by and this period's spend", () => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    const rows = screen.getAllByRole("row").slice(1); // drop the header row
    expect(rows[0]).toHaveTextContent("Production backend");
    expect(rows[0]).toHaveTextContent("admin@label.test");
    expect(rows[1]).toHaveTextContent("Staging");
    expect(within(rows[1]).getAllByRole("cell")[COL.createdBy]).toHaveTextContent("—");
    expect(rows[1]).toHaveTextContent("67");
  });

  it("shows the empty state with no keys", () => {
    keysData = { keys: [], folders: [] };
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    expect(screen.getByText(/No API keys yet/)).toBeInTheDocument();
  });

  // Fix 2: a failed load must not read as "you have no keys".
  it("shows a failure card, not the empty state, when the keys query errors", () => {
    keysError = true;
    keysData = undefined;
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    expect(screen.getByText(/Couldn't load API keys/i)).toBeInTheDocument();
    expect(screen.queryByText(/No API keys yet/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /new api key/i })).not.toBeInTheDocument();
  });

  it("the inactive count matches the rows revealed", () => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    expect(screen.getAllByRole("row").slice(1)).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: /show inactive \(2\)/i }));
    expect(screen.getAllByRole("row").slice(1)).toHaveLength(4);
  });

  // Fix 3: a 0 that means "not loaded" reads as "this key spent nothing".
  it("renders — for spend while usage is unresolved, and 0 only once it has resolved", () => {
    usageData = undefined;
    usageSuccess = false;
    const { rerender } = render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    let cells = within(screen.getAllByRole("row")[1]).getAllByRole("cell");
    expect(cells[COL.credits]).toHaveTextContent("—");
    expect(cells[COL.runs]).toHaveTextContent("—");

    // Resolved with no row for this key: 0 is the truth.
    usageData = { byKey: [] };
    usageSuccess = true;
    rerender(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    cells = within(screen.getAllByRole("row")[1]).getAllByRole("cell");
    expect(cells[COL.credits]).toHaveTextContent("0");
    expect(cells[COL.runs]).toHaveTextContent("0");
  });

  // Fix 4: last_used_at is stamped on every resolve; usage.lastUsedAt only on a debited run this period.
  it("Last used prefers the key's own stamp over the period's newest debited run", () => {
    usageData = { byKey: [spend("root", 30, 1, "2025-01-15T00:00:00+00:00")] };
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    const cells = within(screen.getAllByRole("row")[1]).getAllByRole("cell");
    expect(cells[COL.lastUsed]).toHaveTextContent("2026");
    expect(cells[COL.lastUsed]).not.toHaveTextContent("2025");
  });

  it("shows each key's folder, and a dash when it has none", () => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getAllByRole("cell")[COL.folder]).toHaveTextContent("Ingest");
    expect(within(rows[1]).getAllByRole("cell")[COL.folder]).toHaveTextContent("—");
  });

  it("says inactive keys disappear after 30 days but keep counting", () => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    expect(
      screen.getByText(/Revoked and expired keys stay listed for 30 days, then disappear\. Their spend still counts\./),
    ).toBeInTheDocument();
  });

  it("names the row's Revoke button after its key", () => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    expect(screen.getByRole("button", { name: "Revoke key for Production backend" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Revoke key for Staging" })).toBeInTheDocument();
  });
});

describe("revoke confirm", () => {
  it("names the key, says it stops immediately, and revokes that key", () => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    const rows = screen.getAllByRole("row").slice(1);
    fireEvent.click(within(rows[1]).getByRole("button", { name: /revoke/i }));
    expect(screen.getByText(/Revoke "Staging"\?/)).toBeInTheDocument();
    expect(screen.getByText("Anything using this key stops working immediately.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^revoke key$/i }));
    expect(revokeMutate).toHaveBeenCalledWith({ orgId: ORG, keyId: "staging" }, expect.anything());
  });
});

describe("CreateApiKeyDialog", () => {
  it("offers no key type, and sends name plus the chosen expiry preset", () => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: /new api key/i }));
    // The only radios in the dialog are the expiry presets — no key-type selector.
    expect(screen.getAllByRole("radio")).toHaveLength(EXPIRY_PRESETS.length);
    const create = screen.getByRole("button", { name: /^create$/i });
    // aria-disabled, not disabled: it stays reachable and says what is missing.
    expect(create).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(create);
    expect(createMutate).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText(/^name/i), { target: { value: "Production backend" } });
    fireEvent.click(screen.getByRole("radio", { name: "1 month" }));
    expect(create).toHaveAttribute("aria-disabled", "false");
    fireEvent.click(create);
    const [input] = createMutate.mock.calls[0];
    expect(input).toMatchObject({ orgId: ORG, label: "Production backend" });
    expect("user_ref" in input).toBe(false);
    expect(input.expires_at).toBe(endOfLocalDayIso(expiryFromPreset("1m", NOW)));
  });

  it("says a name is still needed before one is typed", () => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: /new api key/i }));
    expect(screen.getByText(/Add a name to create this key/i)).toBeInTheDocument();
  });

  // The product's central promise: the secret is shown once, and only Done dismisses it.
  it("shows the secret after a successful create, survives Escape, and is gone after Done", async () => {
    createMutate.mockImplementation(() => {
      createState = { isPending: false, error: null, data: { id: "new", secret: "mk_live_SECRET" } };
    });
    const { rerender } = render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: /new api key/i }));
    fireEvent.change(screen.getByLabelText(/^name/i), { target: { value: "Production backend" } });
    expect(screen.queryByText("mk_live_SECRET")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^create$/i }));
    rerender(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    expect(screen.getByText("mk_live_SECRET")).toBeInTheDocument();
    expect(screen.getByText(/we can't show it again/i)).toBeInTheDocument();

    // Escape and a backdrop click must not throw away the one chance to copy it.
    fireEvent.keyDown(document, { key: "Escape", code: "Escape" });
    // Radix attaches its outside-pointer listener a macrotask after mount.
    await new Promise((r) => setTimeout(r, 0));
    fireEvent.pointerDown(document.body);
    fireEvent.click(document.body);
    // The dialog's own X calls onOpenChange directly — only the open-change guard stops it.
    fireEvent.click(screen.getByRole("button", { name: /^close$/i }));
    expect(screen.getByText("mk_live_SECRET")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^done$/i }));
    expect(createReset).toHaveBeenCalled();
    rerender(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    expect(screen.queryByText("mk_live_SECRET")).not.toBeInTheDocument();
    // Reopening shows the form again, never the spent secret.
    fireEvent.click(screen.getByRole("button", { name: /new api key/i }));
    expect(screen.queryByText("mk_live_SECRET")).not.toBeInTheDocument();
    expect(screen.getByLabelText(/^name/i)).toHaveValue("");
  });

  it("points at the manual fallback when the clipboard is unavailable", async () => {
    createState = { isPending: false, error: null, data: { id: "new", secret: "mk_live_SECRET" } };
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: vi.fn().mockRejectedValue(new Error("not allowed")) },
      configurable: true,
    });
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: /new api key/i }));
    fireEvent.click(screen.getByRole("button", { name: /^copy$/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/select the key above and copy it yourself/i);
    expect(screen.getByText("mk_live_SECRET")).toBeInTheDocument();
  });
});

describe("move to folder", () => {
  const openMove = (label: string) => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: `Move ${label} to a folder` }));
    return screen.getByLabelText("Folder");
  };

  it("assigns an existing folder to that key", () => {
    fireEvent.change(openMove("Staging"), { target: { value: "f2" } });
    fireEvent.click(screen.getByRole("button", { name: /^move key$/i }));
    expect(setFolderMutate).toHaveBeenCalledWith(
      { orgId: ORG, keyId: "staging", folderId: "f2" },
      expect.anything(),
    );
  });

  it("clears the folder back to none", () => {
    // Production backend starts in f1; the picker opens on it.
    const select = openMove("Production backend");
    expect(select).toHaveValue("f1");
    fireEvent.change(select, { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: /^move key$/i }));
    expect(setFolderMutate).toHaveBeenCalledWith(
      { orgId: ORG, keyId: "root", folderId: null },
      expect.anything(),
    );
  });

  it("creates the folder first when a new name is typed, then assigns it", async () => {
    fireEvent.change(openMove("Staging"), { target: { value: "__new__" } });
    // Nothing to move to until the folder is named.
    expect(screen.getByRole("button", { name: /^move key$/i })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("New folder name"), { target: { value: "Batch jobs" } });
    fireEvent.click(screen.getByRole("button", { name: /^move key$/i }));
    await vi.waitFor(() => expect(setFolderMutate).toHaveBeenCalled());
    expect(createFolderAsync).toHaveBeenCalledWith({ orgId: ORG, name: "Batch jobs" });
    expect(setFolderMutate).toHaveBeenCalledWith(
      { orgId: ORG, keyId: "staging", folderId: "f-new" },
      expect.anything(),
    );
  });
});

describe("create dialog folders", () => {
  const openCreate = () => {
    render(<MemoryRouter><OrgApiKeysPanel orgId={ORG} /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: /new api key/i }));
    fireEvent.change(screen.getByLabelText(/^name/i), { target: { value: "Ingest worker" } });
  };

  it("sends the chosen folder with the new key", async () => {
    openCreate();
    fireEvent.change(screen.getByLabelText("Folder"), { target: { value: "f1" } });
    fireEvent.click(screen.getByRole("button", { name: /^create$/i }));
    await vi.waitFor(() => expect(createMutate).toHaveBeenCalled());
    expect(createMutate.mock.calls[0][0]).toMatchObject({ orgId: ORG, label: "Ingest worker", folder_id: "f1" });
  });

  it("defaults to no folder", async () => {
    openCreate();
    fireEvent.click(screen.getByRole("button", { name: /^create$/i }));
    await vi.waitFor(() => expect(createMutate).toHaveBeenCalled());
    expect(createMutate.mock.calls[0][0].folder_id).toBeNull();
  });

  it("creates a typed folder first, then mints the key into it", async () => {
    openCreate();
    fireEvent.change(screen.getByLabelText("Folder"), { target: { value: "__new__" } });
    // The name is required before the key can be minted.
    fireEvent.click(screen.getByRole("button", { name: /^create$/i }));
    expect(createMutate).not.toHaveBeenCalled();
    expect(screen.getByText(/Name the new folder/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("New folder name"), { target: { value: "Batch jobs" } });
    fireEvent.click(screen.getByRole("button", { name: /^create$/i }));
    await vi.waitFor(() => expect(createMutate).toHaveBeenCalled());
    expect(createFolderAsync).toHaveBeenCalledWith({ orgId: ORG, name: "Batch jobs" });
    expect(createMutate.mock.calls[0][0].folder_id).toBe("f-new");
  });
});
