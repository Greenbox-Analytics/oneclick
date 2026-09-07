// CreateApiKeyDialog's expiry preset picker — the create panel's own test
// covers the rest of the dialog (name, folder, secret screen). Same mocking
// idiom as api-keys-panel.test.tsx: mock the mutation hooks, render for real.
import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { CreateApiKeyDialog } from "../CreateApiKeyDialog";
import { endOfLocalDayIso, expiryFromPreset } from "@/lib/partnerKeys";
import type { PartnerKeyFolder } from "@/hooks/usePartnerKeys";

const ORG = "org-1";
const NOW = new Date("2026-09-04T12:00:00Z");
const FOLDERS: PartnerKeyFolder[] = [];

const createMutate = vi.fn();
const createReset = vi.fn();
let createState: { isPending: boolean; error: Error | null; data?: { id: string; secret: string } } = {
  isPending: false,
  error: null,
};
const createFolderAsync = vi.fn();

vi.mock("@/hooks/usePartnerKeys", () => ({
  useCreatePartnerKey: () => ({ mutate: createMutate, reset: createReset, ...createState }),
  useCreatePartnerKeyFolder: () => ({ mutateAsync: createFolderAsync, isPending: false }),
}));

beforeEach(() => {
  vi.useFakeTimers({ now: NOW, toFake: ["Date"] });
  createState = { isPending: false, error: null };
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  createMutate.mockReset();
  createReset.mockClear();
  createFolderAsync.mockClear();
});

const openAndName = (name = "Production backend") => {
  render(<CreateApiKeyDialog orgId={ORG} folders={FOLDERS} open onOpenChange={() => {}} />);
  fireEvent.change(screen.getByLabelText(/^name/i), { target: { value: name } });
};

describe("CreateApiKeyDialog expiry", () => {
  it("defaults to Never checked and sends no expires_at", () => {
    openAndName();
    expect(screen.getByRole("radio", { name: "Never" })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByRole("button", { name: /^create$/i }));
    const [input] = createMutate.mock.calls[0];
    expect("expires_at" in input).toBe(false);
  });

  it("sends the resolved date when a duration preset is chosen", () => {
    openAndName();
    fireEvent.click(screen.getByRole("radio", { name: "1 month" }));
    fireEvent.click(screen.getByRole("button", { name: /^create$/i }));
    const [input] = createMutate.mock.calls[0];
    expect(input.expires_at).toBe(endOfLocalDayIso(expiryFromPreset("1m", NOW)!));
  });

  it("shows the date-picker trigger and nudge copy when Custom is chosen", () => {
    openAndName();
    fireEvent.click(screen.getByRole("radio", { name: "Custom" }));
    expect(screen.getByRole("button", { name: /pick a date/i })).toBeInTheDocument();
    expect(screen.getByText("Pick a date, or the key won't expire.")).toBeInTheDocument();
  });
});
