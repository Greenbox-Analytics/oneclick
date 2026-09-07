import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import "@testing-library/jest-dom/vitest";
import Documentation from "../Documentation";

const BASE = "https://partner.msanii.test";

// The API section reads VITE_PARTNER_API_URL at module scope, so the env must
// be stubbed BEFORE the page is imported — hoisted, rather than re-importing
// the page, which would hand it a second copy of react-router and no Router.
vi.hoisted(() => {
  vi.stubEnv("VITE_PARTNER_API_URL", "https://partner.msanii.test");
});

vi.mock("@/contexts/AuthContext", () => ({ useAuth: () => ({ user: null, signOut: vi.fn() }) }));
vi.mock("@/hooks/useSmartBack", () => ({ useSmartBack: () => vi.fn() }));
vi.mock("@/hooks/useCreditPacks", () => ({ useToolPrices: () => ({ data: undefined, isLoading: false }) }));

class FakeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

function mountDocs(query: string) {
  render(
    <MemoryRouter initialEntries={[`/docs?${query}`]}>
      <Documentation />
    </MemoryRouter>
  );
}

const sidebar = () => within(document.querySelector("nav.hidden") as HTMLElement);
const tabs = () => within(screen.getByRole("tablist"));
// The console that is showing (the others stay mounted, hidden).
const shownConsole = () => within(document.querySelector("section[data-console]:not([hidden])") as HTMLElement);

beforeEach(() => {
  vi.stubGlobal("scrollTo", vi.fn());
  vi.stubGlobal("fetch", vi.fn());
  vi.stubGlobal("IntersectionObserver", FakeObserver);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Documentation — sidebar", () => {
  it("has two folds, Platform and API, each collapsible", () => {
    mountDocs("section=getting-started");
    const nav = sidebar();
    const platform = nav.getByRole("button", { name: /^Platform$/ });
    const api = nav.getByRole("button", { name: /^API v1$/ });
    expect(platform).toHaveAttribute("aria-expanded", "true");
    expect(api).toHaveAttribute("aria-expanded", "true");
    // Platform keeps the product groups; API carries the reference nav.
    expect(nav.getByText("Roster & projects")).toBeInTheDocument();
    expect(nav.getByText("Tools")).toBeInTheDocument();
    expect(nav.getAllByText("Workspace")).toHaveLength(2); // the group and its first entry
    expect(nav.getByText("Endpoints")).toBeInTheDocument();
    expect(nav.getByText("Schemas")).toBeInTheDocument();
    // One row per tool; no account-level routes.
    expect(nav.getByText("/oneclick/v1/royalties")).toBeInTheDocument();
    expect(nav.getByText("/registry/v1/splits")).toBeInTheDocument();
    expect(nav.getByText("/splitsheet/v1/documents")).toBeInTheDocument();
    expect(nav.getByText("/zoe/v1/chat/completions")).toBeInTheDocument();
    expect(nav.queryByText("/me")).not.toBeInTheDocument();
    expect(nav.queryByText("/test")).not.toBeInTheDocument();
    expect(nav.queryByText("Authentication")).not.toBeInTheDocument();

    fireEvent.click(api);
    expect(api).toHaveAttribute("aria-expanded", "false");
    expect(nav.queryByText("Schemas")).not.toBeInTheDocument();
  });

  it("opens the API section on the right tab from a nav row, and lights that row", () => {
    mountDocs("section=getting-started");
    fireEvent.click(sidebar().getByText("contract_terms"));
    expect(tabs().getByRole("tab", { name: "Royalty calculation" })).toHaveAttribute("aria-selected", "true");
    expect(sidebar().getByText("contract_terms").closest("button")).toHaveClass("text-primary");
    expect(sidebar().getByText("/oneclick/v1/royalties").closest("button")).not.toHaveClass("text-primary");
    fireEvent.click(sidebar().getByText("contributor"));
    expect(tabs().getByRole("tab", { name: "Split sheet" })).toHaveAttribute("aria-selected", "true");
  });
});

describe("Documentation — API section", () => {
  it("is deep-linkable at /docs?section=api, opens on Overview, and shows the key check", () => {
    mountDocs("section=api");
    expect(tabs().getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    expect(tabs().getAllByRole("tab")).toHaveLength(7);
    // Connect lives on Overview now: base URL, bearer header, the free check.
    expect(screen.getByRole("heading", { name: "Connect" })).toBeInTheDocument();
    expect(screen.getByText(BASE)).toBeInTheDocument();
    expect(screen.getByText(/Authorization: Bearer/)).toBeInTheDocument();
    // Overview: every tool with its price, tool-first paths only.
    expect(screen.getAllByText("/oneclick/v1/royalties").length).toBeGreaterThan(0);
    expect(screen.getAllByText("/registry/v1/splits").length).toBeGreaterThan(0);
    expect(screen.getAllByText("/splitsheet/v1/documents").length).toBeGreaterThan(0);
    expect(screen.getAllByText("/zoe/v1/chat/completions").length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toContain("/partner/v1");
    expect(document.body.textContent).not.toMatch(/GET \/me\b|POST \/test\b|"\/me"|"\/test"/);
    expect(document.body.textContent).not.toMatch(/\bcurl\b/);
    // The free key check sits beside the non-billed tabs.
    const c = shownConsole();
    expect(c.getByLabelText("API key")).toBeInTheDocument();
    expect(c.getByRole("button", { name: /check the key/i })).toBeInTheDocument();
    expect(c.getByText("free")).toBeInTheDocument();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("sends the retired ?tab=auth to Overview, and the models row to its Connect block", () => {
    mountDocs("section=api&tab=auth");
    expect(tabs().getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    expect(tabs().queryByRole("tab", { name: "Authentication" })).not.toBeInTheDocument();
    fireEvent.click(tabs().getByRole("tab", { name: "Zoe" }));
    fireEvent.click(sidebar().getByText("/zoe/v1/models"));
    expect(tabs().getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("heading", { name: "Connect" })).toBeInTheDocument();
    expect(document.getElementById("connect")).not.toBeNull();
  });

  it("deep-links a tab with ?tab= and swaps the console to match", () => {
    mountDocs("section=api&tab=royalties");
    expect(tabs().getByRole("tab", { name: "Royalty calculation" })).toHaveAttribute("aria-selected", "true");
    // Inputs / Outputs, with the field-level reference under each.
    expect(screen.getByRole("heading", { name: "Inputs" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Outputs" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "The statement file" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Result event" })).toBeInTheDocument();
    expect(screen.getAllByText(/import json, requests/).length).toBeGreaterThan(0);
    const c = shownConsole();
    expect(c.getByLabelText("Sample input")).toBeInTheDocument();
    expect(c.getByLabelText("Party 1 Name")).toBeInTheDocument();
    expect(c.getByRole("button", { name: /run request/i })).toBeInTheDocument();
    // The result walkthrough, one section per top-level key.
    expect(screen.getByText("summary")).toBeInTheDocument();
    expect(screen.getByText("total_payable")).toBeInTheDocument();
    expect(screen.getAllByText("billing").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/"type": "result"/).length).toBeGreaterThan(0);
    expect(document.body.textContent).not.toMatch(/total_payments|amount_to_pay/);
  });

  it("moves between tabs from the strip, each with its own console", () => {
    mountDocs("section=api");
    fireEvent.click(tabs().getByRole("tab", { name: "Splits" }));
    expect(screen.getByRole("heading", { name: "Unreadable contracts" })).toBeInTheDocument();
    expect(screen.getByText("CONTRACT_UNREADABLE")).toBeInTheDocument();
    expect(screen.getByText("main_artist")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("main_artist_found");
    expect(shownConsole().getByLabelText("contracts")).toHaveAttribute("type", "file");
    expect(shownConsole().getByRole("button", { name: /run request/i })).toBeInTheDocument();

    fireEvent.click(tabs().getByRole("tab", { name: "Split sheet" }));
    expect(screen.getByRole("heading", { name: "Contributors" })).toBeInTheDocument();
    expect(screen.getAllByText("master_percentage").length).toBeGreaterThan(0);
    expect(shownConsole().getByLabelText("Request body")).toBeInTheDocument();
    expect(shownConsole().getByLabelText("format")).toBeInTheDocument();
    expect(screen.getByText("Msanii-Credits")).toBeInTheDocument();
    expect(shownConsole().getByRole("button", { name: /run request/i })).toBeInTheDocument();

    fireEvent.click(tabs().getByRole("tab", { name: "Errors" }));
    expect(screen.getByText("NO_SONG_MATCHES")).toBeInTheDocument();
    fireEvent.click(tabs().getByRole("tab", { name: "Zoe" }));
    expect(screen.getByRole("heading", { name: "OpenAI compatibility" })).toBeInTheDocument();
    expect(shownConsole().getByLabelText("Message")).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("prompt_tokens");
    expect(screen.getAllByText("billing").length).toBeGreaterThan(0);
    fireEvent.click(tabs().getByRole("tab", { name: "Billing & limits" }));
    expect(screen.getByText(/before the results arrived, costs nothing/i)).toBeInTheDocument();
  });

  it("does not render the API reference on another section", () => {
    mountDocs("section=integrations");
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("API key")).not.toBeInTheDocument();
    expect(screen.queryByText("NO_SONG_MATCHES")).not.toBeInTheDocument();
  });
});
