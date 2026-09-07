import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { useState } from "react";

const BASE = "https://partner.msanii.test";

type Kind = "check" | "royalties" | "registry" | "splitsheet" | "zoe";

// PARTNER_API_URL is read at module scope, so the env is stubbed before the
// component is imported and the module cache reset per mount.
// null = no partner URL configured (undefined would pick the default).
async function mount(kind: Kind, baseUrl: string | null = BASE) {
  vi.resetModules();
  vi.stubEnv("VITE_PARTNER_API_URL", baseUrl ?? "");
  const { PartnerApiConsole } = await import("../PartnerApiConsole");
  function Harness() {
    const [key, setKey] = useState("mk_live_secret");
    return <PartnerApiConsole kind={kind} apiKey={key} onApiKeyChange={setKey} />;
  }
  render(<Harness />);
}

const fetchMock = () => globalThis.fetch as ReturnType<typeof vi.fn>;

// The one console that is visible: the others are mounted with hidden.
function visible() {
  const shown = document.querySelectorAll("section[data-console]:not([hidden])");
  expect(shown.length).toBe(1);
  return within(shown[0] as HTMLElement);
}

const run = (v: ReturnType<typeof within>) => fireEvent.click(v.getByRole("button", { name: /run request|check the key/i }));
const status = (v: ReturnType<typeof within>) => v.getByRole("status");

const jsonResponse = (status: number, body: unknown) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => body,
});

// A text/event-stream body the way fetch hands it back: a reader of byte chunks.
function sseResponse(text: string) {
  const bytes = new TextEncoder().encode(text);
  let sent = false;
  return {
    ok: true,
    status: 200,
    body: {
      getReader: () => ({
        read: async () => (sent ? { done: true, value: undefined } : ((sent = true), { done: false, value: bytes })),
      }),
    },
    json: async () => ({}),
    text: async () => text,
  };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

afterEach(() => {
  cleanup();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("PartnerApiConsole — check", () => {
  it("GETs the model list as the free key check", async () => {
    fetchMock().mockResolvedValue(jsonResponse(200, { object: "list", data: [{ id: "zoe" }] }));
    await mount("check");
    const v = visible();
    // No endpoint picker, no body, no question to type: the only free route.
    expect(v.queryByLabelText(/endpoint|message|question/i)).not.toBeInTheDocument();
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    const [url, init] = fetchMock().mock.calls[0];
    expect(url).toBe(`${BASE}/zoe/v1/models`);
    expect(init.method).toBeUndefined();
    expect(init.headers.Authorization).toBe("Bearer mk_live_secret");
    expect(status(v).textContent).toContain("free");
    expect(status(v).textContent).toContain("The key works");
    expect(document.body.textContent).not.toMatch(/\/me\b|\/test\b/);
  });

  it("explains a rejected key in words, on top of the status", async () => {
    fetchMock().mockResolvedValue(jsonResponse(401, { detail: { code: "invalid_key" } }));
    await mount("check");
    run(visible());
    await waitFor(() => expect(screen.getByText(/wasn't accepted/i)).toBeInTheDocument());
    expect(screen.getByText("401")).toBeInTheDocument();
  });

  it("is replaced by a note when no partner URL is configured", async () => {
    await mount("check", null);
    expect(screen.queryByLabelText("API key")).not.toBeInTheDocument();
    expect(screen.getByText(/isn't available because no partner API URL/i)).toBeInTheDocument();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });
});

describe("PartnerApiConsole — royalties", () => {
  const RESULT = {
    type: "result",
    summary: { payments: 1, total_payable: 400, expense_review_required: true },
    payments: [
      { song: "Blue Sky", payee: { name: "Jane Doe", role: "producer" }, share: { type: "master", percentage: 50, basis: "net" },
        amounts: { gross: 1000, expenses: 200, net: 800, payable: 400 } },
    ],
    billing: { credits: 30, request_id: "r1" },
  };

  it("sends one multipart POST with the statement as a file and contract_terms built from rows, then reads the stream", async () => {
    fetchMock().mockResolvedValueOnce(sseResponse(`: ping\n\ndata: ${JSON.stringify(RESULT)}\n\n`));
    await mount("royalties");
    const v = visible();
    fireEvent.change(v.getByLabelText("Sample input"), { target: { value: "exp" } });
    // A second party, typed into rows — never into JSON.
    fireEvent.click(v.getByRole("button", { name: "Add party" }));
    fireEvent.change(v.getByLabelText("Party 2 Name"), { target: { value: "Sam Ray" } });
    fireEvent.change(v.getByLabelText("Party 2 Role"), { target: { value: "writer" } });
    const preview = JSON.parse(v.getByLabelText("Request body").textContent!);
    expect(preview.contract_terms.parties).toEqual([{ name: "Jane Doe", role: "producer" }, { name: "Sam Ray", role: "writer" }]);
    run(v);

    await waitFor(() => expect(status(v)).toBeInTheDocument());
    expect(fetchMock()).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock().mock.calls[0];
    expect(url).toBe(`${BASE}/oneclick/v1/royalties`);
    expect(init.method).toBe("POST");
    expect(init.headers.Authorization).toBe("Bearer mk_live_secret");
    const form = init.body as FormData;
    expect((form.get("statement") as File).name).toBe("statement.csv");
    expect(form.getAll("contracts")).toHaveLength(0);
    // The preview IS the body sent.
    expect(JSON.parse(form.get("contract_terms") as string)).toEqual(preview.contract_terms);
    expect(JSON.parse(form.get("expenses") as string)).toEqual([{ description: "Studio time", amount: 200, work_titles: ["Blue Sky"] }]);
    const s = status(v).textContent;
    expect(s).toContain("400.00");
    expect(s).toContain("review flagged");
    expect(s).toContain("Blue Sky — Jane Doe");
    expect(s).toContain("50% of net 800.00 (after 200.00 expenses)");
    // The charge comes from the response, never from a local price table.
    expect(s).toContain("30 credits");
    expect(s).not.toContain("base");
  });

  it("blocks the send on an invalid row and says which cell", async () => {
    await mount("royalties");
    const v = visible();
    fireEvent.change(v.getByLabelText("Share 1 %"), { target: { value: "150" } });
    expect(v.getByRole("alert")).toHaveTextContent("% must be between 0 and 100");
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    expect(status(v).textContent).toContain("Fix the highlighted fields");
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("shows a stream error event as unbilled", async () => {
    fetchMock().mockResolvedValueOnce(
      sseResponse(`data: ${JSON.stringify({ type: "error", code: "NO_SONG_MATCHES", message: "The contract covers songs that don't appear in this royalty statement.", suggestion: "…", details: {}, billing: { credits: 0 } })}\n\n`)
    );
    await mount("royalties");
    const v = visible();
    fireEvent.change(v.getByLabelText("Sample input"), { target: { value: "none" } });
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    expect(status(v).textContent).toContain("no credits spent");
    expect(status(v).textContent).toContain("NO_SONG_MATCHES");
  });

  it("labels a replay as not charged again", async () => {
    fetchMock().mockResolvedValueOnce(sseResponse(`data: ${JSON.stringify({ ...RESULT, billing: { credits: 0, replayed: true, request_id: "r1" } })}\n\n`));
    await mount("royalties");
    const v = visible();
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    expect(status(v).textContent).toContain("replay · 0 credits");
  });

  it("still reports the charge when the result frame carries no summary", async () => {
    const { summary, ...noSummary } = RESULT;
    fetchMock().mockResolvedValueOnce(sseResponse(`data: ${JSON.stringify(noSummary)}\n\n`));
    await mount("royalties");
    const v = visible();
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    const s = status(v).textContent;
    expect(s).toContain("400.00");
    expect(s).toContain("review flagged");
    expect(s).toContain("30 credits");
    expect(s).not.toContain("no credits spent");
  });

  it("switches to a PDF picker for the PDF preset and drops contract_terms", async () => {
    await mount("royalties");
    const v = visible();
    fireEvent.change(v.getByLabelText("Sample input"), { target: { value: "pdf" } });
    expect(v.getByLabelText("contracts")).toHaveAttribute("type", "file");
    expect(v.queryByLabelText("Party 1 Name")).not.toBeInTheDocument();
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    expect(status(v).textContent).toContain("Choose at least one contract PDF");
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("treats a 402 as nothing started", async () => {
    fetchMock().mockResolvedValueOnce(jsonResponse(402, { detail: { code: "insufficient_credits", price: 30, balance: 10 } }));
    await mount("royalties");
    run(visible());
    await waitFor(() => expect(screen.getByText(/below the price of this run/i)).toBeInTheDocument());
    expect(screen.getByText("402")).toBeInTheDocument();
    expect(screen.getByRole("status").textContent).toContain("no credits spent");
  });
});

describe("PartnerApiConsole — registry", () => {
  const RESULT = {
    type: "result",
    contract_terms: { parties: [{ name: "Jane Doe", role: "producer", aliases: [] }], works: [{ title: "Blue Sky" }], royalty_shares: [{ party_name: "Jane Doe", royalty_type: "master", percentage: 50 }] },
    splits: { main_artist: "Jane Doe", parties: [{ name: "Jane Doe", role: "producer", master_pct: 50, publishing_pct: 0, soundexchange_pct: 0 }] },
    billing: { credits: 30, request_id: "r2" },
  };

  it("posts the PDFs and the main artist, and renders both views", async () => {
    fetchMock().mockResolvedValueOnce(sseResponse(`: ping\n\ndata: ${JSON.stringify(RESULT)}\n\n`));
    await mount("registry");
    const v = visible();
    fireEvent.change(v.getByLabelText("contracts"), { target: { files: [new File(["%PDF-1.4"], "deal.pdf", { type: "application/pdf" })] } });
    fireEvent.change(v.getByLabelText(/main_artist_name/), { target: { value: "Jane Doe" } });
    run(v);

    await waitFor(() => expect(status(v)).toBeInTheDocument());
    const [url, init] = fetchMock().mock.calls[0];
    expect(url).toBe(`${BASE}/registry/v1/splits`);
    const form = init.body as FormData;
    expect((form.getAll("contracts") as File[]).map((f) => f.name)).toEqual(["deal.pdf"]);
    expect(form.get("main_artist_name")).toBe("Jane Doe");
    const s = status(v).textContent;
    expect(s).toContain("1 parties · 1 works · 1 shares");
    expect(s).toContain("Jane Doe (main artist)");
    expect(s).toContain("50% master");
    expect(s).toContain("30 credits");
  });

  it("says when the main artist was not found", async () => {
    fetchMock().mockResolvedValueOnce(sseResponse(`data: ${JSON.stringify({ ...RESULT, splits: { ...RESULT.splits, main_artist: null } })}\n\n`));
    await mount("registry");
    const v = visible();
    fireEvent.change(v.getByLabelText("contracts"), { target: { files: [new File(["x"], "deal.pdf")] } });
    fireEvent.change(v.getByLabelText(/main_artist_name/), { target: { value: "Nobody" } });
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    expect(status(v).textContent).toContain("main artist not found");
    expect(status(v).textContent).not.toContain("(main artist)");
  });

  it("refuses to send without a PDF", async () => {
    await mount("registry");
    const v = visible();
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    expect(status(v).textContent).toContain("Choose at least one contract PDF");
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("shows an unreadable contract as an unbilled error event", async () => {
    fetchMock().mockResolvedValueOnce(
      sseResponse(`data: ${JSON.stringify({ type: "error", code: "CONTRACT_UNREADABLE", message: "We couldn't read this contract.", suggestion: "…", details: { reason: "no text" }, billing: { credits: 0 } })}\n\n`)
    );
    await mount("registry");
    const v = visible();
    fireEvent.change(v.getByLabelText("contracts"), { target: { files: [new File(["x"], "scan.pdf")] } });
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    expect(status(v).textContent).toContain("couldn't read this contract");
    expect(status(v).textContent).toContain("no credits spent");
  });
});

describe("PartnerApiConsole — split sheet", () => {
  const fileResponse = (headers: Record<string, string>) => ({
    ok: true,
    status: 200,
    headers: { get: (k: string) => headers[k] ?? null },
    blob: async () => new Blob(["%PDF-1.4 sheet"], { type: "application/pdf" }),
    json: async () => ({}),
  });

  it("POSTs the body built from rows with the chosen format, reads the credits header and offers the file", async () => {
    vi.stubGlobal("URL", { ...URL, createObjectURL: vi.fn(() => "blob:sheet"), revokeObjectURL: vi.fn() });
    fetchMock().mockResolvedValueOnce(fileResponse({ "Msanii-Credits": "20", "Msanii-Request-Id": "r3" }));
    await mount("splitsheet");
    const v = visible();
    fireEvent.change(v.getByLabelText("format"), { target: { value: "docx" } });
    fireEvent.click(v.getByRole("button", { name: "Add contributor" }));
    fireEvent.change(v.getByLabelText("Contributor 3 Name"), { target: { value: "Ada Lee" } });
    fireEvent.change(v.getByLabelText("Contributor 3 Role"), { target: { value: "Mixer" } });
    fireEvent.change(v.getByLabelText("Contributor 3 Master %"), { target: { value: "10" } });
    const preview = JSON.parse(v.getByLabelText("Request body").textContent!);
    run(v);

    await waitFor(() => expect(status(v)).toBeInTheDocument());
    const [url, init] = fetchMock().mock.calls[0];
    expect(url).toBe(`${BASE}/splitsheet/v1/documents`);
    expect(init.headers["Content-Type"]).toBe("application/json");
    const sent = JSON.parse(init.body);
    expect(sent).toEqual(preview);
    expect(sent.format).toBe("docx");
    expect(sent.work_title).toBe("Blue Sky");
    expect(sent.contributors).toHaveLength(3);
    expect(sent.contributors[2]).toEqual({ name: "Ada Lee", role: "Mixer", master_percentage: 10 });
    const link = v.getByRole("link", { name: /Split_Sheet_Blue_Sky\.docx/ });
    expect(link).toHaveAttribute("href", "blob:sheet");
    expect(link).toHaveAttribute("download", "Split_Sheet_Blue_Sky.docx");
    expect(status(v).textContent).toContain("20 credits");
  });

  it("labels a replay from the headers", async () => {
    vi.stubGlobal("URL", { ...URL, createObjectURL: vi.fn(() => "blob:sheet"), revokeObjectURL: vi.fn() });
    fetchMock().mockResolvedValueOnce(fileResponse({ "Msanii-Credits": "0", "Msanii-Replayed": "true", "Msanii-Request-Id": "r3" }));
    await mount("splitsheet");
    const v = visible();
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    expect(status(v).textContent).toContain("replay · 0 credits");
  });

  it("blocks the send on an invalid contributor", async () => {
    await mount("splitsheet");
    const v = visible();
    fireEvent.change(v.getByLabelText("Contributor 1 Name"), { target: { value: "" } });
    expect(v.getByRole("alert")).toHaveTextContent("Name is required");
    run(v);
    await waitFor(() => expect(status(v)).toBeInTheDocument());
    expect(status(v).textContent).toContain("Fix the highlighted fields");
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });
});

describe("PartnerApiConsole — zoe", () => {
  it("posts an OpenAI-shaped completion for model zoe and shows the body with its credits", async () => {
    const body = { id: "chatcmpl-1", object: "chat.completion", model: "zoe", choices: [{ index: 0, message: { role: "assistant", content: "A mechanical royalty is…" }, finish_reason: "stop" }], billing: { credits: 5, request_id: "r4" } };
    fetchMock().mockResolvedValueOnce(jsonResponse(200, body));
    await mount("zoe");
    const v = visible();
    fireEvent.change(v.getByLabelText("Message"), { target: { value: "What is a mechanical royalty?" } });
    run(v);

    await waitFor(() => expect(status(v)).toBeInTheDocument());
    const [url, init] = fetchMock().mock.calls[0];
    expect(url).toBe(`${BASE}/zoe/v1/chat/completions`);
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(init.body)).toEqual({ model: "zoe", messages: [{ role: "user", content: "What is a mechanical royalty?" }] });
    expect(status(v).textContent).toContain("A mechanical royalty is…");
    expect(status(v).textContent).toContain("5 credits");
    expect(status(v).textContent).not.toContain("tokens");
  });

  it("shows a 502 as unbilled", async () => {
    fetchMock().mockResolvedValueOnce(jsonResponse(502, { detail: { code: "zoe_failed" } }));
    await mount("zoe");
    run(visible());
    await waitFor(() => expect(screen.getByText("502")).toBeInTheDocument());
    expect(screen.getByRole("status").textContent).toContain("no credits spent");
  });
});
