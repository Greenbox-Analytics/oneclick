// src/components/docs/PartnerApiConsole.tsx
// The "Trial a request" console beside the docs' API section
// (Documentation.tsx → ApiContent). It calls the partner host directly — never
// through our own backend — so what it proves is the exact path a partner's
// server will take, CORS and all. One console per kind of tab:
//   check       GET /zoe/v1/models                the free key check
//   royalties   POST /oneclick/v1/royalties       a real, billed run
//   registry    POST /registry/v1/splits  a real, billed run
//   splitsheet  POST /splitsheet/v1/documents     a real, billed run
//   zoe         POST /zoe/v1/chat/completions     a real, billed run
// All stay mounted and only the current one is shown, so a run in flight
// survives a tab switch (the server finishes — and bills — either way). A
// billed 200 is labelled with the tool's base price: the API has no balance
// read, and the true charge sits in the team's ledger. The key lives in page
// state for the visit and is never persisted.
import { useId, useState, type ReactNode } from "react";
import { Code2, Download, Loader2, Play } from "lucide-react";
import { Tag } from "./apiBits";
import {
  CONSOLE_PRESETS,
  PARTNER_API_URL,
  REGISTRY_PRICE,
  ROYALTIES_PRICE,
  SPLIT_SHEET_PRICE,
  SPLIT_SHEET_SAMPLE,
  ZOE_PRICE,
  ZOE_SAMPLE_MESSAGE,
} from "./partnerApiSamples";

export type ConsoleKind = "check" | "royalties" | "registry" | "splitsheet" | "zoe";

export interface PartnerApiConsoleProps {
  kind: ConsoleKind;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
}

// ---- response model -----------------------------------------------------------

interface Payment {
  song_title: string;
  party_name: string;
  percentage: number;
  basis: string;
  net_amount: number;
  expenses_applied: number;
  amount_to_pay: number;
}

interface SplitParty {
  name: string;
  role: string;
  master_pct: number;
  publishing_pct: number;
  soundexchange_pct: number;
  is_main_artist: boolean;
}

interface ConsoleResult {
  badge: string;
  ok: boolean;
  contentType?: string;
  meta: string;
  summary?: ReactNode;
  rows?: { title: string; value: string; detail: string }[];
  download?: { href: string; name: string; bytes: number };
  body?: string;
}

const pretty = (x: unknown) => JSON.stringify(x, null, 2);
const money = (n: number) => n.toFixed(2);
const pct = (n: number) => `${Number.isInteger(n) ? n : n.toFixed(1)}%`;
const secs = (t0: number) => `${((performance.now() - t0) / 1000).toFixed(1)} s`;
const auth = (key: string) => ({ Authorization: `Bearer ${key.trim()}` });
// A billed deliverable. The base is the published price and a floor; a large
// PDF run can cost more, which only the team's ledger shows.
const billed = (base: number) => `billed (base ${base})`;

// What a person can do about a pre-stream HTTP error. Nothing was started, so
// nothing was charged — every one of these is "no credits spent".
function explain(status: number): string {
  if (status === 401) return "That key wasn't accepted. Check you copied the whole key, and that it hasn't been revoked.";
  if (status === 402) return "The team's balance is below the price of this run. Nothing was started.";
  if (status === 422) return "The API rejected the request. The body says which field.";
  if (status === 413) return "Too large: the contracts are over 20 MB or 10 files, or the statement is over 10 MB.";
  return "The API couldn't answer just now. Try again in a moment.";
}

const UNREACHABLE: ConsoleResult = {
  badge: "no response",
  ok: false,
  meta: "no credits spent",
  summary: "Couldn't reach the API. Check the base URL and try again.",
};

const notSent = (why: string): ConsoleResult => ({
  badge: "not sent",
  ok: false,
  meta: "no credits spent",
  summary: `${why} — fix it and run again.`,
});

async function httpError(res: Response, t0: number): Promise<ConsoleResult> {
  const json = await res.json().catch(() => null);
  return {
    badge: String(res.status),
    ok: false,
    contentType: "application/json",
    meta: `${secs(t0)} · no credits spent`,
    summary: explain(res.status),
    body: pretty(json),
  };
}

// Read a text/event-stream to the end and return the raw text plus the last
// `data:` event (there is exactly one on this API; heartbeats are comments).
async function readStream(res: Response): Promise<{ raw: string; last: Record<string, unknown> | null }> {
  let raw = "";
  const reader = res.body?.getReader();
  if (reader) {
    const dec = new TextDecoder();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      raw += dec.decode(value, { stream: true });
    }
  } else {
    raw = await res.text();
  }
  let last: Record<string, unknown> | null = null;
  for (const line of raw.split("\n")) {
    if (!line.startsWith("data:")) continue;
    try {
      last = JSON.parse(line.slice(5));
    } catch {
      // A partial frame — keep the previous one.
    }
  }
  return { raw, last };
}

function streamBody(raw: string, last: unknown): string {
  const pings = raw.split("\n").filter((l) => l.startsWith(":")).length;
  const heartbeat = pings ? `: ping${pings > 1 ? ` ×${pings}` : ""}\n\n` : "";
  return `${heartbeat}data: ${last ? pretty(last) : "(none)"}`;
}

// An error event on a 200 stream: nothing was delivered, nothing is billed.
const streamError = (t0: number, last: Record<string, unknown> | null, body: string, fallback: string): ConsoleResult => ({
  badge: "error",
  ok: false,
  contentType: "text/event-stream",
  meta: `${secs(t0)} · no credits spent`,
  summary: (last?.message as string) || fallback,
  body,
});

// ---- shared pieces ------------------------------------------------------------

const INPUT =
  "block h-[34px] w-full rounded-lg border border-border bg-background px-2.5 font-mono text-[12px] text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-[3px] focus:ring-primary/15";
const SELECT =
  "block h-[34px] w-full rounded-lg border border-border bg-background px-2 text-[12.5px] text-foreground focus:border-primary focus:outline-none focus:ring-[3px] focus:ring-primary/15";
const AREA =
  "block w-full resize-none rounded-lg border border-border bg-muted/50 px-2.5 py-2 font-mono text-[11.5px] leading-relaxed text-foreground focus:border-primary focus:outline-none focus:ring-[3px] focus:ring-primary/15";
const FILE =
  "block w-full text-[12px] text-muted-foreground file:mr-2 file:rounded-md file:border file:border-border file:bg-background file:px-2 file:py-1 file:text-[12px] file:font-semibold file:text-foreground";

function Field({ label, htmlFor, children }: { label: string; htmlFor: string; children: ReactNode }) {
  return (
    <div className="mb-2.5">
      <label htmlFor={htmlFor} className="mb-1 block text-[11.5px] font-semibold text-foreground">
        {label}
      </label>
      {children}
    </div>
  );
}

function KeyField({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const id = useId();
  return (
    <Field label="API key" htmlFor={id}>
      <input
        id={id}
        type="password"
        autoComplete="off"
        spellCheck={false}
        placeholder="mk_live_…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={INPUT}
      />
    </Field>
  );
}

function RunButton({ running, disabled, onClick, children }: {
  running: boolean; disabled?: boolean; onClick: () => void; children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={running || disabled}
      className="mt-0.5 inline-flex h-[38px] w-full items-center justify-center gap-1.5 rounded-lg bg-primary text-[13.5px] font-semibold text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
    >
      {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3 w-3 fill-current" />}
      {running ? "Running…" : children}
    </button>
  );
}

const Hint = ({ children }: { children: ReactNode }) => (
  <p className="mt-2 text-[11.5px] leading-relaxed text-muted-foreground">{children}</p>
);

function ResultView({ r }: { r: ConsoleResult }) {
  return (
    <div className="border-t border-border bg-muted/50" role="status">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <span
          className={`inline-flex h-5 items-center rounded-[5px] px-1.5 font-mono text-[10.5px] font-semibold ${r.ok ? "bg-primary/10 text-primary" : "bg-destructive/10 text-destructive"}`}
        >
          {r.badge}
        </span>
        {r.contentType && <span className="font-mono text-[10.5px] text-muted-foreground">{r.contentType}</span>}
        <span className="ml-auto font-mono text-[10.5px] text-muted-foreground">{r.meta}</span>
      </div>
      {r.summary && (
        <div className="flex items-baseline gap-2 border-b border-border px-3 py-2 text-[12.5px] text-muted-foreground">{r.summary}</div>
      )}
      {r.rows?.map((row, i) => (
        <div key={i} className="grid grid-cols-[1fr_auto] gap-x-2.5 gap-y-0.5 border-b border-border px-3 py-2 text-[12.5px]">
          <span className="font-semibold text-foreground">{row.title}</span>
          <span className="text-right font-mono font-semibold text-primary">{row.value}</span>
          <span className="col-span-2 font-mono text-[11.5px] text-muted-foreground">{row.detail}</span>
        </div>
      ))}
      {r.download && (
        <div className="flex items-center gap-2 border-b border-border px-3 py-2 text-[12.5px]">
          <a href={r.download.href} download={r.download.name} className="inline-flex items-center gap-1.5 font-semibold text-primary hover:underline">
            <Download className="h-3.5 w-3.5" /> {r.download.name}
          </a>
          <span className="ml-auto font-mono text-[11px] text-muted-foreground">{(r.download.bytes / 1024).toFixed(1)} KB</span>
        </div>
      )}
      {r.body && (
        <pre className="overflow-x-auto px-3 py-2.5 font-mono text-[11px] leading-relaxed text-foreground">{r.body}</pre>
      )}
    </div>
  );
}

function Shell({ tag, hidden, result, children }: {
  tag: string; hidden: boolean; result: ConsoleResult | null; children: ReactNode;
}) {
  return (
    <section hidden={hidden} data-console="" className="overflow-hidden rounded-xl border border-border bg-card">
      <header className="flex items-center gap-2 border-b border-border bg-muted/50 px-3 py-2.5">
        <Code2 className="h-3.5 w-3.5 text-primary" />
        <span className="text-[13px] font-bold text-foreground">Trial a request</span>
        <Tag className="ml-auto">{tag}</Tag>
      </header>
      <div className="p-3">{children}</div>
      {result && <ResultView r={result} />}
    </section>
  );
}

type ConsoleProps = Omit<PartnerApiConsoleProps, "kind"> & { hidden: boolean };

// ---- check: GET /zoe/v1/models --------------------------------------------------

function CheckConsole({ apiKey, onApiKeyChange, hidden }: ConsoleProps) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ConsoleResult | null>(null);

  const run = async () => {
    setRunning(true);
    setResult(null);
    const t0 = performance.now();
    try {
      const res = await fetch(`${PARTNER_API_URL}/zoe/v1/models`, { headers: auth(apiKey) });
      const json = await res.json().catch(() => null);
      setResult({
        badge: String(res.status),
        ok: res.ok,
        contentType: "application/json",
        meta: `${secs(t0)} · free`,
        summary: res.ok ? "The key works. This check didn't use any credits." : explain(res.status),
        body: pretty(json),
      });
    } catch {
      setResult(UNREACHABLE);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Shell tag="free" hidden={hidden} result={result}>
      <KeyField value={apiKey} onChange={onApiKeyChange} />
      <RunButton running={running} disabled={!apiKey.trim()} onClick={run}>
        Check the key · GET /zoe/v1/models
      </RunButton>
      <Hint>
        The model list is the one free route, so it doubles as the key check: a bad, revoked or expired key is a 401 here. Your
        key is used for the request only and never saved.
      </Hint>
    </Shell>
  );
}

// ---- royalties: POST /oneclick/v1/royalties -----------------------------------

function RoyaltiesConsole({ apiKey, onApiKeyChange, hidden }: ConsoleProps) {
  const id = useId();
  const [presetId, setPresetId] = useState(CONSOLE_PRESETS[0].id);
  const preset = CONSOLE_PRESETS.find((p) => p.id === presetId) ?? CONSOLE_PRESETS[0];
  const [statement, setStatement] = useState(preset.statement);
  const [terms, setTerms] = useState(preset.terms);
  const [expenses, setExpenses] = useState(preset.expenses);
  const [files, setFiles] = useState<File[]>([]);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ConsoleResult | null>(null);

  const choosePreset = (next: string) => {
    const p = CONSOLE_PRESETS.find((x) => x.id === next) ?? CONSOLE_PRESETS[0];
    setPresetId(p.id);
    setStatement(p.statement);
    setTerms(p.terms);
    setExpenses(p.expenses);
    setResult(null);
  };

  const run = async () => {
    // Malformed input never leaves the page: the server would 422 it anyway.
    if (preset.pdf && files.length === 0) return setResult(notSent("Choose at least one contract PDF"));
    if (!preset.pdf) {
      try {
        JSON.parse(terms);
      } catch {
        return setResult(notSent("contract_terms is not valid JSON"));
      }
    }
    if (expenses.trim()) {
      try {
        JSON.parse(expenses);
      } catch {
        return setResult(notSent("expenses is not valid JSON"));
      }
    }
    const form = new FormData();
    form.append("statement", new File([statement], "statement.csv", { type: "text/csv" }));
    if (preset.pdf) files.forEach((f) => form.append("contracts", f));
    else form.append("contract_terms", terms);
    if (expenses.trim()) form.append("expenses", expenses);

    setRunning(true);
    setResult(null);
    const t0 = performance.now();
    try {
      const res = await fetch(`${PARTNER_API_URL}/oneclick/v1/royalties`, { method: "POST", headers: auth(apiKey), body: form });
      if (!res.ok) return setResult(await httpError(res, t0));
      const { raw, last } = await readStream(res);
      const body = streamBody(raw, last);
      if (last?.type !== "result") return setResult(streamError(t0, last, body, "The calculation failed."));
      const payments = (last.payments as Payment[]) ?? [];
      const total = payments.reduce((a, p) => a + p.amount_to_pay, 0);
      setResult({
        badge: "200",
        ok: true,
        contentType: "text/event-stream",
        meta: `${secs(t0)} · ${billed(ROYALTIES_PRICE)}`,
        summary: (
          <>
            <b className="text-[14px] text-foreground">{money(total)}</b> across {payments.length} payment
            {payments.length === 1 ? "" : "s"}
            {last.expense_review_required ? " · review flagged" : ""}
          </>
        ),
        rows: payments.map((p) => ({
          title: `${p.song_title} — ${p.party_name}`,
          value: money(p.amount_to_pay),
          detail: `${p.percentage}% of ${p.basis} ${money(p.net_amount)}${p.expenses_applied ? ` (after ${money(p.expenses_applied)} expenses)` : ""}`,
        })),
        body,
      });
    } catch {
      setResult(UNREACHABLE);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Shell tag={`${ROYALTIES_PRICE} credits`} hidden={hidden} result={result}>
      <KeyField value={apiKey} onChange={onApiKeyChange} />
      <Field label="Sample input" htmlFor={`${id}-preset`}>
        <select id={`${id}-preset`} value={presetId} onChange={(e) => choosePreset(e.target.value)} className={SELECT}>
          {CONSOLE_PRESETS.map((p) => (
            <option key={p.id} value={p.id}>{p.label}</option>
          ))}
        </select>
      </Field>
      <Field label="statement.csv" htmlFor={`${id}-stmt`}>
        <textarea id={`${id}-stmt`} rows={4} value={statement} onChange={(e) => setStatement(e.target.value)} spellCheck={false} className={AREA} />
      </Field>
      {preset.pdf ? (
        <Field label="contracts" htmlFor={`${id}-pdf`}>
          <input id={`${id}-pdf`} type="file" accept=".pdf,application/pdf" multiple onChange={(e) => setFiles(Array.from(e.target.files ?? []))} className={FILE} />
          <Hint>Msanii&apos;s AI reads the terms from the PDFs, so <code>contract_terms</code> isn&apos;t sent. Up to 10 files, 20 MB in total.</Hint>
        </Field>
      ) : (
        <Field label="contract_terms" htmlFor={`${id}-terms`}>
          <textarea id={`${id}-terms`} rows={8} value={terms} onChange={(e) => setTerms(e.target.value)} spellCheck={false} className={AREA} />
        </Field>
      )}
      <Field label="expenses" htmlFor={`${id}-exp`}>
        <textarea id={`${id}-exp`} rows={2} value={expenses} onChange={(e) => setExpenses(e.target.value)} spellCheck={false} className={AREA} />
      </Field>
      <RunButton running={running} disabled={!apiKey.trim()} onClick={run}>
        Run request
      </RunButton>
      <Hint>
        A real run against your key. It spends {ROYALTIES_PRICE} credits from the team balance, the same as a call from your own server
        — the response below is what the API returned for the inputs above. A run that ends in an error costs nothing.
      </Hint>
    </Shell>
  );
}

// ---- registry: POST /registry/v1/splits --------------------------------

function RegistryConsole({ apiKey, onApiKeyChange, hidden }: ConsoleProps) {
  const id = useId();
  const [files, setFiles] = useState<File[]>([]);
  const [artist, setArtist] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ConsoleResult | null>(null);

  const run = async () => {
    if (files.length === 0) return setResult(notSent("Choose at least one contract PDF"));
    const form = new FormData();
    files.forEach((f) => form.append("contracts", f));
    if (artist.trim()) form.append("main_artist_name", artist.trim());

    setRunning(true);
    setResult(null);
    const t0 = performance.now();
    try {
      const res = await fetch(`${PARTNER_API_URL}/registry/v1/splits`, { method: "POST", headers: auth(apiKey), body: form });
      if (!res.ok) return setResult(await httpError(res, t0));
      const { raw, last } = await readStream(res);
      const body = streamBody(raw, last);
      if (last?.type !== "result") return setResult(streamError(t0, last, body, "The contract couldn't be parsed."));
      const terms = last.contract_terms as { parties: unknown[]; works: unknown[]; royalty_shares: unknown[] };
      const splits = last.splits as { parties: SplitParty[]; main_artist_found: boolean };
      setResult({
        badge: "200",
        ok: true,
        contentType: "text/event-stream",
        meta: `${secs(t0)} · ${billed(REGISTRY_PRICE)}`,
        summary: (
          <>
            <b className="text-[14px] text-foreground">{terms.parties.length}</b> parties · {terms.works.length} works ·{" "}
            {terms.royalty_shares.length} shares{artist.trim() && !splits.main_artist_found ? " · main artist not found" : ""}
          </>
        ),
        rows: splits.parties.map((p) => ({
          title: `${p.name}${p.is_main_artist ? " (main artist)" : ""}`,
          value: `${pct(p.master_pct)} master`,
          detail: `${p.role} · publishing ${pct(p.publishing_pct)}${p.soundexchange_pct ? ` · SoundExchange ${pct(p.soundexchange_pct)}` : ""}`,
        })),
        body,
      });
    } catch {
      setResult(UNREACHABLE);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Shell tag={`${REGISTRY_PRICE} credits`} hidden={hidden} result={result}>
      <KeyField value={apiKey} onChange={onApiKeyChange} />
      <Field label="contracts" htmlFor={`${id}-pdf`}>
        <input id={`${id}-pdf`} type="file" accept=".pdf,application/pdf" multiple onChange={(e) => setFiles(Array.from(e.target.files ?? []))} className={FILE} />
      </Field>
      <Field label="main_artist_name (optional)" htmlFor={`${id}-artist`}>
        <input id={`${id}-artist`} type="text" value={artist} onChange={(e) => setArtist(e.target.value)} placeholder="Jane Doe" className={INPUT} />
      </Field>
      <RunButton running={running} disabled={!apiKey.trim()} onClick={run}>
        Run request
      </RunButton>
      <Hint>
        A real parse against your key: Msanii&apos;s AI reads the PDFs and returns the deal as data. Up to 10 files, 20 MB in total;
        a contract that can&apos;t be read costs nothing.
      </Hint>
    </Shell>
  );
}

// ---- splitsheet: POST /splitsheet/v1/documents ---------------------------------

function SplitSheetConsole({ apiKey, onApiKeyChange, hidden }: ConsoleProps) {
  const id = useId();
  const [body, setBody] = useState(SPLIT_SHEET_SAMPLE);
  const [format, setFormat] = useState<"pdf" | "docx">("pdf");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ConsoleResult | null>(null);

  const run = async () => {
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(body);
    } catch {
      return setResult(notSent("The request body is not valid JSON"));
    }
    setRunning(true);
    setResult((prev) => {
      if (prev?.download) URL.revokeObjectURL?.(prev.download.href);
      return null;
    });
    const t0 = performance.now();
    try {
      const res = await fetch(`${PARTNER_API_URL}/splitsheet/v1/documents`, {
        method: "POST",
        headers: { ...auth(apiKey), "Content-Type": "application/json" },
        body: JSON.stringify({ ...parsed, format }),
      });
      if (!res.ok) return setResult(await httpError(res, t0));
      const blob = await res.blob();
      const title = String(parsed.work_title ?? "sheet").replace(/[^a-zA-Z0-9._-]/g, "_");
      const name = `Split_Sheet_${title}.${format}`;
      const href = typeof URL.createObjectURL === "function" ? URL.createObjectURL(blob) : "";
      setResult({
        badge: "200",
        ok: true,
        contentType: format === "pdf" ? "application/pdf" : "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        meta: `${secs(t0)} · ${billed(SPLIT_SHEET_PRICE)}`,
        summary: "The finished document. Save it, or open it to check the layout.",
        download: { href, name, bytes: blob.size },
      });
    } catch {
      setResult(UNREACHABLE);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Shell tag={`${SPLIT_SHEET_PRICE} credits`} hidden={hidden} result={result}>
      <KeyField value={apiKey} onChange={onApiKeyChange} />
      <Field label="Request body" htmlFor={`${id}-body`}>
        <textarea id={`${id}-body`} rows={12} value={body} onChange={(e) => setBody(e.target.value)} spellCheck={false} className={AREA} />
      </Field>
      <Field label="format" htmlFor={`${id}-format`}>
        <select id={`${id}-format`} value={format} onChange={(e) => setFormat(e.target.value as "pdf" | "docx")} className={SELECT}>
          <option value="pdf">pdf</option>
          <option value="docx">docx</option>
        </select>
      </Field>
      <RunButton running={running} disabled={!apiKey.trim()} onClick={run}>
        Run request
      </RunButton>
      <Hint>
        A real document against your key. No AI runs, so a sheet always costs exactly {SPLIT_SHEET_PRICE} credits — per document: the pdf
        and the docx of one sheet are two.
      </Hint>
    </Shell>
  );
}

// ---- zoe: POST /zoe/v1/chat/completions ---------------------------------------

function ZoeConsole({ apiKey, onApiKeyChange, hidden }: ConsoleProps) {
  const id = useId();
  const [message, setMessage] = useState(ZOE_SAMPLE_MESSAGE);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ConsoleResult | null>(null);

  const run = async () => {
    setRunning(true);
    setResult(null);
    const t0 = performance.now();
    try {
      const res = await fetch(`${PARTNER_API_URL}/zoe/v1/chat/completions`, {
        method: "POST",
        headers: { ...auth(apiKey), "Content-Type": "application/json" },
        body: JSON.stringify({ model: "zoe", messages: [{ role: "user", content: message }] }),
      });
      if (!res.ok) return setResult(await httpError(res, t0));
      const json = await res.json().catch(() => null);
      setResult({
        badge: "200",
        ok: true,
        contentType: "application/json",
        meta: `${secs(t0)} · ${billed(ZOE_PRICE)}`,
        body: pretty(json),
      });
    } catch {
      setResult(UNREACHABLE);
    } finally {
      setRunning(false);
    }
  };

  return (
    <Shell tag={`${ZOE_PRICE} credits`} hidden={hidden} result={result}>
      <KeyField value={apiKey} onChange={onApiKeyChange} />
      <Field label="Message" htmlFor={id}>
        <textarea id={id} rows={3} value={message} onChange={(e) => setMessage(e.target.value)} className={AREA} />
      </Field>
      <RunButton running={running} disabled={!apiKey.trim() || !message.trim()} onClick={run}>
        Run request
      </RunButton>
      <Hint>
        Charged per completion, the same as a call through the OpenAI SDK. Zoe on the API has no memory and no access to
        stored documents — this one message is all she sees.
      </Hint>
    </Shell>
  );
}

// ---- entry ----------------------------------------------------------------------

export function PartnerApiConsole({ kind, ...props }: PartnerApiConsoleProps) {
  if (!PARTNER_API_URL) {
    return (
      <Shell tag="unavailable" hidden={false} result={null}>
        <p className="text-[12.5px] leading-relaxed text-muted-foreground">
          The console isn&apos;t available because no partner API URL is configured for this site.
        </p>
      </Shell>
    );
  }
  return (
    <>
      <CheckConsole {...props} hidden={kind !== "check"} />
      <RoyaltiesConsole {...props} hidden={kind !== "royalties"} />
      <RegistryConsole {...props} hidden={kind !== "registry"} />
      <SplitSheetConsole {...props} hidden={kind !== "splitsheet"} />
      <ZoeConsole {...props} hidden={kind !== "zoe"} />
    </>
  );
}
