// src/components/docs/PartnerApiConsole.tsx
// The "Trial a request" console beside the docs' API section. It calls the
// partner host directly — never our own backend — so it proves the exact path
// a partner's server takes, CORS and all. One console per tab (the free key
// check, then one per billed endpoint); all stay mounted and only the current
// one is shown, so a run in flight survives a tab switch (the server finishes,
// and bills, either way). Credits shown are what the response reported. The
// key lives in page state for the visit and is never persisted.
import { useId, useState, type ReactNode } from "react";
import { Code2, Download, Loader2, Play } from "lucide-react";
import { Tag } from "./apiBits";
import { RequestPreview, RowsEditor, rowsToObjects, validateRows, type Column, type Row } from "./RowsEditor";
import {
  CONSOLE_PRESETS,
  PARTNER_API_URL,
  REGISTRY_PRICE,
  ROYALTIES_PRICE,
  SPLIT_SHEET_PRESET,
  SPLIT_SHEET_PRICE,
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
  song: string;
  payee: { name: string; role: string };
  share: { type: string; percentage: number; basis: string };
  amounts: { gross: number; expenses: number; net: number; payable: number };
}

interface SplitParty {
  name: string;
  role: string;
  master_pct: number;
  publishing_pct: number;
  soundexchange_pct: number;
}

interface Billing {
  credits: number;
  replayed?: boolean;
  request_id?: string;
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
// What the response said it cost — never a local price table.
const credits = (b?: Billing | null) =>
  b ? (b.replayed ? "replay · 0 credits" : `${b.credits} credit${b.credits === 1 ? "" : "s"}`) : "billed";
const headerCredits = (res: Response) => {
  // Number(null) is 0, so check for the header before casting.
  const raw = res.headers.get("Msanii-Credits");
  const n = Number(raw);
  return raw !== null && Number.isFinite(n)
    ? credits({ credits: n, replayed: res.headers.get("Msanii-Replayed") === "true" })
    : "billed";
};

// A pre-stream HTTP error: nothing started, so nothing was charged.
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

// Read an event-stream to the end: the raw text plus the last `data:` event
// (there is exactly one on this API; heartbeats are comments).
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

// An error event on a 200 stream: nothing delivered, nothing billed.
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

// ---- rows: the list inputs -------------------------------------------------------
const ROYALTY_TYPES = ["master", "streaming"] as const; // what a calculation pays from
const BASES = ["gross", "net"] as const;
const PARTY_COLS: Column[] = [
  { key: "name", label: "Name", kind: "text", required: true, placeholder: "Jane Doe" },
  { key: "role", label: "Role", kind: "text", required: true, placeholder: "producer" },
];
const WORK_COLS: Column[] = [{ key: "title", label: "Title", kind: "text", required: true, placeholder: "Blue Sky" }];
const shareCols = (partyNames: () => string[]): Column[] => [
  { key: "party_name", label: "Party", kind: "select", options: partyNames, required: true },
  { key: "royalty_type", label: "Type", kind: "select", options: ROYALTY_TYPES, required: true },
  { key: "percentage", label: "%", kind: "number", required: true, min: 0, max: 100, width: "w-16" },
  { key: "basis", label: "Basis", kind: "select", options: BASES, placeholder: "contract default" },
];
const EXPENSE_COLS: Column[] = [
  { key: "description", label: "Description", kind: "text", placeholder: "Mastering" },
  { key: "amount", label: "Amount", kind: "number", required: true, min: 0, width: "w-20" },
  { key: "work_titles", label: "Songs", kind: "text", placeholder: "Blue Sky, Red Sun (blank = all)" },
];
const CONTRIBUTOR_COLS: Column[] = [
  { key: "name", label: "Name", kind: "text", required: true, placeholder: "Jane Doe" },
  { key: "role", label: "Role", kind: "text", required: true, placeholder: "Producer" },
  { key: "publishing_share", label: "Publishing %", kind: "number", min: 0, max: 100, width: "w-[76px]" },
  { key: "master_percentage", label: "Master %", kind: "number", min: 0, max: 100, width: "w-[76px]" },
];

// options only drive the <select>; rowsToObjects reads kind.
const buildTerms = (parties: Row[], works: Row[], shares: Row[]) => ({
  parties: rowsToObjects(PARTY_COLS, parties),
  works: rowsToObjects(WORK_COLS, works),
  royalty_shares: rowsToObjects(shareCols(() => []), shares),
});
const buildExpenses = (expenses: Row[]) =>
  expenses.map((e) => {
    const titles = (e.work_titles ?? "").split(",").map((s) => s.trim()).filter(Boolean);
    return {
      ...(e.description?.trim() ? { description: e.description.trim() } : {}),
      amount: Number(e.amount),
      ...(titles.length ? { work_titles: titles } : {}),
    };
  });

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

/** Every console's request lifecycle: clear the last result, time the call,
 * turn anything thrown into UNREACHABLE. The caller only RETURNS the result
 * for a response it actually got. */
function useRun() {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ConsoleResult | null>(null);

  const run = async (fn: (t0: number) => Promise<ConsoleResult>) => {
    setRunning(true);
    setResult((prev) => {
      // A document result holds an object URL — release it before dropping it.
      if (prev?.download) URL.revokeObjectURL?.(prev.download.href);
      return null;
    });
    const t0 = performance.now();
    try {
      setResult(await fn(t0));
    } catch {
      setResult(UNREACHABLE);
    } finally {
      setRunning(false);
    }
  };

  return { running, result, setResult, run };
}

// ---- check: GET /zoe/v1/models --------------------------------------------------

function CheckConsole({ apiKey, onApiKeyChange, hidden }: ConsoleProps) {
  const { running, result, run } = useRun();

  const submit = () =>
    run(async (t0) => {
      const res = await fetch(`${PARTNER_API_URL}/zoe/v1/models`, { headers: auth(apiKey) });
      const json = await res.json().catch(() => null);
      return {
        badge: String(res.status),
        ok: res.ok,
        contentType: "application/json",
        meta: `${secs(t0)} · free`,
        summary: res.ok ? "The key works. This check didn't use any credits." : explain(res.status),
        body: pretty(json),
      };
    });

  return (
    <Shell tag="free" hidden={hidden} result={result}>
      <KeyField value={apiKey} onChange={onApiKeyChange} />
      <RunButton running={running} disabled={!apiKey.trim()} onClick={submit}>
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
  const [parties, setParties] = useState<Row[]>(preset.parties);
  const [works, setWorks] = useState<Row[]>(preset.works);
  const [shares, setShares] = useState<Row[]>(preset.shares);
  const [expenses, setExpenses] = useState<Row[]>(preset.expenses);
  const [files, setFiles] = useState<File[]>([]);
  const { running, result, setResult, run } = useRun();

  const choosePreset = (next: string) => {
    const p = CONSOLE_PRESETS.find((x) => x.id === next) ?? CONSOLE_PRESETS[0];
    setPresetId(p.id);
    setStatement(p.statement);
    setParties(p.parties);
    setWorks(p.works);
    setShares(p.shares);
    setExpenses(p.expenses);
    setResult(null);
  };

  const SHARE_COLS = shareCols(() => parties.map((p) => p.name.trim()).filter(Boolean));
  const partyErrors = preset.pdf ? [] : validateRows(PARTY_COLS, parties);
  const workErrors = preset.pdf ? [] : validateRows(WORK_COLS, works);
  const shareErrors = preset.pdf ? [] : validateRows(SHARE_COLS, shares);
  const expenseErrors = validateRows(EXPENSE_COLS, expenses);
  const blocked = partyErrors.length + workErrors.length + shareErrors.length + expenseErrors.length > 0;
  const terms = buildTerms(parties, works, shares);
  const expenseBody = buildExpenses(expenses);
  const preview = {
    statement: "statement.csv (the text above)",
    ...(preset.pdf ? { contracts: "<the PDF files>" } : { contract_terms: terms }),
    ...(expenseBody.length ? { expenses: expenseBody } : {}),
  };

  const submit = () => {
    // Bad input never leaves the page; the server would 422 it anyway.
    if (preset.pdf && files.length === 0) return setResult(notSent("Choose at least one contract PDF"));
    if (blocked) return setResult(notSent("Fix the highlighted fields first"));
    const form = new FormData();
    form.append("statement", new File([statement], "statement.csv", { type: "text/csv" }));
    if (preset.pdf) files.forEach((f) => form.append("contracts", f));
    else form.append("contract_terms", JSON.stringify(terms));
    if (expenseBody.length) form.append("expenses", JSON.stringify(expenseBody));

    run(async (t0) => {
      const res = await fetch(`${PARTNER_API_URL}/oneclick/v1/royalties`, { method: "POST", headers: auth(apiKey), body: form });
      if (!res.ok) return httpError(res, t0);
      const { raw, last } = await readStream(res);
      const body = streamBody(raw, last);
      if (last?.type !== "result") return streamError(t0, last, body, "The calculation failed.");
      const payments = (last.payments as Payment[]) ?? [];
      // A billed 200 must never read as unbilled, so an absent summary is
      // derived rather than thrown.
      const summary = (last.summary as { payments: number; total_payable: number; expense_review_required: boolean }) ?? {
        payments: payments.length,
        total_payable: payments.reduce((a, p) => a + p.amounts.payable, 0),
        expense_review_required: payments.some((p) => p.share.basis === "net"),
      };
      return {
        badge: "200",
        ok: true,
        contentType: "text/event-stream",
        meta: `${secs(t0)} · ${credits(last.billing as Billing)}`,
        summary: (
          <>
            <b className="text-[14px] text-foreground">{money(summary.total_payable)}</b> across {summary.payments} payment
            {summary.payments === 1 ? "" : "s"}
            {summary.expense_review_required ? " · review flagged" : ""}
          </>
        ),
        rows: payments.map((p) => ({
          title: `${p.song} — ${p.payee.name}`,
          value: money(p.amounts.payable),
          detail: `${p.share.percentage}% of ${p.share.basis} ${money(p.amounts.net)}${p.amounts.expenses ? ` (after ${money(p.amounts.expenses)} expenses)` : ""}`,
        })),
        body,
      };
    });
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
        <>
          <RowsEditor label="Party" columns={PARTY_COLS} rows={parties} onChange={setParties} blank={() => ({ name: "", role: "" })} addLabel="Add party" errors={partyErrors} />
          <RowsEditor label="Work" columns={WORK_COLS} rows={works} onChange={setWorks} blank={() => ({ title: "" })} addLabel="Add work" errors={workErrors} />
          <RowsEditor label="Share" columns={SHARE_COLS} rows={shares} onChange={setShares} blank={() => ({ party_name: "", royalty_type: "master", percentage: "", basis: "" })} addLabel="Add share" errors={shareErrors} />
        </>
      )}
      <RowsEditor label="Expense" columns={EXPENSE_COLS} rows={expenses} onChange={setExpenses} blank={() => ({ description: "", amount: "", work_titles: "" })} addLabel="Add expense" errors={expenseErrors} min={0} />
      <RequestPreview body={preview} />
      <RunButton running={running} disabled={!apiKey.trim()} onClick={submit}>
        Run request
      </RunButton>
      <Hint>
        A real run against your key. It spends credits from the team balance, the same as a call from your own server — the
        response below is what the API returned for the inputs above, and the result below says what it cost. A run that ends in an
        error costs nothing.
      </Hint>
    </Shell>
  );
}

// ---- registry: POST /registry/v1/splits --------------------------------

function RegistryConsole({ apiKey, onApiKeyChange, hidden }: ConsoleProps) {
  const id = useId();
  const [files, setFiles] = useState<File[]>([]);
  const [artist, setArtist] = useState("");
  const { running, result, setResult, run } = useRun();

  const submit = () => {
    if (files.length === 0) return setResult(notSent("Choose at least one contract PDF"));
    const form = new FormData();
    files.forEach((f) => form.append("contracts", f));
    if (artist.trim()) form.append("main_artist_name", artist.trim());

    run(async (t0) => {
      const res = await fetch(`${PARTNER_API_URL}/registry/v1/splits`, { method: "POST", headers: auth(apiKey), body: form });
      if (!res.ok) return httpError(res, t0);
      const { raw, last } = await readStream(res);
      const body = streamBody(raw, last);
      if (last?.type !== "result") return streamError(t0, last, body, "The contract couldn't be parsed.");
      const terms = last.contract_terms as { parties: unknown[]; works: unknown[]; royalty_shares: unknown[] };
      const splits = last.splits as { main_artist: string | null; parties: SplitParty[] };
      return {
        badge: "200",
        ok: true,
        contentType: "text/event-stream",
        meta: `${secs(t0)} · ${credits(last.billing as Billing)}`,
        summary: (
          <>
            <b className="text-[14px] text-foreground">{terms.parties.length}</b> parties · {terms.works.length} works ·{" "}
            {terms.royalty_shares.length} shares{artist.trim() && !splits.main_artist ? " · main artist not found" : ""}
          </>
        ),
        rows: splits.parties.map((p) => ({
          title: `${p.name}${p.name === splits.main_artist ? " (main artist)" : ""}`,
          value: `${pct(p.master_pct)} master`,
          detail: `${p.role} · publishing ${pct(p.publishing_pct)}${p.soundexchange_pct ? ` · SoundExchange ${pct(p.soundexchange_pct)}` : ""}`,
        })),
        body,
      };
    });
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
      <RunButton running={running} disabled={!apiKey.trim()} onClick={submit}>
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
  const [sheet, setSheet] = useState({
    work_title: SPLIT_SHEET_PRESET.work_title,
    work_type: SPLIT_SHEET_PRESET.work_type,
    split_type: SPLIT_SHEET_PRESET.split_type,
    date: SPLIT_SHEET_PRESET.date,
  });
  const [format, setFormat] = useState<"pdf" | "docx">("pdf");
  const [contributors, setContributors] = useState<Row[]>(SPLIT_SHEET_PRESET.contributors);
  const { running, result, setResult, run } = useRun();
  const setField = (key: keyof typeof sheet) => (e: { target: { value: string } }) => setSheet((s) => ({ ...s, [key]: e.target.value }));

  const errors = validateRows(CONTRIBUTOR_COLS, contributors);
  const titleMissing = !sheet.work_title.trim();
  const dateMissing = !sheet.date.trim();
  const body = { ...sheet, format, contributors: rowsToObjects(CONTRIBUTOR_COLS, contributors) };

  const submit = () => {
    if (titleMissing || dateMissing || errors.length) return setResult(notSent("Fix the highlighted fields first"));
    run(async (t0) => {
      const res = await fetch(`${PARTNER_API_URL}/splitsheet/v1/documents`, {
        method: "POST",
        headers: { ...auth(apiKey), "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) return httpError(res, t0);
      const blob = await res.blob();
      const title = sheet.work_title.replace(/[^a-zA-Z0-9._-]/g, "_");
      const name = `Split_Sheet_${title}.${format}`;
      const href = typeof URL.createObjectURL === "function" ? URL.createObjectURL(blob) : "";
      return {
        badge: "200",
        ok: true,
        contentType: format === "pdf" ? "application/pdf" : "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        meta: `${secs(t0)} · ${headerCredits(res)}`,
        summary: "The finished document. Save it, or open it to check the layout.",
        download: { href, name, bytes: blob.size },
      };
    });
  };

  return (
    <Shell tag={`${SPLIT_SHEET_PRICE} credits`} hidden={hidden} result={result}>
      <KeyField value={apiKey} onChange={onApiKeyChange} />
      <Field label="work_title" htmlFor={`${id}-title`}>
        <input id={`${id}-title`} type="text" value={sheet.work_title} onChange={setField("work_title")} aria-invalid={titleMissing} className={`${INPUT} ${titleMissing ? "border-destructive" : ""}`} />
        {titleMissing && <p role="alert" className="mt-0.5 text-[11px] text-destructive">Work title is required</p>}
      </Field>
      <Field label="date" htmlFor={`${id}-date`}>
        <input id={`${id}-date`} type="text" value={sheet.date} onChange={setField("date")} aria-invalid={dateMissing} className={`${INPUT} ${dateMissing ? "border-destructive" : ""}`} />
        {dateMissing && <p role="alert" className="mt-0.5 text-[11px] text-destructive">Date is required</p>}
      </Field>
      <div className="grid grid-cols-3 gap-2">
        <Field label="work_type" htmlFor={`${id}-type`}>
          <select id={`${id}-type`} value={sheet.work_type} onChange={setField("work_type")} className={SELECT}>
            <option value="single">single</option>
            <option value="album">album</option>
            <option value="ep">ep</option>
          </select>
        </Field>
        <Field label="split_type" htmlFor={`${id}-split`}>
          <select id={`${id}-split`} value={sheet.split_type} onChange={setField("split_type")} className={SELECT}>
            <option value="both">both</option>
            <option value="publishing">publishing</option>
            <option value="master">master</option>
          </select>
        </Field>
        <Field label="format" htmlFor={`${id}-format`}>
          <select id={`${id}-format`} value={format} onChange={(e) => setFormat(e.target.value as "pdf" | "docx")} className={SELECT}>
            <option value="pdf">pdf</option>
            <option value="docx">docx</option>
          </select>
        </Field>
      </div>
      <RowsEditor label="Contributor" columns={CONTRIBUTOR_COLS} rows={contributors} onChange={setContributors} blank={() => ({ name: "", role: "", publishing_share: "", master_percentage: "" })} addLabel="Add contributor" errors={errors} />
      <RequestPreview body={body} />
      <RunButton running={running} disabled={!apiKey.trim()} onClick={submit}>
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
  const { running, result, run } = useRun();

  const submit = () =>
    run(async (t0) => {
      const res = await fetch(`${PARTNER_API_URL}/zoe/v1/chat/completions`, {
        method: "POST",
        headers: { ...auth(apiKey), "Content-Type": "application/json" },
        body: JSON.stringify({ model: "zoe", messages: [{ role: "user", content: message }] }),
      });
      if (!res.ok) return httpError(res, t0);
      const json = await res.json().catch(() => null);
      return {
        badge: "200",
        ok: true,
        contentType: "application/json",
        meta: `${secs(t0)} · ${credits(json?.billing as Billing)}`,
        body: pretty(json),
      };
    });

  return (
    <Shell tag={`${ZOE_PRICE} credits`} hidden={hidden} result={result}>
      <KeyField value={apiKey} onChange={onApiKeyChange} />
      <Field label="Message" htmlFor={id}>
        <textarea id={id} rows={3} value={message} onChange={(e) => setMessage(e.target.value)} className={AREA} />
      </Field>
      <RunButton running={running} disabled={!apiKey.trim() || !message.trim()} onClick={submit}>
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
