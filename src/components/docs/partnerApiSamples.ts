// src/components/docs/partnerApiSamples.ts
// Base URL, prices, Python samples, console presets and response walkthroughs
// for the API section of the docs page. Kept out of the component files so fast
// refresh keeps working (only-export-components).
//
// SOURCE OF TRUTH for everything the API section states about the wire format.
// The docs page (Documentation.tsx) and the trial console (PartnerApiConsole)
// both render FROM here, so those two cannot disagree. The partner handout,
// docs/partner-api-reference.md, is a hand-written copy of the same facts and
// is the ONLY file that has to be re-checked when these change — update it in
// the same commit. Direction is one-way: change this file first.

import type { Row } from "./RowsEditor";

export const PARTNER_API_URL = (import.meta.env.VITE_PARTNER_API_URL || "").replace(/\/$/, "");

// Base prices at the time of writing; a 402 body carries the live `price`.
export const ROYALTIES_PRICE = 30;
export const REGISTRY_PRICE = 30;
export const SPLIT_SHEET_PRICE = 20;
export const ZOE_PRICE = 5;

const BASE = PARTNER_API_URL || "https://<your-partner-api-url>";

const PREAMBLE = `import json, requests

BASE = "${BASE}"
AUTH = {"Authorization": "Bearer mk_live_…"}`;

export const API_SAMPLES = {
  models: {
    title: "Check a key — GET /zoe/v1/models",
    code: `${PREAMBLE}

r = requests.get(f"{BASE}/zoe/v1/models", headers=AUTH)
print(r.json())
# {"object": "list", "data": [{"id": "zoe", "object": "model", "created": 0, "owned_by": "msanii"}]}
# A bad, revoked or expired key is a 401 here — and on every other route.`,
  },
  royaltiesPdf: {
    title: "From contract PDFs — POST /oneclick/v1/royalties",
    code: `${PREAMBLE}

r = requests.post(
    f"{BASE}/oneclick/v1/royalties",
    headers={**AUTH, "Idempotency-Key": "statement-2026-09-001"},
    files=[
        ("statement", open("statement.xlsx", "rb")),
        ("contracts", open("producer-agreement.pdf", "rb")),  # repeat "contracts" for more PDFs
    ],
    data={"expenses": json.dumps([{"description": "Mastering", "amount": 250}])},
    stream=True,
)
r.raise_for_status()
# The reply is a stream of server-sent events: heartbeat lines start with ":",
# and the one "data:" line is the answer.
for line in r.iter_lines(decode_unicode=True):
    if line.startswith("data:"):
        event = json.loads(line[5:])
# event == {"type": "result", "summary": {...}, "payments": [...], "billing": {"credits": 30, ...}}
#       or {"type": "error", "code": "…", "message": "…", "suggestion": "…", "details": {...}}`,
  },
  royaltiesTerms: {
    title: "From terms you already hold (no PDF, no AI)",
    code: `terms = {
    "parties": [
        {"name": "Jane Doe", "role": "producer"},
        {"name": "Sam Ray", "role": "featured artist"},
    ],
    "works": [{"title": "Blue Sky"}, {"title": "Red Sun"}],
    "royalty_shares": [
        {"party_name": "Jane Doe", "royalty_type": "master", "percentage": 50, "basis": "gross"},
        {"party_name": "Sam Ray", "royalty_type": "master", "percentage": 10, "basis": "net"},
    ],
}
r = requests.post(
    f"{BASE}/oneclick/v1/royalties",
    headers=AUTH,
    files={"statement": open("statement.xlsx", "rb")},
    data={"contract_terms": json.dumps(terms)},
    stream=True,
)
# Lists are plain JSON arrays: add as many parties, works and shares as the deal has.`,
  },
  contractTerms: {
    title: "Splits: the deal as data — POST /registry/v1/splits",
    code: `${PREAMBLE}

r = requests.post(
    f"{BASE}/registry/v1/splits",
    headers={**AUTH, "Idempotency-Key": "deal-2026-09-001"},
    files=[("contracts", open("producer-agreement.pdf", "rb"))],  # repeat for more PDFs
    data={"main_artist_name": "Jane Doe"},  # optional: who the splits are built around
    stream=True,
)
r.raise_for_status()
for line in r.iter_lines(decode_unicode=True):
    if line.startswith("data:"):
        event = json.loads(line[5:])
terms = event["contract_terms"]   # exactly what /oneclick/v1/royalties takes as contract_terms
splits = event["splits"]          # {"main_artist": "Jane Doe" | None, "parties": [{"name", "role", "master_pct", "publishing_pct", "soundexchange_pct"}]}`,
  },
  splitSheet: {
    title: "A finished split sheet — POST /splitsheet/v1/documents",
    code: `${PREAMBLE}

r = requests.post(
    f"{BASE}/splitsheet/v1/documents",
    headers={**AUTH, "Idempotency-Key": "sheet-blue-sky-1"},
    json={
        "work_title": "Blue Sky",
        "date": "6 September 2026",
        "format": "pdf",  # or "docx"
        "contributors": [
            {"name": "Jane Doe", "role": "Producer", "publishing_share": 50, "master_percentage": 50},
            {"name": "Sam Ray", "role": "Writer", "publishing_share": 50, "master_percentage": 50},
        ],
    },
)
r.raise_for_status()
open("Split_Sheet_Blue_Sky.pdf", "wb").write(r.content)`,
  },
  zoe: {
    title: "Zoe with the OpenAI SDK — POST /zoe/v1/chat/completions",
    code: `from openai import OpenAI  # pip install openai

zoe = OpenAI(api_key="mk_live_…", base_url="${BASE}/zoe/v1")

r = zoe.chat.completions.create(
    model="zoe",
    messages=[{"role": "user", "content": "What is a mechanical royalty?"}],
)
print(r.choices[0].message.content)

# Streaming
for chunk in zoe.chat.completions.create(model="zoe", messages=[...], stream=True):
    print(chunk.choices[0].delta.content or "", end="")`,
  },
} as const;

// ---- console presets ----
// Every preset is a real, valid request; its label says what the API does.

const STATEMENT_SAMPLE = `Title,Net Payable
Blue Sky,1000.00
Red Sun,500.00`;

const JANE: Row = { name: "Jane Doe", role: "producer" };
const share = (basis: string): Row => ({ party_name: "Jane Doe", royalty_type: "master", percentage: "50", basis });

export interface ConsolePreset {
  id: string;
  label: string;
  statement: string;
  /** True when the preset sends PDFs — exactly one of the two goes on the wire. */
  pdf: boolean;
  parties: Row[];
  works: Row[];
  shares: Row[];
  expenses: Row[];
}

export const CONSOLE_PRESETS: ConsolePreset[] = [
  { id: "min", label: "Smallest valid statement + terms", statement: STATEMENT_SAMPLE, pdf: false, parties: [JANE], works: [{ title: "Blue Sky" }], shares: [share("gross")], expenses: [] },
  { id: "pdf", label: "Statement + contract PDF", statement: STATEMENT_SAMPLE, pdf: true, parties: [], works: [], shares: [], expenses: [] },
  { id: "exp", label: "Net basis with expenses", statement: STATEMENT_SAMPLE, pdf: false, parties: [JANE], works: [{ title: "Blue Sky" }], shares: [share("net")], expenses: [{ description: "Studio time", amount: "200", work_titles: "Blue Sky" }] },
  { id: "none", label: "No song matches (error)", statement: STATEMENT_SAMPLE, pdf: false, parties: [JANE], works: [{ title: "Purple Rain" }], shares: [share("gross")], expenses: [] },
];

// The split sheet console's starting rows; every cell is a string because the
// rows editor holds strings.
export const SPLIT_SHEET_PRESET = {
  work_title: "Blue Sky",
  work_type: "single",
  split_type: "both",
  date: "6 September 2026",
  contributors: [
    { name: "Jane Doe", role: "Producer", publishing_share: "50", master_percentage: "50" },
    { name: "Sam Ray", role: "Writer", publishing_share: "50", master_percentage: "50" },
  ] as Row[],
};

// The docs page's JSON example, DERIVED from the preset so the two can't
// describe different documents. Shares are numbers on the wire.
export const SPLIT_SHEET_SAMPLE = JSON.stringify(
  {
    ...SPLIT_SHEET_PRESET,
    contributors: SPLIT_SHEET_PRESET.contributors.map((c) => ({
      ...c,
      publishing_share: Number(c.publishing_share),
      master_percentage: Number(c.master_percentage),
    })),
  },
  null,
  2,
);

// Zoe on the API is stateless, so the sample is answerable from general
// knowledge.
export const ZOE_SAMPLE_MESSAGE = "What is a mechanical royalty, and who collects it?";

// ---- response walkthroughs ----
// One entry per top-level response key, rendered by ResponseExample.

export interface ResponseSection {
  key: string;
  note: string;
  json: string;
  fields?: [string, string, string][];
}

const BILLING_FIELDS: [string, string, string][] = [
  ["credits", "integer", "Credits charged for this call. 0 on a replay under the same Idempotency-Key."],
  ["replayed", "true", "Present only on a replay: this result was charged on its first run, not again."],
  ["request_id", "string", "The id the charge is recorded under — quote it to support."],
];

export const ROYALTIES_RESPONSE: ResponseSection[] = [
  {
    key: "summary",
    note: "The totals, first.",
    json: `"type": "result",
"summary": {"payments": 2, "total_payable": 600.0, "expense_review_required": true}`,
    fields: [
      ["payments", "integer", "How many payment lines follow."],
      ["total_payable", "number", "Sum of every line's payable amount, 2 dp."],
      ["expense_review_required", "boolean", "True when any line is on a net basis: the expense list changed the amounts and deserves a human check."],
    ],
  },
  {
    key: "payments",
    note: "One line per party per matched song.",
    json: `"payments": [
  {
    "song": "Blue Sky",
    "payee": {"name": "Jane Doe", "role": "producer"},
    "share": {"type": "master", "percentage": 50.0, "basis": "net"},
    "amounts": {"gross": 1000.0, "expenses": 200.0, "net": 800.0, "payable": 400.0}
  },
  {
    "song": "Red Sun",
    "payee": {"name": "Jane Doe", "role": "producer"},
    "share": {"type": "master", "percentage": 50.0, "basis": "net"},
    "amounts": {"gross": 500.0, "expenses": 100.0, "net": 400.0, "payable": 200.0}
  }
]`,
    fields: [
      ["song", "string", "The work's title as given in the contract."],
      ["payee.name, payee.role", "string", "Who is paid, and the role the contract gives them (or unknown)."],
      ["share.type", "string", "The income the share is paid from, as written in the contract (master, streaming…)."],
      ["share.percentage", "number", "The share applied, 0–100."],
      ["share.basis", "\"gross\" | \"net\"", "The basis actually applied."],
      ["amounts.gross", "number", "What the statement paid for this song, all matching rows summed."],
      ["amounts.expenses", "number", "Expenses deducted for this song; 0 on a gross share."],
      ["amounts.net", "number", "gross − expenses, floored at 0."],
      ["amounts.payable", "number", "net × percentage ÷ 100 — the figure to pay. Amounts are rounded to 2 dp."],
    ],
  },
  { key: "billing", note: "What this call cost.", json: `"billing": {"credits": ${ROYALTIES_PRICE}, "request_id": "a1b2c3…"}`, fields: BILLING_FIELDS },
];

export const ERROR_EVENT_RESPONSE: ResponseSection[] = [
  {
    key: "error event",
    note: "Instead of a result. HTTP is already 200, so check type.",
    json: `{
  "type": "error",
  "code": "NO_SONG_MATCHES",
  "message": "The contract covers songs that don't appear in this royalty statement.",
  "suggestion": "Make sure the statement is for the same release as the contract…",
  "details": {"contract_works": ["Blue Sky"], "statement_songs": ["Blue Skies (Live)"]},
  "billing": {"credits": 0}
}`,
    fields: [
      ["code", "string", "Stable, for your code to branch on (the list is below)."],
      ["message, suggestion", "string", "Safe to show a person."],
      ["details", "object", "Structured context to fix the input without a person in the loop."],
      ["billing.credits", "0", "An error event is never billed, and says so."],
    ],
  },
];

export const SPLITS_RESPONSE: ResponseSection[] = [
  {
    key: "contract_terms",
    note: "The deal, in exactly the shape the royalty calculation accepts.",
    json: `"type": "result",
"contract_terms": {
  "parties": [{"name": "Jane Doe", "role": "producer", "aliases": []}],
  "works": [{"title": "Blue Sky", "work_type": "song"}],
  "royalty_shares": [
    {"party_name": "Jane Doe", "royalty_type": "master", "percentage": 50.0, "terms": "…", "basis": "net"}
  ],
  "contract_summary": "…",
  "default_basis": null
}`,
  },
  {
    key: "splits",
    note: "The Registry's ownership view, one line per party.",
    json: `"splits": {
  "main_artist": "Jane Doe",
  "parties": [
    {"name": "Jane Doe", "role": "producer", "master_pct": 50.0, "publishing_pct": 0.0, "soundexchange_pct": 0.0}
  ]
}`,
    fields: [
      ["main_artist", "string | null", "The party matching main_artist_name, by the name the contract uses; null when the name wasn't found or none was sent."],
      ["parties[].name, role", "string", "The party as named in the contract. Parties with no master, publishing or SoundExchange share are left out; the main artist is always kept."],
      ["parties[].master_pct", "number", "Share of the sound recording's income, 0–100."],
      ["parties[].publishing_pct", "number", "Share of the composition's income, 0–100."],
      ["parties[].soundexchange_pct", "number", "Share of neighbouring-rights income the contract assigns, where it does."],
    ],
  },
  { key: "billing", note: "What this call cost.", json: `"billing": {"credits": ${REGISTRY_PRICE}, "request_id": "a1b2c3…"}`, fields: BILLING_FIELDS },
];

export const ZOE_RESPONSE: ResponseSection[] = [
  {
    key: "choices",
    note: "OpenAI's shape: the answer is choices[0].message.content.",
    json: `"id": "chatcmpl-…",
"object": "chat.completion",
"created": 1757030400,
"model": "zoe",
"choices": [
  {
    "index": 0,
    "message": {"role": "assistant", "content": "A mechanical royalty is paid to the songwriter and publisher each time a composition is reproduced…"},
    "finish_reason": "stop"
  }
]`,
  },
  {
    key: "billing",
    note: "What this answer cost. On a stream it rides on the final finish_reason: \"stop\" frame, before data: [DONE].",
    json: `"billing": {"credits": ${ZOE_PRICE}, "request_id": "…"}`,
    fields: [
      ["credits", "integer", "Credits charged for this answer. No token counts are returned."],
      ["request_id", "string", "Quote it to support."],
    ],
  },
];

export const SPLIT_SHEET_HEADERS: [string, string, string][] = [
  ["Content-Type", "application/pdf", "or application/vnd.openxmlformats-officedocument.wordprocessingml.document for docx."],
  ["Content-Disposition", "attachment; filename=\"Split_Sheet_<title>.<format>\"", "Characters outside letters, digits, . _ - are replaced by _."],
  ["Content-Length", "bytes", ""],
  ["Msanii-Credits", `${SPLIT_SHEET_PRICE}`, "Credits charged for this document; 0 on a replay."],
  ["Msanii-Request-Id", "id", "The id the charge is recorded under — quote it to support."],
  ["Msanii-Replayed", "true", "Present only on a replay under the same Idempotency-Key."],
];
