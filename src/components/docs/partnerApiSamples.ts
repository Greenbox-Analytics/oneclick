// src/components/docs/partnerApiSamples.ts
// Base URL, prices, Python samples and console presets for the API section of
// the docs page. Kept out of the component files so fast refresh keeps working
// (only-export-components). Mirrors docs/partner-api-reference.md — keep the
// two in step.

export const PARTNER_API_URL = (import.meta.env.VITE_PARTNER_API_URL || "").replace(/\/$/, "");

// Base prices at the time of writing (credit_prices: partner_oneclick_run,
// partner_registry_parse, partner_split_sheet, partner_zoe_message). The 402
// body carries the live `price`.
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
# event == {"type": "result", "payments": [...], "total_payments": 3, "expense_review_required": false}
#       or {"type": "error", "code": "…", "message": "…", "suggestion": "…", "details": {...}}`,
  },
  royaltiesTerms: {
    title: "From terms you already hold (no PDF, no AI)",
    code: `terms = {
    "parties": [{"name": "Jane Doe", "role": "producer"}],
    "works": [{"title": "Blue Sky"}],
    "royalty_shares": [
        {"party_name": "Jane Doe", "royalty_type": "master", "percentage": 50, "basis": "gross"}
    ],
}
r = requests.post(
    f"{BASE}/oneclick/v1/royalties",
    headers=AUTH,
    files={"statement": open("statement.xlsx", "rb")},
    data={"contract_terms": json.dumps(terms)},
    stream=True,
)`,
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
splits = event["splits"]          # {"parties": [{"name", "master_pct", "publishing_pct", ...}], "main_artist_found"}`,
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

// ---- console presets ----------------------------------------------------------
// Sample inputs for the "Trial a request" console. Every preset is a real,
// valid request; the outcome described in its label is what the API does.

export const STATEMENT_SAMPLE = `Title,Net Payable
Blue Sky,1000.00
Red Sun,500.00`;

const TERMS_GROSS = `{
  "parties": [{"name": "Jane Doe", "role": "producer"}],
  "works": [{"title": "Blue Sky"}],
  "royalty_shares": [
    {"party_name": "Jane Doe", "royalty_type": "master",
     "percentage": 50, "basis": "gross"}
  ]
}`;

const TERMS_NET = TERMS_GROSS.replace('"basis": "gross"', '"basis": "net"');

const TERMS_NO_MATCH = TERMS_GROSS.replace('"Blue Sky"', '"Purple Rain"');

export interface ConsolePreset {
  id: string;
  label: string;
  statement: string;
  /** Empty when the preset sends PDFs: exactly one of the two goes on the wire. */
  terms: string;
  expenses: string;
  pdf: boolean;
}

export const CONSOLE_PRESETS: ConsolePreset[] = [
  { id: "min", label: "Smallest valid statement + terms", statement: STATEMENT_SAMPLE, terms: TERMS_GROSS, expenses: "[]", pdf: false },
  { id: "pdf", label: "Statement + contract PDF", statement: STATEMENT_SAMPLE, terms: "", expenses: "[]", pdf: true },
  {
    id: "exp",
    label: "Net basis with expenses",
    statement: STATEMENT_SAMPLE,
    terms: TERMS_NET,
    expenses: `[{"description": "Studio time", "amount": 200,
  "work_titles": ["Blue Sky"]}]`,
    pdf: false,
  },
  { id: "none", label: "No song matches (error)", statement: STATEMENT_SAMPLE, terms: TERMS_NO_MATCH, expenses: "[]", pdf: false },
];

// The split sheet console's editable body (format is picked separately).
export const SPLIT_SHEET_SAMPLE = `{
  "work_title": "Blue Sky",
  "work_type": "single",
  "split_type": "both",
  "date": "6 September 2026",
  "contributors": [
    {"name": "Jane Doe", "role": "Producer",
     "publishing_share": 50, "master_percentage": 50},
    {"name": "Sam Ray", "role": "Writer",
     "publishing_share": 50, "master_percentage": 50}
  ]
}`;

// Zoe on the API is stateless — no stored contracts — so the sample question
// is one she can answer from general knowledge.
export const ZOE_SAMPLE_MESSAGE = "What is a mechanical royalty, and who collects it?";
