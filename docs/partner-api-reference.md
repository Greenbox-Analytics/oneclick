# Msanii Partner API — reference (v1)

The Partner API lets your own software run Msanii's tools: royalty calculations (a statement plus the contract that governs it, back comes who is owed what), splits (a contract PDF, back comes the deal as data), split sheets (the shares, back comes the signed-ready document) and Zoe, the music-business assistant. It is for your servers — a key is your team's credential and spends your team's credits, so it never belongs in a browser or a mobile app.

Everything on this page is what a partner integrating with Msanii needs. Nothing here requires a Msanii account beyond the team that issued your key.

## Base URL and authentication

Your base URL is shown on your team's page in Msanii, under **API documentation**. All paths below are relative to it.

Every request carries your key as a bearer token:

```
Authorization: Bearer mk_live_…
```

Keys are created by a team admin in Msanii (**Teams → API keys**). A key is shown once at creation and cannot be recovered, only revoked and replaced. Treat it like a password.

## Quick start (Python)

A drop-in client lives at [`examples/partner-api/msanii_partner.py`](../examples/partner-api/msanii_partner.py). It needs only `requests`.

```python
from msanii_partner import MsaniiPartner

api = MsaniiPartner("mk_live_…", base_url="https://<partner-api-host>")

api.models()   # free — the key check; a bad key is a 401 here
# ["zoe"]

api.chat([{"role": "user", "content": "What is a mechanical royalty?"}])   # Zoe
# "A mechanical royalty is paid to a songwriter or publisher each time a composition is reproduced…"

deal = api.parse_contract(["producer-agreement.pdf"], main_artist_name="Jane Doe")
deal["contract_terms"]   # exactly what calculate() takes — parse once, run every statement at the base price
deal["splits"]  # {"main_artist": "Jane Doe", "parties": [{"name": "Jane Doe", "master_pct": 50.0, "publishing_pct": 0.0, …}]}

pdf = api.split_sheet(
    work_title="Blue Sky",
    date="6 September 2026",
    contributors=[{"name": "Jane Doe", "role": "Producer", "publishing_share": 50, "master_percentage": 50}],
)
open("Split_Sheet_Blue_Sky.pdf", "wb").write(pdf)

result = api.calculate(
    "statement.xlsx",
    contract_terms={
        "parties": [{"name": "Jane Doe", "role": "producer"}],
        "works": [{"title": "Blue Sky"}],
        "royalty_shares": [
            {"party_name": "Jane Doe", "royalty_type": "master", "percentage": 50, "basis": "gross"}
        ],
    },
    idempotency_key="run-2026-09-05-001",
)
for p in result["payments"]:
    print(p["song"], p["payee"]["name"], p["amounts"]["payable"])
```

Zoe is OpenAI-compatible, so the official OpenAI SDK works without this client at all:

```python
from openai import OpenAI

zoe = OpenAI(api_key="mk_live_…", base_url="https://<partner-api-host>/zoe/v1")
r = zoe.chat.completions.create(model="zoe", messages=[{"role": "user", "content": "What is a mechanical royalty?"}])
print(r.choices[0].message.content)
```

Any HTTP client works; the sections below describe the wire format.

The same file doubles as a smoke test that exercises every endpoint against your key:

```bash
export MSANII_API_URL=https://<partner-api-host>
export MSANII_API_KEY=mk_live_…
python msanii_partner.py                                          # model probe, bad-key 401, two terms-mode runs, Zoe x2, split sheet
python msanii_partner.py --statement statement.xlsx --contract deal.pdf   # adds a PDF-parse run
```

The terms-mode runs are billed once (they reuse an `Idempotency-Key`); the two Zoe answers are billed each; the PDF run is billed each time its inputs change.

## Endpoints

| Method | Path | Billed | Purpose |
|---|---|---|---|
| `POST` | `/oneclick/v1/royalties` | Yes | Royalty calculation |
| `POST` | `/registry/v1/splits` | Yes | Splits: the deal as data, from contract PDFs |
| `POST` | `/splitsheet/v1/documents` | Yes | A finished split sheet, PDF or DOCX |
| `POST` | `/zoe/v1/chat/completions` | Yes | Zoe, OpenAI-compatible chat completions |
| `GET` | `/zoe/v1/models` | No | Lists the one model, `zoe` — also the free key check |

Paths are tool-first: each tool's routes live under its own prefix and version (`/oneclick/v1`, `/registry/v1`, `/splitsheet/v1`, `/zoe/v1`). There are no account-level routes: to check a key, call `GET /zoe/v1/models` — a missing, revoked or expired key is a `401` there, as on every other route.

### POST /oneclick/v1/royalties

One royalty statement plus one contract — either as PDF files that Msanii's AI reads, or as structured terms you already hold — and the response is the list of payments due. The request is `multipart/form-data`; the response is a stream of server-sent events (see [Response](#response)).

**Request** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `statement` | file | yes | The royalty statement: `.csv`, `.xlsx` or `.xls`, up to 10 MB. See [The statement file](#the-statement-file) |
| `contracts` | file, repeated | one of | Contract PDFs, up to 10 files and 20 MB in total. Send the field once per file. Several PDFs are merged into one set of terms |
| `contract_terms` | JSON string | one of | The contract, already structured. No AI runs. See [Contract terms](#contract-terms) |
| `expenses` | JSON string | no | Recoupable costs to deduct from net-basis shares. See [Expenses](#expenses) |

Send **exactly one** of `contracts` or `contract_terms`; both or neither is a `422`.

**Headers**

| Header | Notes |
|---|---|
| `Idempotency-Key` | Optional, any string. A retry with the same key and the same inputs in the same billing period is charged once. Recommended for every call — see [Billing](#billing) |

#### The statement file

One row per song earnings line. Msanii finds the two columns it needs by header name, case-insensitively:

- **Title** — a header containing `title`, `song`, `track`, `release title`, `track name`, `song name` or `release name`.
- **Amount payable** — a header containing `net payable`, `net payment`, `net earnings`, `net revenue`, `net amount`, `payable to artist` or `artist payable`; failing that, one containing `payable`, `amount`, `earnings`, `payment` or `revenue` that does not say withheld, deduction, fee, commission or advance. Headers that match nothing are resolved by fuzzy and then AI matching.

Rows with the same title are summed, so a statement with one line per month or per platform needs no pre-aggregation. The smallest valid statement:

```csv
Title,Net Payable
Blue Sky,1000.00
Red Sun,500.00
```

#### Contract terms

`contract_terms` is the JSON form of a contract. Percentages are per share, applied to each matched work.

| Field | Type | Required | Notes |
|---|---|---|---|
| `parties` | array of Party | yes | Everyone named in the contract |
| `works` | array of Work | yes | The songs the contract covers. Only works that also appear in the statement produce payments |
| `royalty_shares` | array of RoyaltyShare | yes | The splits |
| `default_basis` | `"gross"` \| `"net"` | no | Basis for shares that set none. Defaults to `gross` |
| `contract_summary` | string | no | Free text, returned nowhere; kept for your own records |

**Party**

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | yes | Matched to `royalty_shares[].party_name`, ignoring case and punctuation |
| `role` | string | yes | Free text: `producer`, `artist`, `label`, `mixer`… Copied into each payment |
| `aliases` | array of string | no | Other spellings of the name |

**Work**

| Field | Type | Required | Notes |
|---|---|---|---|
| `title` | string | yes | Matched to statement titles fuzzily: case, punctuation and suffixes such as "(Remix)" are tolerated |
| `work_type` | string | no | Defaults to `song` |

**RoyaltyShare**

| Field | Type | Required | Notes |
|---|---|---|---|
| `party_name` | string | yes | Must name one of `parties` |
| `royalty_type` | string | yes | What income the share is paid from. The calculation covers streaming and master income, so use `master` or `streaming` (variants such as `digital`, `DSP revenue`, `master recording royalties` also count). Publishing, mechanical, sync and performance shares are ignored: they are paid from different statements |
| `percentage` | number | yes | 0–100 |
| `basis` | `"gross"` \| `"net"` | no | `gross` pays the percentage of the statement amount; `net` deducts the work's share of `expenses` first. Falls back to `default_basis`, then `gross` |
| `terms` | string | no | The clause, verbatim if you have it. Returned by `/registry/v1/splits`, not on a payment. If it names a direct-pay collector (SoundExchange, a PRO, the MLC) as the payer, the share is treated as paid outside this statement and skipped |

#### Expenses

`expenses` is an array of costs recouped before net-basis shares are paid. Gross-basis shares ignore them.

| Field | Type | Required | Notes |
|---|---|---|---|
| `amount` | number | yes | ≥ 0, in the statement's currency |
| `description` | string | no | For your records |
| `work_titles` | array of string | no | Songs this cost is tied to. Empty means project-wide |

- A **project-wide** expense is spread across every song in the statement in proportion to each song's earnings.
- A **tagged** expense is applied in full to each listed song that appears in the statement. A tag that matches nothing in the statement is dropped.
- A song's net amount never goes below zero.

#### Response

The response is `200 text/event-stream`. While a PDF is being read the server sends a heartbeat comment line (`: ping`) every 15 seconds; ignore lines starting with `:`. Exactly one `data:` event follows, and it is either a result or an error. Every event carries a `billing` block saying what the call cost (see [Billing](#billing)).

**Result event** — three sections, here for the statement above with one 50 % net-basis master share and a project-wide 300.00 expense:

```json
{
  "type": "result",
  "summary": {"payments": 2, "total_payable": 600.0, "expense_review_required": true},
  "payments": [
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
  ],
  "billing": {"credits": 30, "request_id": "a1b2c3…"}
}
```

`summary`

| Field | Type | Notes |
|---|---|---|
| `payments` | integer | How many payment lines follow |
| `total_payable` | number | The sum of every line's `amounts.payable`. Amounts are rounded to 2 dp before summing, so the lines you are shown add up to this figure exactly |
| `expense_review_required` | boolean | `true` when any line is on a net basis, so the expense list affected the amounts and deserves a human check |

`payments[]` — one line per party per matched work

| Field | Type | Notes |
|---|---|---|
| `song` | string | The work's title as given in the contract |
| `payee.name`, `payee.role` | string | Who is paid, and the role the contract gives them (or `unknown`) |
| `share.type` | string | The income the share is paid from, as written in the contract (master, streaming…) |
| `share.percentage` | number | The share applied, 0–100 |
| `share.basis` | `"gross"` \| `"net"` | The basis actually applied |
| `amounts.gross` | number | What the statement paid for this song, all matching rows summed |
| `amounts.expenses` | number | Expenses deducted for this song; `0` on a gross share |
| `amounts.net` | number | `gross − expenses`, floored at 0 |
| `amounts.payable` | number | `net × percentage / 100` — the figure to pay. All four amounts are rounded to 2 dp |

The share's clause text is not repeated here; `/registry/v1/splits` returns it under `contract_terms.royalty_shares[].terms`.

**Error event** — the calculation could not produce a result. HTTP is already `200` by this point, so check `type`:

```json
{
  "type": "error",
  "code": "NO_SONG_MATCHES",
  "message": "The contract covers songs that don't appear in this royalty statement.",
  "suggestion": "Make sure the statement is for the same release as the contract…",
  "details": {"contract_works": ["Blue Sky"], "statement_songs": ["Blue Skies (Live)"], "statement_song_total_count": 1},
  "billing": {"credits": 0}
}
```

`message` and `suggestion` are safe to show to a person. An error event is never billed, and says so.

### POST /registry/v1/splits

The Registry's contract parse: send contract PDFs, get the deal back as data. The response carries two views of one contract — `contract_terms`, in exactly the shape `/oneclick/v1/royalties` accepts (parse a contract once, then run every statement against it at the base price, with no AI), and `splits`, the per-party master / publishing / SoundExchange percentages the Registry uses for ownership stakes. Nothing is stored.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `contracts` | file, repeated, required | Contract PDFs — up to 10 files, 20 MB in total. Send the field once per file; several PDFs are merged into one set of terms |
| `main_artist_name` | string | Optional. The artist the splits are built around: they are kept in `splits` even at 0 / 0 and named in `splits.main_artist`, by the name the contract uses. When the name is not found, `main_artist` is `null` and the artist is left out of `parties` |
| `Idempotency-Key` | header | Optional. The same key with the same files and artist in the same billing period is charged once. Recommended |

**Response** — `200 text/event-stream`, the same framing as a calculation: heartbeat lines (`: ping`) while the parse runs, then exactly one `data:` event, a result or an error.

```json
{
  "type": "result",
  "contract_terms": {
    "parties": [{"name": "Jane Doe", "role": "producer", "aliases": []}],
    "works": [{"title": "Blue Sky", "work_type": "song"}],
    "royalty_shares": [
      {"party_name": "Jane Doe", "royalty_type": "master", "percentage": 50.0, "terms": "…", "basis": "net"}
    ],
    "contract_summary": "…",
    "default_basis": null
  },
  "splits": {
    "main_artist": "Jane Doe",
    "parties": [
      {"name": "Jane Doe", "role": "producer", "master_pct": 50.0, "publishing_pct": 0.0, "soundexchange_pct": 0.0}
    ]
  },
  "billing": {"credits": 30, "request_id": "…"}
}
```

`contract_terms` fields are documented under the calculation's `contract_terms` above. In `splits`, `main_artist` is the party matching `main_artist_name` (or `null`); parties with no master, publishing or SoundExchange share are omitted (the main artist is always kept); `soundexchange_pct` is the share of neighbouring-rights income the contract assigns, where it does.

A contract that cannot be read ends in an error event with code `CONTRACT_UNREADABLE` (a scanned image, an encrypted or empty file). An error event is never billed, and says so in `billing`.

### POST /splitsheet/v1/documents

The finished split sheet, from the same generator as Msanii's Split Sheet tool. No AI runs, so a sheet always costs exactly the base price — per document: the PDF and the DOCX of one sheet are two deliverables. Nothing is stored.

**Request** — `application/json`

| Field | Type | Description |
|---|---|---|
| `work_title` | string, required | Printed on the sheet and used for the file name |
| `work_type` | string | Default `single`. Printed as given (`single`, `album track`, …) |
| `split_type` | `"publishing"` \| `"master"` \| `"both"` | Which sides the sheet covers. Default `both` |
| `date` | string, required | Printed verbatim, so use the wording you want on the sheet |
| `format` | `"pdf"` \| `"docx"` | Default `pdf` |
| `contributors` | array, required | 1–50 lines, below |
| `Idempotency-Key` | header | Optional. The same body in the same billing period is charged once |

Each contributor:

| Field | Type | Description |
|---|---|---|
| `name`, `role` | string, required | The person and what they did (Producer, Writer, Artist, …) |
| `publishing_share` | number | Their share of the composition, 0–100, when self-published |
| `writer_share`, `publisher_share` | number | Used instead of `publishing_share` when `is_published` is true: the writer's and the publisher's halves |
| `is_published`, `publisher_name`, `publisher_ipi` | | The contributor's publisher, if they have one |
| `ipi_number` | string | The writer's IPI / CAE number |
| `master_percentage` | number | Their share of the sound recording, 0–100 |
| `label` | string | The label on the master side, if any |

```json
{
  "work_title": "Blue Sky",
  "date": "6 September 2026",
  "format": "pdf",
  "contributors": [
    {"name": "Jane Doe", "role": "Producer", "publishing_share": 50, "master_percentage": 50},
    {"name": "Sam Ray", "role": "Writer", "publishing_share": 50, "master_percentage": 50}
  ]
}
```

**Response** — `200`, the document itself: `Content-Type: application/pdf` or `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `Content-Disposition: attachment; filename="Split_Sheet_<title>.<format>"`, `Content-Length` set. Save the body as the file.

The body is the document, so the charge rides in headers:

| Header | Meaning |
|---|---|
| `Msanii-Credits` | Credits charged for this document; `0` on a replay |
| `Msanii-Request-Id` | The id the charge is recorded under — quote it to support |
| `Msanii-Replayed` | `true`, present only on a replay under the same `Idempotency-Key` |

Header names are case-insensitive; HTTP/2 clients see them lower-cased.

### POST /zoe/v1/chat/completions

Zoe, Msanii's music-business assistant, behind the OpenAI chat-completions protocol. Any OpenAI SDK works unchanged: `base_url` is your base URL plus `/zoe/v1`, the API key is your Msanii key, and `model` is `zoe`.

Zoe on the API is **stateless**. She keeps no memory between calls and has no access to documents stored in Msanii — send the whole context (earlier turns, a contract's text) in `messages`, exactly as you would with OpenAI. She answers music-business questions from general knowledge and from whatever you include, and politely declines unrelated topics.

**Request** `application/json`

| Field | Type | Required | Notes |
|---|---|---|---|
| `model` | `"zoe"` | yes | Anything else is a `404 model_not_found` |
| `messages` | array | yes | 1–100 messages of `{role, content}`; roles `system`, `user`, `assistant`; `content` a string or OpenAI text parts (`[{"type": "text", "text": "…"}]`). 100,000 characters in total |
| `stream` | boolean | no | Default `false` |
| `temperature` | number | no | 0–2 |
| `max_tokens` | integer | no | 1–4,000. Default 4,000 |

Other OpenAI fields (`n`, `top_p`, `stop`, `tools`, `response_format`, `user`…) are accepted and ignored. Your own `system` message is honoured; it sits after Zoe's own persona.

**Response** `200` — OpenAI's `chat.completion` object:

```json
{
  "id": "chatcmpl-6d9f3c2e…",
  "object": "chat.completion",
  "created": 1788000000,
  "model": "zoe",
  "choices": [
    {"index": 0, "message": {"role": "assistant", "content": "A mechanical royalty is paid…"}, "finish_reason": "stop"}
  ],
  "billing": {"credits": 5, "request_id": "…"}
}
```

No token counts are returned: `billing.credits` is what the answer cost.

With `"stream": true` the response is `text/event-stream` of `chat.completion.chunk` frames — a first delta carrying `role`, then content deltas, then a final frame with `finish_reason: "stop"` that also carries `billing` — followed by `data: [DONE]`. If Zoe fails mid-stream a frame `{"error": {"code": "zoe_failed", "message": "…"}, "billing": {"credits": 0}}` arrives before `[DONE]` and nothing is billed.

### GET /zoe/v1/models

OpenAI's model listing: `{"object": "list", "data": [{"id": "zoe", …}]}`. Free.

## Errors

**Before the stream opens** the response is a plain HTTP error with a JSON body `{"detail": {"code": …}}`:

| Status | `code` | Meaning |
|---|---|---|
| `401` | `invalid_key` | Missing, unknown, revoked or expired key — or the team's API access has been turned off |
| `402` | `insufficient_credits` | The team's balance is below the price of a run. Body also carries `price` and `balance`. No work was started |
| `413` | `file_too_large` | Statement over 10 MB, or contracts over 20 MB in total |
| `413` | `too_many_contracts` | More than 10 contract files |
| `404` | `model_not_found` | Zoe: `model` was not `zoe` |
| `422` | `invalid_request` | Both or neither of `contracts` / `contract_terms`; a non-PDF contract; malformed JSON in `contract_terms` or `expenses`; no `contracts` on a parse; a split sheet body that fails validation; for Zoe an empty, over-long or badly-shaped `messages` list. `detail` may also be a list of field validation errors |
| `500` | `internal_error` | A split sheet could not be rendered. Quote `request_id` to support |
| `502` | `zoe_failed` | Zoe did not answer. Retry. On a Zoe stream this arrives as an error frame before `[DONE]` instead of a status |

**Inside a stream** (calculations and splits) the failure arrives as an error event:

| `code` | Meaning | `details` |
|---|---|---|
| `STATEMENT_UNSUPPORTED_FORMAT` | Not a CSV or Excel file | |
| `STATEMENT_EMPTY` | No earnings rows could be read | |
| `STATEMENT_COLUMNS_UNDETECTABLE` | The title or amount column could not be identified | `available_columns` |
| `NO_WORKS_IN_CONTRACT` | The contract lists no songs | |
| `NO_ROYALTY_SHARES_IN_CONTRACT` | The contract has no percentage splits | |
| `NO_STREAMING_EARNABLE_SHARES` | Splits exist but none are paid from streaming or master income | `excluded_payor_count` |
| `NO_SONG_MATCHES` | No contract work appears in the statement | `contract_works`, `statement_songs`, `statement_song_total_count` |
| `CONTRACT_UNREADABLE` | The contract could not be read: a scanned image, an encrypted or empty file (parse only) | `reason` |
| `internal_error` | Something failed on Msanii's side. Quote `request_id` to support | `request_id` |

## Billing

Every deliverable draws credits from the team's pool, whichever key ran it.

- **Every delivered response says what it cost.** A result event, a Zoe body and a Zoe stream's final `stop` frame carry `"billing": {"credits": n, "request_id": "…"}`; a split sheet carries the same in `Msanii-Credits` and `Msanii-Request-Id` headers. Error events and frames carry `"billing": {"credits": 0}`. A replay under the same `Idempotency-Key` in the same billing period reports `credits: 0` and `replayed: true` (header `Msanii-Replayed: true`): it was charged on the first run and not again. Two identical calls sent at the same moment may both report the price even though only one is charged — the report can over-state, never under-state. An HTTP error before the stream opens (401, 402, 413, 422, 500, 502) carries no `billing` block — nothing was charged.
- **You pay only for an answer you received.** The charge is applied after the result event, the document, or the last content chunk of a Zoe stream is on the wire. A failed call, or one whose connection dropped before the answer arrived, costs nothing.
- **Each call has a base price** — at the time of writing 30 credits per calculation, 30 per splits run, 20 per split sheet and 5 per Zoe answer, shown as the `price` in a `402`. A calculation or splits run over an unusually large set of PDFs, or a very long Zoe exchange, can cost more than the base; structured `contract_terms` runs and split sheets always cost exactly the base.
- **Idempotency (calculations, splits and split sheets).** With an `Idempotency-Key`, the same key + the same inputs (files and JSON byte-for-byte) is charged once per billing period; a `contract_terms` run returns the same result on the replay, while a PDF run is parsed again and may differ slightly. Without the header every call is billed. Use a key that identifies the run on your side, such as `"<statement id>-<attempt>"`. Zoe answers have no idempotency, as with OpenAI: every delivered answer is billed.
- **The balance is checked first.** A `402` is returned before any work starts. Only a team admin can add credits, in Msanii.
- `GET /zoe/v1/models` is always free.

## Limits

| | |
|---|---|
| Contract files per request | 10, PDF only |
| Contracts, total size | 20 MB |
| Statement size | 10 MB |
| Splits run time | Typically 30–120 s per run; heartbeats keep the connection alive |
| Split sheet contributors | 50 per document |
| Calculation time | A structured-terms run returns in seconds; a PDF run typically takes 30–120 s. Keep read timeouts above 60 s; heartbeats keep the connection alive |
| Zoe messages per call | 100, 100,000 characters in total |
| Zoe `max_tokens` | 4,000 (also the default) |

## Versioning

Everything under `/oneclick/v1`, `/registry/v1`, `/splitsheet/v1` and `/zoe/v1` is frozen. Fields may be added to responses; existing fields, codes and semantics will not change. Breaking changes ship as a `v2` of that tool.
