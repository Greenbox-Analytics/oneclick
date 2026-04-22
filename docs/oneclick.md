# OneClick — Royalty Calculator

OneClick is a royalty calculation tool that cross-references one or more contract files against a royalty statement to produce a per-song, per-party payment breakdown. Users select contracts and a royalty statement (from existing project files or by uploading new ones), trigger a streaming calculation, review the results, and optionally export or share the report.

---

## Backend Endpoints

All OneClick routes are registered under the `/oneclick` prefix in `src/backend/main.py` and `src/backend/oneclick/share.py`.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/oneclick/calculate-royalties-stream` | Stream SSE progress events and final payment results. Checks the cache first; accepts `force_recalculate=true` to bypass it. |
| `POST` | `/oneclick/calculate-royalties` | Non-streaming version of the same calculation (synchronous). |
| `POST` | `/oneclick/confirm` | Persist confirmed calculation results to `royalty_calculations` + `royalty_calculation_contracts`. Replaces any existing cached record for the same statement + contract set. |
| `POST` | `/oneclick/share` | Generate a PDF of the results and upload it to Google Drive or Slack. |

### `GET /oneclick/calculate-royalties-stream` — query parameters

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `project_id` | `string` (UUID) | Yes | Project the files belong to |
| `royalty_statement_file_id` | `string` (UUID) | Yes | `project_files.id` for the royalty statement |
| `contract_id` | `string` (UUID) | No | Single contract (use when selecting one file) |
| `contract_ids` | `string[]` | No | Multiple contracts; takes precedence over `contract_id` |
| `force_recalculate` | `bool` | No | Default `false`. Pass `true` to skip the cache. |

The stream emits `data:` lines of JSON objects with the shape:

```json
{ "type": "status|error|result", "message": "...", "progress": 0-100, "stage": "starting|downloading|extracting_royalty|processing|complete" }
```

On completion the `type: "result"` event carries the full `OneClickRoyaltyResponse` payload.

### `POST /oneclick/calculate-royalties` — request body (`OneClickRoyaltyRequest`)

```json
{
  "project_id": "<uuid>",
  "royalty_statement_file_id": "<uuid>",
  "contract_id": "<uuid | null>",
  "contract_ids": ["<uuid>", "..."]
}
```

### `POST /oneclick/confirm` — request body (`ConfirmCalculationRequest`)

```json
{
  "contract_ids": ["<uuid>"],
  "royalty_statement_id": "<uuid>",
  "project_id": "<uuid>",
  "results": { }
}
```

---

## Share Endpoint

**`POST /oneclick/share`**

Generates a PDF using ReportLab and uploads it to the specified integration. Requires the user to have the relevant OAuth token stored (Google Drive or Slack).

### Request body (`ShareRequest`)

```json
{
  "target": "drive",
  "artist_name": "Artist Name",
  "payments": [
    {
      "song_title": "Track Name",
      "party_name": "Payee Name",
      "role": "Producer",
      "royalty_type": "streaming",
      "percentage": 25.0,
      "total_royalty": 4000.00,
      "amount_to_pay": 1000.00,
      "terms": "optional free-text"
    }
  ],
  "total_payments": 1000.00,
  "channel_id": null,
  "folder_id": null
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `target` | `"drive" \| "slack"` | Yes | Destination integration |
| `artist_name` | `string` | Yes | Appears in the PDF title and filename |
| `payments` | `list[dict]` | Yes | Full payment rows from the calculation result |
| `total_payments` | `float` | Yes | Grand total displayed in the PDF header |
| `channel_id` | `string \| null` | Conditional | Required when `target` is `"slack"` |
| `folder_id` | `string \| null` | No | Google Drive folder ID; uploads to root if omitted |

The generated PDF filename follows the pattern:
`OneClick_Royalties_{artist_name}_{YYYYMMDD}.pdf`

### Response

```json
{ "success": true, "target": "drive", "file": { } }
{ "success": true, "target": "slack", "result": { } }
```

---

## Share UI Flow

1. User completes a calculation and `CalculationResults` renders the results table and pie chart.
2. User clicks the **Share** button (Share2 icon), which opens a `Popover`.
3. The popover lists connected integrations. `useIntegrations()` determines which of Google Drive and Slack have an active connection. Disconnected integrations are shown as disabled.
4. User clicks **Save to Google Drive** or **Share to Slack**.
5. `handleShare(target)` in `CalculationResults.tsx` calls `POST /oneclick/share` with the current `calculationResult.payments`, `total_payments`, and the selected `target`.
6. The backend generates the PDF in-memory (ReportLab), then calls the appropriate integration service (`export_pdf_to_drive` or `upload_file_to_channel`).
7. A toast notification confirms success or surfaces the error message.

---

## Frontend Components

| Component | File | Purpose |
|-----------|------|---------|
| `OneClickDocuments` | `src/pages/OneClickDocuments.tsx` | Main page. Orchestrates file selection, SSE streaming, progress state, and confirmation flow. Entry point for the `/tools/oneclick` route (artist-scoped via `artistId` param). |
| `ContractSelector` | `src/components/oneclick/ContractSelector.tsx` | Tabbed card for selecting contracts. Supports uploading new files or picking existing `project_files` (filtered to `folder_category = "contracts"`). Allows multi-select. |
| `RoyaltyStatementSelector` | `src/components/oneclick/RoyaltyStatementSelector.tsx` | Tabbed card for selecting the royalty statement. Supports uploading new files or picking existing `project_files` (filtered to `folder_category = "royalty_statement"`). Single-select. |
| `CalculationResults` | `src/components/oneclick/CalculationResults.tsx` | Renders the SSE progress ring, the results table, a pie chart of payment distribution, export buttons (CSV, Excel, chart PNG), and the Share popover for Drive/Slack. |

---

## Database Tables

### `royalty_calculations`

Stores the confirmed output of a single calculation run.

| Column | Type | Notes |
|--------|------|-------|
| `id` | `UUID` | Primary key |
| `royalty_statement_id` | `UUID` | FK → `project_files.id` (CASCADE delete) |
| `project_id` | `UUID` | FK → `projects.id` (CASCADE delete) |
| `user_id` | `UUID` | Matches `auth.uid()` for RLS |
| `results` | `JSONB` | Full `OneClickRoyaltyResponse` payload |
| `created_at` | `TIMESTAMPTZ` | |
| `updated_at` | `TIMESTAMPTZ` | Auto-updated via trigger |

RLS: users can SELECT, INSERT, and DELETE their own rows only.

### `royalty_calculation_contracts`

Junction table linking a calculation to the contract files used.

| Column | Type | Notes |
|--------|------|-------|
| `calculation_id` | `UUID` | FK → `royalty_calculations.id` (CASCADE delete) |
| `contract_id` | `UUID` | FK → `project_files.id` (CASCADE delete) |

Composite primary key `(calculation_id, contract_id)`. Deleting a junction row triggers `delete_orphan_calculation()`, which removes the parent `royalty_calculations` row (and cascades the remaining junction rows).

RLS: users can access junction rows only when the parent `royalty_calculations.user_id = auth.uid()`.

---

## Local Testing

### Share endpoint (Google Drive target)

```bash
curl -X POST http://localhost:8000/oneclick/share \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <JWT>" \
  -d '{
    "target": "drive",
    "artist_name": "Test Artist",
    "payments": [
      {
        "song_title": "Track A",
        "party_name": "Jane Doe",
        "role": "Producer",
        "royalty_type": "streaming",
        "percentage": 50.0,
        "total_royalty": 2000.00,
        "amount_to_pay": 1000.00
      }
    ],
    "total_payments": 1000.00
  }'
```

### Share endpoint (Slack target)

```bash
curl -X POST http://localhost:8000/oneclick/share \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <JWT>" \
  -d '{
    "target": "slack",
    "artist_name": "Test Artist",
    "channel_id": "C012AB3CD",
    "payments": [
      {
        "song_title": "Track A",
        "party_name": "Jane Doe",
        "role": "Producer",
        "royalty_type": "streaming",
        "percentage": 50.0,
        "total_royalty": 2000.00,
        "amount_to_pay": 1000.00
      }
    ],
    "total_payments": 1000.00
  }'
```

### Streaming calculation (SSE)

```bash
curl -N "http://localhost:8000/oneclick/calculate-royalties-stream\
?project_id=<uuid>\
&royalty_statement_file_id=<uuid>\
&contract_ids=<uuid1>\
&contract_ids=<uuid2>" \
  -H "Authorization: Bearer <JWT>"
```

### Run backend locally

```bash
cd src/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
