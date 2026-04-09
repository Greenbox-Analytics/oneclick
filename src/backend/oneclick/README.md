# OneClick Royalty Calculator

Automated royalty payment calculation from music contracts and streaming statements.

## Overview

OneClick analyzes music contracts using AI and calculates royalty payments by matching contract terms with streaming revenue data. It eliminates manual spreadsheet work and reduces errors in royalty distribution.

## How It Works

### 1. Contract Analysis
- Contracts are parsed via AI (GPT-5.2) to extract structured data:
  - **Parties** — artists, producers, songwriters, labels, publishers, etc.
  - **Works** — song titles and composition types covered by the contract
  - **Royalty shares** — per-party percentages, royalty types, and terms
- Role taxonomy mapping standardizes verbose contract language (e.g., "master points" -> Streaming)
- Multi-contract support with intelligent merging and conflict resolution

### 2. Royalty Statement Processing
- Supports both CSV and Excel formats
- 3-layer auto-detection for column identification:
  1. **Keyword matching** — priority terms like "net payable", "total payable"
  2. **Fuzzy matching** — sequence matching with 80% similarity threshold
  3. **Semantic (LLM)** — GPT-based fallback for ambiguous headers
- Aggregates revenue by song title across all rows

### 3. Payment Calculation
- 3-tier fuzzy song matching (exact normalized, partial, word-level)
- Aggregates amounts from all matching statement entries per song
- Filters for streaming-equivalent royalties (master, producer, digital, etc.)
- Computes payment: `total_royalty * (percentage / 100)` per party per song

### 4. Output
- Real-time progress via Server-Sent Events (SSE) streaming endpoint
- Caches results per statement + contract combination to avoid recalculation
- Excel export with formatted headers, currency columns, auto-width, and SUM totals
- JSON export with summary statistics
- Confirmed results saved to `royalty_calculations` table in Supabase

## Key Features

- **Smart Matching**: 3-tier fuzzy matching handles song title variations and aggregates revenue correctly
- **Multi-Contract Support**: Parallel parsing with intelligent merging — deduplicates parties, combines roles, resolves royalty share conflicts
- **Auto-Detection**: 3-layer column identification handles various statement formats
- **Caching**: Statement + contract combination caching with `force_recalculate` override
- **Streaming Progress**: SSE endpoint sends real-time status updates during calculation
- **Validation**: Checks for missing data and provides helpful error messages

## File Structure

```
oneclick/
├── contract_parser.py      # AI-powered contract extraction (GPT-5.2)
├── royalty_calculator.py   # Payment calculation engine + multi-contract merging
├── helpers.py              # Title normalization, fuzzy matching, role simplification
└── README.md               # This file
```

## API Endpoints

### `POST /oneclick/confirm`
Save confirmed calculation results to the database.

**Request:**
```json
{
  "contract_ids": ["uuid", "uuid"],
  "royalty_statement_id": "uuid",
  "project_id": "uuid",
  "results": { ... }
}
```

### `GET /oneclick/calculate-royalties-stream`
SSE streaming endpoint with real-time progress updates.

**Query Params:**
- `project_id` (required)
- `royalty_statement_file_id` (required)
- `contract_id` or `contract_ids` (one required)
- `force_recalculate` (optional, bypasses cache)

**SSE Events:**
```
data: {"type": "progress", "message": "Parsing contracts..."}
data: {"type": "progress", "message": "Processing royalty statement..."}
data: {"type": "result", "payments": [...], "is_cached": false}
```

### `POST /oneclick/calculate-royalties`
Standard blocking endpoint for royalty calculation.

**Request:**
```json
{
  "contract_id": "uuid",
  "contract_ids": ["uuid"],
  "project_id": "uuid",
  "royalty_statement_file_id": "uuid"
}
```

**Response:**
```json
{
  "status": "success",
  "total_payments": 6,
  "payments": [
    {
      "song_title": "Midnight Dreams",
      "party_name": "Luna Rivers",
      "role": "Artist",
      "royalty_type": "Streaming Royalties",
      "percentage": 45.0,
      "total_royalty": 125000.00,
      "amount_to_pay": 56250.00
    }
  ],
  "is_cached": false,
  "message": "Calculated from 2 contracts"
}
```

## Requirements

- OpenAI API key (for contract parsing and semantic column detection)
- Python packages: `openpyxl`, `openai`, `python-dotenv`, `PyMuPDF`

## Error Handling

The system validates:
- Contract contains royalty information
- Royalty statement has valid data
- Songs in contract match songs in statement
- All required fields are present

Common errors and solutions are logged with helpful messages to guide troubleshooting.
