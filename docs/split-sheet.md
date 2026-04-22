# Split Sheet

The Split Sheet tool (`/tools/split-sheet`) generates professional split sheet documents as PDF or DOCX, listing contributors and their ownership percentages for a musical work.

---

## Backend Endpoints

Mounted under `/splitsheet` via `src/backend/splitsheet/router.py`.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/splitsheet/generate` | Generate a split sheet document. When `save_to_artist` is true, also uploads the generated file to storage and creates a `project_files` record. Returns the document as a streaming file download. |

**Request body:**

```json
{
  "work_title": "Song Title",
  "work_type": "single",
  "split_type": "both",
  "date": "2026-04-12",
  "format": "pdf",
  "contributors": [
    { "name": "Alice", "role": "Producer", "master_percentage": 50, "publishing_percentage": 50 },
    { "name": "Bob", "role": "Songwriter", "master_percentage": 50, "publishing_percentage": 50 }
  ],
  "save_to_artist": true,
  "artist_id": "<uuid>",
  "project_id": "<uuid>"
}
```

## Key Backend Files

| File | Purpose |
|------|---------|
| `splitsheet/router.py` | FastAPI route handler |
| `splitsheet/pdf_generator.py` | ReportLab PDF generation |

## Frontend

### Pages

| Page | File | Route |
|------|------|-------|
| SplitSheet | `src/pages/SplitSheet.tsx` | `/tools/split-sheet` |

## Local Testing

```bash
TOKEN="your-supabase-jwt-here"
BASE="http://localhost:8000"

# Generate a split sheet PDF
curl -X POST "$BASE/splitsheet/generate" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "work_title": "Blue Horizon",
    "work_type": "single",
    "split_type": "both",
    "date": "2026-04-12",
    "format": "pdf",
    "contributors": [
      {"name": "Alice", "role": "Producer", "master_percentage": 50, "publishing_percentage": 50},
      {"name": "Bob", "role": "Songwriter", "master_percentage": 50, "publishing_percentage": 50}
    ]
  }' --output split_sheet.pdf
```

```bash
cd src/backend && poetry run pytest tests/test_splitsheet.py -v
```
