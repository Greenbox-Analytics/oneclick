# chunk-pdf-pinecone

Ingest a long-form PDF (a book, manual, or reference doc) into a Pinecone namespace so the app's tools — **Zoe** (chatbot) and, later, **OneClick** — can retrieve it as background knowledge.

> `SKILL.md` is the agent-facing version (used by Claude Code). This README is for humans running the uploader by hand.

## What it does

`src/backend/rag-tools/pinecone_upload_pdf.py` is a thin CLI over the importable `src/backend/knowledge/` package:

1. **Extract** per-page markdown (`pymupdf4llm`, keeps page boundaries + headings)
2. **Chunk** section-aware: split on markdown headers (captures a `section_path` breadcrumb), then token-size to ~800 tokens; page range derived from character offsets (correct at boundaries)
3. **Embed** with OpenAI `text-embedding-3-small` (1536-d), batched
4. **Upsert** into the namespace (embed→upsert interleaved, transient-only retry, idempotent)

Each vector stores rich metadata: `source`, `book_title`, `section_path`, `page_start/end`, `chunk_index`, `token_count`, `doc_type`, and the chunk `text`. The retriever (`knowledge/reference_search.py`) reads those, so an uploader **must** write them or citations come back blank.

## Prerequisites

In `src/backend/.env`:

| Var | Purpose |
|---|---|
| `OPENAI_API_KEY` | embeddings (not needed for `--dry-run`) |
| `PINECONE_API_KEY` | upsert (not needed for `--dry-run`) |
| `PINECONE_INDEX_NAME` | a **1536-dimensional** index (the script pre-flights this) |
| `REFERENCE_NAMESPACE` | the namespace the app queries; the CLI defaults to it so write/read can't drift (default `music-business-reference`) |

`poetry install` in `src/backend/` (deps: `pinecone`, `openai`, `pymupdf4llm`, `tiktoken`, `langchain-text-splitters`).

## Usage

Run from `src/backend/`. **Dry-run first** (free, no keys, prints a section diagnostic):

```bash
poetry run python "rag-tools/pinecone_upload_pdf.py" /abs/path/book.pdf \
    --book-title "Human Readable Title" --dry-run
```

Check the diagnostic — `section_path non-empty` should be high (≈100% means the PDF has detectable headings). Then the real upload (idempotent; `--replace` clears this source's old vectors first):

```bash
poetry run python "rag-tools/pinecone_upload_pdf.py" /abs/path/book.pdf \
    --book-title "Human Readable Title" --page-offset 14 --replace
```

Key flags: `--page-offset N` (PDF-page minus printed-page, so citations match the physical book — a literal integer, **no angle brackets**), `--source SLUG` (ID/prefix-safe; default = slugified stem), `--replace`, `--dry-run`, `--chunk-tokens`/`--overlap-tokens` (default 800/100). The `namespace` is an optional positional that defaults to `$REFERENCE_NAMESPACE`. Run `--help` for the rest.

Verify it landed:

```bash
poetry run python -c "import os;from pinecone import Pinecone;ns=os.getenv('REFERENCE_NAMESPACE','music-business-reference');print(Pinecone(api_key=os.environ['PINECONE_API_KEY']).Index(os.environ['PINECONE_INDEX_NAME']).describe_index_stats().namespaces.get(ns))"
```

## How the app uses it

- **Zoe** queries `REFERENCE_NAMESPACE` via `search_reference()`. In **general mode** (no contract selected) the book is the answer source; in **contract-assist mode** it's strict background. See `src/backend/zoe_chatbot/CHATBOT.md`.
- **OneClick** consults it (read-only) when auditing contract-basis nuances. See `src/backend/oneclick/ONECLICK.md`.

## Gotchas

- **Upload namespace must match `REFERENCE_NAMESPACE`** or the app queries an empty namespace and sees nothing. Don't pass a namespace override unless `.env` matches it. To rename: set `REFERENCE_NAMESPACE` in `.env`, re-upload with `--replace`, delete the old namespace, restart the backend.
- **`zsh: no such file or directory: N`** — you typed `--page-offset <14>`; drop the angle brackets (`--page-offset 14`).
- **Re-tune retrieval** (`min_score`, default 0.45) in `reference_search.py` if live results look off — `0.45` was tuned from a probe (relevant ≈0.52–0.70, off-topic ≤0.18).
- Pinecone can't rename a namespace in place — "renaming" = re-upload to new + delete old.
