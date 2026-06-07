---
name: chunk-pdf-pinecone
description: Use when adding a long-form PDF (book, manual, reference doc) to a Pinecone namespace so Zoe/OneClick can retrieve it as background context — ingesting a new reference book, re-ingesting after edits, or wiring a new knowledge source. Keywords: pinecone, embed, chunk, ingest, namespace, retrieval, RAG.
---

# chunk-pdf-pinecone

## Overview

Ingest a PDF into a Pinecone namespace so the app can retrieve it as extra context: extract per-page markdown, **section-aware** chunk (heading breadcrumb + page range), embed with OpenAI `text-embedding-3-small` (1536-d), and upsert in batches.

The CLI is **`src/backend/rag-tools/pinecone_upload_pdf.py`** (a thin wrapper over the importable `src/backend/knowledge/` package: `chunking.py`, `ingest.py`). Run it from `src/backend/` with `poetry run`.

## When to use

- Loading a book / manual / reference PDF into Pinecone for `search_reference()` retrieval
- Re-ingesting a PDF after edits (use `--replace` for clean idempotency)

**Do NOT use for:**
- Contract ingestion — contracts use full-document markdown context in `src/backend/zoe_chatbot/`, not this namespace
- Non-PDF sources — extend `knowledge/chunking.py` first

## Prerequisites

**Env vars** (read from `src/backend/.env`):

| Var | Purpose |
|---|---|
| `OPENAI_API_KEY` | Embedding calls (NOT needed for `--dry-run`) |
| `PINECONE_API_KEY` | Upsert calls (NOT needed for `--dry-run`) |
| `PINECONE_INDEX_NAME` | Target index (overridable via `--index`) |
| `REFERENCE_NAMESPACE` | The namespace the app queries; the upload CLI **defaults to it**, so read/write can't drift. Defaults to `music-business-reference` if unset. |

**Index:** must be **1536-dimensional** (matches `text-embedding-3-small`). The script's `preflight` checks this and aborts before spending. Confirm: `pc.describe_index(name).dimension`.

**Deps:** `openai`, `pinecone`, `pymupdf4llm`, `tiktoken`, `langchain-text-splitters`, `python-dotenv` — all declared in `src/backend/pyproject.toml`. Just `poetry install`.

## Quick reference

Run from `src/backend/`. The **namespace is whatever `REFERENCE_NAMESPACE` is set to in `.env`** — both this CLI and the app read it, so they can't drift. Set it once; you don't pass it each time (but you can, positionally, to override). When ingesting for someone, ask them what namespace name they want and make sure `.env` matches. The PDF path is a plain positional arg; relative (to `src/backend/`) or **absolute** both work.

```bash
# Dry run FIRST — extract + chunk only, prints the section diagnostic. No keys/spend.
# Namespace defaults to $REFERENCE_NAMESPACE.
poetry run python "rag-tools/pinecone_upload_pdf.py" /abs/path/book.pdf \
    --book-title "Human Readable Title" --dry-run

# Real upload (idempotent). --replace deletes this source's old vectors first.
poetry run python "rag-tools/pinecone_upload_pdf.py" /abs/path/book.pdf \
    --book-title "Human Readable Title" --page-offset 14 --replace

# Override the namespace explicitly (must still match the app's REFERENCE_NAMESPACE to be retrievable):
poetry run python "rag-tools/pinecone_upload_pdf.py" /abs/path/book.pdf my-namespace --replace
```

| Flag | Meaning |
|---|---|
| `--book-title` | Human label stored on every vector (default: PDF stem) |
| `--source` | ID-safe slug for vector IDs / `--replace` prefix (default: slugified stem) |
| `--page-offset N` | PDF-page-minus-printed-page; stores true printed pages. **Literal integer, no `<>`** |
| `--replace` | Delete existing `{source}-*` vectors before upsert (serverless-safe) |
| `--dry-run` | Extract + chunk + diagnostic only; no embed/upsert |
| `--chunk-tokens` / `--overlap-tokens` | Defaults 800 / 100 |
| `--index` | Override `$PINECONE_INDEX_NAME` |

`namespace` is an **optional positional** (defaults to `$REFERENCE_NAMESPACE`). Run `--help` for all flags.

## What the script does

1. **Extract** — `pymupdf4llm.to_markdown(page_chunks=True)` → per-page markdown (keeps page boundaries + headings)
2. **Chunk** — section-aware: split on markdown headers (captures `section_path` breadcrumb), then a tiktoken-length splitter to ~800 tokens; page range derived from char offsets (correct at boundaries)
3. **Embed** — `text-embedding-3-small`, batched
4. **Upsert** — embed→upsert interleaved per batch, transient-only retry

**Vector ID:** `{source}-{chunk_index}-{sha256_prefix}`.

## Metadata schema (the retrieval contract)

Every vector stores these keys. `knowledge/reference_search.py::_to_passage` READS `text`, `section_path`, `page_start`, `page_end`, `book_title` — **any uploader must write them or citations come back blank**:

| Key | Notes |
|---|---|
| `source` | slug; ID prefix |
| `book_title` | human label |
| `section_path` | heading breadcrumb, e.g. `Part V ▸ Record Deals` |
| `page_start` / `page_end` | printed pages (after `--page-offset`) |
| `chunk_index`, `token_count` | bookkeeping |
| `doc_type` | `"reference_book"` |
| `text` | the chunk (~3–4 KB at 800 tok; well under Pinecone's 40 KB cap) |

## Making a new PDF actually retrievable

The retriever queries the namespace named by **`REFERENCE_NAMESPACE`** (`reference_search.py` reads it from the env; defaults to `"music-business-reference"`). The upload CLI defaults its namespace to the **same** env var, so as long as you don't override it, write and read always agree.

**To change the knowledge-base namespace name:**
1. Set `REFERENCE_NAMESPACE=<new-name>` in `src/backend/.env` (no code edit — it's not hardcoded).
2. Re-upload with `--replace` (the CLI now targets `<new-name>` automatically).
3. Delete the old namespace's vectors so you don't pay to store them (see below).

Pinecone can't rename a namespace in place — "renaming" is always re-upload-to-new + delete-old. Uploading to a namespace that `REFERENCE_NAMESPACE` doesn't point at = vectors nothing reads.

```bash
# Delete an old namespace after switching
poetry run python -c "import os;from pinecone import Pinecone;Pinecone(api_key=os.environ['PINECONE_API_KEY']).Index(os.environ['PINECONE_INDEX_NAME']).delete(namespace='OLD_NAME', delete_all=True)"
```

## Page numbers & `--page-offset`

Pages are numbered by PDF position. A book's front matter means PDF page ≠ printed page. Find the PDF page that prints "1", set `--page-offset = (that page) − 1`, and re-run `--dry-run` to confirm a sampled `p.X-Y` matches the book. Omit the flag (offset 0) if printed-page-exact citations don't matter yet — retrieval still works.

## Common mistakes

| Mistake | Fix |
|---|---|
| `zsh: no such file or directory: N` (or `14`) | You typed `--page-offset <14>`; zsh reads `<…>` as file redirection. Pass a bare integer: `--page-offset 14` |
| Index dimension ≠ 1536 | `preflight` aborts; create a 1536-d index or switch the model in `knowledge/ingest.py` |
| Re-running with new `--chunk-tokens`, expecting overwrite | Chunk boundaries change → new IDs → orphans. Use `--replace` (deletes by `{source}-` prefix) |
| Source slug is a prefix of another (`book` vs `book-2`) | `--replace` over-deletes. Pick slugs that aren't prefixes of each other |
| Uploaded but Zoe sees nothing | Upload namespace ≠ `REFERENCE_NAMESPACE`. Don't pass a namespace override, or make it match `.env` |
| Section diagnostic shows ~0% `section_path` | PDF emits no markdown headers; revisit chunking before spending |

## Verify after upload

```python
from pinecone import Pinecone
import os
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
ns = os.getenv("REFERENCE_NAMESPACE", "music-business-reference")
print(pc.Index(os.getenv("PINECONE_INDEX_NAME")).describe_index_stats().namespaces.get(ns))
```

Expect `vector_count` ≈ `total_tokens / (chunk_tokens − overlap_tokens)` (~700 for a 1000-page book at defaults).
