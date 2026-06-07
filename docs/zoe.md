# Zoe — AI Contract Analyst

Zoe (`/tools/zoe`) is the AI-powered chatbot for analyzing music contracts. Upload a PDF, ask a question, and Zoe extracts royalty splits, payment terms, rights, and more — with source chips that open the **relevant page** of the original PDF.

---

## Backend Endpoints

Defined directly in `src/backend/main.py`.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/zoe/ask-stream` | Streaming AI response (Server-Sent Events). Accepts query, project_id, contract_ids, artist_id, session_id, context, and pre-fetched contract_markdowns. SSE event types: `start`, `sources`, `token`, `data`, `done`, `complete`, `error`. |
| DELETE | `/zoe/session/{session_id}` | Clear in-memory conversation history for a session. |
| GET | `/zoe/session/{session_id}/history` | Retrieve conversation history for a session. |

## How It Works

1. **Select context (optional)** — Pick an artist, project, and one or more contracts from the sidebar. Zoe fetches and caches the full markdown for each selected contract. You can also ask with **nothing selected** — Zoe answers general music-business questions from its knowledge base.
2. **Ask a question** — The frontend sends query + conversation context + cached contract markdowns to `/zoe/ask-stream`.
3. **Routing — knowledge base vs your contracts** — An LLM router picks the answer source per question, and it **emphasizes the contract**:
   - **No contracts selected**, or a **general concept/clarification** whose answer doesn't depend on a specific deal ("what's a deficit vs net-payable royalty?") → answered from the **knowledge base** (a music-business reference book retrieved via Pinecone RAG), in Zoe's own words with no page citations. This mode uses the book + recent conversation history; it does **not** re-read the contract document.
   - A **deal-specific question** ("what's my master split in this contract?") → answered from the **full contract document(s)**. The reference book is still retrieved and injected as silent **background**, so Zoe can append a short "Supplemental (general industry context)" note when the book adds context the contract doesn't cover — but the contract always governs.
   - Genuinely ambiguous questions lean toward the contract.
4. **Streaming response** — The LLM streams its answer back via SSE. The frontend renders tokens in real-time with confidence scores, source citations, and extracted structured data.
5. **Page-jump** — For contract answers, after the answer finishes, the backend makes one cheap follow-up call that maps the answer back to the contract **page** it drew from, then emits a second `sources` event so the source chip opens the PDF at that page. Best-effort/model-attributed (a sensible page, not an exact clause). Full mechanics in `src/backend/zoe_chatbot/CHATBOT.md` → "Contract Page-Jump".

> For the full routing rules, retrieval tuning, and reference-book mechanics, see `src/backend/zoe_chatbot/CHATBOT.md` → "Routing Decision" and "Reference Knowledge (Book RAG)".

## Frontend

### Hooks

| Hook | File | Returns |
|------|------|---------|
| `useStreamingChat(...)` | `src/hooks/useStreamingChat.ts` | State + `sendMessage` handler for consuming SSE endpoint |

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `ZoeChatMessages` | `src/components/zoe/ZoeChatMessages.tsx` | Streaming chat display + source chips (open `ContractSlideOver` at the cited page) |
| `ZoeInputBar` | `src/components/zoe/ZoeInputBar.tsx` | Message input + quick actions |
| `ContractSlideOver` | `src/components/zoe/ContractSlideOver.tsx` | Right-side PDF viewer; opens the contract at `#page=N` via a short-lived signed URL |

### Pages

| Page | File | Route |
|------|------|-------|
| Zoe | `src/pages/Zoe.tsx` | `/tools/zoe` |

## Key Backend Files

| File | Purpose |
|------|---------|
| `zoe_chatbot/contract_chatbot.py` | RAG chatbot, streaming, conversation memory |
| `zoe_chatbot/contract_ingestion.py` | PDF ingestion pipeline |
| `zoe_chatbot/contract_search.py` | Semantic search with metadata filtering |
| `zoe_chatbot/helpers.py` | PDF → markdown, section splitting, table extraction |

## Local Testing

```bash
TOKEN="your-supabase-jwt-here"
BASE="http://localhost:8000"

# Send a Zoe query (streaming — prints raw SSE)
curl -X POST "$BASE/zoe/ask-stream" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the royalty rates in this contract?",
    "project_id": "<project-uuid>",
    "contract_ids": ["<contract-uuid>"]
  }'

# Clear a session
curl -X DELETE -H "Authorization: Bearer $TOKEN" "$BASE/zoe/session/<session-id>"
```

```bash
cd src/backend && poetry run pytest tests/test_zoe.py -v
```
