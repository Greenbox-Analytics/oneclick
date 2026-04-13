# Zoe — AI Contract Analyst

Zoe (`/tools/zoe`) is the AI-powered chatbot for analyzing music contracts. Upload a PDF, ask a question, and Zoe extracts royalty splits, payment terms, rights, and more — with source citations back to the original document.

---

## Backend Endpoints

Defined directly in `src/backend/main.py`.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/zoe/ask-stream` | Streaming AI response (Server-Sent Events). Accepts query, project_id, contract_ids, artist_id, session_id, context, and pre-fetched contract_markdowns. SSE event types: `start`, `sources`, `token`, `data`, `done`, `complete`, `error`. |
| DELETE | `/zoe/session/{session_id}` | Clear in-memory conversation history for a session. |
| GET | `/zoe/session/{session_id}/history` | Retrieve conversation history for a session. |

## How It Works

1. **Select context** — Pick an artist, project, and one or more contracts from the sidebar. Zoe fetches and caches the full markdown for each selected contract.
2. **Ask a question** — The frontend sends query + conversation context + cached contract markdowns to `/zoe/ask-stream`.
3. **Smart retrieval** — If all selected contracts fit under ~100k tokens, the backend sends the full documents as context (best accuracy). If not, it falls back to semantic vector search via Pinecone, pulling only the most relevant chunks.
4. **Streaming response** — The LLM streams its answer back via SSE. The frontend renders tokens in real-time with confidence scores, source citations, and extracted structured data.

## Frontend

### Hooks

| Hook | File | Returns |
|------|------|---------|
| `useStreamingChat(...)` | `src/hooks/useStreamingChat.ts` | State + `sendMessage` handler for consuming SSE endpoint |

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `ZoeChatMessages` | `src/components/zoe/ZoeChatMessages.tsx` | Streaming chat display + source citations |
| `ZoeInputBar` | `src/components/zoe/ZoeInputBar.tsx` | Message input + quick actions |
| `ZoeDocumentPanel` | `src/components/zoe/ZoeDocumentPanel.tsx` | Contract/document selection sidebar |

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
