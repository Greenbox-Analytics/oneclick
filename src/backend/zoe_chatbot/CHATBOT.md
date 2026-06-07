# Zoe Chatbot

An AI assistant for music contracts, artist profiles, and **general music-business questions**. Streams responses in real-time via SSE (like ChatGPT/Claude), with intelligent routing that decides where to pull answers from.

> **Two answer sources, picked per question by an LLM router** (`_llm_route_decision`):
> - **Contract-assist** — answers from the **full contract document(s)** (passed as markdown into a long-context model), with the reference book retrieved and injected as labeled **background** (the contract always governs). Used for any question whose answer depends on the selected contract(s).
> - **General (knowledge base)** — answers a music-business question from the **reference book** (Pinecone RAG) + the model's knowledge, in Zoe's own words with no page citations. Used when **no contracts are selected**, AND when contracts *are* selected but the question is a general concept/clarification whose answer doesn't depend on the specific deal (e.g. "what's a deficit vs net-payable royalty?"). Needs no artist or project.
>
> The router **emphasizes the contract**: deal-specific questions stay in contract-assist, and genuinely ambiguous questions lean contract. See "Routing Decision" below.
>
> Note: contract answers use the **full-document** approach, not per-chunk Pinecone search of contracts. The only Pinecone namespace Zoe queries is the shared reference book.

## Why Streaming?

- **Perceived speed** — tokens appear in ~200ms instead of waiting 5-15s for a full response
- **User can read as it generates** — no staring at a spinner
- **Stop mid-stream** — user can abort if the answer is going in the wrong direction
- Uses Server-Sent Events (SSE) over POST via `fetch` + `ReadableStream` (not `EventSource`, which doesn't support POST bodies)

## Architecture Overview

```
User Query
    │
    ├─ Tier 1: Conversational fast-path (greetings, thanks)
    │   → Instant response, no LLM or search needed
    │
    ├─ Artist intent detected? → Skip Tier 2, go to Artist Routing
    │   → Disambiguation: "Artist Profile" vs "Conversation History" buttons
    │
    ├─ Tier 2: Conversation history
    │   → LLM checks if history/context can answer the question
    │   → Avoids redundant document searches for follow-ups
    │
    ├─ No contracts selected? → General mode (knowledge base)
    │   → search_reference() over the music-business reference book (Pinecone)
    │   → Answers a general music-business question as a general knowledge base (no page citations)
    │   → No artist/project required
    │
    ├─ Contracts selected → Routing decision (_llm_route_decision: contract vs general)
    │   → "general"  → general concept/clarification (answer doesn't depend on the deal)
    │   │              → General mode (knowledge base), same as above
    │   └→ "contract" → deal-specific question (default; ambiguous leans here) → Tier 3
    │
    └─ Tier 3: Full-document contract context + LLM generation
        → Selected contracts' full markdown is passed into a long-context model
        → Reference book retrieved (strict) and injected as labeled BACKGROUND
        → LLM generates a grounded answer; the contract always governs
        → Structured data extracted for context tracking
        → After generation, the answer is attributed to a contract page so the
          source chip can open the PDF there (see "Contract Page-Jump")
```

## Entry Points

| Method | When Used |
|--------|-----------|
| `smart_ask()` | Main entry — single contract or project-wide search |
| `ask_without_project()` | No project selected — handles artist queries, prompts for project on contract questions |
| `ask_multiple_contracts()` | Specific contracts selected — searches across all of them |
| `ask_stream()` | **Streaming version** — same routing logic, returns SSE events instead of a dict |

## Three-Tier Query Routing

### Tier 1: Conversational Fast-Path
- Detects greetings, thanks, farewells via `_is_conversational_query()`
- Returns a friendly response immediately — no LLM call, no document search
- Includes quick action buttons to guide the user

### Tier 2: Conversation History
- `_try_answer_from_history()` sends the query + conversation history to the LLM
- If the LLM can answer from what's already been discussed, it does — avoiding a redundant Pinecone search
- **Skipped for artist intent queries** to prevent history from intercepting profile questions (see below)
- **Skipped when contracts change** — if the user switches contract selection between queries, Tier 2 is bypassed to force fresh retrieval on the new contract (prevents stale answers from old contract history)

### Routing Decision: general vs contract (`_llm_route_decision`)

Once contracts are selected, an LLM router classifies each question as **`contract`** or **`general`** (`max_completion_tokens=2000` — a reasoning model needs headroom, or it returns empty and falls back to `contract`):

- **`contract`** — the answer depends on the selected contract(s): their terms, parties, splits, percentages, dates, amounts, or obligations; or the question refers to "this/my/the contract/deal"; or it's a follow-up building on a prior contract answer. → **Tier 3**.
- **`general`** — a clarification or concept question whose answer does **not** depend on the specific contract: definitions/terminology ("what does 'controlled composition' mean?"), how the industry works broadly ("how are publishers paid?"), what a clause type typically does, or whether something is standard. → **General mode (knowledge base)**, even though contracts are selected.

The router **emphasizes the contract**: when a question could be read either way (the litmus test is "does the answer change based on what *this* contract says?"), and when it is genuinely unsure, it chooses **`contract`**. With no contracts selected the router is skipped — everything is general mode.

**What general mode does NOT load: the contract document.** A clarification answered in general mode draws on the **reference book + the last ~5 conversation turns** (so it stays aware of what you were just discussing) but does **not** re-read the selected contract — it's a clean general answer, not tailored to the deal's numbers. To get a clarification grounded in a specific deal, phrase it against the contract ("in *this* contract, is the royalty deficit-based?") so it routes to `contract` (which has both the contract and the book background).

### Tier 3: Full-Document Contract Context + LLM
- The selected contracts' **full markdown** is passed into a long-context model (combined size capped ~370k chars). No per-chunk Pinecone search of contracts.
- The reference book is retrieved **strictly** (`search_reference(query, floor_count=0)`) and, if it clears threshold, appended as a clearly-labeled background block — *the user's contract always governs; the book never overrides it*.
- LLM generates a grounded answer; structured data extracted post-generation (royalty splits, parties, payment terms, etc.).
- Contract sources are not chunk-cited (the whole doc is in context). The reference book is injected as background only — the answer reads as general knowledge with **no page citations or source naming**; book passages still ride the `reference_sources` SSE field for observability (not user-shown).
- The answer **prose** carries no page numbers, but after generation Zoe attributes the answer to a contract **page** so the source chip opens the PDF there — see **"Contract Page-Jump"** below.

## Artist Intent Detection & Disambiguation

**Problem solved:** When a user asks "tell me about the artist" after discussing contracts, Tier 2 would answer from the sparse conversation history instead of using the rich Supabase artist profile.

**How it works:**
- `_is_artist_intent_query()` does deterministic phrase matching (e.g., "tell me about", "bio", "social media", "genre")
- If artist phrases are found WITHOUT contract keywords → classified as artist intent
- Artist intent queries **skip Tier 2 entirely** and go to the artist routing logic
- If conversation history exists → user sees disambiguation buttons:
  - **"Artist Profile"** — answers from Supabase data (bio, socials, streaming links, genres)
  - **"Conversation"** — answers from what was discussed so far
- If no history exists → answers directly from Supabase artist profile
- The `source_preference` parameter (`"artist_profile"` | `"conversation_history"` | `"contract_context"`) controls explicit routing when the user clicks a button

## Reference Knowledge (Book RAG)

Zoe is backed by a long-form music-business reference book ingested into a shared Pinecone namespace (see the `chunk-pdf-pinecone` skill and `knowledge/` package). It works as **semantic RAG, not a whole-book context dump**: each turn, only the few passages most relevant to *that* question are retrieved and injected — the model never sees the full book at once.

- **Namespace:** `REFERENCE_NAMESPACE` (env-configurable; default `music-business-reference`). It is the **only** namespace Zoe queries. The upload CLI defaults to the same env var so write/read can't drift.
- **Retriever:** `knowledge.reference_search.search_reference(query, top_k=10, min_score=0.45, floor_count, floor_min=0.2)`. Tuned from a live probe (relevant passages ~0.52–0.70, off-topic ≤0.18); `0.45` admits genuine matches while excluding noise.
- **General mode** (no contracts selected, **or** a clarification/concept question the router classified `general` even with contracts selected): `floor_count=5` — always feed the top-ranked book passages so the book contributes even when nothing clears 0.45. The book informs the answer invisibly (RAG): answers are **concise, in Zoe's own words, with no page citations or source naming**. The contract document is **not** loaded in this mode — only the book and the last ~5 conversation turns. Book passages still ride the `reference_sources` SSE field for observability.
- **Contract-assist mode** (contracts selected): `floor_count=0` (strict) — only ≥0.45 matches injected as background, so marginal book text never pollutes contract-specific reasoning. The model is instructed to **answer from the contract first**; if the question is fully answered by the contract it adds nothing, but if the background offers genuinely applicable general industry context the contract doesn't cover, it appends a short, clearly-separated **"Supplemental (general industry context)"** section (1–3 concise bullets). The book never overrides or adds contract terms.
- **Resilience:** a retrieval/Pinecone failure never breaks Zoe — general mode falls back to a general-knowledge answer; contract mode proceeds with no book context. (So Zoe works even before the book is uploaded.)
- **Analytics:** the `/zoe/ask-stream` endpoint tags every event with `mode` (`general` | `contract`).

## Contract Queries

- Contract answers use the **full contract markdown** (cached in `project_files.contract_markdown`, lazily converted on first use), not Pinecone chunk search.
- LLM is instructed to answer ONLY from the contract; the reference book is background only and never overrides it.
- Extracted data (splits, terms, parties) stored in frontend context for follow-ups.
- **No artist/project is required to ask a general question** — the frontend allows sending with nothing selected, and deselecting an artist back to none.

## Contract Page-Jump (source chip → PDF page)

Clicking the answer's **source chip** opens the original PDF at the page the answer drew from (browser-native `#page=N`). Attribution is best-effort and model-derived — a *sensible* page, not an exact clause (clause-exact highlighting is a future tier). The flow is all backend; the frontend and viewer were already wired.

1. **Page markers at conversion.** `pdf_to_markdown` (`helpers.py`) prefixes each page with a `[[PAGE n]]` marker (1-based, from `pymupdf4llm(..., page_chunks=True)` — `metadata["page"]` is unreliable in the pinned version, so the list index drives `n`). Stored in `project_files.contract_markdown`.
2. **Stale-cache healing.** `GET /contracts/{id}/markdown` re-converts when the cached markdown lacks markers (`markdown_has_page_markers`), so contracts cached before this feature gain pages on next use. (Open sessions heal on reload, not mid-session — the frontend reuses in-memory markdown.)
3. **Leak-proof answer.** The answer model is fed a **marker-stripped** copy (`strip_page_markers`) so it can never echo a `[[PAGE n]]` marker into prose. The marked copy is reserved for extraction only.
4. **Gated post-generation extraction.** *After* the `done` event (so the message finalizes instantly with page-less fallback chips), and only when the answer is non-empty and not a fallback, `_extract_page_citations` makes one cheap `DEFAULT_LLM_MODEL` call: it sends the **marked** contract + the answer and asks for JSON `[{contract, page}]`, parsed tolerantly by the module-level `parse_page_citations`. ⚠️ `DEFAULT_LLM_MODEL` is a **reasoning** model — `max_completion_tokens` caps *reasoning + output*, so the budget must be generous (a small cap is consumed entirely by reasoning → empty output).
5. **Page-aware `sources` event.** A second `sources` SSE event is emitted with one entry per **selected** contract (`{contract_file, page_number}`), pages overlaid only on cited contracts (uncited → `page_number: null` → opens page 1). Built from `contract_names` so only known filenames are emitted (every chip stays resolvable on the frontend). The frontend overwrites the page-less fallback chips → upgrade in place. `reference_sources` is `[]` here (book cites already rode the first `sources` event; re-sending would double-count analytics `source_count`).
6. **Viewer.** Chip → `ContractSlideOver` (`src/components/zoe/`) → `<iframe src={signedUrl}#page=N&view=FitH>` via a short-lived Supabase signed URL.

**OneClick isolation:** markers are stripped (`strip_page_markers`) where OneClick builds its `contract_markdowns` (`main.py`), so the royalty parser and `nuance_adjuster`'s verbatim substring check never see them; royalty math is unaffected.

## Context Tracking

The frontend sends a `ConversationContext` object with every request:

- **Current artist/project** — who and what is selected in the sidebar
- **Contracts discussed** — each with extracted data (splits, terms, parties, advances)
- **Artists discussed** — each with extracted profile fields (bio, socials, links)
- **Context switches** — timestamps of when the user changed artist/project/contract

This enables:
- **Cross-contract comparisons** ("compare royalty splits between both contracts") answered from context — no re-search
- **Follow-up questions** about previously extracted data
- **Divider messages** in the chat when switching artists, projects, or contracts

### Contract Switch Detection

The backend tracks the last-used `contract_ids` per session (`InMemoryChatMessageHistory._last_contract_ids`). When a query arrives with different contract IDs than the previous query:

1. Tier 2 (conversation history) is **skipped** — prevents stale answers from old contracts
2. Context-based answer path is **skipped** — forces fresh retrieval
3. The query goes straight to Tier 3 (vector search on the newly selected contract)
4. Subsequent same-contract queries resume normal Tier 2 behavior

## Session Memory (Backend)

- Conversation history stored per `session_id` via `_add_to_memory()` / `_get_conversation_context()`
- Automatic summarization when approaching token limits (`_summarize_if_needed()`)
- Pending suggestion tracking for affirmative responses ("yes", "sure")
- Last-used `contract_ids` tracked per session for contract switch detection
- `clear_session()` wipes history, pending suggestions, and contract tracking for a session

## Frontend Features

- **SSE streaming** — `useStreamingChat` hook parses SSE events, appends tokens to message bubbles in real-time
- **Stop generation** — `AbortController` aborts the fetch; partial answer stays visible
- **Contract chips** — selected contracts shown as dismissible badges above the chat input (like file attachments), with click-to-remove
- **Context switch dividers** — visual separator inserted in chat when artist, project, or contract selection changes
- **Copy response** — clipboard copy on hover of assistant messages
- **Retry/regenerate** — re-sends the last user query, removes the previous answer
- **Conversation persistence** — `useConversationPersistence` hook saves messages, session, and selections to `localStorage`
  - Auto-saves on every state change (debounced 500ms)
  - Restores on page load if session is < 2 hours old
  - Caps at 5 stored sessions
- **New Chat** button — clears session and starts fresh
- **Quick action buttons** — both static (greeting screen) and dynamic (returned by backend in disambiguation/greeting responses)

## SSE Event Protocol

| Event | When | Payload |
|-------|------|---------|
| `start` | Stream begins | `session_id` |
| `sources` | Before streaming | `sources[]` (contract sources, usually empty under full-doc), `reference_sources[]` (book passages: title/section/pages/score), `highest_score` |
| `token` | Each generated token | `content` (text fragment) |
| `data` | After full answer generated | `extracted_data`, `confidence`, `answered_from` |
| `done` | Stream complete | `answered_from` |
| `sources` (again) | **After `done`**, contract mode only, when a page is attributed | Page-aware `sources[]` (`{contract_file, page_number}`, one per selected contract; `null` page = uncited). `reference_sources: []`. Upgrades the chips to page-jump — see "Contract Page-Jump". |
| `complete` | Instant responses (Tier 1/2, disambiguation) | Full response object (no token streaming) |
| `error` | Something went wrong | `message` |

## Dependencies

| Dependency | Purpose |
|------------|---------|
| **OpenAI** | LLM for routing decisions, answer generation, data extraction |
| **Pinecone** | Vector search over the music-business **reference book** (general + contract-assist background). NOT used for contract documents — those use full-document markdown context. |
| **Supabase** | Artist profile data (bio, socials, streaming links, genres) |
| **FastAPI `StreamingResponse`** | SSE endpoint (`/zoe/ask-stream`) |
| **React `ReadableStream`** | Client-side SSE parsing in `useStreamingChat` hook |