# Zoe Chatbot

A RAG-based AI assistant for music contracts and artist profiles. Streams responses in real-time via SSE (like ChatGPT/Claude), with intelligent routing that decides where to pull answers from.

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
    └─ Tier 3: Vector search + LLM generation
        → Pinecone semantic search across contract documents
        → LLM generates grounded answer with source citations
        → Structured data extracted for context tracking
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

### Tier 3: Vector Search + LLM
- Pinecone semantic search with automatic query categorization
- LLM generates a grounded answer strictly from retrieved document context
- Structured data extracted post-generation (royalty splits, parties, payment terms, etc.)
- Sources cited with contract file names, page numbers, and relevance scores

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

## Contract Queries

- Requires a project to be selected (user is prompted if not)
- Uses Pinecone semantic search via `ContractSearch.smart_search()` or `search_multiple_contracts()`
- Query categorization adjusts search strategy (e.g., royalty queries search financial sections)
- Similarity threshold check before generating answers — low-quality matches are rejected
- LLM is instructed to answer ONLY from retrieved context — no hallucination
- Extracted data (splits, terms, parties) stored in frontend context for follow-ups

## Context Tracking

The frontend sends a `ConversationContext` object with every request:

- **Current artist/project** — who and what is selected in the sidebar
- **Contracts discussed** — each with extracted data (splits, terms, parties, advances)
- **Artists discussed** — each with extracted profile fields (bio, socials, links)
- **Context switches** — timestamps of when the user changed artist/project/contract

This enables:
- **Cross-contract comparisons** ("compare royalty splits between both contracts") answered from context — no re-search
- **Follow-up questions** about previously extracted data
- **Divider messages** in the chat when switching artists or projects

## Session Memory (Backend)

- Conversation history stored per `session_id` via `_add_to_memory()` / `_get_conversation_context()`
- Automatic summarization when approaching token limits (`_summarize_if_needed()`)
- Pending suggestion tracking for affirmative responses ("yes", "sure")
- `clear_session()` wipes history for a session

## Frontend Features

- **SSE streaming** — `useStreamingChat` hook parses SSE events, appends tokens to message bubbles in real-time
- **Stop generation** — `AbortController` aborts the fetch; partial answer stays visible
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
| `sources` | After Pinecone search (before streaming) | `sources[]`, `highest_score` |
| `token` | Each generated token | `content` (text fragment) |
| `data` | After full answer generated | `extracted_data`, `confidence`, `answered_from` |
| `done` | Stream complete | `answered_from` |
| `complete` | Instant responses (Tier 1/2, disambiguation) | Full response object (no token streaming) |
| `error` | Something went wrong | `message` |

## Dependencies

| Dependency | Purpose |
|------------|---------|
| **OpenAI** | LLM for routing decisions, answer generation, data extraction |
| **Pinecone** | Vector search across indexed contract documents |
| **Supabase** | Artist profile data (bio, socials, streaming links, genres) |
| **FastAPI `StreamingResponse`** | SSE endpoint (`/zoe/ask-stream`) |
| **React `ReadableStream`** | Client-side SSE parsing in `useStreamingChat` hook |
