# Zoe Chatbot - Query Routing Documentation

A RAG-based chatbot for answering questions about music contracts and artist information, with intelligent context tracking for follow-up questions.

## Query Flow

```
User Query
    ↓
Context-Based? → _can_answer_from_context() → _answer_from_context()
    ↓ No
Conversational? → _is_conversational_query() → _handle_conversational_query()
    ↓ No
Classify Query → _classify_query() → "artist" or "contract"
    ↓
┌─────────────────────────────────────────┐
│ ARTIST                                  │
│ → _handle_artist_query()                │
│ → Uses artist profile data from DB      │
└─────────────────────────────────────────┘
    or
┌─────────────────────────────────────────┐
│ CONTRACT                                │
│ → Requires project selection            │
│ → Queries Pinecone via smart_search()   │
│ → _generate_answer() with LLM           │
│ → Extracts data for context tracking    │
└─────────────────────────────────────────┘
```

## Entry Points

| Method | Use Case |
|--------|----------|
| `smart_ask()` | Main entry - handles artist/contract queries with auto-classification and context awareness |
| `ask_without_project()` | When no project selected - answers artist queries, prompts for project on contract queries |
| `ask_multiple_contracts()` | When specific contracts are selected - searches across all selected contracts |
| `ask_project()` | Delegates to `smart_ask()` with project context |

## Context-Based Answering

The chatbot tracks structured context from the frontend to answer follow-up questions without re-searching documents.

### How It Works

1. **Frontend Tracking**: The UI extracts data from each response (royalty splits, parties, terms, etc.) and associates it with the source contract
2. **Context Passing**: Each request includes a `ConversationContext` object with:
   - Current artist and project
   - List of contracts discussed with extracted data
   - Context switches (when user changes artist/project)
3. **Detection**: `_can_answer_from_context()` checks if the query can be answered from context:
   - **Comparison keywords**: "compare", "difference", "versus", "both contracts"
   - **Summary keywords**: "summarize", "recap", "what did we discuss"
   - **Follow-up keywords**: "you said", "you mentioned", "earlier"
4. **Context Answer**: `_answer_from_context()` uses the structured context + conversation history to generate a response

### Context Data Structure

```typescript
interface ConversationContext {
  session_id: string;
  artist: { id: string; name: string } | null;
  project: { id: string; name: string } | null;
  contracts_discussed: Array<{
    id: string;
    name: string;
    data_extracted: {
      royalty_splits?: Array<{ party: string; percentage: number }>;
      payment_terms?: string;
      parties?: string[];
      advances?: string;
      term_length?: string;
    };
  }>;
  context_switches: Array<{
    timestamp: string;
    type: 'artist' | 'project' | 'contract';
    from: string;
    to: string;
  }>;
}
```

### Example Flow

1. User: "What are the royalty splits in Contract A?" → RAG search, extracts splits
2. User: "What are the royalty splits in Contract B?" → RAG search, extracts splits
3. User: "Compare the royalty splits between both contracts" → **Answered from context** (no document search)

## Query Classification

`_classify_query()` uses LLM to determine query type:

- **Artist queries**: bio, social media, genres, streaming links, contact info, EPK
- **Contract queries**: royalties, payment terms, splits, advances, parties, clauses

## Key Components

### Conversational Handling
- Greetings, thanks, farewells handled without document search
- Returns friendly responses with optional quick action buttons

### Artist Queries
- Answered from artist profile data (not contracts)
- Responses are concise and direct - no embellishment

### Contract Queries
- Requires a project to be selected
- Uses Pinecone semantic search with query categorization
- Similarity threshold check before generating answers
- Sources cited from matched contract sections
- Data extracted and stored in context for follow-ups

## Session Memory

- `_add_to_memory()` - stores conversation history
- `_get_conversation_context()` - retrieves history for LLM context
- `clear_session()` - clears conversation history
- Memory is session-based using `session_id`

## Response Metadata

Each response includes:
- `answer`: The generated answer text
- `confidence`: "high", "medium", "low", "context_based", "needs_project", etc.
- `sources`: List of source documents with page numbers and scores
- `answered_from`: "conversation_context" when answered from context (no RAG)
- `session_id`: Current session identifier

## Dependencies

- **OpenAI**: LLM for classification and answer generation
- **Pinecone**: Vector search for contract documents
- **ContractSearch**: Semantic search with filters

## Private Methods Reference

| Method | Purpose |
|--------|---------|
| `_can_answer_from_context()` | Detects if query can be answered from structured context |
| `_answer_from_context()` | Generates answer using context without document search |
| `_format_context_for_llm()` | Formats structured context as readable text |
| `_is_conversational_query()` | Detects greetings/thanks/farewells |
| `_handle_conversational_query()` | Returns friendly response for conversational queries |
| `_classify_query()` | LLM-based classification: "artist" or "contract" |
| `_handle_artist_query()` | Answers artist questions from profile data |
| `_format_artist_context()` | Formats artist profile for LLM |
| `_check_similarity_threshold()` | Validates search result quality |
| `_create_system_prompt()` | Builds system prompt for answer generation |
| `_format_context()` | Formats search results as context |
| `_generate_answer()` | LLM answer generation with grounding |
