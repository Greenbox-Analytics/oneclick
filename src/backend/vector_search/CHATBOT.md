# Zoe Chatbot - Query Routing Documentation

A RAG-based chatbot for answering questions about music contracts and artist information.

## Query Flow

```
User Query
    ↓
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
└─────────────────────────────────────────┘
```

## Entry Points

| Method | Use Case |
|--------|----------|
| `smart_ask()` | Main entry - handles both artist and contract queries with auto-classification |
| `ask_without_project()` | When no project selected - can answer artist queries, prompts for project on contract queries |
| `ask_multiple_contracts()` | When specific contracts are selected |
| `ask_project()` | Delegates to `smart_ask()` with project context |
| `ask_contract()` | Delegates to `smart_ask()` with single contract context |

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

## Session Memory

- `_add_to_memory()` - stores conversation history
- `_get_conversation_context()` - retrieves history for LLM context
- `clear_session()` - clears conversation history
- Memory is session-based using `session_id`

## Dependencies

- **OpenAI**: LLM for classification and answer generation
- **Pinecone**: Vector search for contract documents
- **ContractSearch**: Semantic search with filters
