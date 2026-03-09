# OneClick - Complete Process Documentation

## Overview

OneClick is a full-stack application for music contract analysis and royalty calculation. It ingests PDF contracts, vectorizes them for intelligent search, extracts structured data (parties, works, royalty splits), and calculates payments from royalty statements — all powered by AI.

**Tech Stack:**
- **Frontend**: React + TypeScript (Vite)
- **Backend**: FastAPI (Python)
- **Vector DB**: Pinecone (contract embeddings)
- **SQL DB**: Supabase (PostgreSQL)
- **Storage**: Supabase Storage
- **LLM**: OpenAI (gpt-5-mini, text-embedding-3-small)

---

## Phase 1: Contract Ingestion & Vectorization

**Files**: `src/backend/vector_search/contract_ingestion.py`, `src/backend/vector_search/helpers.py`

### Process Flow

A contract PDF is uploaded → converted to markdown → split into sections → chunked → embedded → stored in Pinecone.

### Functions

#### `pdf_to_markdown(pdf_path)`
Converts a PDF file to structured markdown using `pymupdf4llm`. Preserves headings, tables, and text hierarchy.

#### `detect_and_extract_tables(markdown_text)`
Identifies markdown tables (lines with `|---|`) in the converted text.
- **Returns**: `(has_tables: bool, table_blocks: List[TableBlock], text_without_tables: str)`
- `TableBlock` contains: `raw_text`, `preceding_context`, `start_line`, `end_line`

#### `split_into_sections(markdown_text)`
Splits markdown into logical sections using a 4-layer detection strategy:
1. **Layer 1**: Explicit headings (markdown `#` or numbered like "1. SECTION")
2. **Layer 2**: Heuristic headers (ALL CAPS lines, bold text, short title-like lines)
3. **Layer 3**: Semantic heading detection
4. **Layer 4**: Content-based inference (detects royalty triggers like percentage mentions)
- **Returns**: List of `(header, content)` tuples

#### `categorize_section(section_header)`
Assigns each section a category based on keyword matching:
- `ROYALTY_CALCULATIONS`, `PUBLISHING_RIGHTS`, `PERFORMANCE_RIGHTS`, `COPYRIGHT`, `TERMINATION`, `MASTER_RIGHTS`, `OWNERSHIP_RIGHTS`, `ACCOUNTING_AND_CREDIT`, `OTHER`

#### `_build_table_chunks(table_blocks, markdown_lines)`
Processes each detected table:
- Extracts 30% preceding context for surrounding awareness
- Linearizes the table (converts pipe-delimited format to natural language)
- Categorizes table content
- Splits oversized tables (>4000 chars)

#### `RecursiveCharacterTextSplitter` (from LangChain)
Chunks text sections into smaller pieces:
- **Chunk size**: 750 tokens
- **Overlap**: 225 tokens (30%)
- **Separators**: `["\n\n", "\n", ". ", " ", ""]`

#### `generate_deterministic_id(chunk_text, metadata)`
Creates a SHA256 hash from `chunk_text + normalized metadata` (user_id, contract_id, section_heading, contract_file). Ensures the same chunk always gets the same ID across uploads (deduplication).

#### `create_embeddings(texts)`
Generates vector embeddings using OpenAI's `text-embedding-3-small` model (1536 dimensions).
- **Batch size**: 20
- **Returns**: List of embedding vectors

#### Pinecone Upsert
Vectors are upserted to Pinecone with:
- **Namespace**: `{user_id}-namespace`
- **Batch size**: 20
- **Metadata per chunk**:
  ```python
  {
    "user_id", "project_id", "project_name", "contract_id",
    "contract_file", "section_heading", "section_category",
    "uploaded_at", "chunk_text", "is_table"
  }
  ```

#### Ingestion Stats (returned on completion)
```python
{
  "contract_id": str,
  "total_sections": int,
  "total_chunks": int,
  "table_chunks": int,
  "prose_chunks": int,
  "total_vectors": int,
  "category_distribution": dict,
  "namespace": str,
  "index": str
}
```

### Table vs Prose Retrieval: Key Differences

Tables and prose text follow different paths at both **ingestion** and **retrieval** time because structured tabular data (royalty splits, payment schedules, credit breakdowns) needs special handling to preserve meaning.

#### Ingestion Differences

| Aspect | Prose Chunks | Table Chunks |
|---|---|---|
| **Source** | Text with tables stripped out | Detected markdown tables (`\|---\|` pattern) |
| **Splitting** | `RecursiveCharacterTextSplitter` (750 tokens, 30% overlap) | Kept as atomic units; only split if oversized (>4000 chars) |
| **Context** | Inherits section header from `split_into_sections()` | Gets 30% preceding context extracted from surrounding prose |
| **Embedding text** | Raw chunk text | **Linearized** — table is converted from pipe-delimited format to natural language sentences (e.g., "Row 1: Artist gets 50% of streaming royalties") |
| **Stored text** | Same as embedding text | Original pipe-delimited table (for display), NOT the linearized version |
| **Categorization** | Based on section header keywords | Based on table content + preceding context via `categorize_table_content()` |
| **Metadata flag** | `is_table: False` | `is_table: True` |

The linearization step is critical — raw markdown tables embed poorly because embedding models struggle with pipe-delimited structure. Converting to natural language sentences produces much better vector representations.

#### Retrieval Differences

**Contract Parser** (`contract_parser.py` — used for data extraction & royalty calculation):

Runs **two separate Pinecone queries** per question, one for each content type:

| Aspect | Prose Query | Table Query |
|---|---|---|
| **Filter** | `is_table: {$ne: True}` | `is_table: {$eq: True}` |
| **top_k** | 10 | 20 (2x more, because tables are information-dense) |
| **Reranking** | None (direct cosine similarity) | None (direct cosine similarity) |
| **Fallback** | If no results: retry without category filter, keep `is_table` filter | Same fallback strategy |

Results are concatenated: `prose_chunks + table_chunks` → combined context sent to LLM.

The table query uses a higher `top_k` (20 vs 10) because:
- Tables contain dense, structured data (percentages, names, amounts) that is critical for accurate extraction
- A single table row may contain the exact royalty split needed, but it competes with many other table chunks
- Casting a wider net for tables reduces the risk of missing key financial data

**Zoe Chatbot** (`contract_search.py` — used for conversational Q&A):

Does **not** separate table and prose retrieval. Runs a single unified query against all chunks regardless of `is_table` flag. This is acceptable for conversational use where the answer may come from either source, and the LLM handles mixed content well.

---

## Phase 2: Contract Parsing & Data Extraction

**File**: `src/backend/oneclick/contract_parser.py`

### Data Structures

```python
@dataclass
class Party:
    name: str
    role: str

@dataclass
class Work:
    title: str
    work_type: str = "song"  # composition, master recording, song, album

@dataclass
class RoyaltyShare:
    party_name: str
    royalty_type: str       # Streaming, Master, Publishing, Producer, Mixer, Remixer
    percentage: float
    terms: Optional[str]

@dataclass
class ContractData:
    parties: List[Party]
    works: List[Work]
    royalty_shares: List[RoyaltyShare]
    contract_summary: Optional[str]
```

### Functions

#### `parse_contract(path, user_id, contract_id, use_parallel=True)`
Main entry point for extracting structured data from a vectorized contract. Orchestrates the full extraction pipeline.

1. Extracts parties and works in parallel (ThreadPoolExecutor, max_workers=2)
2. Extracts royalties sequentially (needs party names for matching)
3. Post-processes: simplifies roles, reconciles names
- **Returns**: `ContractData`

#### `_extract_parties(user_id, contract_id)`
Queries Pinecone for party-related chunks, sends to LLM with extraction template.
- **Parse format**: `Name | Role`
- Deduplicates by normalized name
- Tracks aliases (p/k/a, d/b/a, etc.)
- **Returns**: `List[Party]`

#### `_extract_works(user_id, contract_id)`
Queries Pinecone for work-related chunks, extracts musical works.
- **Parse format**: `Canonical Title | Observed Variant | Type`
- Deduplicates by normalized title
- Infers work_type (composition, master recording, song, album)
- **Returns**: `List[Work]`

#### `_extract_royalties(user_id, contract_id, parties)`
Queries Pinecone for royalty-related chunks, extracts percentage splits.
- **Parse format**: `Name | Royalty Type | Percentage | Terms`
- Standardizes royalty type names
- Reconciles share party names with extracted parties list
- **Matching strategy**: exact normalized name → partial match
- **Returns**: `List[RoyaltyShare]`

#### `_query_pinecone(question, user_id, contract_id, top_k=5)`
Retrieves relevant chunks from Pinecone for a given question.
- Queries prose chunks (top_k=10) and table chunks (top_k=20) separately
- Creates query embedding using text-embedding-3-small
- Builds metadata filter: `{section_category, is_table, contract_id}`
- **Fallback**: retries without category filter if results are poor
- **Returns**: concatenated prose + table chunk text

#### `_ask_llm(context, question, template)`
Sends retrieved context + question to gpt-5-mini with a structured extraction template.
- **Returns**: parsed structured output (pipe-separated)

### Post-Processing

#### `normalize_name(name)`
Removes role annotations, lowercases, strips whitespace for comparison.

#### `simplify_role(role)`
Maps verbose roles to standard ones using `ROLE_SIMPLIFICATIONS` dict (e.g., "lyrical writer" → "writer").

---

## Phase 3: Search & Retrieval

**Files**: `src/backend/vector_search/contract_search.py`, `src/backend/vector_search/query_categorizer.py`

### ContractSearch Class

#### `search(query, user_id, project_id, contract_id, section_categories, top_k=8)`
Basic vector search against Pinecone.
- Builds metadata filter with user_id, project_id, contract_id
- Adds section_category filter for specific queries
- Creates query embedding
- **Returns**: top_k matches with relevance scores

#### `smart_search(query, user_id, project_id, contract_id, top_k)`
Enhanced search with automatic query categorization.
- Categorizes query using `categorize_query(query, use_llm=True)`
- Auto-adjusts top_k: general queries → 5, specific → 3
- **Fallback**: if best score < 0.30, retries without category filter
- **Returns**: matches + categorization metadata

#### `search_multiple_contracts(query, user_id, project_id, contract_ids, top_k=8)`
Searches across multiple contracts using Pinecone `$in` operator.
- Filter: `{contract_id: {$in: contract_ids}}`
- **Returns**: matches across contracts with distribution info

### Query Categorization

#### `categorize_query_fast(query)` (keyword-based)
Fast, no-LLM categorization using keyword matching against `CATEGORY_KEYWORDS` dict.
- Detects general queries (summarize, overview, all, etc.)
- **Returns**: `{categories, is_general, confidence, reasoning}`

#### `categorize_query(query, use_llm=True)` (LLM-based)
Sends query + category descriptions to gpt-5-mini for accurate categorization.
- **Returns**: `{categories, is_general, confidence, reasoning}`

#### `build_metadata_filter(categories, is_general, user_id, contract_id)`
Constructs a Pinecone metadata filter dict.
- Specific queries: includes `section_category` filter
- General queries: omits category filter (searches all sections)

---

## Phase 4: Zoe AI Chatbot (RAG & Conversation)

**File**: `src/backend/vector_search/contract_chatbot.py`

### Conversation Memory System

#### `PinnedFact` (dataclass)
Per-session pinned facts extracted from contract context:
```python
{
  id: str,
  fact_type: str,        # royalty_split, payment_terms, parties, etc.
  description: str,
  value: Any,
  confidence: float,     # 0.0 - 1.0
  source_type: str,      # "document" | "user_stated" | "inferred"
  source_reference: str,
  scope: Scope,
  extracted_at: datetime,
  last_verified: datetime
}
```

#### `Scope` (dataclass)
Tracks the context (artist, project, contract) a fact applies to.
- `matches(other)`: checks if scope matches or is a subset
- `is_global()`: checks if fact is unscoped

#### `AssumptionLedger`
Tracks unverified assumptions made during conversation.
- `add_assumption()`: register a new assumption
- `get_active_assumptions()`: get non-invalidated ones
- `invalidate_on_scope_change()`: clear assumptions when context changes
- `verify_assumption()`: mark assumption as confirmed

#### `InMemoryChatMessageHistory`
Per-session message storage:
- Max 100 messages per session
- Session TTL: 3600 seconds
- Provides messages for LLM in `[{role, content}]` format
- Estimates token count (~1 token per 4 chars)

#### `ConversationSummary`
Rolling conversation compression:
- Keeps recent 6 turns verbatim
- Compresses older turns into summary
- Preserves key facts and context

### Confidence Levels
| Level | Range | Behavior |
|---|---|---|
| HIGH | ≥ 0.85 | Answer directly |
| MEDIUM | 0.60 - 0.84 | Answer with caveat |
| LOW | < 0.60 | Reverify or ask for clarification |
| CONTEXT | — | From pinned facts |
| CONVERSATIONAL | — | Greetings, small talk |

### Ask Methods (all return generators for streaming)

#### `ask_without_project(query, user_id, session_id, artist_data, context, source_preference)`
For artist-only queries (no contract selected). Answers about artist profile, bio, social links, streaming, genres.

#### `ask_project(query, user_id, project_id, top_k, session_id, artist_data, context, source_preference)`
Searches all contracts within a project using multi-contract search.

#### `ask_multiple_contracts(query, user_id, project_id, contract_ids, top_k, session_id, artist_data, context, source_preference)`
Searches specific contracts by their IDs. Aggregates results across contracts.

#### `ask_stream(query, user_id, project_id, contract_ids, top_k, session_id, artist_data, context, source_preference)`
Unified streaming endpoint — routes to the appropriate ask method and yields SSE events.

### SSE Event Types
| Event | Payload | Description |
|---|---|---|
| `start` | `{session_id}` | Stream initialized |
| `sources` | `{sources[], highest_score, search_results_count}` | Retrieved sources |
| `token` | `{content}` | Streamed text chunk |
| `data` | `{extracted_data, confidence, answered_from, highest_score}` | Metadata |
| `done` | `{answered_from}` | Stream complete |
| `complete` | `{answer, confidence, sources, session_id, ...}` | Non-streamed response |
| `error` | `{message}` | Error occurred |

---

## Phase 5: Royalty Calculation

**Files**: `src/backend/oneclick/royalty_calculator.py`, `src/backend/oneclick/helpers.py`

### Output Structure

```python
@dataclass
class RoyaltyPayment:
    song_title: str
    party_name: str
    role: str
    royalty_type: str
    percentage: float
    total_royalty: float
    amount_to_pay: float
    terms: Optional[str]
```

### Functions

#### `calculate_payments(contract_path, statement_path, user_id, contract_id)`
Main single-contract calculation flow:
1. Parse contract from Pinecone → get `ContractData`
2. Read royalty statement → get `Dict[song_title → amount]`
3. Match works to statement entries
4. Calculate each party's payment
- **Returns**: `List[RoyaltyPayment]`

#### `calculate_payments_from_contract_ids(contract_ids, user_id, statement_path)`
Multi-contract calculation:
1. Parse all contracts in parallel (ThreadPoolExecutor, max_workers=4)
2. Merge contracts with `merge_contracts()`
3. Calculate payments from merged data
- **Returns**: `List[RoyaltyPayment]`

#### `merge_contracts(contracts)`
Merges multiple `ContractData` objects:
- Deduplicates parties by normalized name
- Merges works (keeps more specific `work_type`)
- Merges royalty shares (checks for duplicates by name + percentage + type)

#### `read_royalty_statement(excel_path, title_column, payable_column)`
Reads a CSV or Excel royalty statement and extracts song-to-amount mappings.
- **Column auto-detection** (3 layers):
  1. Keyword matching ("title", "payable")
  2. Fuzzy matching (80% similarity threshold)
  3. Semantic search via LLM
- Sums duplicate entries by song title
- **Returns**: `Dict[song_title → total_amount]`

#### `find_matching_song(song_title, song_totals)`
Fuzzy-matches a contract work title against statement song titles:
1. **Exact**: normalized string match
2. **Partial**: contains check with 70% length ratio
3. **Fuzzy**: first 3 words comparison, 60% ratio
- **Returns**: `(matching_title, aggregated_amount)` for all matches

#### `_calculate_payments_from_data(contract_data, song_totals)`
Core calculation logic:
- For each work in the contract → find matching song in statement
- For each streaming royalty share → `amount = total_royalty × (percentage / 100)`
- Only processes streaming-equivalent royalty types

#### `is_streaming_equivalent_royalty_type(royalty_type)`
Determines if a royalty type should be used for payment calculation. Matches: streaming, digital, master, producer, master royalties, net master revenue, etc.

### Export Functions

#### `save_payments_to_excel(payments, output_path)`
Exports payments to a formatted Excel file with currency formatting, color coding, and totals row.

#### `save_payments_to_json(payments, output_path)`
Exports payments to JSON with a summary section.

#### `print_payment_summary(payments)`
Prints a console-friendly summary grouped by payee.

---

## Phase 6: API Endpoints

**File**: `src/backend/main.py`

### Contract Management
| Method | Endpoint | Description |
|---|---|---|
| POST | `/contracts/upload` | Upload a single contract PDF |
| POST | `/contracts/upload-multiple` | Upload multiple contracts |
| DELETE | `/contracts/{contract_id}` | Delete contract + its vectors |

### File Management
| Method | Endpoint | Description |
|---|---|---|
| GET | `/files/{project_id}` | List project files |
| GET | `/artists/{artist_id}/category/{category}` | Filter files by category |
| POST | `/upload` | Generic file upload (contract, royalty_statement, etc.) |

### Zoe Chatbot
| Method | Endpoint | Description |
|---|---|---|
| POST | `/zoe/ask` | Non-streaming Q&A (immediate response) |
| POST | `/zoe/ask-stream` | Streaming Q&A via SSE |
| DELETE | `/zoe/session/{session_id}` | Clear conversation history |
| GET | `/zoe/session/{session_id}/history` | Get conversation messages |

### OneClick Royalty
| Method | Endpoint | Description |
|---|---|---|
| GET | `/oneclick/calculate-royalties-stream` | Streaming calculation with progress events |
| POST | `/oneclick/calculate-royalties` | Non-streaming calculation |
| POST | `/oneclick/confirm` | Save confirmed results to database |

### Data Management
| Method | Endpoint | Description |
|---|---|---|
| GET | `/artists` | List artists (filter by user_id) |
| GET | `/artists/{artist_id}` | Get artist details |
| GET | `/projects` | List all projects |
| GET | `/projects/{artist_id}` | Get artist's projects |
| GET | `/projects/{project_id}/contracts` | Get project's contracts |
| POST | `/projects` | Create project |

### Health
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/` | Root endpoint |

---

## Phase 7: Frontend Streaming

**File**: `src/hooks/useStreamingChat.ts`

### `useStreamingChat()` Hook

Manages the chat UI state and SSE connection to the backend.

```typescript
interface UseStreamingChat {
  messages: Message[]
  isStreaming: boolean
  error: string

  sendMessage(query, params, options?): Promise<StreamResult>
  stopGeneration(): void
  retryLastMessage(params): void
  addSystemMessage(content: string): void
  clearMessages(): void
}
```

#### `sendMessage(query, params, options)`
Sends a message to the backend SSE endpoint and processes the event stream in real-time:
- Adds user message to state
- Opens SSE connection to `/zoe/ask-stream`
- Parses incoming events via `parseSSEBuffer()`
- Updates assistant message progressively (token by token)
- **Returns**: `{sessionId, contextCleared?, extractedData?, answeredFrom?, sources?}`

#### `parseSSEBuffer(buffer)`
Parses raw SSE data (`data: {...}\n\n` format) into structured events. Handles partial/incomplete events by buffering.

### Message Structure
```typescript
interface Message {
  id: string
  role: "user" | "assistant" | "system"
  content: string
  confidence?: string
  sources?: MessageSource[]
  timestamp: string
  showQuickActions?: boolean
  quickActions?: AssistantQuickAction[]
  answeredFrom?: string
  extractedData?: Record<string, unknown> | null
  isStreaming?: boolean
}

interface MessageSource {
  contract_file: string
  page_number?: number
  score: number
  project_name?: string
  section_heading?: string
  section_category?: string
}
```

---

## End-to-End Data Flow

```
PDF Upload
  │
  ▼
PDF → Markdown (pymupdf4llm, preserves structure)
  │
  ▼
Detect Tables + Split into Sections (4-layer heading detection)
  │
  ▼
Categorize Sections + Chunk Text (750 tokens, 30% overlap)
  │
  ▼
Create Embeddings (text-embedding-3-small, 1536 dims)
  │
  ▼
Upsert to Pinecone (user-namespace, with metadata)
  │
  ├──────────────────────────────────┐
  ▼                                  ▼
[Zoe Chatbot]                   [OneClick Royalty]
  │                                  │
  ▼                                  ▼
Embed Query                     Parse Contract (extract parties,
  │                             works, royalty shares from vectors)
  ▼                                  │
Retrieve Top-K from Pinecone         ▼
  │                             Read Royalty Statement (Excel/CSV)
  ▼                                  │
Build Prompt + Stream Response       ▼
  │                             Fuzzy-Match Works ↔ Statement Songs
  ▼                                  │
SSE Events → Frontend               ▼
                                Calculate Amounts per Party
                                     │
                                     ▼
                                Export (Excel / JSON / Confirm to DB)
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_SECRET_KEY` | Supabase service role key |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Pinecone index name |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_BASE_URL` | OpenAI base URL (optional) |
| `OPENAI_LLM_MODEL` | LLM model name (default: gpt-5-mini) |
| `ALLOWED_ORIGINS` | CORS allowed origins |
| `VITE_BACKEND_API_URL` | Backend API URL (frontend config) |
