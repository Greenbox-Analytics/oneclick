# Contract RAG System Documentation

A comprehensive semantic search and RAG (Retrieval-Augmented Generation) system for music contract analysis.

## 📁 File Structure

### ✅ **REQUIRED FILES** (Contract RAG System)

These files are part of the new Contract RAG system and should be **KEPT**:

```
src/backend/vector_search/
├── config.py                    # Central configuration for all settings
├── contract_ingestion.py        # PDF upload, chunking, and indexing
├── contract_search.py           # Semantic search with metadata filtering
├── contract_chatbot.py          # RAG-based Q&A chatbot
├── example_usage.py             # Example/test script
└── README.md                    # This documentation file
```

### 📦 **Dependencies**

The Contract RAG system requires these packages (already in `requirements.txt`):

```
# Core dependencies
pinecone          # Vector database
openai            # Embeddings and LLM
pymupdf           # PDF text extraction
tiktoken          # Token counting
python-dotenv     # Environment variables

# Optional (if using LangChain features)
langchain
langchain-community
langchain-openai
```

## 🎯 Overview

This system enables users to:
- Upload and index contract PDFs
- Perform semantic search across contracts
- Ask questions and get AI-powered answers grounded in contract text
- Filter searches by project or specific contracts

## 🏗️ Architecture

### Index Structure

**✅ DO:**
- Use **one shared index per region**: `contracts-us`, `contracts-eu`, `contracts-uk`
- Isolate users with **metadata filtering** (not separate namespaces)
- Use namespace pattern: `{user_id}_data`

**❌ DON'T:**
- Create per-project namespaces
- Create per-contract namespaces
- Create per-user indexes

### Metadata Schema

Each vector includes:
```json
{
  "user_id": "uuid",
  "project_id": "uuid",
  "project_name": "string",
  "contract_id": "uuid",
  "contract_file": "filename.pdf",
  "region": "US|EU|UK",
  "uploaded_at": "ISO timestamp",
  "page_number": 1,
  "chunk_index": 0,
  "token_count": 450,
  "chunk_text": "actual text content"
}
```

## 📦 Modules

### 1. `config.py`
Central configuration for all settings:
- Regional index mapping
- Embedding model settings
- Chunking parameters (300-600 tokens)
- Search parameters (top_k, similarity threshold)
- LLM settings

### 2. `contract_ingestion.py`
Handles PDF upload and indexing:
- Extracts text from PDFs using PyMuPDF
- Chunks text into 300-600 token segments
- Generates deterministic vector IDs (SHA256)
- Creates embeddings with `text-embedding-3-small`
- Upserts to regional Pinecone indexes

**Key Class:** `ContractIngestion`

**Methods:**
- `ingest_contract()` - Complete ingestion pipeline
- `delete_contract()` - Remove contract vectors

### 3. `contract_search.py`
Semantic search with metadata filtering:
- Filters by user_id, project_id, contract_id
- Configurable top_k (default: 8)
- Returns ranked results with scores

**Key Class:** `ContractSearch`

**Methods:**
- `search()` - General search with filters
- `search_project()` - Search within a project
- `search_contract()` - Search within a specific contract
- `get_context_for_llm()` - Format results for LLM

### 4. `contract_chatbot.py`
RAG-based Q&A system:
- Semantic search + LLM answer generation
- Similarity threshold enforcement (≥0.75)
- Grounded responses with source citations
- Conversation history tracking

**Key Class:** `ContractChatbot`

**Methods:**
- `ask()` - Ask a question with filters
- `ask_project()` - Ask about a project
- `ask_contract()` - Ask about a specific contract
- `get_conversation_history()` - Retrieve chat history

## 🚀 Quick Start

### Installation

```bash
cd src/backend
pip install -r requirements.txt
```

### Environment Variables

Add to your `.env` file:
```env
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=optional_custom_endpoint
```

### Example 1: Ingest a Contract

```python
from contract_ingestion import ContractIngestion

# Initialize for US region
ingestion = ContractIngestion(region="US")

# Ingest a contract
stats = ingestion.ingest_contract(
    pdf_path="/path/to/contract.pdf",
    user_id="user-uuid-123",
    project_id="project-uuid-456",
    project_name="My Project",
    contract_id="contract-uuid-789",
    contract_filename="contract.pdf"
)

print(f"Ingested {stats['total_chunks']} chunks")
```

### Example 2: Search Contracts

```python
from contract_search import ContractSearch

# Initialize search
search = ContractSearch(region="US")

# Search within a project
results = search.search_project(
    query="What are the royalty splits?",
    user_id="user-uuid-123",
    project_id="project-uuid-456",
    top_k=8
)

# Display results
for match in results["matches"]:
    print(f"Score: {match['score']}")
    print(f"Source: {match['contract_file']} (Page {match['page_number']})")
    print(f"Text: {match['text'][:200]}...")
```

### Example 3: Ask Questions (RAG)

```python
from contract_chatbot import ContractChatbot

# Initialize chatbot
chatbot = ContractChatbot(region="US")

# Ask a question about a project
result = chatbot.ask_project(
    query="What are the payment terms?",
    user_id="user-uuid-123",
    project_id="project-uuid-456"
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Sources: {len(result['sources'])}")

# View sources
for source in result['sources']:
    print(f"  - {source['contract_file']} (Page {source['page_number']})")
```

## 🔍 Search Workflow

1. **User selects scope:**
   - All contracts in a project
   - Specific contract only

2. **System performs semantic search:**
   - Filters by `user_id` + `project_id` (+ `contract_id` if specified)
   - Retrieves top 5-8 most relevant chunks
   - Ranks by similarity score

3. **Chatbot generates answer:**
   - Checks if highest score ≥ 0.75
   - If yes: Sends context to LLM for grounded answer
   - If no: Returns "I don't know based on the available documents."

## 📊 Chunking Strategy

- **Token-based chunking:** 300-600 tokens per chunk
- **Preserves context:** Splits on paragraph boundaries
- **Page tracking:** Each chunk knows its source page
- **Deterministic IDs:** SHA256 hash ensures consistency

## 🔐 Data Isolation

**Namespace-based isolation:**
- Each user has namespace: `{user_id}_data`
- All vectors in shared regional index
- Metadata filtering ensures user can only access their data

**Metadata filtering:**
```python
filter = {
    "user_id": "user-123",      # Required
    "project_id": "project-456", # Optional
    "contract_id": "contract-789" # Optional
}
```

## 🎯 Similarity Threshold

The chatbot enforces a **0.75 minimum similarity threshold**:

- **Score ≥ 0.75:** Answer the question with LLM
- **Score < 0.75:** Return "I don't know based on the available documents."

This prevents hallucinations and ensures grounded responses.

## 🌍 Regional Indexes

Support for multiple regions:

| Region | Index Name | Use Case |
|--------|------------|----------|
| US | `contracts-us` | North American users |
| EU | `contracts-eu` | European users |
| UK | `contracts-uk` | UK users |

Specify region when initializing:
```python
ingestion = ContractIngestion(region="EU")
search = ContractSearch(region="EU")
chatbot = ContractChatbot(region="EU")
```

## 🛠️ Configuration

All settings in `config.py`:

```python
# Chunking
MIN_CHUNK_TOKENS = 300
MAX_CHUNK_TOKENS = 600

# Search
DEFAULT_TOP_K = 8
MIN_SIMILARITY_THRESHOLD = 0.75

# Embedding
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM
DEFAULT_LLM_MODEL = "gpt-5-mini"
```

## 📝 Best Practices

### 1. Ingestion
- Always provide accurate metadata (user_id, project_id, contract_id)
- Use descriptive project names
- Keep original filenames for reference

### 2. Search
- Use appropriate top_k (5-8 recommended)
- Filter by project_id when possible for better results
- Check similarity scores in results

### 3. Chatbot
- Ask specific questions for better answers
- Review sources to verify accuracy
- Use project-level queries for comparative questions
- Use contract-level queries for specific document questions

### 4. Performance
- Batch process multiple contracts when possible
- Monitor token usage for cost optimization
- Use appropriate region for lower latency

## 🔄 Workflow Integration

### Typical User Flow

1. **User uploads contract PDF** → `contract_ingestion.ingest_contract()`
2. **User selects project/contract** → UI selection
3. **User asks question** → `contract_chatbot.ask_project()` or `ask_contract()`
4. **System returns answer with sources** → Display in UI

### API Integration Example

```python
from fastapi import FastAPI, UploadFile
from contract_ingestion import ContractIngestion
from contract_chatbot import ContractChatbot

app = FastAPI()

@app.post("/upload-contract")
async def upload_contract(
    file: UploadFile,
    user_id: str,
    project_id: str,
    contract_id: str
):
    # Save file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Ingest contract
    ingestion = ContractIngestion(region="US")
    stats = ingestion.ingest_contract(
        pdf_path=temp_path,
        user_id=user_id,
        project_id=project_id,
        project_name="Project Name",
        contract_id=contract_id,
        contract_filename=file.filename
    )
    
    return {"status": "success", "stats": stats}

@app.post("/ask-question")
async def ask_question(
    query: str,
    user_id: str,
    project_id: str,
    contract_id: str = None
):
    chatbot = ContractChatbot(region="US")
    
    if contract_id:
        result = chatbot.ask_contract(
            query=query,
            user_id=user_id,
            project_id=project_id,
            contract_id=contract_id
        )
    else:
        result = chatbot.ask_project(
            query=query,
            user_id=user_id,
            project_id=project_id
        )
    
    return result
```

## 🐛 Troubleshooting

### Issue: No results found
- **Check:** User has uploaded contracts for this project
- **Check:** Metadata filters are correct
- **Check:** Namespace exists in Pinecone

### Issue: Low similarity scores
- **Solution:** Rephrase query to match contract language
- **Solution:** Use more specific terms
- **Solution:** Check if information exists in contracts

### Issue: "I don't know" responses
- **Cause:** Similarity score < 0.75
- **Solution:** Verify information exists in contracts
- **Solution:** Try different query phrasing

### Issue: Slow performance
- **Check:** Batch size configuration
- **Check:** Network latency to Pinecone
- **Consider:** Using regional index closer to users

## 📚 Additional Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

## 🔒 Security Considerations

1. **User Isolation:** Always filter by `user_id`
2. **API Keys:** Store in environment variables, never commit
3. **Input Validation:** Sanitize user queries before processing
4. **Rate Limiting:** Implement rate limits on API endpoints
5. **Access Control:** Verify user owns project/contract before querying

## 📈 Monitoring & Analytics

Track these metrics:
- Ingestion success rate
- Average similarity scores
- Query response times
- LLM token usage
- User satisfaction with answers

## 🚧 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced reranking algorithms
- [ ] Conversation context awareness
- [ ] Batch question answering
- [ ] Export search results
- [ ] Analytics dashboard
- [ ] Custom similarity thresholds per user
- [ ] Support for other document types (DOCX, TXT)

---

## 🤖 Zoe — AI Contract Analyst

Zoe is the AI-powered chatbot for analyzing music contracts. Upload a PDF, ask a question, and Zoe extracts royalty splits, payment terms, rights, and more — with source citations back to the original document.

### The Analogy

Think of Zoe like a **paralegal with a photographic memory**. When you hand her a stack of contracts:

1. She first **reads and highlights** each document (PDF → structured markdown)
2. She **indexes key sections** into a filing cabinet (vector embeddings in Pinecone)
3. When you ask a question, she decides: is this a short stack she can flip through entirely, or does she need to go to the filing cabinet and pull only the relevant folders?
4. She gives you an answer and **tells you exactly which page she found it on**

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
│                                                                 │
│  Zoe.tsx                                                        │
│  ├── Sidebar: Artist → Project → Contract selection             │
│  ├── ZoeChatMessages: streaming chat display + source citations │
│  ├── ZoeInputBar: message input + quick actions                 │
│  └── ContractUploadModal: PDF upload                            │
│                                                                 │
│  useStreamingChat hook                                          │
│  ├── Sends query + full contract markdowns to backend           │
│  └── Reads Server-Sent Events (SSE) stream for live tokens      │
├─────────────────────────────────────────────────────────────────┤
│                        BACKEND (FastAPI)                         │
│                                                                 │
│  PDF Upload Pipeline (/contracts/upload)                        │
│  ├── Store PDF in Supabase Storage                              │
│  ├── Convert PDF → Markdown (pymupdf4llm)                       │
│  ├── Detect sections & categorize (royalties, rights, etc.)     │
│  ├── Chunk text (750 tokens, 30% overlap)                       │
│  └── Embed & upsert into Pinecone (text-embedding-3-small)      │
│                                                                 │
│  Chat Pipeline (/zoe/ask-stream)                                │
│  ├── IF total markdown < 400k chars → Full Document Context     │
│  │   └── Entire contract(s) in system prompt                    │
│  └── IF too large → Vector Search (RAG)                         │
│      ├── Semantic search in Pinecone (top-8, score ≥ 0.50)      │
│      └── Only relevant chunks sent to LLM                       │
│                                                                 │
│  Response: SSE stream → tokens, sources, extracted data         │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Select context** — Pick an artist, project, and one or more contracts from the sidebar. Zoe fetches and caches the full markdown for each selected contract.

2. **Ask a question** — Type a question or use quick actions. The frontend sends your query along with conversation context (session history, artist/contract metadata, and cached contract markdowns) to `/zoe/ask-stream`.

3. **Smart retrieval** — The backend checks if all selected contracts fit under ~100k tokens. If yes, it sends the full documents as context (best accuracy). If not, it falls back to semantic vector search via Pinecone, pulling only the most relevant chunks.

4. **Streaming response** — The LLM streams its answer back via Server-Sent Events. The frontend renders tokens in real-time, along with confidence scores, source citations (contract name + page number), and extracted structured data (royalty splits, payment terms, etc.).

5. **Context tracking** — Zoe maintains a `ConversationContext` object that tracks which artists and contracts have been discussed, what data has been extracted, and any context switches (e.g., switching artists mid-conversation inserts a divider rather than clearing history).

### Gotcha

The full-document path sends **entire contract markdowns on every message** when contracts are small enough. This gives the best accuracy but means token usage scales with contract size × message count. For token-heavy workflows (large contracts, long conversations), the system automatically falls back to RAG — but be aware that cross-section reasoning (e.g., "how does clause 3 interact with clause 12?") may be less reliable in RAG mode since the model only sees retrieved chunks.

### Key Files

- `vector_search/contract_chatbot.py` — RAG chatbot, streaming, conversation memory
- `vector_search/contract_ingestion.py` — PDF ingestion pipeline
- `vector_search/contract_search.py` — Semantic search with metadata filtering
- `vector_search/helpers.py` — PDF → markdown, section splitting, table extraction
- Frontend: `src/pages/Zoe.tsx`, `src/hooks/useStreamingChat.ts`

---

## 🎵 OneClick — Royalty Calculator

OneClick is the automated royalty calculation engine. It extracts structured data (parties, works, royalty splits) from vectorized contracts, matches them against royalty statements, and calculates what each party is owed.

### The Analogy

Think of OneClick like an **accountant who can read contracts and do math at the same time**. Normally you'd need:

1. A lawyer to read the contract and list who gets what percentage
2. A bookkeeper to read the royalty statement and total up each song's earnings
3. An accountant to cross-reference the two and calculate payments

OneClick does all three — it reads the contract via AI (extracting parties, works, and splits from Pinecone vectors), reads the royalty statement (Excel/CSV), fuzzy-matches songs across both, and calculates each party's payment.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONTRACT PARSING                             │
│                                                                 │
│  contract_parser.py                                             │
│  ├── _extract_parties()   → queries Pinecone → LLM extraction  │
│  ├── _extract_works()     → queries Pinecone → LLM extraction  │
│  └── _extract_royalties() → queries Pinecone → LLM extraction  │
│       (runs parties + works in parallel, royalties sequential)  │
│                                                                 │
│  Pinecone queries split by content type:                        │
│  ├── Prose chunks:  top_k = 10                                  │
│  └── Table chunks:  top_k = 20 (tables are information-dense)   │
├─────────────────────────────────────────────────────────────────┤
│                    ROYALTY CALCULATION                            │
│                                                                 │
│  royalty_calculator.py                                           │
│  ├── Read royalty statement (CSV/Excel)                          │
│  │   └── Auto-detect columns (keyword → fuzzy → LLM semantic)   │
│  ├── Fuzzy-match contract works ↔ statement songs               │
│  │   └── 3-layer: exact → partial (70%) → fuzzy (60%)           │
│  ├── Calculate: total_royalty × (percentage / 100)               │
│  └── Export: Excel (formatted) / JSON / Confirm to DB            │
│                                                                 │
│  Multi-contract support:                                        │
│  ├── Parse contracts in parallel (ThreadPoolExecutor, 4 workers)│
│  ├── Merge: deduplicate parties, merge works & royalty shares   │
│  └── Calculate from merged data                                 │
├─────────────────────────────────────────────────────────────────┤
│                      API ENDPOINTS                               │
│                                                                 │
│  GET  /oneclick/calculate-royalties-stream  → SSE progress      │
│  POST /oneclick/calculate-royalties         → immediate result  │
│  POST /oneclick/confirm                     → save to database  │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Parse the contract** — `contract_parser.py` queries Pinecone vectors to extract parties (names + roles), works (song titles + types), and royalty shares (who gets what %). Parties and works are extracted in parallel; royalties run sequentially because they need party names for matching.

2. **Read the royalty statement** — The calculator reads an Excel or CSV file and auto-detects the title and payable columns using a 3-layer strategy: keyword matching, fuzzy matching (80% threshold), and LLM-based semantic search as a fallback.

3. **Match works to songs** — Each contract work is fuzzy-matched against statement song titles using exact match → partial match (70% length ratio) → fuzzy first-3-words comparison (60% ratio). Multiple matches are aggregated.

4. **Calculate payments** — For each streaming-equivalent royalty share: `amount = total_royalty × (percentage / 100)`. Results are grouped by party with role and terms preserved.

5. **Export or confirm** — Results can be exported to formatted Excel (with currency formatting and color coding), JSON, or confirmed directly to the database.

### Gotcha

The contract parser runs **separate Pinecone queries for prose and table chunks** (10 and 20 respectively) because tables contain dense financial data (percentages, amounts) that competes with many other chunks in similarity search. If you're getting inaccurate royalty extractions, the issue is often that the table chunk containing the actual split wasn't retrieved — increasing the table `top_k` or checking that tables were properly linearized during ingestion usually fixes it.

### Key Files

- `oneclick/contract_parser.py` — Structured data extraction from vectors
- `oneclick/royalty_calculator.py` — Payment calculation engine
- `oneclick/helpers.py` — Statement reading, column detection, fuzzy matching
- `vector_search/contract_ingestion.py` — Shared ingestion pipeline (table linearization)
- `main.py` — API endpoints (`/oneclick/*`)
