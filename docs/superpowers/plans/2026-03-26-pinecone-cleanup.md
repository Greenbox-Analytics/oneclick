# Pinecone Dead Code Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all Pinecone-related dead code, the unused non-streaming `/zoe/ask` endpoint, vector search modules, debug scripts, pip dependencies, and env vars — completing the Pinecone removal started in the prior plan.

**Architecture:** Work from the leaves inward: delete standalone files first, then remove references in main.py and contract_chatbot.py, then clean up dependencies. The streaming `/zoe/ask-stream` endpoint and `ContractChatbot.ask_stream()` remain as the only Q&A path.

**Tech Stack:** Python/FastAPI backend, pip requirements

---

## File Structure

| Action | File | Reason |
|--------|------|--------|
| **Delete** | `src/backend/vector_search/contract_ingestion.py` | `ingest_contract()` / `delete_contract()` no longer called |
| **Delete** | `src/backend/vector_search/contract_search.py` | `search_multiple_contracts()` / `smart_search()` no longer called from streaming path; non-streaming endpoint being removed |
| **Delete** | `src/backend/vector_search/query_categorizer.py` | Only imported by `contract_search.py` |
| **Delete** | `src/backend/oneclick/search_chunks.py` | Standalone debug script, not part of app |
| **Modify** | `src/backend/main.py` | Remove dead imports, `get_contract_ingestion()`, `contract_ingestion` global, `/zoe/ask` endpoint |
| **Modify** | `src/backend/vector_search/contract_chatbot.py` | Remove `ContractSearch` import, `self.search_engine` init, `ask_project()`, `ask_multiple_contracts()`, `ask_without_project()` methods |
| **Modify** | `src/backend/oneclick/helpers.py` | Remove `find_chunks_with_text()` function and its Pinecone import |
| **Modify** | `src/backend/requirements.txt` | Remove `pinecone` and `langchain-text-splitters` packages |
| **Modify** | `.env.example` | Remove Pinecone env vars |

---

### Task 1: Delete standalone dead files

**Goal:** Remove the 4 files that are entirely dead code.

**Files:**
- Delete: `src/backend/vector_search/contract_ingestion.py`
- Delete: `src/backend/vector_search/contract_search.py`
- Delete: `src/backend/vector_search/query_categorizer.py`
- Delete: `src/backend/oneclick/search_chunks.py`

**Acceptance Criteria:**
- [ ] All 4 files deleted
- [ ] No import errors in remaining code (verified in later tasks)

**Verify:** `ls src/backend/vector_search/contract_ingestion.py src/backend/vector_search/contract_search.py src/backend/vector_search/query_categorizer.py src/backend/oneclick/search_chunks.py` → all "No such file"

**Steps:**

- [ ] **Step 1: Delete the files**

```bash
rm src/backend/vector_search/contract_ingestion.py
rm src/backend/vector_search/contract_search.py
rm src/backend/vector_search/query_categorizer.py
rm src/backend/oneclick/search_chunks.py
```

- [ ] **Step 2: Commit**

```bash
git add -u
git commit -m "chore: delete dead Pinecone modules (ingestion, search, categorizer, debug script)"
```

---

### Task 2: Clean up main.py — remove dead imports, factory, and /zoe/ask endpoint

**Goal:** Remove all references to deleted modules and the unused non-streaming endpoint from main.py.

**Files:**
- Modify: `src/backend/main.py`

**Acceptance Criteria:**
- [ ] `ContractIngestion` and `ContractSearch` imports removed (lines 26-27)
- [ ] `contract_ingestion = None` global removed (line 285)
- [ ] `get_contract_ingestion()` function removed (lines 294-299)
- [ ] `/zoe/ask` endpoint removed (lines 885-999)
- [ ] `ZoeAskRequest` and `ZoeAskResponse` models kept (still used by `/zoe/ask-stream`)

**Verify:** `python -c "import main"` from backend dir → no ImportError

**Steps:**

- [ ] **Step 1: Remove dead imports (lines 26-27)**

  Remove these two lines:
  ```python
  from vector_search.contract_ingestion import ContractIngestion
  from vector_search.contract_search import ContractSearch
  ```

- [ ] **Step 2: Remove `contract_ingestion` global (line 285)**

  Current code:
  ```python
  # Initialize Zoe chatbot and contract ingestion (singletons)
  zoe_chatbot = None
  contract_ingestion = None
  ```

  Replace with:
  ```python
  # Initialize Zoe chatbot (singleton)
  zoe_chatbot = None
  ```

- [ ] **Step 3: Remove `get_contract_ingestion()` function (lines 294-299)**

  Delete:
  ```python
  def get_contract_ingestion():
      """Get or create contract ingestion instance"""
      global contract_ingestion
      if contract_ingestion is None:
          contract_ingestion = ContractIngestion()
      return contract_ingestion
  ```

- [ ] **Step 4: Remove `/zoe/ask` endpoint (lines 885-999)**

  Delete the entire `zoe_ask_question` function:
  ```python
  @app.post("/zoe/ask", response_model=ZoeAskResponse)
  async def zoe_ask_question(request: ZoeAskRequest):
      ...entire function through line 999...
  ```

  The `/zoe/ask-stream` endpoint at line 1002 remains.

- [ ] **Step 5: Commit**

  ```bash
  git add src/backend/main.py
  git commit -m "chore: remove dead Pinecone imports, factory function, and /zoe/ask endpoint"
  ```

---

### Task 3: Clean up contract_chatbot.py — remove search engine and non-streaming methods

**Goal:** Remove `ContractSearch` import, `self.search_engine` initialization, and the three non-streaming ask methods that used vector search.

**Files:**
- Modify: `src/backend/vector_search/contract_chatbot.py`

**Acceptance Criteria:**
- [ ] `from vector_search.contract_search import ContractSearch` import removed (line 31)
- [ ] `self.search_engine = ContractSearch()` removed from `__init__` (line 581)
- [ ] `ask_without_project()` method removed (starts line 3161)
- [ ] `ask_multiple_contracts()` method removed (starts line 3347)
- [ ] `ask_project()` method removed (starts line 3311)
- [ ] `ask_stream()` and `_generate_answer_stream_full_doc()` remain unchanged

**Verify:** `python -c "from vector_search.contract_chatbot import ContractChatbot; print('OK')"` from backend dir → prints "OK"

**Steps:**

- [ ] **Step 1: Remove ContractSearch import (line 31)**

  Delete:
  ```python
  from vector_search.contract_search import ContractSearch
  ```

- [ ] **Step 2: Remove search_engine init from `__init__` (line 581)**

  Current:
  ```python
      self.search_engine = ContractSearch()
  ```

  Delete this line entirely.

- [ ] **Step 3: Remove `ask_without_project()` method**

  Find the method starting at line 3161 (`def ask_without_project(self,`) and delete the entire method through to the next `def` at the same indentation level.

- [ ] **Step 4: Remove `ask_project()` method**

  Find the method starting at line 3311 (`def ask_project(self,`) and delete the entire method.

- [ ] **Step 5: Remove `ask_multiple_contracts()` method**

  Find the method starting at line 3347 (`def ask_multiple_contracts(self,`) and delete the entire method.

  Note: These methods internally reference `self.search_engine.smart_search()` (line 3106 area) and `self.search_engine.search_multiple_contracts()` (line 3505 area). Also check if there are helper methods like `smart_ask()` that are ONLY called by these dead methods — if so, remove those too.

- [ ] **Step 6: Verify `ask_stream()` still intact**

  Read the `ask_stream` method and confirm the guards (missing markdown, 370k limit) and full-doc path are untouched.

- [ ] **Step 7: Commit**

  ```bash
  git add src/backend/vector_search/contract_chatbot.py
  git commit -m "chore: remove non-streaming ask methods and ContractSearch dependency"
  ```

---

### Task 4: Clean up oneclick/helpers.py — remove find_chunks_with_text

**Goal:** Remove the `find_chunks_with_text()` function and its Pinecone import from the helpers module. Keep all other functions (they're actively used).

**Files:**
- Modify: `src/backend/oneclick/helpers.py`

**Acceptance Criteria:**
- [ ] `find_chunks_with_text()` function removed (lines 198-258)
- [ ] Pinecone import removed (line 9: `from pinecone import Pinecone`)
- [ ] Other functions in the file (`normalize_title`, `find_matching_song`, `simplify_role`, `normalize_name`) remain untouched

**Verify:** `python -c "from oneclick.helpers import normalize_title, find_matching_song, simplify_role, normalize_name; print('OK')"` → "OK"

**Steps:**

- [ ] **Step 1: Remove Pinecone import (line 9)**

  Delete:
  ```python
  from pinecone import Pinecone
  ```

- [ ] **Step 2: Remove `find_chunks_with_text()` function (lines 198-258)**

  Delete the entire function from its definition through the end.

- [ ] **Step 3: Commit**

  ```bash
  git add src/backend/oneclick/helpers.py
  git commit -m "chore: remove find_chunks_with_text and Pinecone import from helpers"
  ```

---

### Task 5: Clean up dependencies and env vars

**Goal:** Remove Pinecone pip package and langchain-text-splitters (only used by deleted contract_ingestion.py). Remove Pinecone env var documentation.

**Files:**
- Modify: `src/backend/requirements.txt`
- Modify: `.env.example`

**Acceptance Criteria:**
- [ ] `pinecone==8.1.0` removed from requirements.txt
- [ ] `langchain-text-splitters==1.1.1` removed from requirements.txt (only used by contract_ingestion.py)
- [ ] Pinecone env vars removed from `.env.example`
- [ ] `langchain`, `langchain-community`, `langchain-openai` checked — remove if not used elsewhere

**Verify:** `grep -i pinecone src/backend/requirements.txt` → no output. `grep -i pinecone .env.example` → no output.

**Steps:**

- [ ] **Step 1: Check if langchain packages are used elsewhere**

  Search for `langchain` imports in all Python files EXCEPT the deleted ones. If no remaining code imports them, remove all langchain packages.

- [ ] **Step 2: Edit requirements.txt**

  Remove `pinecone==8.1.0` (line 9).
  Remove `langchain-text-splitters==1.1.1` (line 13) — only used by deleted contract_ingestion.py.
  Check and remove other langchain packages if unused.

- [ ] **Step 3: Edit .env.example**

  Remove the Pinecone section:
  ```
  # Pinecone Configuration (for backend)
  PINECONE_API_KEY=your-pinecone-api-key-here
  ```

- [ ] **Step 4: Note about .env**

  The actual `.env` file still has `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`. These can be left (they're harmless unused env vars) or removed manually by the developer. Do NOT commit `.env` changes.

- [ ] **Step 5: Commit**

  ```bash
  git add src/backend/requirements.txt .env.example
  git commit -m "chore: remove pinecone and langchain-text-splitters from dependencies"
  ```

---

## Verification Checklist

After all tasks:

1. **No import errors:** `cd src/backend && python -c "from main import app; print('OK')"` → OK
2. **No Pinecone references in active code:** `grep -r "pinecone\|Pinecone\|PINECONE" src/backend/ --include="*.py"` → only hits in `.env` (not committed) or comments
3. **Streaming Q&A works:** `/zoe/ask-stream` endpoint still functions (table linearization, 370k limit, missing-markdown guard all intact)
4. **Upload works:** `/contracts/upload` converts PDF to markdown and stores directly
5. **Delete works:** `/contracts/{id}` deletes from Storage and DB only

## What's Left After Cleanup

**Active Pinecone-free code paths:**
- `/contracts/upload` → `pdf_to_markdown()` → store markdown in DB
- `/contracts/upload-multiple` → delegates to above
- `/zoe/ask-stream` → `ContractChatbot.ask_stream()` → full-doc path only
- `/contracts/{id}` DELETE → Storage + DB deletion only

**Removed:**
- `contract_ingestion.py` (422 lines)
- `contract_search.py` (~530 lines)
- `query_categorizer.py` (329 lines)
- `search_chunks.py` (50 lines)
- `/zoe/ask` endpoint (~115 lines)
- `ask_project()`, `ask_multiple_contracts()`, `ask_without_project()` methods
- `find_chunks_with_text()` function
- `pinecone` and `langchain-text-splitters` pip packages
