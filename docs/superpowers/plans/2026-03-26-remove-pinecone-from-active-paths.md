# Remove Pinecone from Active Code Paths — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Pinecone from upload, Q&A, and delete flows — always use full-doc context. If selected contracts exceed 370k chars, tell the user to deselect. Keep Pinecone code files in the repo as dead code.

**Architecture:** The upload endpoint will call `pdf_to_markdown()` directly instead of routing through the ingestion pipeline. The Q&A streaming endpoint will enforce a 370k char hard limit and remove the vector search fallback branch entirely. The delete endpoint will skip Pinecone vector cleanup.

**Tech Stack:** Python/FastAPI (backend), React/TypeScript (frontend), Supabase (storage + DB), pymupdf4llm (PDF conversion)

---

## File Structure

| File | Changes |
|------|---------|
| `src/backend/main.py` | Upload: replace ingestion with direct markdown storage (L713-740). Delete: remove Pinecone call (L867-872). |
| `src/backend/vector_search/contract_chatbot.py` | Replace 400k threshold with 370k hard limit, add missing-markdown guard, remove vector search else branch (L3999-4078). |
| `src/components/ContractUploadModal.tsx` | Fix success message condition for `total_chunks=0` (L302). |
| **Kept as dead code (no changes):** | `contract_ingestion.py`, `contract_search.py`, `query_categorizer.py` |

---

### Task 1: Upload — Replace Pinecone ingestion with direct markdown storage

**Goal:** Upload endpoint converts PDF to markdown and stores it directly, skipping Pinecone chunking/embedding.

**Files:**
- Modify: `src/backend/main.py:713-740`

**Acceptance Criteria:**
- [ ] Upload calls `pdf_to_markdown()` directly instead of `ingest_contract()`
- [ ] Markdown stored in `contract_markdown` column
- [ ] Returns `total_chunks=0`
- [ ] No Pinecone calls during upload

**Verify:** Upload a PDF contract via the UI or API → confirm `contract_markdown` populated in Supabase `project_files` table, response has `total_chunks: 0`

**Steps:**

- [ ] **Step 1: Replace the ingestion block (lines 713-740)**

  Replace the current try block contents inside the `with tempfile` block at `src/backend/main.py:713-740`. The `from vector_search.helpers import pdf_to_markdown` import already exists at line 817 in the lazy migration endpoint, but we need it here too.

  **Current code (lines 713-740):**
  ```python
  try:
      # 4. Process and ingest to Pinecone
      ingestion = get_contract_ingestion()
      stats = ingestion.ingest_contract(
          pdf_path=tmp_path,
          user_id=user_id,
          project_id=project_id,
          project_name=project_name,
          contract_id=contract_id,
          contract_filename=file.filename
      )

      # Store full markdown text for full-document context
      if stats.get("markdown_text"):
          try:
              get_supabase_client().table("project_files").update(
                  {"contract_markdown": stats["markdown_text"]}
              ).eq("id", contract_id).execute()
          except Exception as md_err:
              print(f"Warning: Failed to store contract markdown: {md_err}")

      return ContractUploadResponse(
          status="success",
          contract_id=contract_id,
          contract_filename=file.filename,
          total_chunks=stats["total_chunks"],
          message=f"Contract uploaded and processed successfully. {stats['total_chunks']} chunks created."
      )
  ```

  **New code:**
  ```python
  try:
      # 4. Convert PDF to markdown and store (skip Pinecone)
      from vector_search.helpers import pdf_to_markdown
      markdown_text = pdf_to_markdown(tmp_path)

      if not markdown_text.strip():
          raise ValueError("No text content found in PDF")

      try:
          get_supabase_client().table("project_files").update(
              {"contract_markdown": markdown_text}
          ).eq("id", contract_id).execute()
      except Exception as md_err:
          print(f"Warning: Failed to store contract markdown: {md_err}")

      return ContractUploadResponse(
          status="success",
          contract_id=contract_id,
          contract_filename=file.filename,
          total_chunks=0,
          message="Contract uploaded successfully."
      )
  ```

- [ ] **Step 2: Verify the upload-multiple endpoint needs no changes**

  The `/contracts/upload-multiple` endpoint (line 751) delegates to `upload_contract()` in a loop, so it inherits the change automatically. Confirm `result.total_chunks` at line 778 still works (it will be 0).

- [ ] **Step 3: Commit**

  ```bash
  git add src/backend/main.py
  git commit -m "feat: skip Pinecone ingestion on upload, store markdown directly"
  ```

---

### Task 2: Delete — Remove Pinecone cleanup

**Goal:** Delete endpoint skips Pinecone vector deletion, only removes from Supabase Storage and DB.

**Files:**
- Modify: `src/backend/main.py:867-872`

**Acceptance Criteria:**
- [ ] No `get_contract_ingestion()` or `delete_contract()` calls in delete endpoint
- [ ] Supabase Storage and DB deletion unchanged
- [ ] Delete succeeds without Pinecone connection

**Verify:** Delete a contract via the UI → success response, no Pinecone errors in logs

**Steps:**

- [ ] **Step 1: Remove the Pinecone deletion block (lines 867-872)**

  **Current code (lines 867-872):**
  ```python
        # 2. Delete from Pinecone
        ingestion = get_contract_ingestion()
        delete_result = ingestion.delete_contract(user_id=user_id, contract_id=contract_id)

        if delete_result["status"] != "success":
            raise HTTPException(status_code=500, detail=f"Failed to delete vectors: {delete_result.get('error')}")
  ```

  **Replace with:**
  ```python
        # 2. (Pinecone deletion removed — vectors are no longer created)
  ```

  The remaining steps (3. Delete from Supabase Storage at line 874, 4. Delete from Database at line 882) stay exactly as they are.

- [ ] **Step 2: Commit**

  ```bash
  git add src/backend/main.py
  git commit -m "feat: skip Pinecone deletion on contract delete"
  ```

---

### Task 3: Q&A — Remove vector search fallback, add 370k limit and missing-markdown guard

**Goal:** Q&A always uses full-doc path. If contracts exceed 370k chars, return a limit message. If markdown is missing, tell user to re-upload. Remove the entire vector search else branch.

**Files:**
- Modify: `src/backend/vector_search/contract_chatbot.py:3999-4078`

**Acceptance Criteria:**
- [ ] 370k char limit with user-friendly SSE "complete" message
- [ ] Missing markdown guard with re-upload SSE "complete" message
- [ ] Entire `else` branch (vector search, lines 4038-4078) removed
- [ ] Full-doc path (table linearization, context building, streaming) preserved unchanged
- [ ] Threshold log message updated from 400k to 370k

**Verify:**
- Normal Q&A with contracts < 370k chars → correct answers using full-doc path (check logs for `[Stream] Using full document context`)
- Select contracts totaling > 370k chars → "context limit" message
- Contract with no markdown in DB → "re-upload" message

**Steps:**

- [ ] **Step 1: Replace the branching logic (lines 3999-4078)**

  **Current code (lines 3999-4078):**
  ```python
        # Check if full document context is available and within token limits
        use_full_doc = False
        if contract_markdowns:
            total_chars = sum(len(md) for md in contract_markdowns.values())
            # ~100k tokens ≈ 400k chars — leave room for system prompt + history
            if total_chars < 400_000:
                use_full_doc = True
                logger.info(f"[Stream] Using full document context ({total_chars} chars, {len(contract_markdowns)} contract(s))")
            else:
                logger.info(f"[Stream] Full document too large ({total_chars} chars), falling back to chunk retrieval")

        # Use larger model for multi-contract comparisons (2+ contracts)
        multi_contract_model = DEFAULT_LLM_MODEL_LARGE if (contract_ids and len(contract_ids) >= 2) else None
        if multi_contract_model:
            logger.info(f"[Stream] Multi-contract detected ({len(contract_ids)} contracts) — using {multi_contract_model}")

        if use_full_doc:
            # Linearize tables for clearer LLM comprehension
            processed_markdowns = {}
            for cid, md in contract_markdowns.items():
                has_tables, table_blocks, text_without_tables = detect_and_extract_tables(md)
                if has_tables:
                    processed = text_without_tables
                    for tb in table_blocks:
                        linearized = linearize_table(tb.raw_text, tb.preceding_context)
                        processed = processed.replace("[TABLE_REMOVED]", linearized, 1)
                    processed_markdowns[cid] = processed
                else:
                    processed_markdowns[cid] = md

            # Build full document context with filenames as labels
            full_doc_context = ""
            for cid, md in processed_markdowns.items():
                label = contract_names.get(cid, cid) if contract_names else cid
                full_doc_context += f"\n\n=== CONTRACT: {label} ===\n{md}\n"
            full_doc_context = full_doc_context.strip()

            # Stream answer using full document
            yield from self._generate_answer_stream_full_doc(query, full_doc_context, session_id, model_override=multi_contract_model)
        else:
            search_query = query
            retrieval_reason = route_decision.retrieval_reason or "general_retrieval"
            if retrieval_reason.startswith("missing_"):
                search_query = self._get_targeted_query(retrieval_reason, query)

            # Perform search based on contract selection
            if contract_ids and len(contract_ids) > 0:
                search_results = self.search_engine.search_multiple_contracts(
                    query=search_query, user_id=user_id,
                    project_id=project_id, contract_ids=contract_ids,
                    top_k=top_k or DEFAULT_TOP_K
                )
            else:
                search_results = self.search_engine.smart_search(
                    query=search_query, user_id=user_id,
                    project_id=project_id, contract_id=contract_id,
                    top_k=top_k
                )

            if not search_results["matches"]:
                # ── Fallback: try conversation history before giving up ──
                if not contracts_changed:
                    history_answer = self._try_answer_from_history(query, session_id, context)
                    if history_answer:
                        logger.info("[Stream][Fallback] No contract matches, answered from history")
                        yield self._sse_event("complete", history_answer)
                        return
                # Neither contracts nor history could answer
                no_result_answer = self._no_result_message(contract_ids)
                self._add_to_memory(session_id, "assistant", no_result_answer)
                yield self._sse_event("complete", {
                    "query": query, "answer": no_result_answer, "confidence": "low",
                    "sources": [], "search_results_count": 0, "session_id": session_id
                })
                return

            formatted_context = self._format_context(search_results)

            # Stream the answer generation
            yield from self._generate_answer_stream(query, formatted_context, search_results, session_id)
  ```

  **New code:**
  ```python
        # ── Guard: missing markdown ──
        if not contract_markdowns:
            no_md_msg = "I couldn't load the contract text. Please try re-uploading the contract."
            self._add_to_memory(session_id, "assistant", no_md_msg)
            yield self._sse_event("complete", {
                "query": query, "answer": no_md_msg, "confidence": "low",
                "sources": [], "search_results_count": 0, "session_id": session_id,
            })
            return

        # ── Guard: combined size exceeds context limit ──
        total_chars = sum(len(md) for md in contract_markdowns.values())
        if total_chars >= 370_000:
            limit_msg = (
                "The combined size of your selected contracts exceeds the context limit. "
                "Please deselect one or more contracts and try again."
            )
            logger.info(f"[Stream] Contract text too large ({total_chars} chars >= 370k limit)")
            self._add_to_memory(session_id, "assistant", limit_msg)
            yield self._sse_event("complete", {
                "query": query, "answer": limit_msg, "confidence": "low",
                "sources": [], "search_results_count": 0, "session_id": session_id,
            })
            return

        logger.info(f"[Stream] Using full document context ({total_chars} chars, {len(contract_markdowns)} contract(s))")

        # Use larger model for multi-contract comparisons (2+ contracts)
        multi_contract_model = DEFAULT_LLM_MODEL_LARGE if (contract_ids and len(contract_ids) >= 2) else None
        if multi_contract_model:
            logger.info(f"[Stream] Multi-contract detected ({len(contract_ids)} contracts) — using {multi_contract_model}")

        # Linearize tables for clearer LLM comprehension
        processed_markdowns = {}
        for cid, md in contract_markdowns.items():
            has_tables, table_blocks, text_without_tables = detect_and_extract_tables(md)
            if has_tables:
                processed = text_without_tables
                for tb in table_blocks:
                    linearized = linearize_table(tb.raw_text, tb.preceding_context)
                    processed = processed.replace("[TABLE_REMOVED]", linearized, 1)
                processed_markdowns[cid] = processed
            else:
                processed_markdowns[cid] = md

        # Build full document context with filenames as labels
        full_doc_context = ""
        for cid, md in processed_markdowns.items():
            label = contract_names.get(cid, cid) if contract_names else cid
            full_doc_context += f"\n\n=== CONTRACT: {label} ===\n{md}\n"
        full_doc_context = full_doc_context.strip()

        # Stream answer using full document
        yield from self._generate_answer_stream_full_doc(query, full_doc_context, session_id, model_override=multi_contract_model)
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add src/backend/vector_search/contract_chatbot.py
  git commit -m "feat: enforce 370k char limit, remove vector search fallback in Q&A"
  ```

---

### Task 4: Frontend — Fix upload success message when total_chunks=0

**Goal:** The success message in `ContractUploadModal` renders even when `total_chunks` is 0 (falsy in JavaScript).

**Files:**
- Modify: `src/components/ContractUploadModal.tsx:302`

**Acceptance Criteria:**
- [ ] Success message shows when `status === "success"` regardless of `total_chunks` value

**Verify:** Upload a contract via UI → "Uploaded successfully" message appears

**Steps:**

- [ ] **Step 1: Fix the conditional render (line 302)**

  **Current code:**
  ```tsx
  {result.status === "success" && result.total_chunks && (
    <p className="text-xs text-green-600 mt-1">
      ✓ Uploaded successfully
    </p>
  )}
  ```

  **New code:**
  ```tsx
  {result.status === "success" && (
    <p className="text-xs text-green-600 mt-1">
      ✓ Uploaded successfully
    </p>
  )}
  ```

  The `total_chunks` check was only needed when success depended on Pinecone processing. Now upload success means the markdown was stored.

- [ ] **Step 2: Commit**

  ```bash
  git add src/components/ContractUploadModal.tsx
  git commit -m "fix: show upload success message when total_chunks is 0"
  ```

---

## Verification Checklist

After all tasks are complete, verify end-to-end:

1. **Upload a contract** → fast (no chunking/embedding), `contract_markdown` populated in DB, response shows `total_chunks: 0`, UI shows success
2. **Ask Zoe a question** about an uploaded contract → correct answer, logs show `[Stream] Using full document context`
3. **Select many contracts totaling > 370k chars** → "context limit" message returned
4. **Delete a contract** → succeeds without Pinecone errors
5. **Existing contracts** (already in DB with markdown) → continue working normally

## Dead Code (kept in repo, not modified)

- `src/backend/vector_search/contract_ingestion.py` — `ingest_contract()`, `delete_contract()` no longer called
- `src/backend/vector_search/contract_search.py` — `search_multiple_contracts()`, `smart_search()` no longer called from streaming path
- `src/backend/vector_search/query_categorizer.py` — no longer called
- Non-streaming `/zoe/ask` endpoint methods (`ask_project`, `ask_multiple_contracts`, `ask_without_project`) — frontend only uses `/zoe/ask-stream`
