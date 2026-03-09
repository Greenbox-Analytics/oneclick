"""
Music Contract Parser - Pinecone-based extraction
Extracts structured data (parties, works, royalty shares) from contracts already stored in Pinecone
"""

import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from pathlib import Path
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add backend directory to path to allow imports from vector_search
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from vector_search.query_categorizer import categorize_query, build_metadata_filter

load_dotenv()

# Initialize clients
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not found in .env file")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")
    

pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None
)

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5-mini")
INDEX_NAME = PINECONE_INDEX_NAME
STREAMING_EQUIVALENT_TERMS = [
    "streaming",
    "digital",
    "master",
    "master royalties",
    "producer",
    "producer royalties",
    "master points",
    "royalty participation",
    "revenue participation",
    "net master revenue",
    "sound recording royalty splits",
]


import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Party:
    name: str
    role: str


@dataclass
class Work:
    title: str
    work_type: str = "song"


@dataclass
class RoyaltyShare:
    party_name: str
    royalty_type: str
    percentage: float
    terms: Optional[str] = None


@dataclass
class ContractData:
    parties: List[Party]
    works: List[Work]
    royalty_shares: List[RoyaltyShare]
    contract_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Parser Class
# ---------------------------------------------------------------------------

class MusicContractParser:
    """Extract structured data from contracts stored in Pinecone"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise ValueError("Missing or invalid OpenAI API key.")

        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None
        )
        self.index = pc.Index(INDEX_NAME)

    def parse_contract(self, path: str, user_id: str = None, contract_id: str = None, use_parallel: bool = True) -> ContractData:
        """
        Parse contract by querying Pinecone for relevant sections.
        
        Note: This method signature is kept for compatibility with royalty_calculator.py
        In practice, you should provide user_id and contract_id to query Pinecone.
        
        Args:
            path: Contract file path (not used, kept for compatibility)
            user_id: User ID for Pinecone namespace filtering
            contract_id: Contract ID for filtering specific contract
            use_parallel: If True, runs extractions in parallel (default: True)
            
        Returns:
            ContractData with extracted information
        """
        if not user_id or not contract_id:
            # If path is provided, try to assume it's legacy mode, but we really need user_id and contract_id for Pinecone
            # For now, raise the error as before, but clarify the message
            raise ValueError(
                "user_id and contract_id are required to query Pinecone. "
                "The contract must already be uploaded and indexed."
            )

        start_time = time.time()
        logger.info(f"📄 Extracting contract data from Pinecone")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Contract ID: {contract_id}")
        logger.info(f"   Mode: {'Parallel' if use_parallel else 'Sequential'}")

        if use_parallel:
            # 1. Extract parties and works FIRST in parallel
            t0 = time.time()
            parties = []
            works = []
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_task = {
                    executor.submit(self._extract_parties, user_id, contract_id): "parties",
                    executor.submit(self._extract_works, user_id, contract_id): "works"
                }
                
                for future in as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        result = future.result()
                        if task_name == "parties":
                            parties = result
                        elif task_name == "works":
                            works = result
                    except Exception as e:
                        logger.error(f"   ⚠️ Error in {task_name} extraction: {e}")
            
            logger.info(f"   ⏱️  Parties & Works extraction took: {time.time() - t0:.2f}s")

            # 2. Extract royalties using parties context
            t1 = time.time()
            royalty_shares = self._extract_royalties(user_id, contract_id, parties)
            logger.info(f"   ⏱️  Royalties extraction took: {time.time() - t1:.2f}s")
            
            summary = ""
        else:
            # Sequential execution
            t0 = time.time()
            parties = self._extract_parties(user_id, contract_id)
            logger.info(f"   ⏱️  Parties extraction took: {time.time() - t0:.2f}s")

            t0 = time.time()
            works = self._extract_works(user_id, contract_id)
            logger.info(f"   ⏱️  Works extraction took: {time.time() - t0:.2f}s")

            t0 = time.time()
            royalty_shares = self._extract_royalties(user_id, contract_id, parties)
            logger.info(f"   ⏱️  Royalties extraction took: {time.time() - t0:.2f}s")
            
            summary = ""

        # Simplify party roles
        from oneclick.helpers import normalize_name, simplify_role
        for party in parties:
            party.role = simplify_role(party.role)
        
        # Reconcile royalty share names with extracted parties
        
        for share in royalty_shares:
            share_name_norm = normalize_name(share.party_name)
            best_match = None
            
            # 1. Exact normalized match
            for party in parties:
                if normalize_name(party.name) == share_name_norm:
                    best_match = party
                    break
            
            # 2. Partial match (if no exact match)
            if not best_match:
                for party in parties:
                    party_norm = normalize_name(party.name)
                    # Check if one name contains the other (e.g. "Kenji" in "Kenji Niyokindi")
                    if share_name_norm in party_norm or party_norm in share_name_norm:
                        best_match = party
                        break
            
            # Update share name if a better match is found from the parties list
            if best_match:
                logger.info(f"   🔄 Reconciling name: '{share.party_name}' -> '{best_match.name}'")
                share.party_name = best_match.name
                # If we want to attach the role, we might need to modify the RoyaltyShare dataclass or handle it downstream.
                # For now, ensuring the name matches allows the frontend/calculator to look up the role from the parties list.

        total_time = time.time() - start_time
        logger.info(f"✅ Extraction complete in {total_time:.2f}s")
        logger.info(f"   → {len(parties)} parties, {len(works)} works, {len(royalty_shares)} shares")

        return ContractData(
            parties=parties,
            works=works,
            royalty_shares=royalty_shares,
            contract_summary=summary
        )

    def _query_pinecone(self, query: str, user_id: str, contract_id: str, top_k: int = 5, use_fast_categorization: bool = True) -> str:
        """
        Query Pinecone with separate paths for prose and table chunks.

        Prose path: fetches 10 chunks directly by cosine similarity (no reranker).
        Table path: fetches 20 chunks directly by cosine similarity, no reranker.

        Both paths share the same query embedding and category filter.
        Results are concatenated (prose first, then tables).
        """
        PROSE_TOP_K = 10
        TABLE_TOP_K = 20

        logger.info(f"\n   🧠 Categorizing query: '{query}'")

        t_cat = time.time()
        categorization = categorize_query(query, self.openai_client, use_llm=not use_fast_categorization)
        logger.info(f"      → Categories: {categorization.get('categories')}")
        logger.info(f"      → Confidence: {categorization.get('confidence')}")
        logger.info(f"      → Method: {'Keyword-based (fast)' if use_fast_categorization else 'LLM-based'}")
        logger.info(f"      ⏱️  Categorization took: {time.time() - t_cat:.2f}s")

        base_filter = build_metadata_filter(
            categories=categorization.get('categories', []),
            is_general=categorization.get('is_general', False),
            user_id=None,
            contract_id=contract_id
        )

        t_embed = time.time()
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = response.data[0].embedding
        logger.info(f"      ⏱️  Embedding creation took: {time.time() - t_embed:.2f}s")

        namespace = f"{user_id}-namespace"

        # --- Query 1: Prose (direct cosine similarity, no reranker) ---
        prose_filter = dict(base_filter) if base_filter else {}
        prose_filter["is_table"] = {"$ne": True}

        logger.info(f"      → [Prose] filter={prose_filter}, top_k={PROSE_TOP_K}")
        t_query = time.time()
        prose_chunks = self._fetch_chunks(namespace, query_embedding, prose_filter, PROSE_TOP_K, contract_id, label="Prose")
        logger.info(f"      ⏱️  Prose query took: {time.time() - t_query:.2f}s")
        logger.info(f"      → [Prose] {len(prose_chunks)} chunks (no reranking)")

        # --- Query 2: Tables (direct cosine similarity, no reranker) ---
        table_filter = dict(base_filter) if base_filter else {}
        table_filter["is_table"] = {"$eq": True}

        logger.info(f"      → [Table] filter={table_filter}, top_k={TABLE_TOP_K}")
        t_query = time.time()
        table_chunks = self._fetch_chunks(namespace, query_embedding, table_filter, TABLE_TOP_K, contract_id, label="Table")
        logger.info(f"      ⏱️  Table query took: {time.time() - t_query:.2f}s")
        logger.info(f"      → [Table] {len(table_chunks)} chunks (no reranking)")

        all_chunks = prose_chunks + table_chunks
        logger.info(f"      → Combined context: {len(prose_chunks)} prose + {len(table_chunks)} table = {len(all_chunks)} total")
        return "\n\n".join(all_chunks)

    def _fetch_chunks(self, namespace: str, query_embedding: List[float],
                      filter_dict: dict, top_k: int, contract_id: str,
                      label: str = "") -> List[str]:
        """Run a single Pinecone query and return chunk texts. Falls back to
        a broad contract_id-only filter if the category filter returns nothing."""
        try:
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
        except Exception as e:
            logger.warning(f"      ⚠️ [{label}] Filtered search error: {e}. Falling back.")
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                filter={"contract_id": contract_id},
                include_metadata=True
            )

        chunks = []
        if hasattr(results, 'matches') and results.matches:
            for i, match in enumerate(results.matches):
                section = match.metadata.get("section_heading", "N/A")
                category = match.metadata.get("section_category", "N/A")
                logger.info(f"      [{label}] Chunk {i+1}: Score={match.score:.4f} | Section='{section}' | Category='{category}'")
                text = match.metadata.get("chunk_text", "")
                if text:
                    chunks.append(text)
        else:
            logger.warning(f"      ⚠️ [{label}] No matches. Retrying without category filter.")
            fallback_filter = {"contract_id": contract_id}
            if "is_table" in filter_dict:
                fallback_filter["is_table"] = filter_dict["is_table"]
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                filter=fallback_filter,
                include_metadata=True
            )
            if hasattr(results, 'matches') and results.matches:
                for i, match in enumerate(results.matches):
                    section = match.metadata.get("section_heading", "N/A")
                    logger.info(f"      [{label}] Fallback {i+1}: Score={match.score:.4f} | Section='{section}'")
                    text = match.metadata.get("chunk_text", "")
                    if text:
                        chunks.append(text)
        return chunks

    def _rerank_results(self, query: str, chunks: List[str], top_k: int) -> List[str]:
        """
        Rerank chunks by relevance using a fast LLM call.
        Returns only the top_k most relevant chunks.
        """
        logger.info(f"   🔄 Reranking {len(chunks)} chunks → keeping top {top_k}...")
        t_rerank = time.time()
        
        indexed_chunks = []
        for i, chunk in enumerate(chunks):
            indexed_chunks.append(f"[{i}]: {chunk}")
        
        prompt = (
            f"Query: \"{query}\"\n\n"
            f"Chunks:\n" + "\n\n".join(indexed_chunks) + "\n\n"
            f"Return a JSON list of chunk indices ordered by relevance to the query (most relevant first). "
            f"Only return the JSON list, nothing else."
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100
            )
            content = response.choices[0].message.content.strip()
            if "```" in content:
                content = content.replace("```json", "").replace("```", "").strip()
            
            indices = json.loads(content)
            reranked = [chunks[i] for i in indices if isinstance(i, int) and 0 <= i < len(chunks)]
            logger.info(f"   ✓ Reranking complete in {time.time() - t_rerank:.2f}s (order: {indices[:top_k]})")
            return reranked[:top_k]
        except Exception as e:
            logger.warning(f"   ⚠️ Reranking failed ({e}). Returning original top {top_k}.")
            return chunks[:top_k]

    def _ask_llm(self, context: str, question: str, template: str) -> str:
        """
        Send context and question to LLM.
        
        Args:
            context: Retrieved context from Pinecone
            question: Question to ask
            template: Prompt template
            
        Returns:
            LLM response
        """
        prompt = template.format(context=context, question=question)
        
        t_llm = time.time()
        response = self.openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        logger.info(f"      ⏱️  LLM response took: {time.time() - t_llm:.2f}s")
        
        return response.choices[0].message.content.strip()

    def _extract_parties(self, user_id: str, contract_id: str) -> List[Party]:
        """Extract all parties/contributors from contract"""
        question_definitions = (
            "Identify all specific legal entities and individuals named as parties to this agreement. "
            "Prioritize sections such as recitals/opening paragraph ('by and between'), Definitions, "
            "and signature blocks. Capture aliases like p/k/a, professionally known as, or d/b/a. "
            "For each named party, extract every defined role/term assigned to that party "
            "(e.g., Artist, Producer, Company, Licensor, Licensee, Publisher, Distributor)."
        )
        question_role_mapping = (
            "Find clauses where parties are referenced by role labels only (e.g., 'Producer', 'Artist', "
            "'Songwriter', 'Licensor'). Map each such role reference back to the previously defined named party. "
            "Return only mappings that are explicitly supported by contract text."
        )
        template = """You are a contract analyst. Use the context below to answer.

Context:
{context}

Question:
{question}

Instructions:
1. Look for the introductory paragraph or "Parties" section where the agreement is made "by and between".
2. Extract the exact defined term used for each party (e.g., "hereinafter referred to as 'Artist'").
3. Track where named parties are later referenced only by role labels and merge those roles into the same party record.
4. Include aliases in the name field when present (p/k/a, d/b/a, professionally known as).
5. If a party has no specific defined role, use a generic description based on their function (e.g., "Label", "Distributor").
6. Ignore generic references like "third parties" or "licensees" unless a specific name is attached.
7. If one party has multiple roles, separate roles with semicolons.

List each party strictly in this format:
Name | Role

Answer:
"""
        context_a = self._query_pinecone(question_definitions, user_id, contract_id, top_k=5)
        context_b = self._query_pinecone(question_role_mapping, user_id, contract_id, top_k=5)
        context = context_a + "\n\n" + context_b
        question = question_definitions + " " + question_role_mapping
        result = self._ask_llm(context, question, template)
        
        parties = []
        from oneclick.helpers import normalize_name
        seen_party_names = set()
        for line in result.splitlines():
            if "|" in line:
                parts = line.split("|", 1)
                if len(parts) == 2:
                    name, role = [x.strip() for x in parts]
                    if name and role:
                        key = normalize_name(name)
                        if key in seen_party_names:
                            continue
                        seen_party_names.add(key)
                        parties.append(Party(name, role.lower()))
        
        logger.info(f"👥 Extracted {len(parties)} parties")
        for i, party in enumerate(parties):
            logger.info(f"   {i+1}. {party.name} ({party.role})")
        return parties

    def _extract_works(self, user_id: str, contract_id: str) -> List[Work]:
        """Extract all songs/works from contract"""
        question_primary = (
            "Identify all musical works, compositions, masters, recordings, tracks, or releases covered by this agreement. "
            "Search body clauses and schedules/exhibits/annexes/tables (track lists)."
        )
        question_variant_mapping = (
            "Group references that point to the same underlying title despite formatting differences, "
            "including parenthetical qualifiers like '(composition)', '(master recording)', quoted/unquoted forms, "
            "or minor spelling/punctuation differences."
        )
        template = """Context:
{context}

Question:
{question}

Instructions:
1. Return one line per observed work reference.
2. Provide a canonical title that removes qualifiers/formatting noise.
3. Keep observed variant exactly as shown in the contract.
4. Include a specific work type (composition, master recording, song, album, release).
5. Do not invent titles from generic placeholders.

List each as: Canonical Title | Observed Variant | Type
Answer:
"""
        context_a = self._query_pinecone(question_primary, user_id, contract_id, top_k=5)
        context_b = self._query_pinecone(question_variant_mapping, user_id, contract_id, top_k=5)
        context = context_a + "\n\n" + context_b
        question = question_primary + " " + question_variant_mapping
        result = self._ask_llm(context, question, template)
        
        works = []
        from oneclick.helpers import normalize_title
        seen_works = {}
        for line in result.splitlines():
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    canonical_title, observed_variant, typ = parts[:3]
                    title = canonical_title or observed_variant
                elif len(parts) == 2:
                    title, typ = parts
                else:
                    continue

                if title:
                    normalized = normalize_title(title)
                    work_type = typ.lower() or "song"
                    if normalized in seen_works:
                        existing = seen_works[normalized]
                        if existing.work_type in ("song", "work") and work_type not in ("song", "work"):
                            existing.work_type = work_type
                    else:
                        work_obj = Work(title, work_type)
                        seen_works[normalized] = work_obj
                        works.append(work_obj)
        
        logger.info(f"🎵 Extracted {len(works)} works")
        for i, work in enumerate(works):
            logger.info(f"   {i+1}. {work.title} ({work.work_type})")
        return works

    def _extract_royalties(self, user_id: str, contract_id: str, parties: List[Party] = None) -> List[RoyaltyShare]:
        """Extract royalty shares from contract"""
        
        # Format known parties for context
        parties_context = ""
        if parties:
            parties_list = [f"{p.name} ({p.role})" for p in parties]
            parties_context = "\nKNOWN PARTIES & ROLES:\n" + "\n".join(parties_list) + "\n"
            logger.info(f"   ℹ️  Providing known parties context to LLM: {len(parties)} parties")
            for p in parties:
                logger.info(f"      - {p.name} ({p.role})")

        streaming_terms_str = ", ".join([f"'{term}'" for term in STREAMING_EQUIVALENT_TERMS])
        question = (
            "What are the explicit royalty percentage splits defined in the contract? "
            f"Look for terms like {streaming_terms_str}. "
            "Prioritize schedules/exhibits/annexes/tables where economic terms are listed. "
            "Also scan body clauses for wording such as 'shall receive', 'entitled to', 'payable', 'points', "
            "'gross receipts', 'net receipts', and participation mechanics. "
            "Note that 'master' or 'producer royalties' typically encompass streaming revenue. "
            "Only list parties with a specific numeric percentage."
        )
        template = """Context:
{context}

{parties_context}

Question:
{question}

Instructions:
1. Identify the percentage split for each party.
2. If a royalty split refers to a generic role (e.g., 'Songwriter', 'Producer', 'Artist'), substitute it with the actual name from the KNOWN PARTIES list above.
3. If the role is ambiguous or no matching name is found, keep the role name.
4. Simplify the 'Royalty Type' to one of these standard terms if applicable: 'Streaming',Master', 'Publishing', 'Producer', 'Mixer', 'Remixer'. If it doesn't fit, use a short descriptive term (max 3 words).


List each as: Name | Royalty Type | Percentage | Terms
Answer:
"""
        context = self._query_pinecone(question, user_id, contract_id, top_k=5)
        # Inject parties_context into template before formatting with context/question
        full_template = template.replace("{parties_context}", parties_context)
        
        result = self._ask_llm(context, question, full_template)
        
        shares = []
        for line in result.splitlines():
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    name, typ, pct = parts[:3]
                    terms = parts[3] if len(parts) > 3 else None
                    try:
                        pct_val = float(pct.replace("%", "").strip())
                        shares.append(RoyaltyShare(name, typ.lower(), pct_val, terms))
                    except ValueError:
                        continue
        
        logger.info(f"💰 Extracted {len(shares)} royalty shares")
        for i, share in enumerate(shares):
            logger.info(f"   {i+1}. {share.party_name} | {share.royalty_type} | {share.percentage}%" + (f" | {share.terms}" if share.terms else ""))
        return shares

    def _extract_summary(self, user_id: str, contract_id: str) -> str:
        """Generate contract summary"""
        question = (
            "Provide a concise 2-3 paragraph summary of this music contract, "
            "including the main parties, terms, and royalty arrangements."
        )
        template = """Context:
{context}

Question:
{question}

Answer (2-3 paragraphs):
"""
        context = self._query_pinecone(question, user_id, contract_id, top_k=8)
        result = self._ask_llm(context, question, template)
        
        logger.info("🧾 Summary generated")
        return result
