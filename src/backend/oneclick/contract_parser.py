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

        # Reconcile royalty share names with extracted parties
        from oneclick.helpers import normalize_name
        
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
        Query Pinecone and return concatenated context.
        
        Args:
            query: Search query
            user_id: User ID for namespace
            contract_id: Contract ID for filtering
            top_k: Number of results to retrieve
            use_fast_categorization: If True, uses fast keyword matching instead of LLM (default: True)
            
        Returns:
            Concatenated text from top results
        """
        logger.info(f"\n   🧠 Categorizing query: '{query}'")
        
        # 1. Categorize the query (use fast keyword-based categorization by default)
        t_cat = time.time()
        categorization = categorize_query(query, self.openai_client, use_llm=not use_fast_categorization)
        logger.info(f"      → Categories: {categorization.get('categories')}")
        logger.info(f"      → Confidence: {categorization.get('confidence')}")
        logger.info(f"      → Method: {'Keyword-based (fast)' if use_fast_categorization else 'LLM-based'}")
        logger.info(f"      ⏱️  Categorization took: {time.time() - t_cat:.2f}s")

        # 2. Build metadata filter
        filter_dict = build_metadata_filter(
            categories=categorization.get('categories', []),
            is_general=categorization.get('is_general', False),
            user_id=None, # user_id is handled by namespace
            contract_id=contract_id
        )
        
        # Create query embedding
        t_embed = time.time()
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = response.data[0].embedding
        logger.info(f"      ⏱️  Embedding creation took: {time.time() - t_embed:.2f}s")

        # Query Pinecone
        namespace = f"{user_id}-namespace"
        
        logger.info(f"      → Using filter: {filter_dict}")
        
        t_query = time.time()
        try:
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
        except Exception as e:
            logger.warning(f"      ⚠️ Error with filtered search: {e}. Falling back to broad search.")
            # Fallback to simple contract_id filter if category search fails or returns nothing
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                filter={"contract_id": contract_id},
                include_metadata=True
            )
        logger.info(f"      ⏱️  Pinecone query took: {time.time() - t_query:.2f}s")

        # Concatenate results
        context_parts = []
        if hasattr(results, 'matches') and len(results.matches) > 0:
            logger.info(f"   🔍 Retrieving chunks for: '{query}'")
            for i, match in enumerate(results.matches):
                chunk_id = match.id
                section = match.metadata.get("section_heading", "N/A")
                category = match.metadata.get("section_category", "N/A")
                score = match.score
                logger.info(f"      Chunk {i+1}: ID={chunk_id} | Score={score:.4f} | Section='{section}' | Category='{category}'")
                
                text = match.metadata.get("chunk_text", "")
                if text:
                    context_parts.append(text)
        else:
            logger.warning("      ⚠️ No matches found with filters. Retrying without category filter.")
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                filter={"contract_id": contract_id},
                include_metadata=True
            )
            if hasattr(results, 'matches') and len(results.matches) > 0:
                 for i, match in enumerate(results.matches):
                    chunk_id = match.id
                    section = match.metadata.get("section_heading", "N/A")
                    score = match.score
                    logger.info(f"      Fallback Chunk {i+1}: ID={chunk_id} | Score={score:.4f} | Section='{section}'")
                    text = match.metadata.get("chunk_text", "")
                    if text:
                        context_parts.append(text)

        return "\n\n".join(context_parts)

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
        question = "Identify all specific legal entities and individuals named as parties to this agreement. For each party, extract their full legal name AND the specific role or defined term assigned to them in the contract (e.g., 'Artist', 'Producer', 'Company', 'Licensor'). Do NOT list generic placeholders like 'The Artist', 'The Producer', or 'The Songwriter' as the name."
        template = """You are a contract analyst. Use the context below to answer.

Context:
{context}

Question:
{question}

Instructions:
1. Look for the introductory paragraph or "Parties" section where the agreement is made "by and between".
2. Extract the exact defined term used for each party (e.g., "hereinafter referred to as 'Artist'").
3. If a party has no specific defined role, use a generic description based on their function (e.g., "Label", "Distributor").
4. Ignore generic references like "third parties" or "licensees" unless a specific name is attached.

List each party strictly in this format:
Name | Role

Answer:
"""
        context = self._query_pinecone(question, user_id, contract_id, top_k=5)
        result = self._ask_llm(context, question, template)
        
        parties = []
        for line in result.splitlines():
            if "|" in line:
                parts = line.split("|", 1)
                if len(parts) == 2:
                    name, role = [x.strip() for x in parts]
                    if name and role:
                        parties.append(Party(name, role.lower()))
        
        logger.info(f"👥 Extracted {len(parties)} parties")
        for i, party in enumerate(parties):
            logger.info(f"   {i+1}. {party.name} ({party.role})")
        return parties

    def _extract_works(self, user_id: str, contract_id: str) -> List[Work]:
        """Extract all songs/works from contract"""
        question = "What is the specific musical work, song, or master recording being licensed or transferred in this agreement?"
        template = """Context:
{context}

Question:
{question}

List each as: Title | Type (song, album, composition)
Answer:
"""
        context = self._query_pinecone(question, user_id, contract_id, top_k=5)
        result = self._ask_llm(context, question, template)
        
        works = []
        for line in result.splitlines():
            if "|" in line:
                parts = line.split("|", 1)
                if len(parts) == 2:
                    title, typ = [x.strip() for x in parts]
                    if title:
                        works.append(Work(title, typ.lower() or "song"))
        
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

        question = (
            "What are the explicit royalty percentage splits defined in the contract? "
            "Look for terms like 'master royalties', 'producer royalties', 'streaming', 'master points', "
            "'royalty participation', 'revenue participation', 'net master revenue', or 'sound recording royalty splits'. "
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
