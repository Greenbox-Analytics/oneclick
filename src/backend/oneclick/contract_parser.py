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

load_dotenv()

# Initialize clients
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

pc = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None
)

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-5-mini"
INDEX_NAME = "test-3-small-index"


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

    def parse_contract(self, path: str, user_id: str = None, contract_id: str = None) -> ContractData:
        """
        Parse contract by querying Pinecone for relevant sections.
        
        Note: This method signature is kept for compatibility with royalty_calculator.py
        In practice, you should provide user_id and contract_id to query Pinecone.
        
        Args:
            path: Contract file path (not used, kept for compatibility)
            user_id: User ID for Pinecone namespace filtering
            contract_id: Contract ID for filtering specific contract
            
        Returns:
            ContractData with extracted information
        """
        if not user_id or not contract_id:
            raise ValueError(
                "user_id and contract_id are required to query Pinecone. "
                "The contract must already be uploaded and indexed."
            )

        print(f"📄 Extracting contract data from Pinecone")
        print(f"   User ID: {user_id}")
        print(f"   Contract ID: {contract_id}")

        # Extract structured elements
        parties = self._extract_parties(user_id, contract_id)
        works = self._extract_works(user_id, contract_id)
        royalty_shares = self._extract_royalties(user_id, contract_id)
        summary = self._extract_summary(user_id, contract_id)

        print(f"✅ Extraction complete")
        print(f"   → {len(parties)} parties, {len(works)} works, {len(royalty_shares)} shares")

        return ContractData(
            parties=parties,
            works=works,
            royalty_shares=royalty_shares,
            contract_summary=summary
        )

    def _query_pinecone(self, query: str, user_id: str, contract_id: str, top_k: int = 5) -> str:
        """
        Query Pinecone and return concatenated context.
        
        Args:
            query: Search query
            user_id: User ID for namespace
            contract_id: Contract ID for filtering
            top_k: Number of results to retrieve
            
        Returns:
            Concatenated text from top results
        """
        # Create query embedding
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = response.data[0].embedding

        # Query Pinecone
        namespace = f"{user_id}-namespace"
        results = self.index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            filter={"contract_id": contract_id},
            include_metadata=True
        )

        # Concatenate results
        context_parts = []
        if hasattr(results, 'matches') and len(results.matches) > 0:
            for match in results.matches:
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
        
        response = self.openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.strip()

    def _extract_parties(self, user_id: str, contract_id: str) -> List[Party]:
        """Extract all parties/contributors from contract"""
        question = "List all parties or contributors in this music contract and their roles."
        template = """You are a contract analyst. Use the context below to answer.

Context:
{context}

Question:
{question}

List each party as: Name | Role
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
        
        print(f"👥 Extracted {len(parties)} parties")
        return parties

    def _extract_works(self, user_id: str, contract_id: str) -> List[Work]:
        """Extract all songs/works from contract"""
        question = "List all songs, albums, or musical works mentioned in this contract."
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
        
        print(f"🎵 Extracted {len(works)} works")
        return works

    def _extract_royalties(self, user_id: str, contract_id: str) -> List[RoyaltyShare]:
        """Extract royalty shares from contract"""
        question = "List all streaming royalty shares, with name, type, percentage, and terms if any."
        template = """Context:
{context}

Question:
{question}

List each as: Name | Royalty Type | Percentage | Terms
Answer:
"""
        context = self._query_pinecone(question, user_id, contract_id, top_k=6)
        result = self._ask_llm(context, question, template)
        
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
        
        print(f"💰 Extracted {len(shares)} royalty shares")
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
        
        print("🧾 Summary generated")
        return result
