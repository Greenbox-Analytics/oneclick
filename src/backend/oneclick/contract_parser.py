"""
Music Contract Parser - Unified full-document extraction
Extracts structured data (parties, works, royalty shares) from contract markdown in a single LLM call.
"""

import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import time
import json

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Configuration
LLM_MODEL_LARGE = os.getenv("OPENAI_LLM_MODEL_LARGE", "gpt-5.2")
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
    """Extract structured data from contracts using full document context"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise ValueError("Missing or invalid OpenAI API key.")

        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None
        )

    def parse_contract(self, full_text: str, path: str = None, user_id: str = None, contract_id: str = None, use_parallel: bool = True) -> ContractData:
        """
        Parse contract by extracting all data in a single LLM call.

        Args:
            full_text: Full markdown text of the contract (required)
            path: Unused, kept for backward compatibility
            user_id: Unused, kept for backward compatibility
            contract_id: Unused, kept for backward compatibility
            use_parallel: Unused, kept for backward compatibility

        Returns:
            ContractData with extracted information
        """
        if not full_text:
            raise ValueError("full_text is required. The contract markdown must be provided.")

        start_time = time.time()
        logger.info(f"📄 Extracting contract data (unified single-call, model={LLM_MODEL_LARGE})")
        logger.info(f"   Document length: {len(full_text)} chars")

        # Single unified extraction
        result = self._extract_all_unified(full_text)

        parties = result["parties"]
        works = result["works"]
        royalty_shares = result["royalty_shares"]

        # Post-processing: simplify roles
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
                    if share_name_norm in party_norm or party_norm in share_name_norm:
                        best_match = party
                        break

            if best_match:
                logger.info(f"   🔄 Reconciling name: '{share.party_name}' -> '{best_match.name}'")
                share.party_name = best_match.name

        total_time = time.time() - start_time
        logger.info(f"✅ Extraction complete in {total_time:.2f}s")
        logger.info(f"   → {len(parties)} parties, {len(works)} works, {len(royalty_shares)} shares")

        return ContractData(
            parties=parties,
            works=works,
            royalty_shares=royalty_shares,
            contract_summary=""
        )

    def _extract_all_unified(self, full_text: str) -> dict:
        """
        Extract parties, works, and royalty shares in a single LLM call.

        Args:
            full_text: Full markdown text of the contract

        Returns:
            Dict with keys: "parties", "works", "royalty_shares"
        """
        streaming_terms_str = ", ".join([f"'{term}'" for term in STREAMING_EQUIVALENT_TERMS])

        prompt = f"""You are a music contract analyst. Extract ALL of the following from this contract in a single pass.

CONTRACT TEXT:
{full_text}

STANDARD ROLE TAXONOMY (use ONLY these terms for party roles):
Writer, Artist, Producer, Label, Publisher, Distributor, Manager, Mixer, Licensor, Licensee

EXTRACTION INSTRUCTIONS:

1. PARTIES: Identify all named legal entities and individuals who are parties to this agreement.
   - Look in: opening paragraph ("by and between"), definitions, signature blocks, AND tables.
   - Identify each party's role using ANY of these sources (in priority order):
     a. Parenthetical labels after the name (e.g., 'John Smith ("Producer")')
     b. Table cells or column headers associating a name with a role
     c. Defined terms (e.g., '"Producer" shall mean...')
     d. Signature block role labels
     e. Functional descriptions in the contract body
   - Map roles to the STANDARD ROLE TAXONOMY above. For example: "Songwriter" -> Writer, "Recording Artist" -> Artist, "Company" -> Label.
     If no standard term fits, use a short descriptive term (max 2 words).
   - If a party has multiple roles, separate them with semicolons (e.g., "Writer; Producer").
   - Include aliases (p/k/a, d/b/a, professionally known as) in the name field.
   - Ignore generic references like "third parties" unless a specific name is attached.

2. WORKS: Identify all musical works, compositions, masters, recordings, tracks, or releases covered by this agreement.
   - Search body clauses, schedules, exhibits, annexes, and tables (track lists).
   - Deduplicate variations of the same title — use a canonical title.
   - Do not invent titles from generic placeholders.

3. ROYALTY SHARES: Identify all explicit royalty percentage splits defined in this contract.
   - Look for terms like {streaming_terms_str}.
   - Prioritize schedules/exhibits/annexes/tables where economic terms are listed.
   - Also scan body clauses for wording such as 'shall receive', 'entitled to', 'payable', 'points', 'gross receipts', 'net receipts'.
   - Note that 'master' or 'producer royalties' typically encompass streaming revenue.
   - CRITICAL: If a royalty split refers to a generic role (e.g., 'Songwriter', 'Producer', 'Artist'), substitute it with the actual party name from step 1.
   - Simplify royalty type to one of: Streaming, Master, Publishing, Producer, Mixer, Remixer. If it doesn't fit, use a short descriptive term (max 3 words).
   - Only include parties with a specific numeric percentage.

Return your answer as a JSON object with this exact structure:
{{
  "parties": [
    {{"name": "Full Name (including aliases)", "role": "standard_role"}}
  ],
  "works": [
    {{"title": "Canonical Title", "work_type": "composition|master recording|song|album"}}
  ],
  "royalty_shares": [
    {{"party_name": "Full Name", "royalty_type": "Streaming|Master|Publishing|Producer|Mixer", "percentage": 50.0, "terms": "optional terms or null"}}
  ]
}}

Return ONLY valid JSON."""

        logger.info(f"   🧠 Sending unified extraction request to {LLM_MODEL_LARGE}...")
        t_llm = time.time()

        response = self.openai_client.chat.completions.create(
            model=LLM_MODEL_LARGE,
            messages=[
                {"role": "system", "content": "You are a music contract analyst. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content.strip()
        logger.info(f"      ⏱️  LLM response took: {time.time() - t_llm:.2f}s")

        # Parse JSON response
        data = json.loads(response_text)

        # Parse parties
        from oneclick.helpers import normalize_name, normalize_title
        parties = []
        seen_party_names = set()
        for item in data.get("parties", []):
            name = item.get("name", "").strip()
            role = item.get("role", "").strip()
            if name and role:
                key = normalize_name(name)
                if key not in seen_party_names:
                    seen_party_names.add(key)
                    parties.append(Party(name, role.lower()))

        logger.info(f"👥 Extracted {len(parties)} parties")
        for i, party in enumerate(parties):
            logger.info(f"   {i+1}. {party.name} ({party.role})")

        # Parse works
        works = []
        seen_works = set()
        for item in data.get("works", []):
            title = item.get("title", "").strip()
            work_type = item.get("work_type", "song").strip().lower() or "song"
            if title:
                normalized = normalize_title(title)
                if normalized not in seen_works:
                    seen_works.add(normalized)
                    works.append(Work(title, work_type))

        logger.info(f"🎵 Extracted {len(works)} works")
        for i, work in enumerate(works):
            logger.info(f"   {i+1}. {work.title} ({work.work_type})")

        # Parse royalty shares
        shares = []
        for item in data.get("royalty_shares", []):
            party_name = item.get("party_name", "").strip()
            royalty_type = item.get("royalty_type", "").strip()
            percentage = item.get("percentage")
            terms = item.get("terms")
            if party_name and royalty_type and percentage is not None:
                try:
                    pct_val = float(percentage)
                    terms_str = str(terms) if terms and terms != "null" else None
                    shares.append(RoyaltyShare(party_name, royalty_type.lower(), pct_val, terms_str))
                except (ValueError, TypeError):
                    continue

        logger.info(f"💰 Extracted {len(shares)} royalty shares")
        for i, share in enumerate(shares):
            logger.info(f"   {i+1}. {share.party_name} | {share.royalty_type} | {share.percentage}%" + (f" | {share.terms}" if share.terms else ""))

        return {
            "parties": parties,
            "works": works,
            "royalty_shares": shares
        }
