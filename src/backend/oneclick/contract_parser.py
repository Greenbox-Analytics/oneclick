"""
RAG-Based Music Contract Parser (Manual + Logging)
--------------------------------------------------
Features:
- No LangChain chains — direct control of retrieval & prompting
- Qdrant vector database
- OpenAI LLM + embeddings
- Full progress logging, timings, and cost tracking
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# --- LangChain components ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MusicContractParser")


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
    """Manual retrieval + LLM parser for music contracts, with progress logs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        collection_name: str = "music_contracts",
        storage_path: str = "./qdrant_storage",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise ValueError("Missing or invalid OpenAI API key.")

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=self.api_key)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(path=storage_path)
        logger.info("Contract Parser initialized")

    # -----------------------------------------------------------------------
    # Main
    # -----------------------------------------------------------------------

    def parse_contract(self, path: str) -> ContractData:
        """Parse PDF or DOCX contract into structured info."""
        start_time = time.time()
        logger.info(f"📄 Starting parse for: {Path(path).name}")

        # Step 1 — Load and split
        docs = self._load_document(path)
        chunks = self.text_splitter.split_documents(docs)

        # Step 2 — Build retriever
        retriever = self._build_retriever(chunks)

        # Step 3 — Extract structured elements with timing
        parties = self._timed(self._extract_parties, retriever, stage="Extract Parties")
        works = self._timed(self._extract_works, retriever, stage="Extract Works")
        royalty_shares = self._timed(self._extract_royalties, retriever, stage="Extract Royalties")
        summary = self._timed(self._extract_summary, retriever, stage="Generate Summary")

        def ensure_party(p):
            if isinstance(p, Party):
                return p
            elif isinstance(p, dict):
                return Party(name=p.get("name", "").strip(), role=p.get("role", ""))
            else:
                raise TypeError(f"Invalid party format: {p}")

        def ensure_work(w):
            if isinstance(w, Work):
                return w
            elif isinstance(w, dict):
                return Work(title=w.get("title", "").strip(), work_type=w.get("work_type", "song"))
            else:
                raise TypeError(f"Invalid work format: {w}")

        def ensure_share(s):
            if isinstance(s, RoyaltyShare):
                return s
            elif isinstance(s, dict):
                return RoyaltyShare(
                    party_name=s.get("party_name", "").strip(),
                    royalty_type=s.get("royalty_type", "streaming"),
                    percentage=float(s.get("percentage", 0.0)),
                    terms=s.get("terms")
                )
            else:
                raise TypeError(f"Invalid royalty share format: {s}")

        parties = [ensure_party(p) for p in parties if p]
        works = [ensure_work(w) for w in works if w]
        royalty_shares = [ensure_share(s) for s in royalty_shares if s]

        # Step 5 — Log completion
        total_time = time.time() - start_time
        logger.info(f"✅ Contract parsing complete in {total_time:.2f}s")
        logger.info(f"   → {len(parties)} parties, {len(works)} works, {len(royalty_shares)} shares")

        # Step 6 — Return standardized ContractData
        contract_data = ContractData(
            parties=parties,
            works=works,
            royalty_shares=royalty_shares,
            contract_summary=summary
        )

        # Final validation
        if not isinstance(contract_data, ContractData):
            raise TypeError("parse_contract() must return a ContractData instance")

        return contract_data

        # # Step 4 — Log completion
        # total_time = time.time() - start_time
        # logger.info(f"✅ Contract parsing complete in {total_time:.2f}s")

        # # Step 5 — Return structured data
        # return ContractData(
        #     parties=parties,
        #     works=works,
        #     royalty_shares=royalty_shares,
        #     contract_summary=summary
        # )

    # -----------------------------------------------------------------------
    # Helper: Timed execution wrapper
    # -----------------------------------------------------------------------

    def _timed(self, func, *args, stage: str):
        t0 = time.time()
        logger.info(f"▶️ {stage} ...")
        result = func(*args)
        logger.info(f"⏱️  {stage} done in {time.time() - t0:.2f}s")
        return result

    # -----------------------------------------------------------------------
    # Document & Vector Handling
    # -----------------------------------------------------------------------

    def _load_document(self, file_path: str):
        ext = Path(file_path).suffix.lower()
        loader = PyPDFLoader(file_path) if ext == ".pdf" else Docx2txtLoader(file_path)
        docs = loader.load()
        logger.info(f"📚 Loaded {len(docs)} document sections")
        return docs

    def _build_retriever(self, docs):
        """Embed and store chunks in Qdrant, return retriever."""
        for d in docs:
            d.metadata["source"] = "contract"

        # Reset collection if it exists
        try:
            self.qdrant_client.delete_collection(self.collection_name)
        except Exception:
            pass

        # Create local collection
        self.qdrant_client.create_collection(
            self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        # ✅ Initialize the vector store directly (no 'from_documents')
        vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        # ✅ Add documents manually
        vectorstore.add_documents(docs)

        logger.info(f"💾 Created vectorstore '{self.collection_name}' with {len(docs)} chunks")
        return vectorstore.as_retriever(search_kwargs={"k": 6})

    # -----------------------------------------------------------------------
    # Retrieval + LLM Helper
    # -----------------------------------------------------------------------

    def _ask_llm(self, retriever, question: str, template: str) -> str:
        """Retrieve context and send custom prompt to LLM."""
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        logger.info(f"🔍 Retrieved {len(docs)} relevant chunks for: {question[:50]}...")

        prompt = PromptTemplate.from_template(template).format(context=context, question=question)

        t0 = time.time()
        response = self.llm.invoke(prompt)
        duration = time.time() - t0

        logger.info(f"🤖 LLM responded in {duration:.2f}s ({len(prompt)} chars prompt)")
        # Try to print token usage if available
        try:
            usage = response.response_metadata.get("token_usage", {})
            if usage:
                logger.info(f"💰 Tokens: {usage}")
        except Exception:
            pass

        return response.content.strip()

    # -----------------------------------------------------------------------
    # Extraction Methods
    # -----------------------------------------------------------------------

    def _extract_parties(self, retriever) -> List[Party]:
        question = "List all parties or contributors in this music contract and their roles."
        template = """You are a contract analyst. Use the context below to answer.

        Context:
        {context}

        Question:
        {question}

        List each party as: Name | Role
        Answer:
        """
        result = self._ask_llm(retriever, question, template)
        parties = []
        for line in result.splitlines():
            if "|" in line:
                name, role = [x.strip() for x in line.split("|", 1)]
                parties.append(Party(name, role.lower()))
        logger.info(f"👥 Extracted {len(parties)} parties")
        return parties

    def _extract_works(self, retriever) -> List[Work]:
        question = "List all songs, albums, or musical works mentioned in this contract."
        template = """Context:
            {context}

            Question:
            {question}

            List each as: Title | Type (song, album, composition)
            Answer:
            """
        result = self._ask_llm(retriever, question, template)
        works = []
        for line in result.splitlines():
            if "|" in line:
                title, typ = [x.strip() for x in line.split("|", 1)]
                works.append(Work(title, typ.lower() or "song"))
        logger.info(f"🎵 Extracted {len(works)} works")
        return works

    def _extract_royalties(self, retriever) -> List[RoyaltyShare]:
        question = "List all streaming royalty shares, with name, type, percentage, and terms if any."
        template = """Context:
        {context}

        Question:
        {question}

        List each as: Name | Royalty Type | Percentage | Terms
        Answer:
        """
        result = self._ask_llm(retriever, question, template)
        shares = []
        for line in result.splitlines():
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    name, typ, pct = parts[:3]
                    terms = parts[3] if len(parts) > 3 else None
                    try:
                        pct_val = float(pct.replace("%", "").strip())
                    except ValueError:
                        continue
                    shares.append(RoyaltyShare(name, typ.lower(), pct_val, terms))
        logger.info(f"💰 Extracted {len(shares)} royalty shares")
        return shares

    def _extract_summary(self, retriever) -> str:
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
        result = self._ask_llm(retriever, question, template)
        logger.info("🧾 Summary generated")
        return result

    def cleanup(self):
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"🧹 Deleted collection: {self.collection_name}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Example CLI
# ---------------------------------------------------------------------------

def main():
    parser = MusicContractParser()
    path = "/Users/kenjiniyokindi/Documents/GREENBOX ANALYTICS/Personal Projects/Music Data Projects/royalty-automated-calculator copy/Contracts/Scenario 2.1 'Home' - Romes_Kenji Contract.pdf"  # or "contract.docx"

    if not Path(path).exists():
        logger.error(f"❌ Contract not found: {path}")
        return

    data = parser.parse_contract(path)

    print("\n📋 Parties:")
    for p in data.parties:
        print(f" • {p.name} – {p.role}")

    print("\n🎵 Works:")
    for w in data.works:
        print(f" • {w.title} ({w.work_type})")

    print("\n💰 Royalty Shares:")
    for s in data.royalty_shares:
        print(f" • {s.party_name}: {s.percentage}% ({s.royalty_type})")

    print("\n📄 Summary:\n", data.contract_summary)
    parser.cleanup()


if __name__ == "__main__":
    main()
