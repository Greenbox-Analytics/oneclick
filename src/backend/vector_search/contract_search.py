"""
Contract Semantic Search Module
Performs filtered vector similarity search over contract embeddings with section category support.

Features:
- Metadata-based filtering (user_id, project_id, contract_id, section_category)
- Smart query categorization for targeted retrieval
- Configurable top_k (default 5-8)
- Similarity score thresholding
- Regional index support
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from .helpers import create_query_embedding
from .query_categorizer import categorize_query, build_metadata_filter

# Load environment variables
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
DEFAULT_TOP_K = 8
MIN_SIMILARITY_THRESHOLD = 0.50

# Valid section categories (must match contract_ingestion.py)
VALID_SECTION_CATEGORIES = [
    "ROYALTY_CALCULATIONS",
    "PUBLISHING_RIGHTS",
    "PERFORMANCE_RIGHTS",
    "COPYRIGHT",
    "TERMINATION",
    "MASTER_RIGHTS",
    "OWNERSHIP_RIGHTS",
    "ACCOUNTING_AND_CREDIT",
    "OTHER"
]


class ContractSearch:
    """Handles semantic search over contract embeddings"""
    
    def __init__(self):
        """
        Initialize the contract search handler
        
        Uses PINECONE_INDEX_NAME from environment variables
        """
        self.index_name = PINECONE_INDEX_NAME
        self.index = pc.Index(self.index_name)
    
    def search(self,
               query: str,
               user_id: str,
               project_id: Optional[str] = None,
               contract_id: Optional[str] = None,
               section_categories: Optional[List[str]] = None,
               top_k: int = DEFAULT_TOP_K,
               min_score: Optional[float] = None) -> Dict:
        """
        Perform semantic search with metadata filtering including section categories.
        
        Args:
            query: Search query text
            user_id: UUID of the user (required for namespace and filtering)
            project_id: UUID of the project (optional filter)
            contract_id: UUID of specific contract (optional filter)
            section_categories: List of section categories to filter by (optional)
                e.g., ["ROYALTY_CALCULATIONS", "PUBLISHING_RIGHTS"]
            top_k: Number of results to return (default 8)
            min_score: Minimum similarity score threshold (optional)
            
        Returns:
            Dict with search results and metadata
        """
        print("\n" + "=" * 80)
        print("SEMANTIC SEARCH (with category support)")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"User ID: {user_id}")
        if project_id:
            print(f"Project ID: {project_id}")
        if contract_id:
            print(f"Contract ID: {contract_id}")
        if section_categories:
            print(f"Section Categories: {section_categories}")
        print(f"Top K: {top_k}")
        print("-" * 80)
        
        # Build metadata filter
        filter_dict = {"user_id": user_id}
        
        if project_id:
            filter_dict["project_id"] = project_id
        
        if contract_id:
            filter_dict["contract_id"] = contract_id
        
        # Add section category filter if provided
        if section_categories:
            # Validate categories
            valid_cats = [c for c in section_categories if c in VALID_SECTION_CATEGORIES]
            if valid_cats:
                if len(valid_cats) == 1:
                    filter_dict["section_category"] = valid_cats[0]
                else:
                    # Use $in operator for multiple categories
                    filter_dict["section_category"] = {"$in": valid_cats}
        
        # Create query embedding
        print("Creating query embedding...")
        query_embedding = create_query_embedding(query)
        
        # Perform search
        namespace = f"{user_id}-namespace"
        
        print(f"Searching in namespace: {namespace}")
        print(f"Filter: {filter_dict}")
        
        results = self.index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        # Process results
        matches = []
        if hasattr(results, 'matches') and len(results.matches) > 0:
            for match in results.matches:
                # Apply minimum score threshold if specified
                if min_score and match.score < min_score:
                    continue
                
                matches.append({
                    "id": match.id,
                    "score": round(match.score, 4),
                    "contract_id": match.metadata.get("contract_id"),
                    "contract_file": match.metadata.get("contract_file"),
                    "project_id": match.metadata.get("project_id"),
                    "project_name": match.metadata.get("project_name"),
                    "section_heading": match.metadata.get("section_heading", ""),
                    "section_category": match.metadata.get("section_category", "OTHER"),
                    "text": match.metadata.get("chunk_text", ""),
                    "uploaded_at": match.metadata.get("uploaded_at")
                })
        
        # Print results summary
        print(f"\nFound {len(matches)} results")
        if matches:
            print(f"Top score: {matches[0]['score']}")
            print(f"Lowest score: {matches[-1]['score']}")
        print("=" * 80)
        
        return {
            "query": query,
            "total_results": len(matches),
            "matches": matches,
            "filter": filter_dict,
            "top_k": top_k,
            "namespace": namespace
        }
    
    def search_project(self,
                      query: str,
                      user_id: str,
                      project_id: str,
                      top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Search within a specific project
        
        Args:
            query: Search query text
            user_id: UUID of the user
            project_id: UUID of the project
            top_k: Number of results to return
            
        Returns:
            Dict with search results
        """
        return self.search(
            query=query,
            user_id=user_id,
            project_id=project_id,
            top_k=top_k
        )
    
    def search_contract(self,
                       query: str,
                       user_id: str,
                       project_id: str,
                       contract_id: str,
                       top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Search within a specific contract
        
        Args:
            query: Search query text
            user_id: UUID of the user
            project_id: UUID of the project
            contract_id: UUID of the contract
            top_k: Number of results to return
            
        Returns:
            Dict with search results
        """
        return self.search(
            query=query,
            user_id=user_id,
            project_id=project_id,
            contract_id=contract_id,
            top_k=top_k
        )
    
    def search_multiple_contracts(self,
                                  query: str,
                                  user_id: str,
                                  project_id: str,
                                  contract_ids: List[str],
                                  top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Search within multiple specific contracts.
        Uses Pinecone's $in operator to filter by multiple contract IDs.
        
        Args:
            query: Search query text
            user_id: UUID of the user
            project_id: UUID of the project
            contract_ids: List of contract UUIDs to search
            top_k: Number of results to return
            
        Returns:
            Dict with search results
        """
        print("\n" + "=" * 80)
        print("MULTI-CONTRACT SEMANTIC SEARCH")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"User ID: {user_id}")
        print(f"Project ID: {project_id}")
        print(f"Contract IDs: {contract_ids}")
        print(f"Number of contracts: {len(contract_ids)}")
        print(f"Top K: {top_k}")
        print("-" * 80)
        
        # Build metadata filter with $in operator for multiple contracts
        filter_dict = {
            "user_id": user_id,
            "project_id": project_id
        }
        
        # Use $in operator for multiple contract IDs (similar to section_categories)
        if len(contract_ids) == 1:
            filter_dict["contract_id"] = contract_ids[0]
        else:
            filter_dict["contract_id"] = {"$in": contract_ids}
        
        # Create query embedding
        print("Creating query embedding...")
        query_embedding = create_query_embedding(query)
        
        # Perform search
        namespace = f"{user_id}-namespace"
        
        print(f"Searching in namespace: {namespace}")
        print(f"Filter: {filter_dict}")
        
        results = self.index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        # Process results
        matches = []
        if hasattr(results, 'matches') and len(results.matches) > 0:
            for match in results.matches:
                matches.append({
                    "id": match.id,
                    "score": round(match.score, 4),
                    "contract_id": match.metadata.get("contract_id"),
                    "contract_file": match.metadata.get("contract_file"),
                    "project_id": match.metadata.get("project_id"),
                    "project_name": match.metadata.get("project_name"),
                    "section_heading": match.metadata.get("section_heading", ""),
                    "section_category": match.metadata.get("section_category", "OTHER"),
                    "text": match.metadata.get("chunk_text", ""),
                    "uploaded_at": match.metadata.get("uploaded_at")
                })
        
        # Print results summary
        print(f"\nFound {len(matches)} results across {len(contract_ids)} contracts")
        if matches:
            print(f"Top score: {matches[0]['score']}")
            print(f"Lowest score: {matches[-1]['score']}")
            # Show distribution across contracts
            contract_distribution = {}
            for match in matches:
                contract_file = match['contract_file']
                contract_distribution[contract_file] = contract_distribution.get(contract_file, 0) + 1
            print(f"Results per contract: {contract_distribution}")
        print("=" * 80)
        
        return {
            "query": query,
            "total_results": len(matches),
            "matches": matches,
            "filter": filter_dict,
            "top_k": top_k,
            "namespace": namespace,
            "contracts_searched": len(contract_ids)
        }
    
    def smart_search(self,
                    query: str,
                    user_id: str,
                    project_id: Optional[str] = None,
                    contract_id: Optional[str] = None,
                    top_k: Optional[int] = None) -> Dict:
        """
        Perform intelligent search with automatic query categorization.
        Uses the same approach as oneclick_retrieval.py.
        
        Uses LLM to determine which contract sections are most relevant to the query,
        then filters search to those sections for better precision.
        
        Args:
            query: User's question
            user_id: UUID of the user
            project_id: UUID of the project (optional)
            contract_id: UUID of specific contract (optional)
            top_k: Number of results (auto-adjusted if None)
            
        Returns:
            Dict with search results and categorization metadata
        """
        
        print("\n" + "=" * 80)
        print("SMART SEARCH WITH AUTO-CATEGORIZATION (OneClick Method)")
        print("=" * 80)
        print(f"Query: '{query}'")
        
        # Step 1: Categorize the query (same as oneclick_retrieval.py)
        print("\n--- Step 1: Query Categorization ---")
        categorization = categorize_query(query, openai_client)
        
        print(f"Categories: {', '.join(categorization['categories'])}")
        print(f"Is General Query: {categorization['is_general']}")
        print(f"Confidence: {categorization['confidence']}")
        print(f"Reasoning: {categorization.get('reasoning', '')}")
        
        # Step 2: Auto-adjust top_k based on query type (same as oneclick_retrieval.py)
        if top_k is None:
            if categorization['is_general']:
                top_k = 5  # More context for general queries
            else:
                top_k = 3  # Focused results for specific queries
        
        print(f"Top-K: {top_k} (auto-adjusted for {'general' if categorization['is_general'] else 'specific'} query)")
        
        # Step 3: Build metadata filter using oneclick method
        print("\n--- Step 2: Building Metadata Filter ---")
        metadata_filter = build_metadata_filter(
            categories=categorization['categories'],
            is_general=categorization['is_general'],
            user_id=user_id,
            contract_id=contract_id
        )
        
        # Add project_id to filter if provided
        if project_id:
            if metadata_filter is None:
                metadata_filter = {}
            metadata_filter["project_id"] = {"$eq": project_id}
        
        print(f"Metadata Filter: {metadata_filter}")
        
        # Step 4: Create query embedding
        print("\n--- Step 3: Semantic Search ---")
        query_embedding = create_query_embedding(query)
        
        # Step 5: Query Pinecone (same as oneclick_retrieval.py)
        namespace = f"{user_id}-namespace"
        print(f"Searching in namespace: {namespace}")
        
        results = self.index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            filter=metadata_filter,
            include_metadata=True
        )
        
        print(f"Found {len(results.matches) if hasattr(results, 'matches') else 0} matches")
        
        # Step 6: Process results (same format as oneclick_retrieval.py)
        matches = []
        if hasattr(results, 'matches') and len(results.matches) > 0:
            for i, match in enumerate(results.matches):
                print(f"\nMatch {i+1}:")
                print(f"  Score: {match.score:.4f}")
                print(f"  Chunk ID: {match.id}")
                print(f"  Section: {match.metadata.get('section_heading', 'N/A')}")
                print(f"  Category: {match.metadata.get('section_category', 'N/A')}")
                
                matches.append({
                    "id": match.id,
                    "score": round(match.score, 4),
                    "contract_id": match.metadata.get("contract_id"),
                    "contract_file": match.metadata.get("contract_file"),
                    "project_id": match.metadata.get("project_id"),
                    "project_name": match.metadata.get("project_name"),
                    "section_heading": match.metadata.get("section_heading", ""),
                    "section_category": match.metadata.get("section_category", "OTHER"),
                    "text": match.metadata.get("chunk_text", ""),
                    "uploaded_at": match.metadata.get("uploaded_at")
                })
        
        print("=" * 80)
        
        return {
            "query": query,
            "total_results": len(matches),
            "matches": matches,
            "filter": metadata_filter,
            "top_k": top_k,
            "namespace": namespace,
            "categorization": categorization
        }
    
    def get_context_for_llm(self, search_results: Dict) -> str:
        """
        Format search results as context for LLM
        
        Args:
            search_results: Results from search() method
            
        Returns:
            Formatted context string
        """
        if not search_results["matches"]:
            return ""
        
        context_parts = []
        for i, match in enumerate(search_results["matches"], 1):
            context_parts.append(
                f"[Result {i} - Score: {match['score']}]\n"
                f"Source: {match['contract_file']}\n"
                f"Project: {match['project_name']}\n"
                f"Content:\n{match['text']}\n"
            )
        
        return "\n---\n\n".join(context_parts)
