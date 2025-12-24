"""
Contract Semantic Search Module
Performs filtered vector similarity search over contract embeddings.

Features:
- Metadata-based filtering (user_id, project_id, contract_id)
- Configurable top_k (default 5-8)
- Similarity score thresholding
- Regional index support
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
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
DEFAULT_TOP_K = 8
MIN_SIMILARITY_THRESHOLD = 0.50

# Regional index mapping
REGIONAL_INDEXES = {
    "US": "test-3-small-index",
    "EU": "test-3-small-index",
    "UK": "test-3-small-index"
}


class ContractSearch:
    """Handles semantic search over contract embeddings"""
    
    def __init__(self, region: str = "US"):
        """
        Initialize the contract search handler
        
        Args:
            region: Region code (US, EU, UK) - determines which index to use
        """
        if region not in REGIONAL_INDEXES:
            raise ValueError(f"Invalid region: {region}. Must be one of {list(REGIONAL_INDEXES.keys())}")
        
        self.region = region
        self.index_name = REGIONAL_INDEXES[region]
        self.index = pc.Index(self.index_name)
    
    def create_query_embedding(self, query: str) -> List[float]:
        """
        Create embedding for search query
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector
        """
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        return response.data[0].embedding
    
    def search(self,
               query: str,
               user_id: str,
               project_id: Optional[str] = None,
               contract_id: Optional[str] = None,
               top_k: int = DEFAULT_TOP_K,
               min_score: Optional[float] = None) -> Dict:
        """
        Perform semantic search with metadata filtering
        
        Args:
            query: Search query text
            user_id: UUID of the user (required for namespace and filtering)
            project_id: UUID of the project (optional filter)
            contract_id: UUID of specific contract (optional filter)
            top_k: Number of results to return (default 8)
            min_score: Minimum similarity score threshold (optional)
            
        Returns:
            Dict with search results and metadata
        """
        print("\n" + "=" * 80)
        print("SEMANTIC SEARCH")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"User ID: {user_id}")
        if project_id:
            print(f"Project ID: {project_id}")
        if contract_id:
            print(f"Contract ID: {contract_id}")
        print(f"Top K: {top_k}")
        print("-" * 80)
        
        # Build metadata filter
        filter_dict = {"user_id": user_id}
        
        if project_id:
            filter_dict["project_id"] = project_id
        
        if contract_id:
            filter_dict["contract_id"] = contract_id
        
        # Create query embedding
        print("Creating query embedding...")
        query_embedding = self.create_query_embedding(query)
        
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


# Example usage
if __name__ == "__main__":
    # Example: Search for royalty information
    search = ContractSearch(region="US")
    
    # Search within a project
    results = search.search_project(
        query="What are the royalty percentage splits?",
        user_id="test-user-123",
        project_id="test-project-456",
        top_k=5
    )
    
    print("\n" + "=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)
    
    for i, match in enumerate(results["matches"], 1):
        print(f"\nResult #{i}")
        print(f"  Score: {match['score']}")
        print(f"  Contract: {match['contract_file']}")
        print(f"  Project: {match['project_name']}")
        print(f"  Content Preview: {match['text'][:200]}...")
        print("-" * 80)
    
    # Get formatted context for LLM
    context = search.get_context_for_llm(results)
    print("\n" + "=" * 80)
    print("FORMATTED CONTEXT FOR LLM")
    print("=" * 80)
    print(context[:500] + "..." if len(context) > 500 else context)
