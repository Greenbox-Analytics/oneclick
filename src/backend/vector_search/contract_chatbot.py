"""
Contract Chatbot Module
RAG-based Q&A system for contract documents with similarity thresholding.

Features:
- Semantic search with metadata filtering
- Similarity threshold enforcement (≥0.75)
- LLM-powered answer generation with grounding
- Support for project-level and contract-level queries
- Conversation history tracking
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from vector_search.contract_search import ContractSearch

# Load environment variables
load_dotenv()

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None
)

# Configuration
DEFAULT_LLM_MODEL = "gpt-4o-mini"  # Can be changed to gpt-5-nano when available
MIN_SIMILARITY_THRESHOLD = 0.75
DEFAULT_TOP_K = 8
MAX_CONTEXT_LENGTH = 8000  # Characters to send to LLM


class ContractChatbot:
    """RAG-based chatbot for contract Q&A"""
    
    def __init__(self, region: str = "US", llm_model: str = DEFAULT_LLM_MODEL):
        """
        Initialize the contract chatbot
        
        Args:
            region: Region code (US, EU, UK)
            llm_model: LLM model to use for answer generation
        """
        self.region = region
        self.llm_model = llm_model
        self.search_engine = ContractSearch(region=region)
        self.conversation_history = []
    
    def _check_similarity_threshold(self, search_results: Dict) -> bool:
        """
        Check if highest similarity score meets threshold
        
        Args:
            search_results: Results from ContractSearch
            
        Returns:
            True if threshold is met, False otherwise
        """
        if not search_results["matches"]:
            return False
        
        highest_score = search_results["matches"][0]["score"]
        return highest_score >= MIN_SIMILARITY_THRESHOLD
    
    def _create_system_prompt(self) -> str:
        """
        Create system prompt for LLM with grounding instructions
        
        Returns:
            System prompt string
        """
        return """You are a specialized contract analysis assistant for the music industry. Your role is to answer questions about music contracts accurately and precisely.

CRITICAL RULES:
1. ONLY answer based on the provided contract excerpts - do not use external knowledge
2. If the answer is not explicitly stated in the excerpts, respond with: "I don't know based on the available documents."
3. Always cite the source (contract file and page number) when providing information
4. Be precise with numbers, percentages, dates, and legal terms
5. If multiple contracts contain relevant information, clearly distinguish between them
6. Do not make assumptions or inferences beyond what is explicitly stated
7. If asked about something not in the excerpts, acknowledge the limitation

Your answers should be:
- Accurate and grounded in the provided text
- Clear and concise
- Properly cited with sources
- Professional and helpful"""
    
    def _format_context(self, search_results: Dict) -> str:
        """
        Format search results as context for LLM
        
        Args:
            search_results: Results from ContractSearch
            
        Returns:
            Formatted context string
        """
        if not search_results["matches"]:
            return "No relevant contract excerpts found."
        
        context_parts = []
        for i, match in enumerate(search_results["matches"], 1):
            context_parts.append(
                f"[Excerpt {i}]\n"
                f"Contract: {match['contract_file']}\n"
                f"Page: {match['page_number']}\n"
                f"Relevance Score: {match['score']}\n"
                f"Project: {match['project_name']}\n"
                f"\nContent:\n{match['text']}\n"
            )
        
        full_context = "\n" + "="*80 + "\n\n".join(context_parts)
        
        # Truncate if too long
        if len(full_context) > MAX_CONTEXT_LENGTH:
            full_context = full_context[:MAX_CONTEXT_LENGTH] + "\n\n[Context truncated due to length...]"
        
        return full_context
    
    def _generate_answer(self, query: str, context: str, search_results: Dict) -> Dict:
        """
        Generate answer using LLM
        
        Args:
            query: User's question
            context: Formatted context from search results
            search_results: Original search results
            
        Returns:
            Dict with answer and metadata
        """
        # Check similarity threshold
        if not self._check_similarity_threshold(search_results):
            return {
                "answer": "I don't know based on the available documents.",
                "confidence": "low",
                "reason": "No sufficiently relevant information found (similarity threshold not met)",
                "highest_score": search_results["matches"][0]["score"] if search_results["matches"] else 0.0,
                "threshold": MIN_SIMILARITY_THRESHOLD,
                "sources": []
            }
        
        # Create messages for LLM
        system_prompt = self._create_system_prompt()
        
        user_prompt = f"""Based on the following contract excerpts, please answer this question:

QUESTION: {query}

CONTRACT EXCERPTS:
{context}

Remember to:
- Only use information from the excerpts above
- Cite sources (contract file and page number)
- If the answer isn't in the excerpts, say "I don't know based on the available documents."
"""
        
        # Call LLM
        print(f"\nGenerating answer using {self.llm_model}...")
        
        try:
            response = openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Low temperature for factual accuracy
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Extract sources from search results
            sources = [
                {
                    "contract_file": match["contract_file"],
                    "page_number": match["page_number"],
                    "score": match["score"],
                    "project_name": match["project_name"]
                }
                for match in search_results["matches"]
            ]
            
            return {
                "answer": answer,
                "confidence": "high",
                "highest_score": search_results["matches"][0]["score"],
                "threshold": MIN_SIMILARITY_THRESHOLD,
                "sources": sources,
                "model": self.llm_model
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "confidence": "error",
                "error": str(e),
                "sources": []
            }
    
    def ask(self,
            query: str,
            user_id: str,
            project_id: Optional[str] = None,
            contract_id: Optional[str] = None,
            top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Ask a question about contracts
        
        Args:
            query: User's question
            user_id: UUID of the user
            project_id: UUID of the project (optional)
            contract_id: UUID of specific contract (optional)
            top_k: Number of search results to retrieve
            
        Returns:
            Dict with answer, sources, and metadata
        """
        print("\n" + "=" * 80)
        print("CONTRACT CHATBOT")
        print("=" * 80)
        print(f"Question: {query}")
        print(f"User ID: {user_id}")
        if project_id:
            print(f"Project ID: {project_id}")
        if contract_id:
            print(f"Contract ID: {contract_id}")
        print("-" * 80)
        
        # Step 1: Perform semantic search
        search_results = self.search_engine.search(
            query=query,
            user_id=user_id,
            project_id=project_id,
            contract_id=contract_id,
            top_k=top_k
        )
        
        # Step 2: Check if we have results
        if not search_results["matches"]:
            return {
                "query": query,
                "answer": "I don't know based on the available documents.",
                "confidence": "low",
                "reason": "No relevant documents found",
                "sources": [],
                "search_results_count": 0
            }
        
        # Step 3: Format context
        context = self._format_context(search_results)
        
        # Step 4: Generate answer
        result = self._generate_answer(query, context, search_results)
        
        # Step 5: Add query and search metadata
        result["query"] = query
        result["search_results_count"] = search_results["total_results"]
        result["filter"] = search_results["filter"]
        
        # Step 6: Store in conversation history
        self.conversation_history.append({
            "query": query,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "timestamp": search_results["matches"][0]["uploaded_at"] if search_results["matches"] else None
        })
        
        print("\n" + "=" * 80)
        print("ANSWER GENERATED")
        print("=" * 80)
        print(f"Confidence: {result['confidence']}")
        if result.get('highest_score'):
            print(f"Highest Similarity Score: {result['highest_score']}")
        print(f"Sources Used: {len(result['sources'])}")
        print("=" * 80)
        
        return result
    
    def ask_project(self,
                   query: str,
                   user_id: str,
                   project_id: str,
                   top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Ask a question about a specific project's contracts
        
        Args:
            query: User's question
            user_id: UUID of the user
            project_id: UUID of the project
            top_k: Number of search results to retrieve
            
        Returns:
            Dict with answer and metadata
        """
        return self.ask(
            query=query,
            user_id=user_id,
            project_id=project_id,
            top_k=top_k
        )
    
    def ask_contract(self,
                    query: str,
                    user_id: str,
                    project_id: str,
                    contract_id: str,
                    top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Ask a question about a specific contract
        
        Args:
            query: User's question
            user_id: UUID of the user
            project_id: UUID of the project
            contract_id: UUID of the contract
            top_k: Number of search results to retrieve
            
        Returns:
            Dict with answer and metadata
        """
        return self.ask(
            query=query,
            user_id=user_id,
            project_id=project_id,
            contract_id=contract_id,
            top_k=top_k
        )
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Get conversation history
        
        Returns:
            List of conversation turns
        """
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = ContractChatbot(region="US")
    
    # Example 1: Ask about a project
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Project-level question")
    print("=" * 80)
    
    result1 = chatbot.ask_project(
        query="What are the royalty percentage splits in this project?",
        user_id="test-user-123",
        project_id="test-project-456"
    )
    
    print("\nQUESTION:", result1["query"])
    print("\nANSWER:")
    print(result1["answer"])
    print("\nCONFIDENCE:", result1["confidence"])
    print("\nSOURCES:")
    for source in result1["sources"]:
        print(f"  - {source['contract_file']} (Page {source['page_number']}, Score: {source['score']})")
    
    # Example 2: Ask about a specific contract
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Contract-specific question")
    print("=" * 80)
    
    result2 = chatbot.ask_contract(
        query="What is the term length of this contract?",
        user_id="test-user-123",
        project_id="test-project-456",
        contract_id="test-contract-789"
    )
    
    print("\nQUESTION:", result2["query"])
    print("\nANSWER:")
    print(result2["answer"])
    print("\nCONFIDENCE:", result2["confidence"])
    
    # Example 3: Question with no relevant information
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: Question with no relevant information")
    print("=" * 80)
    
    result3 = chatbot.ask_project(
        query="What is the weather like today?",
        user_id="test-user-123",
        project_id="test-project-456"
    )
    
    print("\nQUESTION:", result3["query"])
    print("\nANSWER:")
    print(result3["answer"])
    print("\nCONFIDENCE:", result3["confidence"])
