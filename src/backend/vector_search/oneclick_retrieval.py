import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from typing import Optional
from query_categorizer import categorize_query, build_metadata_filter

# Load environment variables
load_dotenv()

# Initialize clients
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index_name = "test-3-small-index"
index = pinecone_client.Index(index_name)
namespace = "yash_test_namespace"
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

file_path = "../sample_docs/Scenario 3 'Home' - Romes_Yash Contract.pdf"
document_name = Path(file_path).name


def smart_retrieval(
    query_text: str,
    document_name: Optional[str] = None,
    user_id: Optional[str] = None,
    contract_id: Optional[str] = None,
    top_k: Optional[int] = None
):
    """
    Perform intelligent retrieval with automatic query categorization.
    
    Args:
        query_text: The user's question
        document_name: Optional document name filter
        user_id: Optional user ID filter
        contract_id: Optional contract ID filter
        top_k: Number of results to return (auto-adjusted if None)
        
    Returns:
        Tuple of (response, categorization_result, final_answer)
    """
    print("\n" + "="*80)
    print("SMART RETRIEVAL WITH AUTO-CATEGORIZATION")
    print("="*80)
    print(f"\nQuery: '{query_text}'")
    
    # Step 1: Categorize the query
    print("\n--- Step 1: Query Categorization ---")
    categorization = categorize_query(query_text, openai_client)
    
    print(f"Categories: {', '.join(categorization['categories'])}")
    print(f"Is General Query: {categorization['is_general']}")
    print(f"Confidence: {categorization['confidence']}")
    print(f"Reasoning: {categorization['reasoning']}")
    
    # Auto-adjust top_k based on query type
    if top_k is None:
        if categorization['is_general']:
            top_k = 5  # More context for general queries
        else:
            top_k = 3   # Focused results for specific queries
    
    print(f"Top-K: {top_k} (auto-adjusted for {'general' if categorization['is_general'] else 'specific'} query)")
    
    # Step 2: Build metadata filter
    metadata_filter = build_metadata_filter(
        categories=categorization['categories'],
        is_general=categorization['is_general'],
        document_name=document_name,
        user_id=user_id,
        contract_id=contract_id
    )
    
    print(f"Metadata Filter: {metadata_filter}")
    
    # Step 3: Create query embedding
    print("\n--- Step 2: Semantic Search ---")
    query_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query_text]
    ).data[0].embedding
    
    # Step 4: Query Pinecone
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=metadata_filter
    )
    
    print(f"Found {len(response['matches'])} matches")
    
    # Step 5: Generate answer from retrieved chunks
    retrieved_chunks = []
    for i, match in enumerate(response['matches']):
        print(f"\nMatch {i+1}:")
        print(f"  Score: {match['score']:.4f}")
        print(f"  Section: {match['metadata'].get('section_heading', 'N/A')}")
        print(f"  Category: {match['metadata'].get('section_category', 'N/A')}")
        
        chunk_text = match['metadata'].get('chunk_text', '')
        if chunk_text:
            retrieved_chunks.append({
                'text': chunk_text,
                'section': match['metadata'].get('section_heading', 'N/A'),
                'score': match['score']
            })
    
    # Generate final answer
    final_answer = ""
    if len(retrieved_chunks) > 0:
        context_text = "\n\n".join([
            f"[Section: {chunk['section']}]\n{chunk['text']}" 
            for chunk in retrieved_chunks
        ])
        
        system_prompt = """You are a legal contract analyst specializing in music industry agreements. 
Your task is to answer questions about contract documents based on the provided context.
Be precise and only include information that is explicitly stated in the provided context."""

        user_prompt = f"""Based on the following contract excerpts, answer this question:

{query_text}

Contract Excerpts:
{context_text}

Return a clear, concise answer based only on the information provided in the excerpts."""

        llm_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=1000
        )

        final_answer = llm_response.choices[0].message.content
    else:
        final_answer = "No relevant information found in the contract."
    
    return response, categorization, final_answer


# Test with different types of queries
test_queries = [
    "What are the publishing royalty splits in this contract?",  # Specific
    "Summarize the key terms of this agreement",  # General
    "Who owns the master rights?"  # Specific
]

for query_text in test_queries:
    response, categorization, final_answer = smart_retrieval(
        query_text=query_text,
        document_name=document_name
    )
    
    # Display final answer
    print("\n" + "="*80)
    print("FINAL ANSWER:")
    print("="*80)
    print(final_answer)
    print("="*80)
