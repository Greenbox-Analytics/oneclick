"""
PDF Processing and Semantic Search with Pinecone
This script:
1. Loads PDFs from sample_docs folder
2. Chunks them using PyMuPDF and LangChain's RecursiveCharacterTextSplitter
3. Creates embeddings using OpenAI text-embedding-3-small
4. Uploads embeddings to Pinecone namespace "music_text_test"
5. Performs semantic search for royalty percentage splits
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
INDEX_NAME = "test-3-small-index"
NAMESPACE = "music_text_test"
SAMPLE_DOCS_PATH = Path(__file__).parent.parent / "sample_docs"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 154
BATCH_SIZE = 20


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF"""
    print(f"Extracting text from: {pdf_path.name}")
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        text += f"\n--- Page {page_num + 1} ---\n"
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text, source_file):
    """Chunk text using LangChain's RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks from {source_file}")
    return chunks


def create_embeddings(texts):
    """Create embeddings using OpenAI text-embedding-3-small model"""
    print(f"Creating embeddings for {len(texts)} chunks...")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    print(f"Created {len(embeddings)} embeddings")
    return embeddings


def process_pdfs():
    """Process all PDFs in the sample_docs folder"""
    pdf_files = list(SAMPLE_DOCS_PATH.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {SAMPLE_DOCS_PATH}")
        return []
    
    print(f"\nFound {len(pdf_files)} PDF files")
    print("=" * 80)
    
    all_records = []
    record_id = 0
    
    for pdf_file in pdf_files:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
        
        # Chunk the text
        chunks = chunk_text(text, pdf_file.name)
        
        # Create records for each chunk
        for chunk_idx, chunk in enumerate(chunks):
            record_id += 1
            all_records.append({
                "id": f"doc_{record_id}",
                "text": chunk,
                "source_file": pdf_file.name,
                "chunk_index": chunk_idx
            })
    
    print(f"\nTotal records created: {len(all_records)}")
    return all_records


def upload_to_pinecone(records):
    """Upload embeddings to Pinecone in batches"""
    print("\n" + "=" * 80)
    print("UPLOADING TO PINECONE")
    print("=" * 80)
    
    # Get the existing index
    print(f"Connecting to index: {INDEX_NAME}")
    index = pc.Index(INDEX_NAME)
    
    # Process in batches
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, len(records))
        batch = records[start_idx:end_idx]
        
        # Extract texts for embedding
        texts = [record["text"] for record in batch]
        
        # Create embeddings using OpenAI
        print(f"Creating embeddings for batch {batch_num + 1}/{total_batches}...")
        embeddings = create_embeddings(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, record in enumerate(batch):
            vectors.append({
                "id": record["id"],
                "values": embeddings[i],
                "metadata": {
                    "chunk_text": record["text"],
                    "source_file": record["source_file"],
                    "chunk_index": record["chunk_index"]
                }
            })
        
        # Upsert batch
        print(f"Uploading batch {batch_num + 1}/{total_batches} ({len(vectors)} vectors)...")
        index.upsert(vectors=vectors, namespace=NAMESPACE)
    
    print(f"\nSuccessfully uploaded {len(records)} records to namespace '{NAMESPACE}'")
    
    # Wait for indexing to complete
    print("Waiting for indexing to complete...")
    time.sleep(5)
    
    # Check stats
    stats = index.describe_index_stats()
    print(f"\nIndex stats:")
    print(f"  Total vectors: {stats.get('total_vector_count', 0)}")
    if 'namespaces' in stats and NAMESPACE in stats['namespaces']:
        print(f"  Vectors in '{NAMESPACE}': {stats['namespaces'][NAMESPACE].get('vector_count', 0)}")
    
    return index


def semantic_search(index, query, top_k=2):
    """Perform semantic search on the Pinecone index"""
    print("\n" + "=" * 80)
    print("SEMANTIC SEARCH")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Top K: {top_k}")
    print("-" * 80)
    
    # Create embedding for the query
    print("Creating query embedding...")
    query_embedding = create_embeddings([query])[0]
    
    # Search the index
    results = index.query(
        namespace=NAMESPACE,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Print results
    if hasattr(results, 'matches') and len(results.matches) > 0:
        matches = results.matches
        print(f"\nFound {len(matches)} results:\n")
        
        for i, match in enumerate(matches, 1):
            print(f"Result #{i}")
            print(f"  ID: {match.id}")
            print(f"  Score: {round(match.score, 4)}")
            print(f"  Source File: {match.metadata.get('source_file', 'N/A')}")
            print(f"  Chunk Index: {match.metadata.get('chunk_index', 'N/A')}")
            print(f"  Content:")
            print(f"  {'-' * 76}")
            content = match.metadata.get('chunk_text', '')
            # Print content with proper indentation
            for line in content.split('\n'):
                print(f"  {line}")
            print(f"  {'-' * 76}\n")
    else:
        print("No results found")
    
    return results


def generate_final_answer(search_results, user_query):
    """Use GPT-5-nano to generate a concise answer from search results"""
    print("\n" + "=" * 80)
    print("GENERATING FINAL ANSWER WITH GPT-5-nano")
    print("=" * 80)
    
    # Extract chunks from search results
    if not hasattr(search_results, 'matches') or len(search_results.matches) == 0:
        print("No search results to process")
        return None
    
    # Combine all relevant chunks
    context_chunks = []
    for match in search_results.matches:
        chunk_text = match.metadata.get('chunk_text', '')
        source_file = match.metadata.get('source_file', 'Unknown')
        context_chunks.append(f"[Source: {source_file}]\n{chunk_text}")
    
    combined_context = "\n\n---\n\n".join(context_chunks)
    
    # Create the prompt for GPT-5-nano
    system_prompt = """You are a legal contract analyst specializing in music industry agreements. 
Your task is to extract royalty percentage splits from contract documents and present them in a specific format.
Be precise and only include information that is explicitly stated in the provided context."""
    
    user_prompt = f"""Based on the following contract excerpts, answer this question:

{user_query}

Contract Excerpts:
{combined_context}

Return the output concisely in this specific format: 'specific person : percentage split match'
If multiple people are mentioned, list each on a separate line.
Only include information that is explicitly stated in the excerpts."""
    
    # Call GPT-5-nano
    print("Calling GPT-5-nano...")
    response = openai_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    final_answer = response.choices[0].message.content
    
    print("\nFINAL ANSWER:")
    print("-" * 80)
    print(final_answer)
    print("-" * 80)
    
    return final_answer


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("PDF TO PINECONE - MUSIC CONTRACT PROCESSING")
    print("=" * 80)
    
    # Step 1: Process PDFs
    records = process_pdfs()
    
    if not records:
        print("No records to process. Exiting.")
        return
    
    # Step 2: Upload to Pinecone
    index = upload_to_pinecone(records)
    
    # Step 3: Perform semantic search for royalty percentage splits
    query = "royalty percentage splits"
    search_results = semantic_search(index, query, top_k=2)
    
    # Step 4: Generate final answer using GPT-5-nano
    user_query = "What are the royalty percentage splits for the artist and producer in the contract?"
    generate_final_answer(search_results, user_query)
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
