### OneClick Upsert Module
"""
This module provides functionality to perform upsert operations to vector databases for the oneclick feature.  
The method works as follows:

1) Parse PDFs by first converting into markdown format.
2) Split PDFs into sections based on headings. These headings are grouped using certain keywords via a classifier method.
3) Chunk each section into smaller pieces using a the recursive character splitter method.
4) Generate embeddings for each chunk using a specified embedding model. (In this case, we use OpenAI's text-embedding-3-small model.)
5) Add metadata to each chunk, including source document information and section headings.
6) Upsert the generated embeddings into the vector database.
"""

import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pinecone import Pinecone
from openai import OpenAI
from typing import List, Tuple
from helpers import section_categorizer, is_semantic_heading, pdf_to_markdown, split_into_sections, create_vector_id, create_embeddings

# Load environment variables
load_dotenv()
# Initialize clients
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")


# initialise file path for testing
file_path = "../sample_docs/Scenario 3 'Home' - Romes_Yash Contract.pdf"

# Take PDF and convert to markdown
md_pdf = pdf_to_markdown(file_path)

# split into sections
sections = split_into_sections(md_pdf)

# categorize sections
categorized_sections = []
for header , content in sections:
    category = section_categorizer(header)
    categorized_sections.append((category, header, content))


# Chunking per section
chunk_size = 524  # tokens
chunk_overlap = 128  # tokens

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len  # Assuming len gives token count; replace with actual tokenizer if needed
)

# Initialize list to store LangChain Documents
all_chunks: List[Document] = []

# Sample metadata values (replace with actual values from your application)
user_id = "sample-user-uuid"
contract_id = "sample-contract-uuid"
document_name = Path(file_path).name
upload_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

for category, header, content in categorized_sections:
    # Split section content into chunks
    chunks = text_splitter.split_text(content)
    
    # Create a LangChain Document for each chunk with metadata
    for chunk_text in chunks:
        metadata = {
            "user_id": user_id,
            "contract_id": contract_id,
            "section_category": category,
            "section_heading": header,
            "document_name": document_name,
            "upload_time": upload_time,
            "chunk_text": chunk_text  # Store the actual chunk text in metadata for retrieval
        }
        
        # Create LangChain Document with chunk text and metadata
        doc = Document(
            page_content=chunk_text,
            metadata=metadata
        )
        all_chunks.append(doc)

print(f"Total chunks created: {len(all_chunks)}")
print("")

# Create embeddings
openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
texts = [doc.page_content for doc in all_chunks]
embeddings_list = create_embeddings(openai_client, texts)

print(f"Number of embeddings: {len(embeddings_list)}")

# Prepare vectors for Pinecone upsert
vectors_to_upsert = []
for i, doc in enumerate(all_chunks):
    vector_id = create_vector_id(doc.page_content, doc.metadata)
    vectors_to_upsert.append({
        "id": vector_id,
        "values": embeddings_list[i],
        "metadata": doc.metadata
    })

# Upsert to Pinecone namespace
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index_name = "test-3-small-index"
namespace = "yash_test_namespace"

index = pinecone_client.Index(index_name)

# Upsert in batches
batch_size = 20
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    index.upsert(vectors=batch, namespace=namespace)
    print(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors)")

print(f"Successfully upserted {len(vectors_to_upsert)} vectors to Pinecone")


