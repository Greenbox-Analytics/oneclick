# Contract Upload/Deletion Implementation Guide

## Overview

This document describes the implementation of an optimized, scalable contract upload and deletion workflow with deterministic hash IDs for the Zoe AI platform.

---

## Architecture

### 1. User Identification & Namespace

**Single Namespace Per User:**
- Format: `{userID}-namespace`
- One namespace per user across all projects and contracts
- No per-contract or per-project namespaces

**User Identification:**
```typescript
const { data: { user } } = await supabase.auth.getUser()
const namespace = `${user.id}-namespace`
```

---

## 2. Contract Upload Workflow

### UI Flow

1. **Upload Button**: Click "Upload" button in Zoe interface
2. **Modal Dialog**: Opens `ContractUploadModal` component
3. **File Selection**: Select one or multiple PDF files
4. **Upload**: Click "Upload" to trigger processing pipeline

### Processing Pipeline (Per PDF)

For each contract, the system performs the following steps:

#### Step 1: Extract Text
- Uses PyMuPDF (fitz) to extract text from PDF
- Preserves page numbers for metadata

#### Step 2: Chunk Text
- Chunks text into **524 character segments with 50 character overlap**
- Uses LangChain's `RecursiveCharacterTextSplitter`
- Hierarchical splitting with separators: `["\n\n", "\n", ". ", " ", ""]`
- Preserves semantic boundaries (paragraphs, sentences, words)
- Tracks page numbers and chunk indices
- **Improved similarity scores** compared to token-based chunking

#### Step 3: Generate Metadata
```json
{
  "user_id": "uuid",
  "project_id": "uuid",
  "project_name": "Project Name",
  "contract_id": "uuid",
  "contract_filename": "file.pdf",
  "page_number": 1,
  "chunk_index": 0,
  "uploaded_at": "2025-12-07T23:30:00.000Z"
}
```

#### Step 4: Generate Deterministic Chunk ID

**Critical Implementation:**
```python
def generate_deterministic_id(self, chunk_text: str, metadata: Dict) -> str:
    """
    Generate deterministic vector ID using SHA256 of content + metadata
    
    This ensures uniqueness across:
    - Different users (via user_id in metadata)
    - Different projects (via project_id in metadata)
    - Different contracts (via contract_id in metadata)
    - Different versions (content or metadata changes = new hash)
    """
    # Create canonical JSON of metadata (sorted keys for consistency)
    canonical_metadata = json.dumps(metadata, sort_keys=True)
    
    # Combine page content + canonical metadata
    combined_string = chunk_text + canonical_metadata
    
    # Generate SHA256 hash
    return hashlib.sha256(combined_string.encode()).hexdigest()
```

**Why This Ensures Uniqueness:**
- If text changes → hash changes
- If ANY metadata changes (page number, contract ID, etc.) → hash changes
- Prevents collisions across:
  - Different users
  - Different projects
  - Different contracts
  - Different versions of the same contract

#### Step 5: Create Embeddings
- Uses OpenAI `text-embedding-3-small` model
- Processes in batches of 100 for efficiency

#### Step 6: Upsert to Pinecone
- Upserts all chunk vectors to user's namespace
- Batch size: 100 vectors per request
- Includes full metadata for filtering

---

## 3. Contract Deletion Workflow

### Process

1. **User Action**: Click delete icon next to contract
2. **Confirmation**: Show confirmation dialog
3. **Delete from Vector DB**: Query and delete all chunks with matching `contract_id`
4. **Delete from Storage**: Remove file from Supabase Storage
5. **Delete from Database**: Remove record from `project_files` table

### Implementation

```python
def delete_contract(self, user_id: str, contract_id: str) -> Dict:
    """
    Delete all vectors for a specific contract using metadata filter
    
    The vector DB is the authoritative source - no pickle files or local dictionaries.
    """
    namespace = f"{user_id}-namespace"
    
    # Delete all vectors matching the contract_id filter
    self.index.delete(
        namespace=namespace,
        filter={"contract_id": contract_id}
    )
    
    return {
        "status": "success",
        "contract_id": contract_id,
        "namespace": namespace
    }
```

**Key Points:**
- Vector DB is the authoritative source of truth
- No pickle files or local dictionaries
- Uses metadata filter for efficient deletion
- Atomic operation per contract

---

## 4. Contract Replacement / Re-Upload

### Process

If a user re-uploads a contract with the same identifier:

1. **Delete Existing**: First delete all existing chunks with matching `contract_id`
2. **Process New**: Then process and upsert all new chunks
3. **Automatic Updates**: Because chunk IDs include metadata AND content, they automatically reflect changes

### Implementation

```python
def replace_contract(self, pdf_path: str, user_id: str, project_id: str, 
                    project_name: str, contract_id: str, contract_filename: str) -> Dict:
    """
    Replace an existing contract by deleting old chunks and uploading new ones
    """
    # Step 1: Delete existing contract chunks
    delete_result = self.delete_contract(user_id, contract_id)
    
    if delete_result["status"] != "success":
        return {"status": "error", "error": "Failed to delete existing contract"}
    
    # Step 2: Ingest new contract
    ingest_result = self.ingest_contract(
        pdf_path=pdf_path,
        user_id=user_id,
        project_id=project_id,
        project_name=project_name,
        contract_id=contract_id,
        contract_filename=contract_filename
    )
    
    return {"status": "success", "operation": "replace", **ingest_result}
```

---

## API Endpoints

### Upload Contract

**Endpoint:** `POST /contracts/upload`

**Request:**
```
Content-Type: multipart/form-data

file: PDF file
project_id: UUID
user_id: UUID
```

**Response:**
```json
{
  "status": "success",
  "contract_id": "uuid",
  "contract_filename": "contract.pdf",
  "total_chunks": 42,
  "message": "Contract uploaded and processed successfully. 42 chunks created."
}
```

### Upload Multiple Contracts

**Endpoint:** `POST /contracts/upload-multiple`

**Request:**
```
Content-Type: multipart/form-data

files: [PDF file, PDF file, ...]
project_id: UUID
user_id: UUID
```

**Response:**
```json
{
  "total_files": 3,
  "successful": 2,
  "failed": 1,
  "results": [
    {
      "filename": "contract1.pdf",
      "status": "success",
      "contract_id": "uuid",
      "total_chunks": 42
    },
    {
      "filename": "contract2.pdf",
      "status": "error",
      "error": "Invalid PDF format"
    }
  ]
}
```

### Delete Contract

**Endpoint:** `DELETE /contracts/{contract_id}`

**Request:**
```
Content-Type: multipart/form-data

user_id: UUID
```

**Response:**
```json
{
  "status": "success",
  "contract_id": "uuid",
  "message": "Contract and all associated data deleted successfully"
}
```

---

## Frontend Components

### ContractUploadModal

**Location:** `src/components/ContractUploadModal.tsx`

**Features:**
- Multi-file upload support
- Real-time upload progress
- Per-file status tracking
- Success/error indicators
- Chunk count display

**Usage:**
```tsx
<ContractUploadModal
  open={uploadModalOpen}
  onOpenChange={setUploadModalOpen}
  projectId={selectedProject}
  onUploadComplete={handleUploadComplete}
/>
```

### Zoe Page Integration

**Location:** `src/pages/Zoe.tsx`

**Features:**
- Upload button in contract section
- Delete button (hover to reveal) for each contract
- Confirmation dialog for deletions
- Auto-refresh contract list after upload/delete
- User authentication integration

---

## Database Schema

### project_files Table

```sql
CREATE TABLE project_files (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  project_id UUID REFERENCES projects(id),
  folder_category TEXT, -- 'contract', 'royalty_statement', etc.
  file_name TEXT,
  file_url TEXT,
  file_path TEXT,
  file_size BIGINT,
  file_type TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

---

## Vector Database Structure

### Pinecone Index

**Index Name:** `test-3-small-index` (regional)

**Namespace Format:** `{user_id}-namespace`

**Vector Metadata:**
```json
{
  "user_id": "uuid",
  "project_id": "uuid",
  "project_name": "Project Name",
  "contract_id": "uuid",
  "contract_filename": "file.pdf",
  "page_number": 1,
  "chunk_index": 0,
  "uploaded_at": "2025-12-07T23:30:00.000Z",
  "chunk_text": "The actual text content..."
}
```

**Vector ID:** SHA256 hash of (chunk_text + canonical_metadata_json)

---

## Key Benefits

### 1. Deterministic IDs
- Same content + metadata = same ID
- Enables idempotent uploads
- Prevents duplicate vectors
- Simplifies version management

### 2. Single Namespace Per User
- Simplified namespace management
- Efficient cross-project searches
- Reduced complexity
- Better scalability

### 3. Metadata-Based Filtering
- Fast contract deletion
- Precise search scoping
- No external state management
- Vector DB as single source of truth

### 4. Scalable Architecture
- Batch processing for efficiency
- Regional index support
- Handles multiple concurrent uploads
- Optimized for large documents

---

## Testing

### Manual Testing Steps

1. **Upload Single Contract:**
   - Select project in Zoe
   - Click "Upload" button
   - Select a PDF file
   - Verify upload progress
   - Confirm contract appears in list

2. **Upload Multiple Contracts:**
   - Select multiple PDFs
   - Verify individual progress tracking
   - Check success/failure counts

3. **Delete Contract:**
   - Hover over contract
   - Click delete icon
   - Confirm deletion
   - Verify contract removed from list
   - Verify vectors removed from Pinecone

4. **Search After Upload:**
   - Upload a contract
   - Ask Zoe a question about the content
   - Verify correct answers with sources

5. **Re-upload Contract:**
   - Upload same contract again
   - Verify no duplicates created
   - Verify updated content reflected

---

## Configuration

### Environment Variables

**Backend (.env):**
```
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_SECRET_KEY=your_supabase_secret_key
```

### Constants

**Backend (contract_ingestion.py):**
```python
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1024  # Character-based chunk size
CHUNK_OVERLAP = 154  # Character overlap
BATCH_SIZE = 20

REGIONAL_INDEXES = {
    "US": "test-3-small-index",
    "EU": "test-3-small-index",
    "UK": "test-3-small-index"
}
```

---

## Error Handling

### Upload Errors
- Invalid file type → User-friendly error message
- PDF parsing failure → Detailed error in response
- Embedding API failure → Retry logic (future enhancement)
- Storage failure → Rollback transaction

### Delete Errors
- Contract not found → 404 error
- Vector deletion failure → Error message with details
- Storage deletion failure → Warning logged, continues
- Database deletion failure → Transaction rollback

---

## Future Enhancements

1. **Batch Upload Optimization:**
   - Parallel processing of multiple files
   - Progress streaming via WebSockets

2. **Version Control:**
   - Track contract versions
   - Compare changes between versions
   - Rollback to previous versions

3. **Advanced Search:**
   - Cross-contract search
   - Temporal filtering (date ranges)
   - Similarity threshold adjustment

4. **Analytics:**
   - Upload statistics
   - Search performance metrics
   - Storage usage tracking

---

## Troubleshooting

### Common Issues

**Issue:** Uploads fail silently
- **Solution:** Check backend logs, verify API keys, ensure Pinecone index exists

**Issue:** Deletions don't remove vectors
- **Solution:** Verify namespace format, check user_id matches, ensure metadata filter syntax

**Issue:** Duplicate vectors created
- **Solution:** Verify deterministic ID generation, check metadata consistency

**Issue:** Search returns no results
- **Solution:** Verify namespace format matches upload, check metadata filters, confirm vectors uploaded

---

## Conclusion

This implementation provides a robust, scalable contract upload and deletion system with:
- ✅ Deterministic hash IDs for uniqueness
- ✅ Single namespace per user for simplicity
- ✅ Efficient metadata-based filtering
- ✅ Clean separation of concerns
- ✅ User-friendly UI with progress tracking
- ✅ Comprehensive error handling

The system is production-ready and follows best practices for vector database management and document processing.
