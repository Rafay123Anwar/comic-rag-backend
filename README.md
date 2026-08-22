# Comic RAG Backend

A multimodal Retrieval-Augmented Generation (RAG) backend engine for ingesting comic books (CBR, CBZ, PDF, JPG, PNG, WEBP), transcribing & analyzing page contents via Mistral AI Vision, indexing text chunks in ChromaDB with cosine embeddings, managing multi-turn conversational chat memory, and answering user questions strictly grounded in comic evidence.

---

## 1. Architecture Overview

```
[Comic File Upload (CBR/CBZ/PDF/Images)]
                   │
                   ▼
       [Page Extractor (7-Zip / PyMuPDF)]
                   │
                   ▼
     [Vision & OCR Analysis (Mistral AI)]
                   │
                   ▼
       [Structured Storage (comic.json)]
                   │
                   ▼
[RAG Ingestion: Documents ➔ Chunks ➔ Embeddings]
                   │
                   ▼
       [ChromaDB Vector Store (cosine)]
                   ▲
                   │
  [Conversation History + Query Normalization]
                   │
       [Deterministic Retrieval & Sort]
                   │
        [Strict Grounding Prompt]
                   │
    [Answer & Source Validation]
                   │
                   ▼
      [Frontend-Ready Chat API Output]
```

---

## 2. Directory Structure

```
Comic-rag/
├── .env
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── run.py
│
├── app/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application & global exception handler
│   ├── config.py                 # Backwards-compatible configuration re-export proxy
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Centralized settings & environment variables
│   │   └── logging.py            # Standardized application logger
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── comic.py              # Upload, details, and comic QA endpoints
│   │   └── conversation.py       # Conversation creation, details, deletion, and chat QA
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── comic.py              # Pydantic models for comics and QA
│   │   └── conversation.py       # Pydantic models for conversations & chat
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ai_analyzer.py        # Multimodal page vision and OCR analysis
│   │   ├── conversation.py       # JSON-based conversation history persistence
│   │   ├── embedding.py          # Mistral embedding generator
│   │   ├── extractor.py          # Archive, PDF, and image page extractor
│   │   ├── llm.py                # Grounded answer generation
│   │   ├── query_normalizer.py   # Whitespace query normalization
│   │   ├── rag_chunker.py        # Deterministic text chunking
│   │   ├── rag_ingestion.py      # ChromaDB ingestion pipeline
│   │   ├── rag_preprocessor.py   # Page to RAG document converter
│   │   ├── rag_qa.py             # Retrieval QA orchestration & validation
│   │   ├── retriever.py          # Vector search, filtering & deterministic sorting
│   │   ├── storage.py            # Local JSON persistence
│   │   └── vector_store.py       # ChromaDB client & collection management
│   │
│   ├── models/
│   │   └── __init__.py
│   │
│   └── utils/
│       └── __init__.py
│
├── tests/
│   ├── __init__.py
│   └── verification/
│       ├── __init__.py
│       ├── verify_m11.py         # Query normalization verification
│       ├── verify_m12.py         # Multi-page context assembly verification
│       ├── verify_m13.py         # Live grounding & source integrity verification
│       ├── verify_m14.py         # Consistency validation verification
│       ├── verify_m15.py         # API error handling & input validation
│       ├── verify_m16.py         # API contracts & OpenAPI schema verification
│       └── verify_m18.py         # Conversation memory & chat API verification
│
├── notebooks/
│   └── comic.ipynb               # Exploratory prototyping & OCR prompt testing
│
└── storage/
    ├── comics/                   # Analyzed comic pages and comic.json files
    ├── uploads/                  # Uploaded raw archives and images
    ├── vector_db/                # ChromaDB SQLite and HNSW vector index
    └── conversations/            # Persistent conversation session JSON files
```

---

## 3. Setup & Installation

### 1. Prerequisites
- Python 3.11+
- [7-Zip](https://www.7-zip.org/) (for extracting CBR/CBZ archives on Windows)
- Mistral AI API Key

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and set your Mistral API key:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL=mistral-small-2603
LLM_MODEL=mistral-small-latest
EMBEDDING_MODEL=mistral-embed
STORAGE_PATH=storage
MAX_CONVERSATION_MESSAGES=10
MAX_AI_WORKERS=3
MAX_AI_RETRIES=5
SEVEN_ZIP_PATH=C:\Program Files\7-Zip\7z.exe
```

---

## 4. Running the Backend Server

Start the FastAPI application with Uvicorn:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **OpenAPI JSON Contract**: `http://127.0.0.1:8000/openapi.json`

---

## 5. API Endpoints

### Comic Ingestion & QA Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/comics/upload` | Upload a comic file for extraction, analysis, and RAG ingestion. |
| `GET` | `/comics/{comic_id}` | Retrieve analyzed metadata and pages for a specific comic. |
| `POST` | `/comics/{comic_id}/ask` | Ask a question about a specific comic (path-scoped). |
| `POST` | `/comics/ask` | Ask a question about a comic (specifying `comic_id` in request body). |
| `POST` | `/ask` | Root-level convenience endpoint for asking questions. |

### Conversation & Chat Memory Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/conversations` | Create a new conversation session tied to a `comic_id`. |
| `GET` | `/conversations/{conversation_id}` | Retrieve conversation metadata and chronological message history. |
| `DELETE` | `/conversations/{conversation_id}` | Safely delete a conversation session and its stored history. |
| `POST` | `/conversations/{conversation_id}/ask` | Ask a question in the context of an existing conversation. |

---

## 6. Frontend Integration Contract

This section explains how frontend client applications should integrate with the backend API.

### A. Lifecycle & Integration Flow

```
1. UPLOAD COMIC
   POST /comics/upload  ➔  Receive comic_id

2. CREATE CONVERSATION SESSION
   POST /conversations  ➔  Receive conversation_id

3. INITIAL CHAT MESSAGE
   POST /conversations/{conversation_id}/ask
   { "question": "What did Victor say about his mother?" }
   ➔ Receive answer + sources

4. FOLLOW-UP CHAT MESSAGE (Pronouns / References)
   POST /conversations/{conversation_id}/ask
   { "question": "What did he say about her next?" }
   ➔ System uses conversation memory for reference resolution
   ➔ Strictly grounds answer in retrieved comic chunks
```

### B. Endpoint Schemas & Examples

#### 1. Upload Comic (`POST /comics/upload`)
- **Content-Type**: `multipart/form-data`
- **Field**: `file` (Binary file of type `.cbr`, `.cbz`, `.pdf`, `.jpg`, `.jpeg`, `.png`, `.webp`)

**Response (200 OK):**
```json
{
  "comic_id": "dbb3835c-5445-4986-9c6f-c6e2d2fc81c9",
  "comic_name": "sample_comic",
  "source_format": "cbz",
  "total_pages": 12,
  "successful_pages": 12,
  "failed_pages": 0,
  "rag_ingested": true,
  "rag_chunks_stored": 24,
  "rag_error": null,
  "message": "Comic processed and indexed successfully."
}
```

#### 2. Create Conversation (`POST /conversations`)
- **Request Body**:
```json
{
  "comic_id": "dbb3835c-5445-4986-9c6f-c6e2d2fc81c9"
}
```

**Response (200 OK):**
```json
{
  "conversation_id": "c4838a57-dfb6-4efc-8d80-7ffe010be7d5",
  "comic_id": "dbb3835c-5445-4986-9c6f-c6e2d2fc81c9",
  "created_at": "2026-08-15T01:50:00.000000+00:00",
  "updated_at": "2026-08-15T01:50:00.000000+00:00"
}
```

#### 3. Ask Question in Conversation (`POST /conversations/{conversation_id}/ask`)
- **Request Body**:
```json
{
  "question": "What did Victor say about his mother?"
}
```

**Response (200 OK):**
```json
{
  "conversation_id": "c4838a57-dfb6-4efc-8d80-7ffe010be7d5",
  "comic_id": "dbb3835c-5445-4986-9c6f-c6e2d2fc81c9",
  "question": "What did Victor say about his mother?",
  "answer": "Victor said that his mother never doubted him despite his father's disbelief, and that she always knew he was different.",
  "sources": [
    {
      "comic_id": "dbb3835c-5445-4986-9c6f-c6e2d2fc81c9",
      "page_number": 1,
      "chunk_id": "dbb3835c-5445-4986-9c6f-c6e2d2fc81c9_page_1_chunk_1",
      "chunk_index": 1,
      "distance": 0.28325
    }
  ]
}
```

#### 4. Fallback Handling
When no evidence is found in the comic, the backend returns:
```json
{
  "conversation_id": "c4838a57-dfb6-4efc-8d80-7ffe010be7d5",
  "comic_id": "dbb3835c-5445-4986-9c6f-c6e2d2fc81c9",
  "question": "What is Victor's favorite food?",
  "answer": "I could not find relevant information in the comic.",
  "sources": []
}
```
*Frontend Rendering Guideline*: Render the answer directly and hide the source cards accordion/list when `sources` is empty.

#### 5. Empty Question Handling
When the user submits empty or whitespace input:
```json
{
  "conversation_id": "c4838a57-dfb6-4efc-8d80-7ffe010be7d5",
  "comic_id": "dbb3835c-5445-4986-9c6f-c6e2d2fc81c9",
  "question": "   ",
  "answer": "Please provide a valid question.",
  "sources": []
}
```

#### 6. Error Handling Contract
All API errors return a standard JSON object containing a `detail` field:
- **400 Bad Request**: Invalid UUID format, empty file, or unsupported file extension.
  ```json
  { "detail": "Invalid conversation_id format. Expected a valid UUID." }
  ```
- **404 Not Found**: Non-existent comic or conversation.
  ```json
  { "detail": "Conversation not found" }
  ```
- **500 Internal Server Error**: Sanitized message protecting internal server details.
  ```json
  { "detail": "An unexpected server error occurred." }
  ```

---

## 7. RAG Pipeline Invariants & Configuration

- **ChromaDB Collection**: `comic_pages`
- **Distance Metric**: `cosine` (`hnsw:space = cosine`)
- **Distance Threshold**: `0.35` (chunks with distance `> 0.35` are discarded)
- **Default Top K**: `5`
- **Deterministic Ordering**: Chunks are ordered by `(page_number ASC, chunk_index ASC, chunk_id ASC)`
- **Chunk Sizing**: `chunk_size = 1000`, `chunk_overlap = 150`
- **Context Markers**: `[PAGE X | CHUNK Y]` with separator `--------------------`
- **Grounding Pinned Fallback**: `"I could not find relevant information in the comic."`
- **Grounding Rule**: Comic chunks are the authoritative source of truth. Conversation history is utilized solely for follow-up reference resolution.

---

## 8. Running Automated Verification Suites

Execute the milestone verification test suites from the project root:

```bash
python -u tests/verification/verify_m11.py   # Query normalization (28 checks)
python -u tests/verification/verify_m12.py   # Multi-page context assembly (29 checks)
python -u tests/verification/verify_m13.py   # Live grounding & source integrity (55 checks)
python -u tests/verification/verify_m14.py   # Consistency validation (47 checks)
python -u tests/verification/verify_m15.py   # API error handling & validation (59 checks)
python -u tests/verification/verify_m16.py   # API contracts & OpenAPI schema (60 checks)
python -u tests/verification/verify_m18.py   # Conversation memory & chat API (77 checks)
```
