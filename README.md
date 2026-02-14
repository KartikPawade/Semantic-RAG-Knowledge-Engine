# Semantic-RAG-Knowledge-Engine

A production-ready **Retrieval-Augmented Generation (RAG)** API: ingest PDFs and text, run **semantic search**, and get answers grounded only in your documents. Built with **embeddings**, **ChromaDB**, **Ollama** (llama3 + nomic-embed-text), and **LCEL**.

---

## Use Cases

| Use case | How it’s supported |
|----------|---------------------|
| **Document ingestion** | Upload PDFs or text → chunk (recursive character) → embed (Ollama) → store in ChromaDB. Separate collections (e.g. `hr_manual`, `engineering_docs`) for different domains. |
| **Semantic search** | `POST /search`: embed the query, run similarity search over stored chunks, return snippets + scores (no LLM). Good for testing retrieval. |
| **Grounded Q&A (RAG)** | `POST /ask`: retrieve relevant chunks → augment prompt with context → LLM answers only from context; says “I cannot find that in the manual” when not in docs. |
| **Advanced RAG – Query Expansion** | `POST /ask` with `use_query_expansion: true`: expand the question into 2–3 alternative phrasings → retrieve for each → merge/dedupe docs → same RAG answer. Improves recall when user wording differs from the document. |
| **Collection management** | Ingest/search/ask target a collection (default or per request). `DELETE /clear` wipes one collection for testing. |
| **Health / ops** | `GET /status`: check Ollama and ChromaDB availability. |

---

## Concepts Used in This Project

### 1. Embeddings (the “language of numbers”)

- Text is converted into **vectors** (lists of numbers) so we can compare meaning, not just keywords.
- **Normalization** keeps vector length consistent so distances (e.g. between “Paternity Leave” and “Childcare”) are comparable across chunks of different length.
- This project uses **Ollama** with the **nomic-embed-text** model for local embeddings.

### 2. Vector database (ChromaDB)

- Vectors are stored in **ChromaDB**, which acts as a semantic search index.
- **Persistence:** Data is saved to disk (`./chroma_db`) so you don’t re-ingest PDFs after a server restart.
- **Collections:** Data is grouped into named collections (e.g. `hr_manual`, `engineering_docs`) so you can separate HR content from other domains and search per collection.

### 3. RAG (Retrieval-Augmented Generation)

- The model does **not** answer from its own training; it answers using **retrieved chunks** from your documents.
- Flow: user question → **retrieve** relevant chunks from the vector DB → **augment** the prompt with that context → **generate** an answer from the LLM. That’s RAG.

### 4. Recursive character chunking

- A full PDF is too large to send as one “thought”; it’s split into **chunks** (default 1,000 characters).
- **Chunk overlap** (default 200 characters): the last 200 characters of one chunk are repeated at the start of the next so information at chunk boundaries (e.g. a key date) isn’t lost.
- Implemented with **RecursiveCharacterTextSplitter** (LangChain).

### 5. Similarity score and thresholding

- Vector search returns a **score** (distance: lower = more similar).
- **Filtering:** If the best match is below a configured **similarity threshold**, we treat it as “no relevant context” and the system is instructed to say it cannot find the answer in the manual instead of guessing.

### 6. System instructions (grounding)

- A **system prompt** constrains the LLM: answer **only** from the provided context; if the answer isn’t there, say “I cannot find that in the manual.” This keeps answers grounded and avoids hallucination.

### 7. LCEL (LangChain Expression Language)

- The RAG pipeline is built as a **chain** using LCEL: retriever → format docs → prompt → LLM → output. Composable and easy to extend.

### 8. Query Expansion (Advanced RAG)

- **Problem:** A user might ask “How much paternity leave do I get?” while the document says “Fathers are entitled to 2 weeks paid leave.” Single-query retrieval can miss the chunk if phrasing differs.
- **Query Expansion:** The LLM generates 2–3 alternative phrasings (rephrasing, synonyms, or sub-questions) from the user question. We run **retrieval for each** expanded query, **merge and deduplicate** the documents, then pass the combined context to the same RAG prompt and LLM. This improves **recall** without changing how the answer is grounded.
- **Usage:** Set `use_query_expansion: true` in the `POST /ask` request body. Config: `QUERY_EXPANSION_MAX_QUERIES` (default 3).

---

## APIs and Their Flow

### Overview

| # | Endpoint   | Method | Purpose |
|---|------------|--------|---------|
| 1 | `/ingest`  | POST   | Upload files → chunk, embed, store in ChromaDB. |
| 2 | `/search`  | POST   | Semantic search over stored chunks (snippets + scores). |
| 3 | `/ask`     | POST   | Full RAG: search + LLM answer (grounded in your docs). |
| 4 | `/clear`   | DELETE | Wipe a collection (reset for testing). |
| 5 | `/status`  | GET    | Health check: Ollama and ChromaDB. |

---

### 1. `POST /ingest`

**Purpose:** Turn PDFs or text files into searchable vectors.

**Flow:**

1. Accept one or more files (multipart form: `files`; optional query param: `collection`).
2. Save files temporarily, load with **PyPDFLoader** (PDF) or **TextLoader** (.txt).
3. Split into chunks with **RecursiveCharacterTextSplitter** (chunk_size=1000, overlap=200).
4. Embed chunks with **Ollama** (nomic-embed-text) and add to **ChromaDB** in the given (or default) collection.
5. Return `chunks_added` and `files_processed`; temp files are deleted.

**Request:** Multipart form with `files`; optional `?collection=hr_manual`.

**Response:** `{ "status": "ok", "chunks_added": 12, "files_processed": 1 }`

---

### 2. `POST /search`

**Purpose:** Test semantic search: get relevant text snippets (no LLM).

**Flow:**

1. Accept JSON body: `query`, optional `collection`, optional `k` (number of results, 1–20).
2. Embed the query with the same Ollama embedding model.
3. Run **similarity search** in ChromaDB (default collection if not specified).
4. Return matching chunks with **content**, **score** (distance; lower = more similar), and **metadata**.

**Request body:** `{ "query": "paternity leave", "collection": null, "k": 5 }`

**Response:** `{ "query": "paternity leave", "snippets": [ { "content": "...", "score": 0.32, "metadata": {} } ] }`

---

### 3. `POST /ask`

**Purpose:** Full RAG: answer a question using only your ingested documents. Optional **Query Expansion** for better recall.

**Flow:**

1. Accept JSON body: `question`, optional `collection`, optional `use_query_expansion` (default `false`).
2. If `use_query_expansion` is `true`: **expand** the question into 2–3 alternative queries (LLM), **retrieve** for each query, **merge and dedupe** chunks.
3. Otherwise: **Retrieve** top-k chunks above the similarity threshold (retriever with score_threshold).
4. **Format** chunks into a single context string.
5. **Prompt** the LLM (Ollama llama3) with system instructions: “Use only the provided context; if not found, say I cannot find that in the manual.”
6. **Generate** and return the answer.

**Request body (basic):** `{ "question": "How much paternity leave do I get?", "collection": null }`

**Request body (with Query Expansion):** `{ "question": "How much paternity leave do I get?", "collection": null, "use_query_expansion": true }`

**Response:** `{ "question": "...", "answer": "Paternity Leave is 2 weeks paid leave..." }`

---

### 4. `DELETE /clear`

**Purpose:** Wipe a collection (e.g. for testing).

**Flow:** Delete the named (or default) collection from ChromaDB. No request body; optional query param: `?collection=hr_manual`.

**Response:** `{ "status": "ok", "message": "Collection 'hr_manual' cleared." }`

---

### 5. `GET /status`

**Purpose:** Check if Ollama and ChromaDB are reachable.

**Flow:** Call Ollama API (e.g. `/api/tags`) and ChromaDB heartbeat.

**Response:** `{ "ollama": "online" | "offline", "chromadb": "online" | "offline" }`

---

## Setup Guide

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running (for llama3 and nomic-embed-text)

### 1. Clone and go to the project

```bash
git clone https://github.com/YOUR_USERNAME/Semantic-RAG-Knowledge-Engine.git
cd Semantic-RAG-Knowledge-Engine
```

(Or use your actual repo URL and folder name.)

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Cmd):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**WSL / Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your prompt.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ollama: run server and pull models

Ensure Ollama is running (e.g. `ollama serve` or the Ollama app), then pull both models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

Verify with: `ollama list`

### 5. Environment file

```bash
cp .env.example .env
```

Edit `.env` if needed. Defaults (Ollama on localhost) are usually fine:

- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_LLM_MODEL=llama3`
- `OLLAMA_EMBEDDING_MODEL=nomic-embed-text`

If the app runs on **Windows** and Ollama is in **WSL**, and `localhost` doesn’t work, set `OLLAMA_BASE_URL` to your WSL IP (e.g. `http://172.x.x.x:11434`).

### 6. (Optional) Generate a sample HR PDF

```bash
python scripts/generate_sample_hr_pdf.py
```

This creates `data/hr_manual_sample.pdf` for testing ingest and `/ask`.

### 7. Run the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- **API docs (Swagger):** http://localhost:8000/docs  
- **Health:** http://localhost:8000/status  

### 8. Quick test

1. **GET** http://localhost:8000/status → expect `"ollama": "online"` and `"chromadb": "online"`.
2. **POST /ingest** → upload `data/hr_manual_sample.pdf` (or any PDF).
3. **POST /ask** → body: `{"question": "How much paternity leave do I get?"}`.

---

## Project Layout

```
Semantic-RAG-Knowledge-Engine/
├── main.py              # FastAPI app and 5 endpoints
├── config.py            # Settings (Ollama, chunk size, threshold, paths)
├── app/
│   ├── embeddings.py       # Ollama embeddings (nomic-embed-text)
│   ├── vector_store.py     # ChromaDB: persistence, collections, retriever
│   ├── chunking.py         # RecursiveCharacterTextSplitter (1000 / 200)
│   ├── ingestion.py        # PyPDFLoader + chunk + store
│   ├── prompts.py          # System instructions (grounding)
│   ├── query_expansion.py  # Advanced RAG: expand question into multiple queries
│   ├── rag.py              # LCEL RAG chain (+ query expansion chain)
│   └── llm.py              # Chat model (Ollama llama3)
├── scripts/
│   └── generate_sample_hr_pdf.py
├── data/                # Optional sample PDFs
├── requirements.txt
├── .env.example
└── README.md
```

Runtime-created (and in `.gitignore`): `uploads/`, `chroma_db/`, `.venv/`, `.env`.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `OLLAMA_LLM_MODEL` | `llama3` | Model for `/ask` (RAG answers). |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Model for embeddings (ingest + search). |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB data directory. |
| `DEFAULT_COLLECTION` | `hr_manual` | Default collection name. |
| `CHUNK_SIZE` | `1000` | Characters per chunk. |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks. |
| `SIMILARITY_THRESHOLD` | `0.2` | Min relevance; below this, RAG can refuse to answer. |
| `QUERY_EXPANSION_MAX_QUERIES` | `3` | Max alternative queries when using Query Expansion in `/ask`. |

---

## Collections

- **Ingest** and **search/ask** use a **collection name** (default from config, or overridden per request).
- Use the **same** collection name when you ingest and when you search or ask, so the app looks in the right place.
- **Clear** deletes only the specified (or default) collection, so you can reset one dataset without touching others.
