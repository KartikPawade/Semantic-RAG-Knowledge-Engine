# 🚀Semantic-RAG-Knowledge-Engine

A production-ready **Advanced RAG** API with **metadata filtering**, **schema-driven routing**, and **dynamic Pydantic validation**. Ingest PDFs and text via **autonomous ingestion**, run **semantic search** with **collection routing** and **metadata filters**, and get answers grounded only in your documents. Built with **embeddings**, **ChromaDB**, **Ollama** (llama3 + nomic-embed-text), **LCEL**, and a **Schema Registry** for multi-collection **metadata filtering**.

**Keywords:** Advanced RAG · Metadata filtering · Dynamic Pydantic validation · Query expansion · Schema-driven routing · Autonomous ingestion · Background workers · RabbitMQ · Idempotency · Content hashing · Worker pattern · ChromaDB · Ollama · LCEL · Retrieval-Augmented Generation · Semantic search · Multi-collection · Metadata extraction · Filter extraction · Grounded Q&A

---

## Use Cases

| Use case | How it's supported |
|----------|---------------------|
| **Background ingestion (Worker pattern)** | Upload files → land in storage → **RabbitMQ** queue → **worker** picks up → **idempotency** (content hash; skip if duplicate) → parse, chunk, **metadata extraction**, embed → push to ChromaDB. If a 1,000-page PDF crashes the parser, the rest of the system keeps running. |
| **Semantic search with metadata filtering** | `POST /search`: **collection routing** (LLM) → **schema-aware filter extraction** (LLM + **dynamic Pydantic validation**) → **metadata filtering** in ChromaDB → return snippets + scores. |
| **Advanced RAG with Query Expansion** | `POST /ask` with `use_query_expansion: true`: **query expansion** (2–3 alternative phrasings) → retrieve for each → merge/dedupe → RAG answer. Improves recall. |
| **Advanced RAG with metadata filtering** | `POST /ask`: **collection routing** → **filter extraction** (Schema Registry + **dynamic Pydantic validation**) → retrieval with **metadata filters** → **schema hints** in system prompt → grounded answer. |
| **Schema Registry (multi-collection metadata)** | **Schema-driven routing**: each collection has defined metadata fields (e.g. `city`, `department`, `product_id`, `region`), **schema hints** for the AI, and **value normalizers** to avoid zero-result errors. |
| **Health / ops** | `GET /status`: check Ollama and ChromaDB availability. |
| **Collection clear** | `DELETE /clear`: wipe a collection (optional query param). |

---

## 🏗️Concepts Used in This Project

### 1. Advanced RAG (Retrieval-Augmented Generation)

- **RAG** = retrieve relevant chunks → augment prompt with context → LLM answers only from context.
- **Advanced RAG** in this app: **Query Expansion** (multiple phrasings → merge/dedupe), **metadata filtering** (schema-driven filters), **collection routing** (LLM picks collection), and **schema hints** in the system prompt for grounded, filter-aware answers.

### 2. Metadata filtering & Schema-driven routing

- **Metadata filtering**: Search and RAG restrict results by metadata (e.g. `city=NY`, `department=HR`, `product_id=A99`) so users see only relevant docs (e.g. policy for their office, product for their region).
- **Schema Registry**: A code-level map of each collection to its **allowed metadata fields**, **schema hints** for the AI, and **filter strategy**. Decouples metadata schema from LLM logic.
- **Two-step extraction**: (1) **Collection routing** — LLM chooses which collection fits the query. (2) **Schema-aware filter extraction** — LLM extracts only that collection’s fields from the user query; **dynamic Pydantic validation** and **value normalizers** (e.g. "New York" → "NY") avoid zero-result errors.

### 3. Dynamic Pydantic validation

- **Dynamic Pydantic models**: The app builds a temporary Pydantic model per collection from the **Schema Registry** (e.g. `city: Optional[str]`, `department: Optional[str]`). The LLM’s filter JSON is validated and normalized with this model before building the Chroma **where** clause.
- Ensures only **allowed fields** and valid types are used for **metadata filtering**; invalid or unknown values are dropped (None fallback).

### 4. Query Expansion (Advanced RAG)

- **Problem:** User phrasing may differ from document wording; single-query retrieval can miss relevant chunks.
- **Query Expansion:** LLM generates 2–3 alternative phrasings (rephrasing, synonyms, sub-questions). We **retrieve for each** query, **merge and deduplicate** chunks, then run the same RAG prompt. Improves **recall**.
- **Usage:** Set `use_query_expansion: true` in `POST /ask`. Config: `QUERY_EXPANSION_MAX_QUERIES` (default 3).

### 5. Background workers (RabbitMQ) & Worker pattern

- **Flow:** A file lands in storage (`uploads/pending`) → API publishes a task to **RabbitMQ** → returns **202 Accepted** immediately. A **worker** process consumes the queue → picks up the file → **idempotency check** (content hash) → if new: parse, chunk, tag (metadata), push vectors to ChromaDB → record hash; if duplicate: skip. This ensures that if a 1,000-page PDF crashes the parser, only that job fails; the API and other workers keep running.
- **Worker pattern:** Decouple heavy lifting from the API. Run one or more workers with `python worker.py` from the project root (and RabbitMQ running). No collection parameter in the request; routing and metadata extraction happen inside the worker.

### 6. Idempotency & hashing

- Before processing, the worker computes a **SHA-256 hash** of the file content. If the hash exists in the **processed hashes** store (SQLite), the file is **skipped** to prevent duplicate data from poisoning search results. If you upload the same file twice, the second run is a no-op.
- Processed hashes are stored in `data/processed_hashes.db` (configurable). Table: `content_hash`, `filename`, `collection_name`, `created_at`.

### 7. Autonomous ingestion flow (inside the worker)

- **Read & sample:** First 1,000 words of each document.
- **Classify:** LLM compares content to **existing collections** (from ChromaDB); returns an existing collection name, a **new** collection name (snake_case, `_collection` suffix), or **UNCLASSIFIED**.
- **Metadata extraction (Schema Registry):** If the collection has a schema (e.g. `policy_collection` → `city`, `department`), a small LLM call extracts those fields from the document and attaches them to **every chunk** (automatic metadata enrichment).
- **Route:** Ingest into the chosen collection; **dynamic collection creation** if the name is new; **fallback** to **unclassified_knowledge** if UNCLASSIFIED.

### 8. Embeddings & vector database (ChromaDB)

- Text → **vectors** via **Ollama** (**nomic-embed-text**). **ChromaDB** stores vectors with **persistence** (`./chroma_db`). **Collections** separate domains (policies, products, unclassified). **Metadata** on chunks enables **metadata filtering**.

### 9. Recursive character chunking

- **Chunk size** 1,000 characters, **overlap** 200 so boundary information isn’t lost. **RecursiveCharacterTextSplitter** (LangChain).

### 10. Similarity score & thresholding

- Chroma returns **distance** (lower = more similar). **Score threshold** filters out weak matches; RAG says "I cannot find that in the manual" when no chunk passes.

### 11. System instructions & schema hints

- **Grounding:** System prompt instructs the LLM to use **only** the provided context. **Schema hints** (from the Schema Registry) are injected for the chosen collection so the AI knows which **metadata filters** apply (e.g. "Use city and department when the user mentions location or team").

### 12. LCEL (LangChain Expression Language)

- RAG and query-expansion chains are built with **LCEL**: retriever → format docs → prompt → LLM → output. Composable and easy to extend.

---

## 🔗APIs and Their Flow

### Overview

| # | Endpoint   | Method | Purpose |
|---|------------|--------|---------|
| 1 | `/ingest`  | POST   | **Background ingestion**: save file → publish to **RabbitMQ** → **202 Accepted**. Worker does idempotency (hash), then classify → metadata extraction → chunk → embed → store. |
| 2 | `/search`  | POST   | **Semantic search** with **collection routing** + **metadata filtering** (schema-aware filter extraction). |
| 3 | `/ask`     | POST   | **Advanced RAG**: collection routing, **metadata filtering**, optional **Query Expansion**, **schema hints**. |
| 4 | `/clear`   | DELETE | Wipe a collection (optional `?collection=...`). |
| 5 | `/status`  | GET    | Health: Ollama and ChromaDB. |

---

### 1. `POST /ingest` (Background ingestion — Worker pattern)

**Purpose:** Upload files and **queue** them for background processing. Returns **202 Accepted** immediately. A **worker** (RabbitMQ consumer) does the heavy lifting: **idempotency** (content hash; skip duplicates), then classify → metadata extraction → chunk → embed → push to ChromaDB. If a large PDF crashes the parser, the API and other jobs keep running.

**Flow (API):**

1. Accept one or more files (multipart form: `files`). **No collection parameter**.
2. Save each file to `uploads/pending/{task_id}_{filename}`.
3. Publish one message per file to **RabbitMQ** queue `ingestion_tasks` (body: `task_id`, `file_path`, `filename`).
4. Return **202 Accepted** with `tasks` (task_id and file name per file). If RabbitMQ is unavailable, return **503**.

**Flow (Worker):**

1. Consume message from queue; resolve file path (relative to project root).
2. **Idempotency:** Compute **SHA-256** hash of file content. If hash is in `processed_hashes` (SQLite), **skip** and ack (prevents duplicate data from poisoning search).
3. Otherwise: load file → **classify** (LLM, first 1,000 words) → **metadata extraction** (Schema Registry) → chunk → embed → add to ChromaDB → **record hash** → delete file → ack.
4. On parser/LLM failure: nack (no requeue) so the message is not retried indefinitely.

**Request:** Multipart form with `files` only.

**Response (202 Accepted):** `{ "status": "accepted", "message": "Files queued for ingestion. A background worker will process them.", "tasks": [ { "task_id": "abc123", "file": "policy.pdf" } ] }`

**Prerequisites:** RabbitMQ running (e.g. `docker run -d -p 5672:5672 rabbitmq:3`). At least one worker: `python worker.py` (run from project root).

---

### 2. `POST /search` (Semantic search + metadata filtering)

**Purpose:** **Semantic search** with **collection routing** and **schema-aware metadata filtering**. Two-step extraction: route to collection, then extract filters from the query.

**Flow:**

1. **Collection routing:** LLM selects one collection from ChromaDB’s list (or **unclassified_knowledge**).
2. **Filter extraction:** LLM extracts filter values from the query using that collection’s **Schema Registry** schema; **dynamic Pydantic validation** and **value normalizers**; build Chroma **where** clause (or none to avoid zero-result).
3. **Similarity search** in that collection with optional **metadata filter**; return snippets with **content**, **score**, **metadata** (including `collection`).

**Request body:** `{ "query": "HR policy in New York", "k": 5 }`

**Response:** `{ "query": "...", "collection": "policy_collection", "snippets": [ { "content": "...", "score": 0.32, "metadata": { "collection": "policy_collection", "city": "NY", "department": "HR" } } ] }`

---

### 3. `POST /ask` (Advanced RAG + Query Expansion + metadata filtering)

**Purpose:** Full **Advanced RAG**: **collection routing**, **metadata filtering**, **schema hints** in the system prompt. Optional **Query Expansion** for better recall.

**Flow:**

1. **Collection routing** and **schema-aware filter extraction** (same as search).
2. **Retriever** with **metadata filter** and similarity threshold.
3. If `use_query_expansion: true`: **Query Expansion** — 2–3 alternative queries → retrieve for each → **merge and dedupe** chunks.
4. **Schema hints** for the chosen collection are injected into the system prompt.
5. **RAG chain:** format context → prompt (grounding + schema hints) → LLM → answer.

**Request body (basic):** `{ "question": "What is the paternity leave policy in NY?" }`

**Request body (with Query Expansion):** `{ "question": "What is the paternity leave policy in NY?", "use_query_expansion": true }`

**Response:** `{ "question": "...", "collection": "policy_collection", "answer": "..." }`

---

### 4. `DELETE /clear`

**Purpose:** Wipe a collection. Optional query param: `?collection=hr_manual`.

**Response:** `{ "status": "ok", "message": "Collection 'hr_manual' cleared." }`

---

### 5. `GET /status`

**Purpose:** Check Ollama (and that configured LLM/embedding models are pulled), and ChromaDB availability.

**Response:** `{ "ollama": "online" | "offline", "models": "ok" | "missing: [list]", "chromadb": "online" | "offline" }`

---

## 📋Schema Registry (multi-collection metadata)

The **Schema Registry** (`app/schema_registry.py`) is the "system map" for **metadata filtering** and **schema-driven routing**.

| Collection                    | Metadata fields     | Schema hint (for AI) |
|------------------------------|---------------------|-----------------------|
| **policy_collection**        | `city`, `department`| Use city and department when the user mentions location or team. |
| **product_catalog_collection**| `product_id`, `region` | If a product code is mentioned (e.g. A99), extract into product_id; filter by region if user specifies location. |
| **unclassified_knowledge**    | (none)              | No specific filters; semantic search only. |

- **Dynamic Pydantic validation:** `build_filter_model(collection_name)` creates an optional-field Pydantic model from the registry for **filter extraction** validation.
- **Value normalizers:** e.g. `"New York"` → `"NY"` to match stored metadata and avoid zero-result errors (configurable in `VALUE_NORMALIZERS`).
- **Schema hints** are passed into the RAG system prompt so the model is aware of which **metadata filters** apply.

---

## ⚙️Setup Guide

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running (for **llama3** and **nomic-embed-text**)

### 1. Clone and enter the project

```bash
git clone https://github.com/YOUR_USERNAME/Semantic-RAG-Knowledge-Engine.git
cd Semantic-RAG-Knowledge-Engine
```

### 2. Virtual environment

**Windows (PowerShell)** — if scripts are disabled, run once in that session:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

Then:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Or run without activating:**

```powershell
.venv\Scripts\python.exe main.py
```

**Windows (Cmd):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ollama: run server and pull models

```bash
ollama pull llama3
ollama pull nomic-embed-text
ollama list
```

### 5. Environment

```bash
cp .env.example .env
```

Edit `.env` if needed. Defaults: `OLLAMA_BASE_URL`, `OLLAMA_LLM_MODEL`, `OLLAMA_EMBEDDING_MODEL`, `DEFAULT_FALLBACK_COLLECTION=unclassified_knowledge`, etc.

### 6. RabbitMQ (for background ingestion)

Start RabbitMQ (required for `POST /ingest` to queue tasks). Example with Docker:

```bash
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

Default URL: `amqp://guest:guest@localhost:5672/`. Override with `RABBITMQ_URL` in `.env` if needed.

### 7. (Optional) Sample HR PDF

```bash
python scripts/generate_sample_hr_pdf.py
```

### 8. Run the API

```bash
python main.py
```

Or:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- **API docs (Swagger):** http://localhost:8000/docs  
- **Health:** http://localhost:8000/status  

### 9. Run the ingestion worker

From the **project root**, start at least one worker so queued files are actually processed:

```bash
python worker.py
```

Leave it running. It consumes from the `ingestion_tasks` queue, hashes files for **idempotency**, then parses, chunks, and pushes to ChromaDB.

### 10. Quick test

1. **GET** http://localhost:8000/status → expect `ollama` and `chromadb` online.
2. **POST /ingest** → upload a PDF; expect **202 Accepted** and `tasks` with `task_id`. (Worker must be running to process the file.)
3. **POST /search** → `{ "query": "paternity leave", "k": 5 }` (collection and **metadata filtering** resolved automatically).
4. **POST /ask** → `{ "question": "How much paternity leave do I get?" }` or with `"use_query_expansion": true` for **Query Expansion**.

---

## 📂Project Layout

```
Semantic-RAG-Knowledge-Engine/
├── main.py                 # FastAPI app: ingest, search, ask, clear, status
├── config.py               # Settings (provider, Ollama/OpenAI, chunks, similarity threshold)
├── worker.py               # Ingestion worker: RabbitMQ consumer → hash → idempotency → parse/chunk/tag → ChromaDB
├── app/
│   ├── providers/          # LLM + embedding abstraction (OpenAI, Ollama)
│   ├── loaders/            # Format-specific document loaders (PDF, DOCX, Excel, etc.)
│   ├── chunkers/           # Format-aware chunking (dispatcher, table, slide, semantic, etc.)
│   ├── vector_store.py     # ChromaDB: list collections, retriever with metadata filter
│   ├── ingestion.py        # Autonomous ingestion: classify, metadata extraction, chunk, store
│   ├── idempotency.py      # Content hashing (SHA-256) and SQLite store for duplicate detection
│   ├── messaging.py        # RabbitMQ: publish (with retry), consume loop for worker
│   ├── schema_registry.py  # Schema Registry: collection schemas, dynamic Pydantic, normalizers
│   ├── filter_extraction.py # Schema-aware filter extraction from user query
│   ├── prompts.py          # RAG, classification, metadata extraction, filter extraction
│   ├── query_expansion.py  # Advanced RAG: expand question into multiple queries
│   ├── rag.py              # LCEL RAG chain (+ query expansion), ask_rag with schema_hint
│   └── logging_config.py   # Structured JSON logging
├── scripts/
│   └── generate_sample_hr_pdf.py
├── tests/                  # Unit tests (schema registry, chunkers, idempotency, routing)
├── data/                   # processed_hashes.db (idempotency), optional PDFs
├── requirements.txt
├── requirements-dev.txt    # pytest for tests
├── .env.example
└── README.md
```

Runtime-created (in `.gitignore`): `uploads/`, `chroma_db/`, `data/processed_hashes.db`, `.venv/`, `.env`.

---

## 🛠️Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `OLLAMA_LLM_MODEL` | `llama3` | Model for RAG, classification, metadata/filter extraction. |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embeddings (ingest + search). |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory. |
| `DEFAULT_COLLECTION` | `hr_manual` | Default collection when none specified for clear. |
| `DEFAULT_FALLBACK_COLLECTION` | `unclassified_knowledge` | Fallback collection for autonomous ingestion and query routing when UNCLASSIFIED. |
| `CHUNK_SIZE` | `1000` | Characters per chunk. |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks. |
| `SIMILARITY_THRESHOLD` | `0.35` | Max distance for retrieval; above this, chunks are filtered out. |
| `QUERY_EXPANSION_MAX_QUERIES` | `3` | Max alternative queries when using **Query Expansion** in `/ask`. |
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection URL for ingestion queue. |
| `INGESTION_QUEUE_NAME` | `ingestion_tasks` | Queue name for background ingestion tasks. |
| `PROCESSED_HASHES_DB` | `./data/processed_hashes.db` | SQLite DB for **idempotency** (processed content hashes). |

---

## 📂Collections & Schema-Driven Routing

- **Ingestion:** No collection in the request. **Autonomous ingestion** classifies each file to an existing or new collection (or **unclassified_knowledge**). **Metadata extraction** enriches chunks using the **Schema Registry**.
- **Search / Ask:** No collection in the request. **Collection routing** (LLM) and **schema-aware filter extraction** (LLM + **dynamic Pydantic validation**) determine the collection and **metadata filters**; retrieval uses Chroma **where** clauses when filters are present.
- **Schema Registry** defines per-collection metadata fields and **schema hints**; **value normalizers** reduce zero-result errors from value mismatches.
