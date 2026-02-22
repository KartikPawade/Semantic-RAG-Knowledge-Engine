"""
Enterprise Knowledge Engine - Production RAG API.

Endpoints:
  POST /ingest   - Upload files → queue for background worker (RabbitMQ); 202 Accepted
  POST /search   - Semantic search (relevant snippets)
  POST /ask      - Full RAG: search + LLM answer
  DELETE /clear  - Wipe the vector database
  GET  /status   - Health: Ollama and ChromaDB

Ingestion uses the Worker pattern: file lands in storage → worker picks up from
RabbitMQ → parse, chunk, tag, push to ChromaDB. Idempotency via content hashing.
"""
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from pydantic import BaseModel, Field

from config import get_settings, get_provider
from app.filter_extraction import extract_filters_from_query_async
from app.logging_config import setup_logging
from app.ingestion import classify_query_to_collection_async
from app.loaders import SUPPORTED_EXTENSIONS
from app.messaging import publish_ingest_task
from app.rag import build_rag_chain, build_rag_chain_with_query_expansion, ask_rag
from app.schema_registry import get_schema_hint_for_rag
from app.vector_store import get_vector_store, get_retriever, list_collection_names, clear_collection
from langchain_chroma import Chroma

setup_logging()
settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = get_provider(settings)
    yield


app = FastAPI(
    title="Enterprise Knowledge Engine",
    description="Production RAG: ingest PDFs, semantic search, and grounded Q&A.",
    version="1.0.0",
    lifespan=lifespan,
)


# ----- Request/Response models -----


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(5, ge=1, le=20, description="Number of snippets to return")


class AskRequest(BaseModel):
    question: str = Field(..., description="Question for the RAG assistant")
    use_query_expansion: bool = Field(
        False,
        description="Use Query Expansion: generate alternative phrasings and retrieve for each (better recall)",
    )


# ----- Helpers -----


def _collection(collection: str | None) -> str:
    return collection or settings.default_collection


def _ensure_upload_dir():
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_pending_dir.mkdir(parents=True, exist_ok=True)


_vector_store_cache: dict[tuple[str, str], Chroma] = {}


def _get_vector_store(collection: str) -> Chroma:
    key = (str(settings.chroma_persist_dir), collection)
    if key not in _vector_store_cache:
        embedding = app.state.provider.get_embedding_model()
        _vector_store_cache[key] = get_vector_store(
            persist_directory=settings.chroma_persist_dir,
            collection_name=collection,
            embedding=embedding,
        )
    return _vector_store_cache[key]


async def _resolve_collection_for_query(user_query: str, llm) -> str:
    """
    Route user query to a single collection via LLM (like document classification).
    Returns existing collection name or default_fallback_collection if none match.
    """
    existing = list_collection_names(settings.chroma_persist_dir)
    logger.info("existing_collections fetched: %s", existing)
    return await classify_query_to_collection_async(
        user_query=user_query,
        existing_collections=existing,
        fallback_collection=settings.default_fallback_collection,
        llm=llm,
    )


# ----- Endpoints -----


@app.get("/status")
async def status():
    """
    Check if Ollama (and configured models), and ChromaDB are online.
    When using Ollama, verifies that the configured LLM and embedding models are pulled.
    """
    ollama_ok = False
    models_ok = False
    missing_models: list[str] = []
    chroma_ok = False
    try:
        r = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2.0)
        if r.status_code == 200:
            ollama_ok = True
            available = {m["name"].split(":")[0] for m in r.json().get("models", [])}
            required = {settings.ollama_llm_model, settings.ollama_embedding_model}
            missing_models = sorted(required - available)
            models_ok = not missing_models
    except Exception:
        pass
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        client.heartbeat()
        chroma_ok = True
    except Exception:
        pass
    return {
        "ollama": "online" if ollama_ok else "offline",
        "models": "ok" if models_ok else f"missing: {missing_models}",
        "chromadb": "online" if chroma_ok else "offline",
    }


# Worker pattern: file lands in storage → RabbitMQ → worker parses, chunks, tags, pushes to ChromaDB.
# Idempotency: worker hashes file content; duplicates are skipped to avoid poisoning search.

@app.post("/ingest", status_code=202)
async def ingest(files: list[UploadFile] = File(...)):
    """
    Upload files and queue them for background ingestion (RabbitMQ). Returns 202 Accepted.
    A worker picks up each file: content hash (idempotency) → if new, parse, chunk,
    metadata extraction, embed, push to ChromaDB. Duplicate content (same hash) is skipped.
    If a large PDF crashes the parser, only that job fails; the rest of the system keeps running.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    _ensure_upload_dir()
    tasks: list[dict] = []
    for u in files:
        suffix = Path(u.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {u.filename}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            )
        task_id = uuid.uuid4().hex
        safe_name = (u.filename or "upload").replace(" ", "_")
        path = settings.upload_pending_dir / f"{task_id}_{safe_name}"
        path.write_bytes(await u.read())
        # Relative path so worker (run from project root) can resolve it
        rel_path = f"uploads/pending/{task_id}_{safe_name}"
        try:
            publish_ingest_task(
                file_path=rel_path,
                filename=u.filename or "upload",
                task_id=task_id,
                rabbitmq_url=settings.rabbitmq_url,
                queue_name=settings.ingestion_queue_name,
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Queue unavailable. File saved at {rel_path}. Ensure RabbitMQ is running. Error: {e}",
            )
        tasks.append({"task_id": task_id, "file": u.filename or "upload"})
    return {
        "status": "accepted",
        "message": "Files queued for ingestion. A background worker will process them.",
        "tasks": tasks,
    }


@app.post("/search")
async def search(body: SearchRequest):
    """
    Semantic search with schema-driven filters. Step 1: route query to a
    collection. Step 2: extract metadata filters from the query using that
    collection's schema; search runs on that collection with filter (or none).
    Results are filtered by similarity_threshold (same as /ask) so only
    sufficiently relevant snippets are returned.
    """
    fast_llm = app.state.provider.get_fast_model()
    collection = await _resolve_collection_for_query(body.query, fast_llm)
    chroma_filter = await extract_filters_from_query_async(body.query, collection, fast_llm)
    vs = _get_vector_store(collection)
    search_kwargs: dict = {"k": body.k}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter
    results = vs.similarity_search_with_score(body.query, **search_kwargs)
    # Apply same relevance threshold as /ask: filter out weak matches (Chroma returns distance; lower = more similar)
    results = [
        (doc, score)
        for doc, score in results
        if score <= settings.similarity_threshold
    ]
    snippets = [
        {
            "content": doc.page_content,
            "score": float(score),
            "metadata": {**doc.metadata, "collection": collection},
        }
        for doc, score in results
    ]
    return {"query": body.query, "collection": collection, "snippets": snippets}


@app.post("/ask")
async def ask(body: AskRequest):
    """
    Full RAG with schema-driven filters. Step 1: route question to a collection.
    Step 2: extract metadata filters from the question; retrieval uses that
    collection and filter. Schema hints are injected into the system prompt.
    """
    provider = app.state.provider
    fast_llm = provider.get_fast_model()
    llm = provider.get_chat_model()
    collection = await _resolve_collection_for_query(body.question, fast_llm)
    chroma_filter = await extract_filters_from_query_async(body.question, collection, fast_llm)
    vs = _get_vector_store(collection)
    retriever = get_retriever(
        vs,
        k=4,
        score_threshold=settings.similarity_threshold,
        filter=chroma_filter,
    )
    schema_hint = get_schema_hint_for_rag(collection)
    if body.use_query_expansion:
        chain = build_rag_chain_with_query_expansion(
            retriever, llm, max_expanded_queries=settings.query_expansion_max_queries
        )
    else:
        chain = build_rag_chain(retriever, llm)
    answer = ask_rag(chain, body.question, schema_hint=schema_hint)
    return {"question": body.question, "collection": collection, "answer": answer}


@app.delete("/clear")
async def clear(
    collection: str | None = Query(None, description="Collection to clear (default from config)"),
):
    """
    Wipe the vector database (default collection). Use for reset during testing.
    """
    coll = _collection(collection)
    clear_collection(
        persist_directory=settings.chroma_persist_dir,
        collection_name=coll,
    )
    _vector_store_cache.pop((str(settings.chroma_persist_dir), coll), None)
    return {"status": "ok", "message": f"Collection '{coll}' cleared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
