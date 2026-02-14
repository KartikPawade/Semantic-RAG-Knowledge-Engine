"""
Enterprise Knowledge Engine - Production RAG API.

Endpoints:
  POST /ingest   - Upload and process files (PDFs/Text) into searchable vectors
  POST /search   - Semantic search (relevant snippets)
  POST /ask      - Full RAG: search + LLM answer
  DELETE /clear  - Wipe the vector database
  GET  /status   - Health: Ollama and ChromaDB
"""
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import get_settings
from app.filter_extraction import extract_filters_from_query
from app.ingestion import ingest_files, classify_query_to_collection
from app.rag import build_rag_chain, build_rag_chain_with_query_expansion, ask_rag
from app.schema_registry import get_schema_hint_for_rag
from app.vector_store import get_vector_store, get_retriever, list_collection_names, clear_collection
from app.llm import get_chat_model

app = FastAPI(
    title="Enterprise Knowledge Engine",
    description="Production RAG: ingest PDFs, semantic search, and grounded Q&A.",
    version="1.0.0",
)

settings = get_settings()


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


def _get_vector_store(collection: str):
    return get_vector_store(
        persist_directory=settings.chroma_persist_dir,
        collection_name=collection,
        ollama_base_url=settings.ollama_base_url,
        ollama_embedding_model=settings.ollama_embedding_model,
    )


def _resolve_collection_for_query(user_query: str, llm) -> str:
    """
    Route user query to a single collection via LLM (like document classification).
    Returns existing collection name or default_fallback_collection if none match.
    """
    existing = list_collection_names(settings.chroma_persist_dir)
    return classify_query_to_collection(
        user_query=user_query,
        existing_collections=existing,
        fallback_collection=settings.default_fallback_collection,
        llm=llm,
    )


# ----- Endpoints -----


@app.get("/status")
async def status():
    """
    Check if Ollama and ChromaDB are online.
    """
    ollama_ok = False
    chroma_ok = False
    try:
        r = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=2.0)
        ollama_ok = r.status_code == 200
    except Exception:
        pass
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        client.heartbeat()
        chroma_ok = True
    except Exception:
        chroma_ok = False
    return {
        "ollama": "online" if ollama_ok else "offline",
        "chromadb": "online" if chroma_ok else "offline",
    }


# Autonomous ingestion: no collection parameter. Each file is classified (first 1000 words)
# into an existing or new collection, or routed to the default fallback collection.

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    """
    Upload and process files (PDFs/Text). Autonomous flow: for each file, the system
    reads the first 1,000 words, classifies it against known collections (or suggests
    a new one), then chunks and embeds into that collection. Unclassifiable docs
    go to the default fallback collection (unclassified_knowledge).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    _ensure_upload_dir()
    saved: list[Path] = []
    try:
        for u in files:
            suffix = Path(u.filename or "").suffix.lower()
            if suffix not in (".pdf", ".txt", ".text"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {u.filename}. Use .pdf or .txt",
                )
            path = settings.upload_dir / (u.filename or "upload")
            path.write_bytes(await u.read())
            saved.append(path)
        llm = get_chat_model(
            base_url=settings.ollama_base_url,
            model=settings.ollama_llm_model,
        )
        result = ingest_files(
            file_paths=saved,
            persist_directory=settings.chroma_persist_dir,
            fallback_collection=settings.default_fallback_collection,
            llm=llm,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            sample_words=1000,
            ollama_base_url=settings.ollama_base_url,
            ollama_embedding_model=settings.ollama_embedding_model,
        )
        return {
            "status": "ok",
            "chunks_added": result["chunks_added"],
            "files_processed": result["files_processed"],
            "routing": result["routing"],
        }
    finally:
        for p in saved:
            try:
                p.unlink()
            except Exception:
                pass


@app.post("/search")
async def search(body: SearchRequest):
    """
    Semantic search with schema-driven filters. Step 1: route query to a
    collection. Step 2: extract metadata filters from the query using that
    collection's schema; search runs on that collection with filter (or none).
    """
    llm = get_chat_model(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
    )
    collection = _resolve_collection_for_query(body.query, llm)
    chroma_filter = extract_filters_from_query(body.query, collection, llm)
    vs = _get_vector_store(collection)
    search_kwargs: dict = {"k": body.k}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter
    results = vs.similarity_search_with_score(body.query, **search_kwargs)
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
    llm = get_chat_model(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
    )
    collection = _resolve_collection_for_query(body.question, llm)
    chroma_filter = extract_filters_from_query(body.question, collection, llm)
    vs = _get_vector_store(collection)
    retriever = get_retriever(
        vs,
        k=4,
        score_threshold=settings.similarity_threshold,
        filter=chroma_filter,
    )
    schema_hint = get_schema_hint_for_rag([collection])
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
    clear_collection(
        persist_directory=settings.chroma_persist_dir,
        collection_name=_collection(collection),
    )
    return {"status": "ok", "message": f"Collection '{_collection(collection)}' cleared."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
