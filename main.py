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
from app.ingestion import ingest_files
from app.rag import build_rag_chain, build_rag_chain_with_query_expansion, ask_rag
from app.vector_store import get_vector_store, get_retriever, clear_collection
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
    collection: str | None = Field(None, description="Collection name (default from config)")
    k: int = Field(5, ge=1, le=20, description="Number of snippets to return")


class AskRequest(BaseModel):
    question: str = Field(..., description="Question for the RAG assistant")
    collection: str | None = Field(None, description="Collection name (default from config)")
    use_query_expansion: bool = Field(
        False,
        description="Use Query Expansion: generate alternative phrasings and retrieve for each (better recall)",
    )


# ----- Helpers -----


def _collection(collection: str | None) -> str:
    return collection or settings.default_collection


def _ensure_upload_dir():
    settings.upload_dir.mkdir(parents=True, exist_ok=True)


def _get_vector_store(collection: str | None = None):
    return get_vector_store(
        persist_directory=settings.chroma_persist_dir,
        collection_name=_collection(collection),
        ollama_base_url=settings.ollama_base_url,
        ollama_embedding_model=settings.ollama_embedding_model,
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
    Semantic search: returns relevant text snippets from the vector DB.
    Use this to verify the "search" part of RAG.
    """
    vs = _get_vector_store(body.collection)
    # similarity_search_with_score returns (doc, distance); lower distance = more similar
    results = vs.similarity_search_with_score(body.query, k=body.k)
    snippets = [
        {
            "content": doc.page_content,
            "score": float(score),  # distance: lower is better
            "metadata": doc.metadata,
        }
        for doc, score in results
    ]
    return {"query": body.query, "snippets": snippets}


@app.post("/ask")
async def ask(body: AskRequest):
    """
    Full RAG: search + Ollama (llama3) answer. Uses only provided context; says
    "I cannot find that in the manual" when the answer is not in the docs.
    Set use_query_expansion=true for Advanced RAG (Query Expansion) to improve recall.
    """
    vs = _get_vector_store(body.collection)
    retriever = get_retriever(
        vs,
        k=4,
        score_threshold=settings.similarity_threshold,
    )
    llm = get_chat_model(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model,
    )
    if body.use_query_expansion:
        chain = build_rag_chain_with_query_expansion(
            retriever, llm, max_expanded_queries=settings.query_expansion_max_queries
        )
    else:
        chain = build_rag_chain(retriever, llm)
    answer = ask_rag(chain, body.question)
    return {"question": body.question, "answer": answer}


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
