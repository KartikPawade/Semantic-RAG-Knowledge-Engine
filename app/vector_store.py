"""
Vector Database: The "Semantic Library".

- Persistence: Data is saved to disk (e.g. ./chroma_db) so we don't re-read
  PDFs on every server restart.
- Collection Management: Organize data into collections (e.g. "HR Data" vs
  "Engineering Data").
"""
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.embeddings import get_embedding_model


def clear_collection(persist_directory: str | Path, collection_name: str) -> None:
    """Wipe the named collection from ChromaDB (for testing/reset)."""
    path = Path(persist_directory)
    if not path.exists():
        return
    client = chromadb.PersistentClient(path=str(path))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


def get_vector_store(
    persist_directory: str | Path,
    collection_name: str,
    embedding: Embeddings | None = None,
    *,
    ollama_base_url: str = "http://localhost:11434",
    ollama_embedding_model: str = "nomic-embed-text",
) -> Chroma:
    """
    ChromaDB with persistence and a named collection.
    Uses Ollama for embeddings when embedding is not provided.
    """
    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)
    emb = embedding or get_embedding_model(
        base_url=ollama_base_url,
        model=ollama_embedding_model,
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=emb,
        persist_directory=str(path),
    )


def get_retriever( 
    vector_store: VectorStore,
    k: int = 4,
    score_threshold: float | None = None,
):
    """
    Retriever with optional similarity score threshold (filtering).
    Chroma uses distance (lower = more similar). With score_threshold we only
    return documents with distance <= threshold, so we can say "I don't know"
    when no relevant context is found.
    """
    search_kwargs: dict = {"k": k}
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
    search_type = "similarity_score_threshold" if score_threshold is not None else "similarity"
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
