"""
Ingestion: Load documents and chunk them for the Vector DB.

- PyPDFLoader for PDFs; can be extended for plain text.
- Recursive character chunking with configurable size/overlap.
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.chunking import get_text_splitter
from app.vector_store import get_vector_store


def load_document(file_path: str | Path) -> list[Document]:
    """Load a single file (PDF or .txt) into LangChain documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in (".txt", ".text"):
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")
    return loader.load()


def ingest_files(
    file_paths: list[str | Path],
    persist_directory: str | Path,
    collection_name: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    *,
    ollama_base_url: str = "http://localhost:11434",
    ollama_embedding_model: str = "nomic-embed-text",
) -> int:
    """
    Load files, split into chunks, embed with Ollama, and store in ChromaDB.
    Returns the number of chunks added.
    """
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_docs: list[Document] = []
    for fp in file_paths:
        all_docs.extend(load_document(fp))
    if not all_docs:
        return 0
    chunks = splitter.split_documents(all_docs)
    vector_store = get_vector_store(
        persist_directory=persist_directory,
        collection_name=collection_name,
        ollama_base_url=ollama_base_url,
        ollama_embedding_model=ollama_embedding_model,
    )
    vector_store.add_documents(chunks)
    return len(chunks)
