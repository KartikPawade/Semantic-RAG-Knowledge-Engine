"""
Ingestion: now format-aware and provider-agnostic.

Changes from original:
- load_document() → app.loaders (format-specific, structure-preserving)
- get_text_splitter() → ChunkerDispatcher (format-aware, strategy dispatch)
- LLM/embedder → passed in via provider, not constructed here

Flow (unchanged from user perspective):
1. Load: format-specific loader extracts text + structure + metadata
2. Classify: LLM routes to collection (or creates new one)
3. Extract metadata: schema-aware LLM call for collection fields
4. Chunk: dispatcher picks strategy per document type
5. Embed + store: ChromaDB via provider embedding model
"""
from pathlib import Path
import json
import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.chunkers.dispatcher import ChunkerDispatcher
from app.loaders import load_document
from app.prompts import (
    CLASSIFY_COLLECTION_PROMPT,
    CLASSIFY_QUERY_COLLECTION_PROMPT,
    METADATA_EXTRACT_PROMPT,
)
from app.schema_registry import get_collection_schema
from app.vector_store import get_vector_store, list_collection_names


def get_first_n_words(documents: list[Document], n: int = 1000) -> str:
    full_text = "\n".join(doc.page_content or "" for doc in documents)
    words = full_text.split()
    return " ".join(words[:n]) if words else ""


def classify_document_to_collection(
    sample_text: str,
    existing_collections: list[str],
    fallback_collection: str,
    llm: BaseChatModel,
) -> str:
    existing_str = ", ".join(existing_collections) if existing_collections else "(none)"
    chain = CLASSIFY_COLLECTION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "existing_collections": existing_str,
        "document_excerpt": (sample_text or "")[:8000],
    })
    name = (raw or "").strip()
    if not name or name.upper() == "UNCLASSIFIED":
        return fallback_collection
    name = re.sub(r"[^\w_]", " ", name).strip().replace(" ", "_").strip("_")
    if not name or name.upper() == "UNCLASSIFIED":
        return fallback_collection
    name = name.lower()
    if "_collection" not in name:
        name = name + "_collection"
    return name


def classify_query_to_collection(
    user_query: str,
    existing_collections: list[str],
    fallback_collection: str,
    llm: BaseChatModel,
) -> str:
    if not existing_collections:
        return fallback_collection
    existing_str = ", ".join(existing_collections)
    chain = CLASSIFY_QUERY_COLLECTION_PROMPT | llm | StrOutputParser()
    raw = (chain.invoke({
        "existing_collections": existing_str,
        "user_query": (user_query or "").strip()[:2000],
    }) or "").strip()
    if not raw or raw.upper() == "UNCLASSIFIED":
        return fallback_collection
    raw_lower = re.sub(r"[^\w_]", "", raw).strip().lower()
    for col in existing_collections:
        col_norm = re.sub(r"[^\w_]", "", col).strip().lower()
        if col_norm and raw_lower == col_norm:
            return col
    return fallback_collection


def extract_metadata_for_document(
    document_excerpt: str,
    collection_name: str,
    llm: BaseChatModel,
) -> dict:
    schema = get_collection_schema(collection_name)
    if not schema.fields:
        return {}
    field_names = ", ".join(schema.fields.keys())
    chain = METADATA_EXTRACT_PROMPT | llm | StrOutputParser()
    raw = (chain.invoke({
        "field_names": field_names,
        "excerpt": (document_excerpt or "")[:6000],
    }) or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw).strip()
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if k in schema.fields and v is not None and v != ""}
    except json.JSONDecodeError:
        return {}


def ingest_files(
    file_paths: list[str | Path],
    persist_directory: str | Path,
    fallback_collection: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    sample_words: int = 1000,
    use_semantic_chunking: bool = True,
) -> dict:
    """
    Format-aware autonomous ingestion.

    Key changes vs original:
    - load_document() uses the loader registry (format-specific)
    - ChunkerDispatcher routes each document to the right chunker
    - embedding_model passed in (provider-agnostic)
    """
    existing = list_collection_names(persist_directory)
    dispatcher = ChunkerDispatcher(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_semantic=use_semantic_chunking,
    )
    total_chunks = 0
    routing = []

    for fp in file_paths:
        path = Path(fp)
        docs = load_document(path)
        if not docs:
            continue

        doc_type = _infer_document_type(path)
        for doc in docs:
            doc.metadata = doc.metadata or {}
            doc.metadata["document_type"] = doc_type

        sample = get_first_n_words(docs, n=sample_words)
        collection_name = classify_document_to_collection(
            sample, existing, fallback_collection, llm
        )
        if collection_name not in existing:
            existing.append(collection_name)

        vs = get_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding=embedding_model,
        )

        chunks = dispatcher.chunk(docs)

        extracted = extract_metadata_for_document(sample, collection_name, llm)
        if extracted:
            for c in chunks:
                c.metadata = {**(c.metadata or {}), **extracted}

        vs.add_documents(chunks)
        n = len(chunks)
        total_chunks += n
        routing.append({"file": path.name, "collection": collection_name, "chunks": n})

    return {
        "chunks_added": total_chunks,
        "files_processed": len(routing),
        "routing": routing,
    }


def _infer_document_type(path: Path) -> str:
    """Simple heuristic document type tag for metadata."""
    ext = path.suffix.lower()
    return {
        ".pdf": "pdf", ".docx": "word", ".doc": "word",
        ".xlsx": "spreadsheet", ".xls": "spreadsheet", ".csv": "csv",
        ".pptx": "presentation", ".ppt": "presentation",
        ".md": "markdown", ".html": "html", ".htm": "html",
        ".eml": "email", ".msg": "email",
        ".txt": "text", ".text": "text",
    }.get(ext, "unknown")
