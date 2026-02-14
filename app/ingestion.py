"""
Ingestion: Autonomous flow — classify then chunk, enrich metadata, embed.

Flow:
1. Read & sample: first 1,000 words of each document.
2. Classify: LLM compares content to known collections (or suggests new name).
3. Metadata extraction: Schema-aware LLM extracts fields (city, department, etc.)
   from the document and attaches to every chunk (automatic metadata enrichment).
4. Route: ingest into matching collection; create collection on-the-fly if new;
   fallback to default (unclassified_knowledge) if UNCLASSIFIED.
"""
from pathlib import Path
import json
import re

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.chunking import get_text_splitter
from app.schema_registry import get_collection_schema
from app.vector_store import get_vector_store, list_collection_names
from app.prompts import (
    CLASSIFY_COLLECTION_PROMPT,
    CLASSIFY_QUERY_COLLECTION_PROMPT,
    METADATA_EXTRACT_PROMPT,
)


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


def get_first_n_words(documents: list[Document], n: int = 1000) -> str:
    """Concatenate document texts and return the first n words (for classification)."""
    full_text = "\n".join(doc.page_content or "" for doc in documents)
    words = full_text.split()
    return " ".join(words[:n]) if words else ""


def classify_document_to_collection(
    sample_text: str,
    existing_collections: list[str],
    fallback_collection: str,
    llm: BaseChatModel,
) -> str:
    """
    Classify document excerpt into an existing collection, a new collection name, or fallback.
    Returns the collection name to use (existing or new); creates collection later on add.
    """
    existing_str = ", ".join(existing_collections) if existing_collections else "(none)"
    chain = CLASSIFY_COLLECTION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "existing_collections": existing_str,
        "document_excerpt": (sample_text or "")[:8000],  # cap token usage
    })
    name = (raw or "").strip()
    if not name or name.upper() == "UNCLASSIFIED":
        return fallback_collection
    # Normalize: allow only word chars and underscores
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
    """
    Classify user query (search or question) into one existing collection or fallback.
    Uses LLM with list of collections; if the answer is not in any, returns
    fallback_collection (e.g. unclassified_knowledge). Does not create new collections.
    """
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
    # Normalize: strip, lowercase for comparison
    raw_lower = re.sub(r"[^\w_]", "", raw).strip().lower()
    if not raw_lower or raw_lower == "unclassified":
        return fallback_collection
    # Match against existing collections (exact match after normalizing)
    for col in existing_collections:
        col_norm = re.sub(r"[^\w_]", "", col).strip().lower()
        if col_norm and raw_lower == col_norm:
            return col
    # LLM returned something not in list; use fallback
    return fallback_collection


def extract_metadata_for_document(
    document_excerpt: str,
    collection_name: str,
    llm: BaseChatModel,
) -> dict:
    """
    Schema-aware metadata extraction during ingestion. Uses the collection's
    schema from the registry and a small LLM call to fill in fields (city,
    department, product_id, region, etc.) from the document text.
    Returns a dict suitable for chunk.metadata; empty if no schema or parse failure.
    """
    schema = get_collection_schema(collection_name)
    if not schema.fields:
        return {}
    field_names = ", ".join(schema.fields.keys())
    chain = METADATA_EXTRACT_PROMPT | llm | StrOutputParser()
    raw = (chain.invoke({
        "field_names": field_names,
        "excerpt": (document_excerpt or "")[:6000],
    }) or "").strip()
    # Strip markdown code block if present
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw).strip()
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        # Only keep keys that are in the schema
        return {k: v for k, v in data.items() if k in schema.fields and v is not None and v != ""}
    except json.JSONDecodeError:
        return {}


def ingest_files(
    file_paths: list[str | Path],
    persist_directory: str | Path,
    fallback_collection: str,
    llm: BaseChatModel,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    sample_words: int = 1000,
    *,
    ollama_base_url: str = "http://localhost:11434",
    ollama_embedding_model: str = "nomic-embed-text",
) -> dict:
    """
    Autonomous ingestion: for each file, sample first N words, classify to a collection,
    then chunk and embed into that collection (creating it if needed). Unknown docs
    go to fallback_collection.

    Returns dict with: chunks_added (total), files_processed, and routing (list of
    {file, collection, chunks}).
    """
    existing = list_collection_names(persist_directory)
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    total_chunks = 0
    routing: list[dict] = []

    for fp in file_paths:
        path = Path(fp)
        docs = load_document(path)
        if not docs:
            continue
        sample = get_first_n_words(docs, n=sample_words)
        collection_name = classify_document_to_collection(
            sample, existing, fallback_collection, llm
        )
        # Dynamic creation: collection is created on first add_documents
        if collection_name not in existing:
            existing.append(collection_name)
        vs = get_vector_store(
            persist_directory=persist_directory,
            collection_name=collection_name,
            ollama_base_url=ollama_base_url,
            ollama_embedding_model=ollama_embedding_model,
        )
        chunks = splitter.split_documents(docs)
        # Schema-driven: enrich chunks with extracted metadata from registry
        extracted = extract_metadata_for_document(sample, collection_name, llm)
        if extracted:
            for c in chunks:
                c.metadata = {**(c.metadata or {}), **extracted}
        vs.add_documents(chunks)
        n = len(chunks)
        total_chunks += n
        routing.append({
            "file": path.name,
            "collection": collection_name,
            "chunks": n,
        })

    return {
        "chunks_added": total_chunks,
        "files_processed": len(routing),
        "routing": routing,
    }
