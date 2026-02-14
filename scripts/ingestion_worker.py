#!/usr/bin/env python3
"""
Ingestion worker: pick up files from RabbitMQ, hash for idempotency, then parse,
chunk, tag (metadata), and push vectors to ChromaDB.

Run from project root so relative paths in messages resolve. Keeps heavy/crashing
jobs (e.g. a 1,000-page PDF) off the API so the rest of the system keeps running.

Usage:
  python scripts/ingestion_worker.py
"""
import sys
from pathlib import Path

# Run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from app.idempotency import content_hash, is_processed, record_processed
from app.ingestion import ingest_files
from app.llm import get_chat_model
from app.messaging import consume_ingest_tasks


def process_one_task(data: dict, channel, method) -> None:
    task_id = data.get("task_id", "")
    file_path = data.get("file_path", "")
    filename = data.get("filename", "")
    if not file_path:
        channel.basic_nack(method.delivery_tag, requeue=False)
        return
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        channel.basic_nack(method.delivery_tag, requeue=False)
        return
    settings = get_settings()
    # Idempotency: skip if we already processed this content
    try:
        file_hash = content_hash(path)
    except Exception:
        channel.basic_nack(method.delivery_tag, requeue=False)
        return
    if is_processed(settings.processed_hashes_db, file_hash):
        try:
            path.unlink()
        except Exception:
            pass
        channel.basic_ack(method.delivery_tag)
        return
    # Worker pattern: parse → chunk → tag → push to ChromaDB
    try:
        llm = get_chat_model(
            base_url=settings.ollama_base_url,
            model=settings.ollama_llm_model,
        )
        result = ingest_files(
            file_paths=[path],
            persist_directory=settings.chroma_persist_dir,
            fallback_collection=settings.default_fallback_collection,
            llm=llm,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            sample_words=1000,
            ollama_base_url=settings.ollama_base_url,
            ollama_embedding_model=settings.ollama_embedding_model,
        )
        collection = ""
        if result.get("routing"):
            collection = result["routing"][0].get("collection", "")
        record_processed(
            settings.processed_hashes_db,
            file_hash,
            filename=filename,
            collection_name=collection,
        )
    except Exception:
        channel.basic_nack(method.delivery_tag, requeue=False)
        return
    try:
        path.unlink()
    except Exception:
        pass
    channel.basic_ack(method.delivery_tag)


def main() -> None:
    settings = get_settings()
    print(f"Worker consuming from queue: {settings.ingestion_queue_name}")
    consume_ingest_tasks(
        process_one_task,
        rabbitmq_url=settings.rabbitmq_url,
        queue_name=settings.ingestion_queue_name,
    )


if __name__ == "__main__":
    main()
