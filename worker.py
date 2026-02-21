"""
Ingestion worker — now provider-aware.

Change: constructs LLM and embedding model from provider factory.
Everything else (RabbitMQ, idempotency, file handling) unchanged.

Run from project root so relative paths in messages resolve.

Usage:
  python worker.py
"""
import logging
import sys
from pathlib import Path

# Run from project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings, get_provider
from app.idempotency import content_hash, is_processed, record_processed
from app.ingestion import ingest_files
from app.logging_config import setup_logging
from app.messaging import consume_ingest_tasks

setup_logging()
logger = logging.getLogger(__name__)


def process_one_task(data: dict, channel, method) -> None:
    settings = get_settings()
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

    try:
        file_hash = content_hash(path)
    except Exception:
        channel.basic_nack(method.delivery_tag, requeue=False)
        return

    if is_processed(settings.processed_hashes_db, file_hash):
        logger.info("Skipping duplicate", extra={"task_id": task_id, "hash": file_hash})
        path.unlink(missing_ok=True)
        channel.basic_ack(method.delivery_tag)
        return

    logger.info("Processing task", extra={"task_id": task_id, "file": filename})
    try:
        provider = get_provider(settings)
        llm = provider.get_fast_model()
        embedding_model = provider.get_embedding_model()

        result = ingest_files(
            file_paths=[path],
            persist_directory=settings.chroma_persist_dir,
            fallback_collection=settings.default_fallback_collection,
            llm=llm,
            embedding_model=embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            use_semantic_chunking=settings.use_semantic_chunking,
        )
        collection = ""
        if result.get("routing"):
            collection = result["routing"][0].get("collection", "")
        record_processed(settings.processed_hashes_db, file_hash, filename, collection)
        logger.info(
            "Ingestion complete",
            extra={
                "task_id": task_id,
                "collection": collection,
                "chunks": result.get("chunks_added", 0),
            },
        )
    except Exception as e:
        logger.error(
            "Ingestion failed",
            extra={"task_id": task_id, "file": filename},
            exc_info=True,
        )
        channel.basic_nack(method.delivery_tag, requeue=False)
        return

    channel.basic_ack(method.delivery_tag)
    path.unlink(missing_ok=True)


def main() -> None:
    settings = get_settings()
    logger.info("Worker started", extra={"queue": settings.ingestion_queue_name})
    consume_ingest_tasks(
        process_one_task,
        rabbitmq_url=settings.rabbitmq_url,
        queue_name=settings.ingestion_queue_name,
    )


if __name__ == "__main__":
    main()
