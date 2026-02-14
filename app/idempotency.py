"""
Idempotency & hashing: prevent duplicate ingestion from poisoning search results.

Before processing, the worker computes a content hash (SHA-256) of the file.
If the hash is already in the store, the file is skipped. Otherwise the file
is processed and the hash is recorded.
"""
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_hashes (
            content_hash TEXT PRIMARY KEY,
            filename TEXT,
            collection_name TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    return conn


def content_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of file content for idempotency check."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_processed(db_path: Path, content_hash: str) -> bool:
    """Return True if this content hash was already processed (skip duplicate)."""
    conn = _get_conn(db_path)
    try:
        cur = conn.execute(
            "SELECT 1 FROM processed_hashes WHERE content_hash = ?",
            (content_hash,),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def record_processed(
    db_path: Path,
    content_hash: str,
    filename: str = "",
    collection_name: Optional[str] = None,
) -> None:
    """Record a content hash as processed after successful ingestion."""
    conn = _get_conn(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO processed_hashes (content_hash, filename, collection_name) VALUES (?, ?, ?)",
            (content_hash, filename, collection_name or ""),
        )
        conn.commit()
    finally:
        conn.close()
