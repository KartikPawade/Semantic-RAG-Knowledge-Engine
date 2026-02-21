import tempfile
from pathlib import Path

import pytest

from app.idempotency import content_hash, is_processed, record_processed


def test_content_hash_deterministic():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"hello world")
        path = Path(f.name)
    try:
        h1 = content_hash(path)
        h2 = content_hash(path)
        assert h1 == h2
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)
    finally:
        path.unlink(missing_ok=True)


def test_content_hash_different_content():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f1:
        f1.write(b"aaa")
        p1 = Path(f1.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f2:
        f2.write(b"bbb")
        p2 = Path(f2.name)
    try:
        assert content_hash(p1) != content_hash(p2)
    finally:
        p1.unlink(missing_ok=True)
        p2.unlink(missing_ok=True)


def test_is_processed_record_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "hashes.db"
        assert is_processed(db, "abc123") is False
        record_processed(db, "abc123", filename="test.pdf", collection_name="policy_collection")
        assert is_processed(db, "abc123") is True
        assert is_processed(db, "other") is False
