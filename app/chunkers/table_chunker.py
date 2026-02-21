"""
Table Chunker — no splitting, ever.

A serialized row like "Product: A99 | Region: APAC | Price: $120" is already
the right chunk size. Splitting it would destroy the row structure.

If a row is extremely long (>2000 chars, e.g. a cell with a paragraph of text),
we truncate to preserve the row structure. This is rare in practice.
"""
from langchain_core.documents import Document

from app.chunkers.base import BaseChunker


class TableChunker(BaseChunker):
    def __init__(self, max_row_chars: int = 2000):
        self.max_row_chars = max_row_chars

    def chunk(self, documents: list[Document]) -> list[Document]:
        result = []
        for doc in documents:
            content = doc.page_content
            if len(content) > self.max_row_chars:
                content = content[: self.max_row_chars] + "..."
            result.append(Document(page_content=content, metadata=doc.metadata or {}))
        return result
