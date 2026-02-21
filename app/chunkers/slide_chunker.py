"""
Slide Chunker — no splitting. One slide = one chunk.

Rationale: a slide is a natural semantic unit. Its title + bullets form a
complete thought. Splitting mid-slide loses context (bullets without title).
"""
from langchain_core.documents import Document

from app.chunkers.base import BaseChunker


class SlideChunker(BaseChunker):
    def chunk(self, documents: list[Document]) -> list[Document]:
        return list(documents)
