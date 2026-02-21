"""
Recursive Character Chunker — prose fallback.

Splits on paragraph, then line, then sentence, then word boundaries.
Used for plain text and short documents when semantic chunking is disabled
or content is below the semantic threshold.
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunkers.base import BaseChunker


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        return self.splitter.split_documents(documents)
