"""
Structural Chunker — respects document section boundaries.

For DOCX and Markdown documents where loaders preserved heading metadata:
1. Group documents by their section (heading).
2. Within each section, apply recursive character splitting.
3. Every resulting chunk carries the section name as metadata.

This means retrieval can surface "from the Parental Leave section" rather
than an arbitrary 1000-char window that happens to mention "leave".
"""
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunkers.base import BaseChunker


class StructuralChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        sections: dict[str, list[Document]] = defaultdict(list)
        for doc in documents:
            section = (doc.metadata or {}).get("section", "")
            sections[section].append(doc)

        result = []
        for section, docs in sections.items():
            combined = "\n\n".join(d.page_content for d in docs)
            base_metadata = {**(docs[0].metadata or {}), "section": section}
            chunks_text = self.splitter.split_text(combined)
            for c in chunks_text:
                result.append(Document(page_content=c, metadata={**base_metadata}))
        return result
