"""
Chunker dispatcher: picks the right chunking strategy per document type.

Decision logic (in priority order):
1. is_table=True in metadata → TableChunker (no splitting, row IS the chunk)
2. slide=N in metadata → SlideChunker (no splitting, slide IS the chunk)
3. is_heading=True → skip (headings are already atomic)
4. section in metadata (DOCX/Markdown with structure) → StructuralChunker
5. source ends in .pdf and content > threshold → SemanticChunker
6. default → RecursiveChunker

This means you get:
- Excel rows: never split (each row is already optimal chunk size)
- Slides: never split (slide boundary = semantic boundary)
- Structured DOCX: split at heading boundaries, then recursive within sections
- Long PDFs: semantic splits where topics change
- Plain text / short PDFs: standard recursive character splitting
"""
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.chunkers.recursive_chunker import RecursiveChunker
from app.chunkers.semantic_chunker import SemanticChunker
from app.chunkers.structural_chunker import StructuralChunker
from app.chunkers.table_chunker import TableChunker
from app.chunkers.slide_chunker import SlideChunker


class ChunkerDispatcher:
    def __init__(
        self,
        embedding_model: Embeddings,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_semantic: bool = True,
        semantic_threshold: int = 3000,
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic
        self.semantic_threshold = semantic_threshold

        self._table = TableChunker()
        self._slide = SlideChunker()
        self._recursive = RecursiveChunker(chunk_size, chunk_overlap)
        self._structural = StructuralChunker(chunk_size, chunk_overlap)
        self._semantic = SemanticChunker(embedding_model) if use_semantic else None

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Route each document to the appropriate chunker and collect results."""
        tables, slides, headings, structural, prose = [], [], [], [], []

        for doc in documents:
            meta = doc.metadata or {}
            if meta.get("is_table"):
                tables.append(doc)
            elif meta.get("slide"):
                slides.append(doc)
            elif meta.get("is_heading"):
                headings.append(doc)
            elif meta.get("section"):
                structural.append(doc)
            else:
                prose.append(doc)

        result = []
        result.extend(self._table.chunk(tables))
        result.extend(self._slide.chunk(slides))
        result.extend(headings)

        result.extend(self._structural.chunk(structural))

        total_prose_chars = sum(len(d.page_content) for d in prose)
        if self.use_semantic and self._semantic and total_prose_chars > self.semantic_threshold:
            result.extend(self._semantic.chunk(prose))
        else:
            result.extend(self._recursive.chunk(prose))

        return result
