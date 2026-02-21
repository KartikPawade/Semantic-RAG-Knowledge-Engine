"""
Semantic Chunker — splits where embedding distance spikes.

Algorithm:
1. Split text into sentences.
2. Embed each sentence (or small window of sentences).
3. Compute cosine distance between adjacent sentence embeddings.
4. Split where distance > breakpoint_threshold (topic change).

Result: variable-size chunks that respect topic boundaries.

When to use: long policy PDFs, legal documents, research papers.
When NOT to use: tabular data, slides, short docs (<3000 chars).

Cost note (OpenAI): semantic chunking embeds every sentence during ingestion.
For a 100-page PDF (~50,000 words), this is ~200 embedding API calls.
At $0.02/1M tokens for text-embedding-3-small, cost is negligible (<$0.01).
But add a semantic_threshold (default 3000 chars) to skip it for short docs.

We use LangChain's SemanticChunker which handles the sentence splitting,
embedding, and distance computation. Just pass your embedding model.
"""
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.chunkers.base import BaseChunker


class SemanticChunker(BaseChunker):
    def __init__(self, embedding_model: Embeddings, breakpoint_type: str = "percentile"):
        self.embedding_model = embedding_model
        self.breakpoint_type = breakpoint_type

    def chunk(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        try:
            from langchain_experimental.text_splitter import SemanticChunker as LC_SemanticChunker
            splitter = LC_SemanticChunker(
                embeddings=self.embedding_model,
                breakpoint_threshold_type=self.breakpoint_type,
            )
            result = []
            for doc in documents:
                chunks = splitter.split_documents([doc])
                result.extend(chunks)
            return result
        except ImportError:
            from app.chunkers.recursive_chunker import RecursiveChunker
            return RecursiveChunker(1000, 200).chunk(documents)
