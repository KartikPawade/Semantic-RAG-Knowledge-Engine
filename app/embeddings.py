"""
Embeddings: The "Language of Numbers".

Converts text into vectors (lists of numbers). We use Ollama's embedding model
(e.g. nomic-embed-text). ChromaDB uses cosine similarity; embeddings are
normalized so distances are comparable across chunks.
"""
from langchain_ollama import OllamaEmbeddings


def get_embedding_model(
    base_url: str = "http://localhost:11434",
    model: str = "nomic-embed-text",
) -> OllamaEmbeddings:
    """
    Ollama embedding model for local vector generation.
    Pull the model first: ollama pull nomic-embed-text
    """
    return OllamaEmbeddings(
        base_url=base_url,
        model=model,
    )
