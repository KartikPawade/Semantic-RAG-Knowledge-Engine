"""Configuration for the Enterprise Knowledge Engine."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection
    llm_provider: str = "ollama"  # "openai" | "ollama"

    # OpenAI (used when LLM_PROVIDER=openai)
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o"
    openai_fast_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # API Keys (optional; used only if switching back to Gemini)
    google_api_key: str = ""

    # Ollama: local LLM and embeddings (used when LLM_PROVIDER=ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"

    # Paths
    chroma_persist_dir: str = "./chroma_db"
    upload_dir: Path = Path("./uploads")
    upload_pending_dir: Path = Path("./uploads/pending")  # Queued files for workers
    data_dir: Path = Path("./data")
    processed_hashes_db: Path = Path("./data/processed_hashes.db")  # Idempotency store

    # Background workers (RabbitMQ)
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    ingestion_queue_name: str = "ingestion_tasks"

    # Collection management: default for search/ask when no collection specified
    default_collection: str = "hr_manual"
    # Fallback collection for autonomous ingestion when document cannot be classified
    default_fallback_collection: str = "unclassified_knowledge"

    # Chunking: Recursive Character with overlap so edge facts aren't lost
    chunk_size: int = 1000
    chunk_overlap: int = 200
    use_semantic_chunking: bool = True

    # Similarity threshold: if top result < this, say "I don't know"
    similarity_threshold: float = 0.35

    # Query Expansion (Advanced RAG): max alternative queries to generate
    query_expansion_max_queries: int = 3

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def get_provider(settings: Settings | None = None):
    """
    Factory: return the configured provider.
    Import this in main.py and worker.py — nothing else needs to know
    which provider is active.
    """
    s = settings or get_settings()
    if s.llm_provider == "openai":
        from app.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=s.openai_api_key,
            llm_model=s.openai_llm_model,
            fast_model=s.openai_fast_model,
            embedding_model=s.openai_embedding_model,
        )
    if s.llm_provider == "ollama":
        from app.providers.ollama_provider import OllamaProvider
        return OllamaProvider(
            base_url=s.ollama_base_url,
            llm_model=s.ollama_llm_model,
            embedding_model=s.ollama_embedding_model,
        )
    raise ValueError(f"Unknown provider: {s.llm_provider}")
