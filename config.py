"""Configuration for the Enterprise Knowledge Engine."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys (optional; used only if switching back to Gemini)
    google_api_key: str = ""

    # Ollama: local LLM and embeddings
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3"
    ollama_embedding_model: str = "nomic-embed-text"

    # Paths
    chroma_persist_dir: str = "./chroma_db"
    upload_dir: Path = Path("./uploads")
    data_dir: Path = Path("./data")

    # Collection management: separate "HR Data" from "Engineering Data", etc.
    default_collection: str = "hr_manual"

    # Chunking: Recursive Character with overlap so edge facts aren't lost
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Similarity threshold: if top result < this, say "I don't know"
    similarity_threshold: float = 0.2

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir).resolve()


def get_settings() -> Settings:
    return Settings()
