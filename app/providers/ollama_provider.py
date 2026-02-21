"""
Ollama provider: local LLM (llama3) + local embeddings (nomic-embed-text).
Kept for local dev and air-gapped deployments.
"""
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        llm_model: str = "llama3",
        embedding_model: str = "nomic-embed-text",
        temperature: float = 0,
    ):
        self.base_url = base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature

    def get_chat_model(self) -> ChatOllama:
        return ChatOllama(
            base_url=self.base_url,
            model=self.llm_model,
            temperature=self.temperature,
        )

    def get_fast_model(self) -> ChatOllama:
        # Ollama: same model for both (no tiered pricing)
        return self.get_chat_model()

    def get_embedding_model(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            base_url=self.base_url,
            model=self.embedding_model,
        )
