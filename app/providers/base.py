"""
Abstract provider interfaces. Every LLM and embedding backend implements these.
Swap providers by changing config — no other file changes.
"""
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


class LLMProvider(ABC):
    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        """Return a LangChain-compatible chat model."""
        ...

    @abstractmethod
    def get_embedding_model(self) -> Embeddings:
        """Return a LangChain-compatible embedding model."""
        ...
