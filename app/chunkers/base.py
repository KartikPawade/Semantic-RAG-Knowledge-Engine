from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks ready for embedding."""
        ...
