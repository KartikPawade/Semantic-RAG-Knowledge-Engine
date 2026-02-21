from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class BaseLoader(ABC):
    """
    Contract: every loader returns a list of Documents.
    Each Document has page_content (str) and metadata (dict).
    Loaders are responsible for extracting structural metadata
    (headings, page numbers, slide numbers, row indices) so chunkers
    and the schema registry have rich signal to work with.
    """
    SUPPORTED_EXTENSIONS: list[str] = []

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        ...

    @classmethod
    def supports(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS
