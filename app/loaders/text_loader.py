"""Plain text fallback loader."""
from pathlib import Path

from langchain_core.documents import Document

from app.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".txt", ".text", ".log"]

    def load(self, file_path: Path) -> list[Document]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return [Document(
            page_content=text,
            metadata={"source": file_path.name},
        )]
