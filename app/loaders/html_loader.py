"""
HTML Loader — strip boilerplate (nav, scripts, styles), extract main content.
"""
from pathlib import Path

from langchain_core.documents import Document

from app.loaders.base import BaseLoader


class HTMLLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".html", ".htm"]

    def load(self, file_path: Path) -> list[Document]:
        from bs4 import BeautifulSoup
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return [Document(
            page_content=text,
            metadata={"source": file_path.name},
        )]
