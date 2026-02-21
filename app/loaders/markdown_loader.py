"""
Markdown Loader — header-aware splitting.

Preserves heading structure so the structural chunker can split at section
boundaries. Each block (paragraph or code block) carries the nearest
preceding heading as section metadata.
"""
from pathlib import Path
import re

from langchain_core.documents import Document

from app.loaders.base import BaseLoader


class MarkdownLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".md", ".markdown"]

    def load(self, file_path: Path) -> list[Document]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        documents = []
        current_section = ""
        buffer = []
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        def flush_buffer():
            if buffer:
                content = "\n".join(buffer)
                if content.strip():
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_path.name,
                            "section": current_section,
                        },
                    ))
                buffer.clear()

        for line in text.split("\n"):
            match = heading_pattern.match(line)
            if match:
                flush_buffer()
                level = len(match.group(1))
                current_section = match.group(2).strip()
                documents.append(Document(
                    page_content=current_section,
                    metadata={
                        "source": file_path.name,
                        "section": current_section,
                        "heading_level": level,
                        "is_heading": True,
                    },
                ))
            else:
                buffer.append(line)

        flush_buffer()
        return documents if documents else [Document(page_content=text, metadata={"source": file_path.name})]
