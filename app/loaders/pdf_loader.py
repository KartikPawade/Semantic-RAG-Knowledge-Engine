"""
PDF Loader — three-tier strategy:

Tier 1 (default): PyPDF — fast, good for text-native PDFs.
Tier 2: pdfplumber — activates when PyPDF yields <50 chars/page average.
         Better table extraction (returns structured cell data).
Tier 3: OCR fallback (pdf2image + pytesseract) — activates when pdfplumber
         also yields <50 chars/page. Handles scanned documents.

Why three tiers?
- PyPDF is 10x faster than pdfplumber, but misses tables
- pdfplumber recovers tables but is slower and occasionally fails on
  malformed PDFs
- OCR is slow (2-5s per page) so it's last resort only

Table detection: pdfplumber.extract_tables() returns list[list[str]].
We serialize each table as: "TABLE: col1 | col2 | col3\nval1 | val2 | val3"
and attach it as a separate Document with metadata table=True, page=N.
This keeps table rows together and prevents the chunker from splitting mid-row.
"""
import statistics
from pathlib import Path

from langchain_core.documents import Document

from app.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    SUPPORTED_EXTENSIONS = [".pdf"]

    def __init__(self, ocr_enabled: bool = True, min_chars_per_page: int = 50):
        self.ocr_enabled = ocr_enabled
        self.min_chars_per_page = min_chars_per_page

    def load(self, file_path: Path) -> list[Document]:
        docs = self._load_pypdf(file_path)
        avg_chars = self._avg_chars(docs)

        if avg_chars >= self.min_chars_per_page:
            return docs

        # Tier 2: pdfplumber for better table extraction
        try:
            docs = self._load_pdfplumber(file_path)
            avg_chars = self._avg_chars(docs)
            if avg_chars >= self.min_chars_per_page:
                return docs
        except Exception:
            pass

        # Tier 3: OCR
        if self.ocr_enabled:
            try:
                return self._load_ocr(file_path)
            except Exception:
                pass

        return docs

    def _load_pypdf(self, path: Path) -> list[Document]:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(Document(
                page_content=text,
                metadata={"source": path.name, "page": i + 1, "loader": "pypdf"},
            ))
        return docs

    def _load_pdfplumber(self, path: Path) -> list[Document]:
        import pdfplumber
        docs = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                for t_idx, table in enumerate(tables):
                    serialized = self._serialize_table(table)
                    if serialized:
                        docs.append(Document(
                            page_content=serialized,
                            metadata={
                                "source": path.name,
                                "page": i + 1,
                                "is_table": True,
                                "table_index": t_idx,
                                "loader": "pdfplumber",
                            },
                        ))
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": path.name, "page": i + 1, "loader": "pdfplumber"},
                    ))
        return docs

    def _load_ocr(self, path: Path) -> list[Document]:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(str(path))
        docs = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            docs.append(Document(
                page_content=text,
                metadata={"source": path.name, "page": i + 1, "loader": "ocr"},
            ))
        return docs

    def _serialize_table(self, table: list) -> str:
        if not table:
            return ""
        rows = []
        for row in table:
            rows.append(" | ".join(str(cell or "").strip() for cell in row))
        return "TABLE:\n" + "\n".join(rows)

    def _avg_chars(self, docs: list[Document]) -> float:
        lengths = [len(d.page_content) for d in docs if d.page_content]
        return statistics.mean(lengths) if lengths else 0
