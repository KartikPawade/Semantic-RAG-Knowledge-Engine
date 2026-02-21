"""
Loader registry: maps file extension → loader class.
Adding a new format = add one line here. Nothing else changes.
"""
from pathlib import Path

from app.loaders.pdf_loader import PDFLoader
from app.loaders.docx_loader import DocxLoader
from app.loaders.excel_loader import ExcelLoader
from app.loaders.pptx_loader import PptxLoader
from app.loaders.markdown_loader import MarkdownLoader
from app.loaders.html_loader import HTMLLoader
from app.loaders.email_loader import EmailLoader
from app.loaders.text_loader import TextLoader

# Order matters: more specific extensions first
_REGISTRY = [
    PDFLoader,
    DocxLoader,
    ExcelLoader,
    PptxLoader,
    MarkdownLoader,
    HTMLLoader,
    EmailLoader,
    TextLoader,
]

SUPPORTED_EXTENSIONS = {
    ext
    for cls in _REGISTRY
    for ext in cls.SUPPORTED_EXTENSIONS
}


def get_loader(file_path: Path):
    """Return the appropriate loader for this file extension."""
    for cls in _REGISTRY:
        if cls.supports(file_path):
            return cls()
    raise ValueError(
        f"No loader for extension '{file_path.suffix}'. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def load_document(file_path: str | Path):
    """Unified entry point. Replaces old load_document() in ingestion.py."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    loader = get_loader(path)
    return loader.load(path)
