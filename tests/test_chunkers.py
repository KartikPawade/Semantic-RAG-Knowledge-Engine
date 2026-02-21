"""Test ChunkerDispatcher routing and chunker behavior."""
from langchain_core.documents import Document

from app.chunkers import ChunkerDispatcher
from app.chunkers.slide_chunker import SlideChunker
from app.chunkers.table_chunker import TableChunker


def _fake_embeddings():
    try:
        from langchain_core.embeddings import FakeEmbeddings
    except ImportError:
        from langchain_core.embeddings.fake import FakeEmbeddings
    return FakeEmbeddings()


def test_table_chunker_passthrough():
    chunker = TableChunker()
    docs = [
        Document(page_content="Product: A99 | Price: 120", metadata={"is_table": True}),
    ]
    result = chunker.chunk(docs)
    assert len(result) == 1
    assert result[0].page_content == "Product: A99 | Price: 120"


def test_table_chunker_truncates_long_row():
    chunker = TableChunker(max_row_chars=20)
    docs = [
        Document(page_content="a" * 50, metadata={"is_table": True}),
    ]
    result = chunker.chunk(docs)
    assert len(result) == 1
    assert result[0].page_content.endswith("...")
    assert len(result[0].page_content) == 23


def test_slide_chunker_passthrough():
    chunker = SlideChunker()
    docs = [
        Document(page_content="Slide 1: Title", metadata={"slide": 1}),
    ]
    result = chunker.chunk(docs)
    assert len(result) == 1
    assert result[0].page_content == "Slide 1: Title"


def test_dispatcher_routes_table():
    """Table docs go to TableChunker (no splitting)."""
    disp = ChunkerDispatcher(_fake_embeddings(), use_semantic=False)
    docs = [
        Document(page_content="A | B | C", metadata={"is_table": True}),
    ]
    result = disp.chunk(docs)
    assert len(result) == 1
    assert result[0].page_content == "A | B | C"


def test_dispatcher_routes_slide():
    """Slide docs go to SlideChunker (no splitting)."""
    disp = ChunkerDispatcher(_fake_embeddings(), use_semantic=False)
    docs = [
        Document(page_content="Slide content", metadata={"slide": 1}),
    ]
    result = disp.chunk(docs)
    assert len(result) == 1


def test_dispatcher_routes_heading_passthrough():
    """Headings are passed through as-is."""
    disp = ChunkerDispatcher(_fake_embeddings(), use_semantic=False)
    docs = [
        Document(page_content="Section 3", metadata={"is_heading": True}),
    ]
    result = disp.chunk(docs)
    assert len(result) == 1
    assert result[0].page_content == "Section 3"
