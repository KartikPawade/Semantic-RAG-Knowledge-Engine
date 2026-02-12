"""
Recursive Character Chunking.

A PDF is too large to fit into a single "thought." We break it into chunks
of 1,000 characters with 200-character overlap so that if a vital fact (e.g.
a date) sits at the edge of a cut, it appears in both chunks and context
isn't lost.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
