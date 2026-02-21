"""Format-aware chunking. Use ChunkerDispatcher from ingestion."""
from app.chunkers.base import BaseChunker
from app.chunkers.dispatcher import ChunkerDispatcher

__all__ = ["BaseChunker", "ChunkerDispatcher"]
