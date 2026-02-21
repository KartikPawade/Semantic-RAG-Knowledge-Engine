"""
OpenAI provider: GPT-4o for reasoning, text-embedding-3-small for embeddings.

Why text-embedding-3-small over large:
- 1536 dims vs 3072: half the storage, half the Chroma query time
- 5x cheaper per token
- Benchmark difference is marginal for RAG workloads (MTEB ~2% gap)
- Use text-embedding-3-large only if your domain has highly technical jargon
  (legal, genomics) where nuance materially affects retrieval

Why gpt-4o-mini for classification/extraction, gpt-4o for RAG answers:
- Classification (collection routing, filter extraction) = structured output,
  short context. Mini is accurate enough and 10x cheaper.
- RAG answer = needs reasoning over long context. Use full 4o.
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        llm_model: str = "gpt-4o",
        fast_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0,
    ):
        self.api_key = api_key
        self.llm_model = llm_model
        self.fast_model = fast_model
        self.embedding_model = embedding_model
        self.temperature = temperature

    def get_chat_model(self) -> ChatOpenAI:
        """Full model for RAG answers — reasoning over retrieved context."""
        return ChatOpenAI(
            api_key=self.api_key,
            model=self.llm_model,
            temperature=self.temperature,
        )

    def get_fast_model(self) -> ChatOpenAI:
        """
        Lightweight model for classification and filter extraction.
        Called 2x per /search and /ask request (collection routing + filter extraction).
        Use mini to keep latency and cost low for these structured tasks.
        """
        return ChatOpenAI(
            api_key=self.api_key,
            model=self.fast_model,
            temperature=0,
        )

    def get_embedding_model(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            api_key=self.api_key,
            model=self.embedding_model,
        )
