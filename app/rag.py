"""
RAG: Retrieval-Augmented Generation with LCEL.

Recipe: Don't answer from the LLM's memory; use the specific paragraphs
retrieved from the vector DB. LCEL chains: retriever -> format_docs -> prompt -> LLM.
Includes similarity thresholding: if no docs pass the threshold, we say "I don't know."

Advanced RAG: Query Expansion — expand the user question into multiple queries,
retrieve for each, merge and dedupe docs, then generate the answer for better recall.
"""
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from app.prompts import RAG_PROMPT_SIMPLE
from app.query_expansion import expand_queries


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _merge_and_dedupe_docs(doc_lists: list[list[Document]]) -> list[Document]:
    """Merge multiple document lists and deduplicate by page_content."""
    seen: set[str] = set()
    out: list[Document] = []
    for docs in doc_lists:
        for doc in docs:
            key = (doc.page_content or "").strip()
            if key and key not in seen:
                seen.add(key)
                out.append(doc)
    return out


def build_rag_chain(
    retriever: VectorStoreRetriever,
    llm: BaseChatModel,
):
    """
    LCEL chain: question -> retriever -> format_docs -> prompt -> llm -> string.
    If the retriever returns no documents (e.g. all below score threshold),
    we pass empty context and the system instruction tells the LLM to say
    "I cannot find that in the manual."
    """
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: _format_docs(retriever.invoke(x["question"])),
        )
        | RAG_PROMPT_SIMPLE
        | llm
        | StrOutputParser()
    )
    return chain


def build_rag_chain_with_query_expansion(
    retriever: VectorStoreRetriever,
    llm: BaseChatModel,
    max_expanded_queries: int = 3,
):
    """
    RAG chain with Query Expansion: expand question -> retrieve per query ->
    merge/dedupe docs -> format -> prompt -> llm -> string.
    Uses multiple phrasings to improve retrieval recall.
    """
    def get_context(inputs: dict) -> str:
        question = inputs["question"]
        queries = expand_queries(llm, question, max_queries=max_expanded_queries)
        doc_lists = [retriever.invoke(q) for q in queries]
        merged = _merge_and_dedupe_docs(doc_lists)
        return _format_docs(merged)

    chain = (
        RunnablePassthrough.assign(context=get_context)
        | RAG_PROMPT_SIMPLE
        | llm
        | StrOutputParser()
    )
    return chain


def ask_rag(
    chain,
    question: str,
    schema_hint: str = "",
) -> str:
    """Run the RAG chain for one question. schema_hint is injected for schema-driven collections."""
    return chain.invoke({"question": question, "schema_hint": schema_hint or ""})
