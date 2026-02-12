"""
RAG: Retrieval-Augmented Generation with LCEL.

Recipe: Don't answer from the LLM's memory; use the specific paragraphs
retrieved from the vector DB. LCEL chains: retriever -> format_docs -> prompt -> LLM.
Includes similarity thresholding: if no docs pass the threshold, we say "I don't know."
"""
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from app.prompts import RAG_PROMPT_SIMPLE


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


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


def ask_rag(
    chain,
    question: str,
) -> str:
    """Run the RAG chain for one question."""
    return chain.invoke({"question": question})
