"""
Query Expansion — Advanced RAG technique.

Expands the user's question into multiple related queries (rephrasings, synonyms,
or sub-questions) so retrieval can find relevant chunks even when the document
wording differs from the user's phrasing. Improves recall.
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query expander for a document search system. Given a user question, "
            "output 2 to 3 alternative phrasings or related questions that could help find "
            "the same information in documents (rephrasing, synonyms, or sub-questions). "
            "Output ONLY the alternative queries, one per line, no numbering or bullets. "
            "Keep each line concise. Include the original question as the first line.",
        ),
        ("human", "{question}"),
    ]
)


def expand_queries(llm: BaseChatModel, question: str, max_queries: int = 3) -> list[str]:
    """
    Use the LLM to generate alternative queries from the user's question.
    Returns a list of unique queries (original + expansions), capped at max_queries.
    """
    chain = QUERY_EXPANSION_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question})
    # Parse lines: strip, drop empty, dedupe while preserving order
    seen: set[str] = set[str]()
    queries: list[str] = []
    for line in raw.strip().splitlines():
        q = line.strip().strip(".-) ").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        queries.append(q)
        if len(queries) >= max_queries:
            break
    # Always include original as first if not already present
    if question not in seen:
        queries.insert(0, question)
    elif queries and queries[0] != question:
        # Move original to front if it appeared later
        try:
            idx = next(i for i, q in enumerate(queries) if q == question)
            queries.pop(idx)
            queries.insert(0, question)
        except StopIteration:
            pass
    return queries[:max_queries]
