"""
System Instructions (Grounding).

We "lock" the LLM's behavior so it uses ONLY the provided context.
If the answer is not in the context, it must say so instead of using outside knowledge.
"""
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_INSTRUCTION = """You are an HR Assistant. Use ONLY the provided context to answer.
If the answer is not in the context, say: "I cannot find that in the manual."
Do not use outside knowledge. Do not make up policy details."""

RAG_PROMPT_SIMPLE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTION + "\n\nContext:\n{context}"),
        ("human", "{question}"),
    ]
)

# ----- Autonomous ingestion: document classification -----

CLASSIFY_COLLECTION_SYSTEM = """You are a document classifier. Your job is to decide which knowledge collection a document belongs to.

Given:
1) A short excerpt from a document (first ~1000 words).
2) A list of existing collection names (if any).

You must reply with EXACTLY one of:
- One of the existing collection names exactly as written (if the document clearly fits that collection), OR
- A new collection name in snake_case ending with _collection (e.g. company_policy_collection, invoice_collection, product_details_collection) if the document fits a new category, OR
- The word UNCLASSIFIED if the document does not clearly fit any category and you cannot suggest a meaningful new one.

Reply with ONLY the collection name or UNCLASSIFIED. No explanation, no quotes, no punctuation after the name."""

CLASSIFY_COLLECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CLASSIFY_COLLECTION_SYSTEM),
        ("human", "Existing collections: {existing_collections}\n\nDocument excerpt:\n{document_excerpt}"),
    ]
)

# ----- Search/Ask: query-to-collection routing -----

CLASSIFY_QUERY_COLLECTION_SYSTEM = """You are a query router. Given a user search query or question and a list of existing knowledge collections, decide which single collection is most likely to contain the answer.

Rules:
- Reply with EXACTLY one existing collection name as written in the list (if the query clearly relates to that collection), OR
- Reply with the word UNCLASSIFIED if the query does not clearly relate to any of the listed collections.

Do NOT suggest new collection names. Do NOT explain. Reply with ONLY the collection name or UNCLASSIFIED."""

CLASSIFY_QUERY_COLLECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CLASSIFY_QUERY_COLLECTION_SYSTEM),
        ("human", "Existing collections: {existing_collections}\n\nUser query: {user_query}"),
    ]
)
