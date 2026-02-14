"""
System Instructions (Grounding).

We "lock" the LLM's behavior so it uses ONLY the provided context.
If the answer is not in the context, it must say so instead of using outside knowledge.
"""
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_INSTRUCTION = """You are an HR Assistant. Use ONLY the provided context to answer.
If the answer is not in the context, say: "I cannot find that in the manual."
Do not use outside knowledge. Do not make up policy details."""

# schema_hint: cheat sheet of collections and their filters (injected when collection is known; can be empty)
RAG_PROMPT_SIMPLE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTION + "\n\n{schema_hint}\n\nContext:\n{context}"),
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

# ----- Schema-driven: metadata extraction (ingestion) -----

METADATA_EXTRACT_SYSTEM = """You are a metadata extractor. Given a document excerpt and a list of metadata field names, extract values for those fields from the document. Use short, normalized values (e.g. city code 'NY' not 'New York', department name like 'HR' or 'Engineering'). If a value cannot be determined, omit the key or use null. Reply with ONLY a valid JSON object, no other text. Example: {"city": "NY", "department": "HR"}"""

METADATA_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", METADATA_EXTRACT_SYSTEM),
        ("human", "Metadata fields to extract: {field_names}\n\nDocument excerpt:\n{excerpt}"),
    ]
)

# ----- Schema-driven: filter extraction from user query (search/ask) -----

EXTRACT_FILTER_SYSTEM = """You extract filter values from a user query for a document search. You are given the allowed filter field names for the current collection and a hint on when to use them. Output ONLY a JSON object with those field names as keys and extracted values (or null if not mentioned). Use short, normalized values. If the user does not mention a filter, do not include it or set it to null. Reply with ONLY valid JSON, no explanation."""

EXTRACT_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_FILTER_SYSTEM),
        ("human", "Allowed filters: {field_names}\nHint: {schema_hint}\n\nUser query: {user_query}"),
    ]
)
