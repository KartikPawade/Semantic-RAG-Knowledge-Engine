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
