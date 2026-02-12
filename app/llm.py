"""LLM: local Ollama (llama3) for RAG answers."""
from langchain_ollama import ChatOllama


def get_chat_model(
    base_url: str = "http://localhost:11434",
    model: str = "llama3",
    temperature: float = 0,
):
    """
    Ollama chat model for RAG. Pull the model first: ollama pull llama3
    """
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=temperature,
    )
