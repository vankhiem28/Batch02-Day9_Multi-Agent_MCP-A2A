"""Shared LangChain chat-model factory for all agents.

Defaults to a local Ollama endpoint that exposes an OpenAI-compatible API.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a LangChain chat model backed by an OpenAI-compatible endpoint."""
    return ChatOpenAI(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        temperature=0.3,
    )
