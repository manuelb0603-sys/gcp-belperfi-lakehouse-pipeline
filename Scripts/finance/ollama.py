"""Compatibility functions over finance.model_clients.OllamaClient for existing callers and tests."""

from __future__ import annotations

from .model_clients import OllamaClient, api_url  # noqa: F401


def generate_text(
    prompt: str,
    ollama_url: str,
    model_name: str,
    timeout: int = 600,
    stream: bool = False,
) -> str:
    """Generate text with Ollama and raise for transport or response errors (result is stripped)."""
    reply = OllamaClient(ollama_url, model_name, default_timeout=timeout).generate(prompt, stream=stream)
    if not reply.strip():
        raise ValueError("Ollama response did not contain text")
    return str(reply).strip()


def embed_text(text: str, ollama_url: str, model_name: str, timeout: int = 600) -> list[float]:
    """Request and validate a single vector from Ollama's /api/embed endpoint."""
    return OllamaClient(ollama_url, model_name, default_timeout=timeout).embed(text)


def embed_texts(texts: list[str], ollama_url: str, model_name: str, timeout: int = 600) -> list[list[float]]:
    """Embed a batch of texts using Ollama's /api/embed endpoint."""
    return OllamaClient(ollama_url, model_name, default_timeout=timeout).embed_many(texts)
