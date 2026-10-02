"""Shared Ollama text-generation and embedding clients."""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def api_url(base_url: str, endpoint: str) -> str:
    """Build an Ollama API endpoint from a generate, embed, or base URL."""
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    for suffix in ("/api/generate", "/api/chat", "/api/embed", "/api/embeddings", "/api"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parts.scheme, parts.netloc, f"{path}/api/{endpoint}", "", ""))


def generate_text(
    prompt: str,
    ollama_url: str,
    model_name: str,
    timeout: int = 600,
    stream: bool = False,
) -> str:
    """Generate text with Ollama and raise for transport or response errors."""
    import requests

    response = requests.post(
        api_url(ollama_url, "generate"),
        json={"model": model_name, "prompt": prompt, "stream": stream},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("response")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Ollama response did not contain text")
    return result.strip()


def embed_text(
    text: str,
    ollama_url: str,
    model_name: str,
    timeout: int = 600,
) -> list[float]:
    """Request and validate a single vector from Ollama's /api/embed endpoint."""
    import requests

    response = requests.post(
        api_url(ollama_url, "embed"),
        json={"model": model_name, "input": text},
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
        vector = embeddings[0]
    else:
        vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ValueError("Ollama embedding response did not contain a vector")
    values = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Ollama embedding response contained non-finite values")
    return values


def embed_texts(
    texts: list[str],
    ollama_url: str,
    model_name: str,
    timeout: int = 600,
) -> list[list[float]]:
    """Embed a batch of texts using Ollama's /api/embed endpoint."""
    import requests

    if not texts:
        return []
    response = requests.post(
        api_url(ollama_url, "embed"),
        json={"model": model_name, "input": texts},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        if len(texts) == 1 and isinstance(payload.get("embedding"), list):
            embeddings = [payload["embedding"]]
        else:
            raise ValueError("Ollama embedding response did not contain one vector per input")
    vectors = [[float(value) for value in vector] for vector in embeddings]
    if any(not vector or not all(math.isfinite(value) for value in vector) for vector in vectors):
        raise ValueError("Ollama embedding response contained an invalid vector")
    return vectors
