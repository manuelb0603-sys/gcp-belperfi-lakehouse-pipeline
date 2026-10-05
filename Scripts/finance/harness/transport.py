"""Thin Ollama transport for model turns (to be replaced by a shared model client)."""

from __future__ import annotations

from typing import Any

from ..ollama import api_url
from .errors import EmptyModelResponse
from .protocol import ModelReply


def query_ollama(
    prompt: str,
    ollama_url: str,
    model_name: str,
    timeout: int = 600,
    response_format: str | dict[str, Any] | None = "json",
    options: dict[str, Any] | None = None,
    think: bool | None = None,
) -> ModelReply:
    """Call Ollama and return the text response for one model turn with generation metadata.

    ``ollama_url`` may be the server base URL or any Ollama API endpoint; it is normalized to /api/generate.
    """
    import requests

    payload: dict[str, Any] = {"model": model_name, "prompt": prompt, "stream": False}
    if response_format is not None:
        payload["format"] = response_format
    if options:
        payload["options"] = options
    if think is not None:
        payload["think"] = think
    response = requests.post(api_url(ollama_url, "generate"), json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    result = body.get("response")
    if not isinstance(result, str) or not result.strip():
        if body.get("done_reason") == "length":
            # The token limit was spent before any answer (often on hidden thinking); retrying the
            # same call would fail the same way, so let the format/length handling report it.
            thinking_chars = len(body.get("thinking") or "")
            print(
                f"Ollama hit num_predict with no answer text ({thinking_chars} thinking chars); "
                "raise num_predict or set think=false for this role.",
                flush=True,
            )
            result = ""
        else:
            raise EmptyModelResponse("Ollama response did not contain text")
    return ModelReply(
        result,
        done_reason=body.get("done_reason"),
        eval_count=body.get("eval_count"),
        prompt_eval_count=body.get("prompt_eval_count"),
    )


def _response_meta(response: str) -> dict[str, Any]:
    """Generation metadata for trace events; absent for plain-string test doubles."""
    return {
        "done_reason": getattr(response, "done_reason", None),
        "eval_count": getattr(response, "eval_count", None),
        "prompt_eval_count": getattr(response, "prompt_eval_count", None),
    }
