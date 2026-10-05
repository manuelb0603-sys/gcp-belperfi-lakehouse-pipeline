"""Model transport for harness turns: the legacy query_ollama function and adapters over ModelClient."""

from __future__ import annotations

from typing import Any, Callable

from ..model_clients import ModelReply, OllamaClient


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

    Compatibility wrapper over OllamaClient.generate. ``ollama_url`` may be the server base URL
    or any Ollama API endpoint; it is normalized to /api/generate.
    """
    client = OllamaClient(ollama_url, model_name, options=options, think=think, default_timeout=timeout)
    return client.generate(prompt, json_schema=response_format, timeout=timeout)


class QueryFunctionClient:
    """Adapts a legacy ``query(prompt, url, model, timeout=, response_format=, **role_kwargs)`` callable.

    Used when a ReportHarness is given ``query=`` (tests inject fakes this way). Like Ollama, it
    receives the system and user text concatenated into one prompt.
    """

    def __init__(
        self,
        query: Callable[..., str],
        url: str,
        model: str,
        role_kwargs: dict[str, Any] | None = None,
        provider: str = "ollama",
    ) -> None:
        self.query = query
        self.url = url
        self.model = model
        self.role_kwargs = dict(role_kwargs or {})
        self.provider = provider

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | str | None = None,
        timeout: float | None = None,
    ) -> str:
        full_prompt = f"{system}{prompt}" if system else prompt
        return self.query(
            full_prompt,
            self.url,
            self.model,
            timeout=timeout,
            response_format=json_schema,
            **self.role_kwargs,
        )


def _response_meta(response: str) -> dict[str, Any]:
    """Generation metadata for trace events; absent for plain-string test doubles.

    Always has done_reason/eval_count/prompt_eval_count; provider token usage (Anthropic cache
    hits) and refusal details are added only when the reply carries them.
    """
    meta: dict[str, Any] = {
        "done_reason": getattr(response, "done_reason", None),
        "eval_count": getattr(response, "eval_count", None),
        "prompt_eval_count": getattr(response, "prompt_eval_count", None),
    }
    usage = getattr(response, "usage", None)
    if usage:
        meta["usage"] = usage
    metadata = getattr(response, "metadata", None) or {}
    stop_details = metadata.get("stop_details")
    if stop_details is not None:
        meta["stop_details"] = stop_details
    if metadata.get("cost_usd") is not None:
        meta["cost_usd"] = metadata["cost_usd"]  # Claude calls: USD recorded in the cost ledger
    return meta
