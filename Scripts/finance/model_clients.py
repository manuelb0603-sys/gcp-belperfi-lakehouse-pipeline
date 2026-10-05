"""One model-client layer for every text-generation call: Ollama (local) and Anthropic (Claude).

Every client exposes ``generate(prompt, *, system=None, json_schema=None, timeout=None) -> ModelReply``.
``ModelReply`` is a str (so parsers keep working) carrying normalized generation metadata:
``done_reason`` is "stop", "length" (token limit), "refusal" (Claude declined), or None.

Per-role configuration (``model_roles``, with fallback to the legacy model_name /
validator_model_name / ollama_options / model_server_url keys) is resolved by
``resolve_model_role`` and turned into a client by ``build_model_client``.

The ``anthropic`` SDK is imported lazily, so Ollama-only installs never need it. Credentials
come from the environment (ANTHROPIC_API_KEY or an ``ant auth login`` profile), never config.json.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

from .settings import DEFAULTS


logger = logging.getLogger(__name__)

MODEL_ROLES = ("author", "validator", "eval")
DEFAULT_TIMEOUT = DEFAULTS["ollama_timeout"]
DEFAULT_ANTHROPIC_MAX_TOKENS = 16000
# Anthropic rejects an empty user message; used only when the whole prompt went to the system block.
EMPTY_USER_MESSAGE = "Follow the instructions above."


class EmptyModelResponse(ValueError):
    """The model answered without any text; treated like an unavailable model call."""


class ModelReply(str):
    """Model text plus generation metadata; behaves as a plain string for parsing.

    done_reason is normalized across providers ("stop", "length", "refusal"). usage holds
    provider token counts (Anthropic: input/output/cache_read/cache_creation tokens) and
    metadata any extra provider details (Ollama durations, Anthropic stop_details).
    """

    done_reason: str | None
    eval_count: int | None
    prompt_eval_count: int | None
    provider: str | None
    model: str | None
    usage: dict[str, Any] | None
    metadata: dict[str, Any]

    def __new__(
        cls,
        text: str,
        done_reason: str | None = None,
        eval_count: int | None = None,
        prompt_eval_count: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ModelReply":
        reply = super().__new__(cls, text)
        reply.done_reason = done_reason
        reply.eval_count = eval_count
        reply.prompt_eval_count = prompt_eval_count
        reply.provider = provider
        reply.model = model
        reply.usage = usage
        reply.metadata = dict(metadata or {})
        return reply


@runtime_checkable
class ModelClient(Protocol):
    """A text-generation backend for one model."""

    provider: str
    model: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | str | None = None,
        timeout: float | None = None,
    ) -> ModelReply:
        """Generate one reply. system is the stable instruction text; prompt the per-call content."""
        ...


# ----- Ollama -----


def api_url(base_url: str, endpoint: str) -> str:
    """Build an Ollama API endpoint from a generate, embed, or base URL."""
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    for suffix in ("/api/generate", "/api/chat", "/api/embed", "/api/embeddings", "/api"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parts.scheme, parts.netloc, f"{path}/api/{endpoint}", "", ""))


class OllamaClient:
    """Ollama /api/generate and /api/embed client (requests is imported lazily per call)."""

    provider = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
        default_timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.options = dict(options) if options else None
        self.think = think
        self.default_timeout = default_timeout

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | str | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> ModelReply:
        """Call /api/generate. Ollama gets one prompt: ``system + prompt`` concatenated exactly.

        json_schema (a JSON schema dict, or "json") becomes Ollama's ``format``. An empty answer
        raises EmptyModelResponse unless the token limit was spent first (done_reason "length"),
        which returns an empty reply so the caller's format/length handling can report it.
        ``stream`` is kept only for the legacy email path.
        """
        import requests

        full_prompt = f"{system}{prompt}" if system else prompt
        payload: dict[str, Any] = {"model": self.model, "prompt": full_prompt, "stream": stream}
        if json_schema is not None:
            payload["format"] = json_schema
        if self.options:
            payload["options"] = self.options
        if self.think is not None:
            payload["think"] = self.think
        response = requests.post(
            api_url(self.base_url, "generate"),
            json=payload,
            timeout=self.default_timeout if timeout is None else timeout,
        )
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
        metadata = {
            key: body[key]
            for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
            if key in body
        }
        return ModelReply(
            result,
            done_reason=body.get("done_reason"),  # Ollama already reports "stop"/"length"
            eval_count=body.get("eval_count"),
            prompt_eval_count=body.get("prompt_eval_count"),
            provider=self.provider,
            model=self.model,
            metadata=metadata,
        )

    def embed(self, text: str, timeout: float | None = None) -> list[float]:
        """Request and validate a single vector from /api/embed."""
        import requests

        response = requests.post(
            api_url(self.base_url, "embed"),
            json={"model": self.model, "input": text},
            timeout=self.default_timeout if timeout is None else timeout,
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

    def embed_many(self, texts: list[str], timeout: float | None = None) -> list[list[float]]:
        """Embed a batch of texts with one /api/embed request."""
        import requests

        if not texts:
            return []
        response = requests.post(
            api_url(self.base_url, "embed"),
            json={"model": self.model, "input": texts},
            timeout=self.default_timeout if timeout is None else timeout,
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


# ----- Anthropic (Claude) -----

# Claude stop_reason -> the harness's done_reason vocabulary.
_ANTHROPIC_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "refusal",
}


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from an SDK object or a key from a dict (fake clients in tests)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _plain(value: Any) -> Any:
    """SDK model -> plain dict for trace metadata (dicts and None pass through)."""
    if value is None or isinstance(value, (dict, str, int, float, bool)):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    return {key: val for key, val in vars(value).items() if not key.startswith("_")}


class AnthropicClient:
    """Claude Messages API client using the official ``anthropic`` SDK (imported lazily).

    The stable instruction text goes in a ``system`` block marked for prompt caching; the
    per-call content is the single user message. No thinking/sampling parameters are sent
    (current models reject ``thinking: disabled``, ``budget_tokens`` and temperature); depth
    is controlled with ``effort`` only. The SDK is built with max_retries=1 so its own retry
    does not stack with the harness's retry.
    """

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
        effort: str | None = None,
        cache_system: bool = True,
        default_timeout: float = DEFAULT_TIMEOUT,
        client: Any = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = int(max_tokens)
        self.effort = effort
        self.cache_system = cache_system
        self.default_timeout = default_timeout
        self._client = client
        ignored = sorted((options or {}).keys()) + (["think"] if think is not None else [])
        if ignored:
            logger.debug("AnthropicClient(%s) ignores Ollama-only settings: %s", model, ", ".join(ignored))

    def _sdk_client(self) -> Any:
        if self._client is None:
            import anthropic

            # Credentials come from ANTHROPIC_API_KEY or an `ant auth login` profile.
            self._client = anthropic.Anthropic(max_retries=1)
        return self._client

    def request_kwargs(self, prompt: str, system: str | None, json_schema: Any) -> dict[str, Any]:
        """The messages.create arguments for one call (exposed for tests and debugging)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt if prompt.strip() else EMPTY_USER_MESSAGE}],
        }
        if system:
            block: dict[str, Any] = {"type": "text", "text": system}
            if self.cache_system:
                block["cache_control"] = {"type": "ephemeral"}
            kwargs["system"] = [block]
        output_config: dict[str, Any] = {}
        if isinstance(json_schema, dict):
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        elif json_schema is not None:
            logger.debug("AnthropicClient ignores non-schema format %r; send a JSON schema dict instead", json_schema)
        if self.effort:
            output_config["effort"] = self.effort
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | str | None = None,
        timeout: float | None = None,
    ) -> ModelReply:
        """Call messages.create and normalize the reply (refusal -> empty text, max_tokens -> "length")."""
        client = self._sdk_client().with_options(timeout=self.default_timeout if timeout is None else timeout)
        message = client.messages.create(**self.request_kwargs(prompt, system, json_schema))
        stop_reason = _field(message, "stop_reason")
        done_reason = _ANTHROPIC_STOP_REASONS.get(stop_reason, stop_reason)
        usage_obj = _field(message, "usage")
        usage = {
            key: _field(usage_obj, key)
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        } if usage_obj is not None else None
        metadata: dict[str, Any] = {"stop_reason": stop_reason}
        message_id = _field(message, "id")
        if message_id:
            metadata["message_id"] = message_id
        if done_reason == "refusal":
            metadata["stop_details"] = _plain(_field(message, "stop_details"))
            text = ""
        else:
            text = "".join(
                str(_field(block, "text") or "")
                for block in (_field(message, "content") or [])
                if _field(block, "type") == "text"
            )
            if not text.strip() and done_reason != "length":
                raise EmptyModelResponse(f"Claude response did not contain text (stop_reason={stop_reason})")
        return ModelReply(
            text,
            done_reason=done_reason,
            eval_count=usage.get("output_tokens") if usage else None,
            prompt_eval_count=usage.get("input_tokens") if usage else None,
            provider=self.provider,
            model=self.model,
            usage=usage,
            metadata=metadata,
        )


# ----- per-role configuration -----


def _legacy_ollama_section(config: dict[str, Any], role: str) -> dict[str, Any]:
    options = config.get("ollama_options")
    section = options.get(role) if isinstance(options, dict) else None
    return dict(section) if isinstance(section, dict) else {}


def _split_ollama_section(section: dict[str, Any]) -> tuple[dict[str, Any] | None, bool | None]:
    """ollama_options[role] -> (request options without None values, top-level think flag)."""
    options = {key: value for key, value in section.items() if key != "think" and value is not None}
    think = section.get("think")
    return (options or None), (bool(think) if think is not None else None)


def resolve_model_role(config: Any, role: str) -> dict[str, Any]:
    """Resolved settings for one role: provider, model, and provider-specific fields.

    config is a config dict or a finance.settings.ReportSettings (which resolved its roles here).

    model_roles[role] wins when present; otherwise the legacy keys apply (author: model_name;
    validator: validator_model_name, else model_name; ollama_options[role]; model_server_url).
    The "eval" role falls back to the validator's settings when it is not configured.
    """
    if role not in MODEL_ROLES:
        raise ValueError(f"Unknown model role {role!r}; expected one of {', '.join(MODEL_ROLES)}")
    if not isinstance(config, dict) and callable(getattr(config, "role", None)):
        return config.role(role)  # a ReportSettings: roles were resolved once when it was built
    roles = config.get("model_roles")
    entry = roles.get(role) if isinstance(roles, dict) else None
    if not isinstance(entry, dict) or not entry.get("model"):
        if role == "eval":
            return {**resolve_model_role(config, "validator"), "role": "eval"}
        if role == "author":
            model = config.get("model_name")
        else:
            model = config.get("validator_model_name") or config.get("model_name")
        entry = {"provider": "ollama", "model": model}
    provider = str(entry.get("provider") or "ollama").lower()
    resolved: dict[str, Any] = {"role": role, "provider": provider, "model": entry.get("model")}
    timeout = entry.get("timeout", config.get("ollama_timeout", DEFAULT_TIMEOUT))
    resolved["timeout"] = int(timeout) if timeout is not None else DEFAULT_TIMEOUT
    if provider == "ollama":
        if "options" in entry or "think" in entry:
            section = dict(entry.get("options") or {})
            if "think" in entry:
                section["think"] = entry["think"]
        else:
            section = _legacy_ollama_section(config, role)
        resolved["options"], resolved["think"] = _split_ollama_section(section)
        resolved["base_url"] = entry.get("base_url") or config.get("model_server_url")
    elif provider == "anthropic":
        resolved["max_tokens"] = int(entry.get("max_tokens") or DEFAULT_ANTHROPIC_MAX_TOKENS)
        resolved["effort"] = entry.get("effort")
        resolved["cache_system"] = bool(entry.get("cache_system", True))
    else:
        raise ValueError(f"Unsupported provider {provider!r} for model role {role!r} (use 'ollama' or 'anthropic')")
    return resolved


def build_model_client(config: Any, role: str, *, anthropic_client: Any = None) -> ModelClient:
    """Create the ModelClient for a role ("author", "validator", or "eval") from a config dict or ReportSettings.

    anthropic_client lets tests inject a fake SDK client; real runs build anthropic.Anthropic().
    """
    spec = resolve_model_role(config, role)
    if not spec.get("model"):
        raise ValueError(f"No model configured for role {role!r} (set model_roles.{role}.model or model_name)")
    if spec["provider"] == "anthropic":
        return AnthropicClient(
            spec["model"],
            max_tokens=spec["max_tokens"],
            effort=spec["effort"],
            cache_system=spec["cache_system"],
            default_timeout=spec["timeout"],
            client=anthropic_client,
        )
    if not spec.get("base_url"):
        raise ValueError(f"Role {role!r} uses Ollama but model_server_url is not configured")
    return OllamaClient(
        spec["base_url"],
        spec["model"],
        options=spec["options"],
        think=spec["think"],
        default_timeout=spec["timeout"],
    )


# ----- failure classification (no SDK import at module load) -----

_ANTHROPIC_RETRYABLE = {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}


def _anthropic_class_names(exc: BaseException) -> set[str]:
    """Names of the anthropic SDK exception classes exc is an instance of (empty for others)."""
    return {
        cls.__name__ for cls in type(exc).__mro__
        if str(getattr(cls, "__module__", "")).split(".")[0] == "anthropic"
    }


def is_anthropic_timeout(exc: BaseException) -> bool:
    return "APITimeoutError" in _anthropic_class_names(exc)


def is_anthropic_transient(exc: BaseException) -> bool:
    """Timeouts, connection errors, 429 and 5xx from the SDK; 4xx such as BadRequestError are not."""
    names = _anthropic_class_names(exc)
    if names & _ANTHROPIC_RETRYABLE:
        return True
    if "APIStatusError" in names:
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and (status == 429 or status >= 500)
    return False
