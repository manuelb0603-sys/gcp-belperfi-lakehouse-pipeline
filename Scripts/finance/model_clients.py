"""One model-client layer for every text-generation call: Ollama (local) and Anthropic (Claude).

Every client exposes ``generate(prompt, *, system=None, json_schema=None, timeout=None) -> ModelReply``.
Chat-capable clients also expose ``chat(messages, *, system=None, tools=None, json_schema=None,
timeout=None) -> ChatReply`` (provider-neutral messages and tool calls; used by the finance chat).
``ModelReply`` is a str (so parsers keep working) carrying normalized generation metadata:
``done_reason`` is "stop", "length" (token limit), "refusal" (Claude declined), or None.

Per-role configuration (``model_roles``, with fallback to the legacy model_name /
validator_model_name / ollama_options / model_server_url keys) is resolved by
``resolve_model_role`` and turned into a client by ``build_model_client``.

The ``anthropic`` SDK is imported lazily, so Ollama-only installs never need it. Credentials
come from the environment (ANTHROPIC_API_KEY or an ``ant auth login`` profile), never config.json.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

from .budget import BudgetGuard, guard_from_settings
from .pricing import cost_usd, worst_case_call_usd
from .settings import DEFAULTS, GUIDANCE_MODES


logger = logging.getLogger(__name__)

MODEL_ROLES = ("author", "validator", "eval", "chat")
DEFAULT_TIMEOUT = DEFAULTS["ollama_timeout"]
DEFAULT_ANTHROPIC_MAX_TOKENS = 16000
# Upper bound accepted for model_roles.<role>.max_tokens (thinking tokens count toward it).
MAX_ANTHROPIC_MAX_TOKENS = 64000
# With stream unset (auto), requests above this many max_tokens are streamed: the SDK refuses
# non-streaming requests whose expected duration exceeds 10 minutes (about 21k max_tokens).
ANTHROPIC_STREAM_THRESHOLD = 16000
# The only thinking mode a role may request; current models reject "disabled" and budget_tokens.
ANTHROPIC_THINKING_MODES = ("adaptive",)
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


# ----- chat (tool calling) -----


@dataclass
class ChatReply:
    """One chat-model turn, normalized across providers.

    tool_calls is ``[{"id", "name", "arguments": dict}]``; done_reason is "stop", "tool_use",
    "length" (token limit), "refusal" (Claude declined), or None. provider_content keeps the
    provider's own assistant blocks (Claude thinking and tool_use blocks) so the turn can be sent
    back unchanged within the same conversation.
    """

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done_reason: str | None = None
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    provider_content: list[dict[str, Any]] | None = None

    def assistant_message(self) -> dict[str, Any]:
        """The provider-neutral assistant message to append to the conversation."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.text,
            "tool_calls": [dict(call) for call in self.tool_calls],
        }
        if self.provider_content is not None:
            message["provider_content"] = {"provider": self.provider, "model": self.model, "blocks": self.provider_content}
        return message


def tool_message(call_id: str | None, name: str, content: Any, *, is_error: bool = False) -> dict[str, Any]:
    """A provider-neutral tool-result message (content is sent as JSON text)."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    message: dict[str, Any] = {"role": "tool", "tool_call_id": call_id, "name": name, "content": text}
    if is_error:
        message["is_error"] = True
    return message


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _tool_arguments(value: Any) -> dict[str, Any]:
    """Tool-call arguments as a dict; JSON text is parsed, anything else becomes an invalid marker."""
    if isinstance(value, dict):
        return dict(value)
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        return {"__invalid_arguments__": value[:200]}
    plain = _plain(value)
    if isinstance(plain, dict):
        return dict(plain)
    return {"__invalid_arguments__": str(value)[:200]}


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

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | str | None = None,
        timeout: float | None = None,
        think: bool | None = None,
    ) -> ChatReply:
        """Call /api/chat with provider-neutral messages and tools (``{name, description, parameters}``).

        Tool results go back as ``{"role": "tool", "tool_name": ..., "content": json}``. ``think``
        overrides the role's think flag for this call (the chat engine retries once with think off
        when thinking spent the whole token budget). An empty answer is returned, not raised.
        """
        import requests

        payload: dict[str, Any] = {"model": self.model, "messages": _ollama_messages(messages, system), "stream": False}
        if tools:
            payload["tools"] = [
                {"type": "function", "function": {
                    "name": tool["name"], "description": tool.get("description", ""), "parameters": tool["parameters"],
                }}
                for tool in tools
            ]
        if json_schema is not None:
            payload["format"] = json_schema
        if self.options:
            payload["options"] = self.options
        effective_think = self.think if think is None else think
        if effective_think is not None:
            payload["think"] = effective_think
        response = requests.post(
            api_url(self.base_url, "chat"),
            json=payload,
            timeout=self.default_timeout if timeout is None else timeout,
        )
        response.raise_for_status()
        body = response.json()
        message = body.get("message") if isinstance(body.get("message"), dict) else {}
        text = message.get("content") if isinstance(message.get("content"), str) else ""
        calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not function.get("name"):
                continue
            calls.append({
                "id": str(call.get("id") or f"call_{index + 1}"),
                "name": str(function["name"]),
                "arguments": _tool_arguments(function.get("arguments")),
            })
        done_reason = body.get("done_reason")
        if calls and done_reason in (None, "stop"):
            done_reason = "tool_use"
        metadata: dict[str, Any] = {
            key: body[key]
            for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
            if key in body
        }
        metadata["think"] = effective_think
        metadata["thinking_chars"] = len(message.get("thinking") or "")
        return ChatReply(
            text,
            calls,
            done_reason=done_reason,
            usage={"prompt_eval_count": body.get("prompt_eval_count"), "eval_count": body.get("eval_count")},
            metadata=metadata,
            provider=self.provider,
            model=self.model,
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


def _ollama_messages(messages: list[dict[str, Any]], system: str | None) -> list[dict[str, Any]]:
    """Provider-neutral messages -> Ollama /api/chat messages (system first)."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for message in messages:
        role = message.get("role") or "user"
        if role == "tool":
            out.append({"role": "tool", "tool_name": message.get("name"), "content": _content_text(message.get("content"))})
        elif role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": _content_text(message.get("content"))}
            if message.get("tool_calls"):
                item["tool_calls"] = [
                    {"function": {"name": call["name"], "arguments": call.get("arguments") or {}}}
                    for call in message["tool_calls"]
                ]
            out.append(item)
        else:
            out.append({"role": role, "content": _content_text(message.get("content"))})
    return out


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


def _http_class_names(exc: BaseException) -> set[str]:
    """Names of httpx/httpx2 exception classes exc is an instance of (empty for others)."""
    return {
        cls.__name__ for cls in type(exc).__mro__
        if str(getattr(cls, "__module__", "")).split(".")[0] in {"httpx", "httpx2"}
    }


class AnthropicClient:
    """Claude Messages API client using the official ``anthropic`` SDK (imported lazily).

    The stable instruction text goes in a ``system`` block marked for prompt caching; the
    per-call content is the single user message. No sampling parameters, ``thinking:
    disabled`` or ``budget_tokens`` are sent (current models reject them). Claude Opus 5.5
    and Sonnet 5.5 think adaptively by default and ``effort`` controls the depth; a role may
    set ``thinking="adaptive"`` (sent as ``{"type": "adaptive"}``) for models where thinking
    is off by default. Thinking tokens count toward ``max_tokens``; the thinking display is
    left at the API default.

    ``stream`` None (auto) streams when max_tokens exceeds ANTHROPIC_STREAM_THRESHOLD (the SDK
    requires streaming for long requests); the reply is built the same way either way. The SDK
    is built with max_retries=1 so its own retry does not stack with the harness's retry.

    ``budget_guard`` (finance.budget.BudgetGuard, attached by build_model_client) is asked before
    every request and records every billed call in the cost ledger; a refused call raises
    BudgetExceededError without contacting the API. A real SDK client is never built without a
    guard; only an injected fake SDK client (tests) may run unguarded. Each reply carries
    ``metadata["cost_usd"]``.
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
        stream: bool | None = None,
        thinking: str | None = None,
        budget_guard: BudgetGuard | None = None,
        role: str | None = None,
    ) -> None:
        self.model = model
        self.budget_guard = budget_guard
        self.role = role
        self.max_tokens = int(max_tokens)
        self.effort = effort
        self.cache_system = cache_system
        self.default_timeout = default_timeout
        self.stream = stream
        if thinking is not None and thinking not in ANTHROPIC_THINKING_MODES:
            raise ValueError(f"thinking must be 'adaptive' or omitted (Claude thinking cannot be disabled), got {thinking!r}")
        self.thinking = thinking
        self._client = client
        ignored = sorted((options or {}).keys()) + (["think"] if think is not None else [])
        if ignored:
            logger.debug("AnthropicClient(%s) ignores Ollama-only settings: %s", model, ", ".join(ignored))

    def _sdk_client(self) -> Any:
        if self._client is None:
            if self.budget_guard is None:
                # Real API calls are always metered; build clients with build_model_client.
                raise RuntimeError(
                    "AnthropicClient needs a BudgetGuard before it may call the Claude API; use build_model_client"
                )
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
        if self.thinking:
            kwargs["thinking"] = {"type": self.thinking}
        return kwargs

    def uses_streaming(self) -> bool:
        """Whether generate() streams: the explicit stream setting, else max_tokens above the threshold."""
        return self.stream if self.stream is not None else self.max_tokens > ANTHROPIC_STREAM_THRESHOLD

    @staticmethod
    def _streamed_message(client: Any, kwargs: dict[str, Any], timeout: float) -> Any:
        """Run one streamed request and return its final message.

        The SDK timeout applies per network read, so the total call time is also checked between
        stream events. Exceeding it, or an httpx read/transport error mid-stream (the SDK does not
        wrap those), raises TimeoutError/ConnectionError so the harness treats it like any failed call.
        """
        deadline = time.monotonic() + timeout
        try:
            with client.messages.stream(**kwargs) as stream:
                if hasattr(stream, "__iter__"):
                    for _event in stream:
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"Claude stream exceeded the {timeout:g}s call timeout")
                return stream.get_final_message()
        except TimeoutError:
            raise
        except Exception as exc:
            names = _http_class_names(exc)
            if "TimeoutException" in names:
                raise TimeoutError(f"Claude stream timed out: {exc}") from exc
            if names & {"TransportError", "NetworkError", "RemoteProtocolError"}:
                raise ConnectionError(f"Claude stream was interrupted: {exc}") from exc
            raise

    def _record_failed_call(self, exc: Exception, prompt_chars: int) -> None:
        """Record a failed request that may still have been billed.

        A timeout or an interrupted stream may have consumed the prompt, so its input is recorded
        at the list price (estimated from the characters, marked estimated). Errors the API rejects
        up front (4xx) are not billed and are not recorded.
        """
        if self.budget_guard is None:
            return
        if not (isinstance(exc, (TimeoutError, ConnectionError)) or is_anthropic_transient(exc)):
            return
        if "RateLimitError" in _anthropic_class_names(exc):
            return  # rejected before any processing
        estimate = worst_case_call_usd(self.model, prompt_chars, 0)
        try:
            exc.cost_usd = estimate  # type: ignore[attr-defined]
        except AttributeError:
            pass
        self.budget_guard.record(
            self.model, None, role=self.role, outcome=f"error:{type(exc).__name__}", cost=estimate, estimated=True
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | str | None = None,
        timeout: float | None = None,
    ) -> ModelReply:
        """Call the Messages API and normalize the reply (refusal -> empty text, max_tokens -> "length").

        Streams (messages.stream, final message only) when uses_streaming(), else messages.create.
        """
        call_timeout = self.default_timeout if timeout is None else timeout
        kwargs = self.request_kwargs(prompt, system, json_schema)
        prompt_chars = len(system or "") + len(kwargs["messages"][0]["content"])
        if self.budget_guard is not None:
            # Raises BudgetExceededError (nothing is sent) when this call's worst case could pass a cap.
            self.budget_guard.check(self.model, prompt_chars, self.max_tokens, role=self.role)
        client = self._sdk_client().with_options(timeout=call_timeout)
        streamed = self.uses_streaming()
        try:
            if streamed:
                message = self._streamed_message(client, kwargs, call_timeout)
            else:
                message = client.messages.create(**kwargs)
        except Exception as exc:
            self._record_failed_call(exc, prompt_chars)
            raise
        stop_reason = _field(message, "stop_reason")
        done_reason = _ANTHROPIC_STOP_REASONS.get(stop_reason, stop_reason)
        usage_obj = _field(message, "usage")
        usage = {
            key: _field(usage_obj, key)
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        } if usage_obj is not None else None
        call_cost = cost_usd(self.model, usage)
        metadata: dict[str, Any] = {"stop_reason": stop_reason, "streamed": streamed, "cost_usd": call_cost}
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
        empty = done_reason != "refusal" and not text.strip() and done_reason != "length"
        if self.budget_guard is not None:
            # Every billed call is recorded, including refusals and empty answers; a failed write raises.
            outcome = "empty" if empty else (done_reason or "ok")
            self.budget_guard.record(self.model, usage, role=self.role, outcome=outcome, cost=call_cost)
        if empty:
            error = EmptyModelResponse(f"Claude response did not contain text (stop_reason={stop_reason})")
            error.cost_usd = call_cost  # type: ignore[attr-defined]
            raise error
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

    def chat_request_kwargs(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]] | None,
        json_schema: Any,
    ) -> dict[str, Any]:
        """The messages.create arguments for one chat call (exposed for tests and debugging).

        The system block is cached and the last tool carries cache_control (with cache_system), so
        repeated rounds read the instructions and tool definitions from the prompt cache.
        """
        kwargs = self.request_kwargs("", system, json_schema)
        kwargs["messages"] = _anthropic_messages(messages, self.model)
        if tools:
            definitions = [
                {"name": tool["name"], "description": tool.get("description", ""), "input_schema": tool["parameters"]}
                for tool in tools
            ]
            if self.cache_system:
                definitions[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = definitions
            kwargs["tool_choice"] = {"type": "auto"}
        return kwargs

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | str | None = None,
        timeout: float | None = None,
        think: bool | None = None,
    ) -> ChatReply:
        """One Messages API call with tools (tool_choice auto), through the BudgetGuard like generate().

        tool_use blocks become tool_calls with their ids; send results back as tool messages (they
        become one user message of tool_result blocks). The assistant's own blocks (including
        thinking) are kept in provider_content and replayed unchanged. A refusal returns empty text
        with stop_details; a max_tokens stop returns done_reason "length" and drops any tool calls.
        ``think`` is accepted for interface parity and ignored (Claude thinking cannot be disabled).
        """
        return _anthropic_chat(self, messages, system, tools, json_schema, timeout)


def _anthropic_blocks_from_message(message: dict[str, Any], model: str) -> list[dict[str, Any]]:
    """Assistant content blocks: the provider's own blocks when they came from this model, else rebuilt."""
    stored = message.get("provider_content")
    if isinstance(stored, dict) and stored.get("provider") == "anthropic" and stored.get("model") == model and stored.get("blocks"):
        return [dict(block) for block in stored["blocks"]]
    blocks: list[dict[str, Any]] = []
    text = _content_text(message.get("content"))
    if text.strip():
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        blocks.append({"type": "tool_use", "id": call.get("id"), "name": call["name"], "input": call.get("arguments") or {}})
    return blocks


def _anthropic_messages(messages: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Provider-neutral messages -> Messages API messages.

    Consecutive tool results become one user message of tool_result blocks (matching tool_use_id);
    a user note that directly follows them joins the same message after the results.
    """
    out: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    def flush() -> None:
        if results:
            out.append({"role": "user", "content": list(results)})
            results.clear()

    for message in messages:
        role = message.get("role") or "user"
        if role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result", "tool_use_id": message.get("tool_call_id"),
                "content": _content_text(message.get("content")),
            }
            if message.get("is_error"):
                block["is_error"] = True
            results.append(block)
            continue
        if role == "user" and results:
            text = _content_text(message.get("content"))
            if text.strip():
                results.append({"type": "text", "text": text})
            flush()
            continue
        flush()
        if role == "assistant":
            blocks = _anthropic_blocks_from_message(message, model)
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        else:
            text = _content_text(message.get("content"))
            out.append({"role": "user", "content": text if text.strip() else EMPTY_USER_MESSAGE})
    flush()
    return out


def _anthropic_block_param(block: Any) -> dict[str, Any] | None:
    """A response content block as a request param, so the assistant turn can be sent back unchanged."""
    kind = _field(block, "type")
    if kind == "text":
        return {"type": "text", "text": str(_field(block, "text") or "")}
    if kind == "tool_use":
        return {"type": "tool_use", "id": _field(block, "id"), "name": _field(block, "name"),
                "input": _tool_arguments(_field(block, "input"))}
    if kind == "thinking":
        param = {"type": "thinking", "thinking": _field(block, "thinking") or ""}
        if _field(block, "signature") is not None:
            param["signature"] = _field(block, "signature")
        return param
    if kind == "redacted_thinking":
        return {"type": "redacted_thinking", "data": _field(block, "data")}
    plain = _plain(block)
    return {key: value for key, value in plain.items() if value is not None} if isinstance(plain, dict) else None


def _anthropic_chat(client_obj: "AnthropicClient", messages: list[dict[str, Any]], system: str | None,
                    tools: list[dict[str, Any]] | None, json_schema: Any, timeout: float | None) -> ChatReply:
    """AnthropicClient.chat: one guarded Messages API call with tools (see AnthropicClient.chat)."""
    self = client_obj
    call_timeout = self.default_timeout if timeout is None else timeout
    kwargs = self.chat_request_kwargs(messages, system, tools, json_schema)
    prompt_chars = (
        len(system or "")
        + len(json.dumps(kwargs["messages"], ensure_ascii=False, default=str))
        + len(json.dumps(kwargs.get("tools") or [], ensure_ascii=False, default=str))
    )
    if self.budget_guard is not None:
        # Raises BudgetExceededError (nothing is sent) when this call's worst case could pass a cap.
        self.budget_guard.check(self.model, prompt_chars, self.max_tokens, role=self.role)
    sdk = self._sdk_client().with_options(timeout=call_timeout)
    streamed = self.uses_streaming()
    try:
        if streamed:
            message = self._streamed_message(sdk, kwargs, call_timeout)
        else:
            message = sdk.messages.create(**kwargs)
    except Exception as exc:
        self._record_failed_call(exc, prompt_chars)
        raise
    stop_reason = _field(message, "stop_reason")
    done_reason = _ANTHROPIC_STOP_REASONS.get(stop_reason, stop_reason)
    usage_obj = _field(message, "usage")
    usage = {
        key: _field(usage_obj, key)
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    } if usage_obj is not None else None
    call_cost = cost_usd(self.model, usage)
    metadata: dict[str, Any] = {"stop_reason": stop_reason, "streamed": streamed, "cost_usd": call_cost}
    message_id = _field(message, "id")
    if message_id:
        metadata["message_id"] = message_id
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for block in _field(message, "content") or []:
        param = _anthropic_block_param(block)
        if param is None:
            continue
        blocks.append(param)
        if param["type"] == "text":
            texts.append(param["text"])
        elif param["type"] == "tool_use":
            calls.append({"id": param["id"], "name": param["name"], "arguments": dict(param["input"])})
    text = "".join(texts)
    provider_content: list[dict[str, Any]] | None = blocks
    if done_reason == "refusal":
        metadata["stop_details"] = _plain(_field(message, "stop_details"))
        text, calls, provider_content = "", [], None
    elif done_reason == "length":
        if calls:
            metadata["dropped_tool_calls"] = len(calls)  # inputs may be cut off at max_tokens
        calls, provider_content = [], None
    empty = not text.strip() and not calls and done_reason not in ("refusal", "length")
    if self.budget_guard is not None:
        # Every billed call is recorded, including refusals and empty answers; a failed write raises.
        self.budget_guard.record(
            self.model, usage, role=self.role, outcome="empty" if empty else (done_reason or "ok"), cost=call_cost
        )
    return ChatReply(
        text,
        calls,
        done_reason=done_reason,
        usage=usage,
        metadata=metadata,
        provider=self.provider,
        model=self.model,
        provider_content=provider_content,
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
    The "chat" role (finance chat) falls back to model_name with ollama_options["chat"].
    model_roles.author.guidance ("strict" or "guided") is kept as the author's "guidance" key
    (it overrides the top-level guidance; see finance.settings), even when the entry has no model.
    """
    if role not in MODEL_ROLES:
        raise ValueError(f"Unknown model role {role!r}; expected one of {', '.join(MODEL_ROLES)}")
    if not isinstance(config, dict) and callable(getattr(config, "role", None)):
        return config.role(role)  # a ReportSettings: roles were resolved once when it was built
    roles = config.get("model_roles")
    entry = roles.get(role) if isinstance(roles, dict) else None
    guidance = entry.get("guidance") if role == "author" and isinstance(entry, dict) else None
    if guidance is not None:
        if not isinstance(guidance, str) or guidance.strip().lower() not in GUIDANCE_MODES:
            raise ValueError(f"model_roles.author.guidance must be one of {', '.join(GUIDANCE_MODES)}, got {guidance!r}")
        guidance = guidance.strip().lower()
    if not isinstance(entry, dict) or not entry.get("model"):
        if role == "eval":
            return {**resolve_model_role(config, "validator"), "role": "eval"}
        if role in ("author", "chat"):
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
        raw_max = entry.get("max_tokens")
        if raw_max is None:
            raw_max = DEFAULT_ANTHROPIC_MAX_TOKENS
        max_tokens: int | None = None
        if isinstance(raw_max, int) and not isinstance(raw_max, bool):
            max_tokens = raw_max
        elif isinstance(raw_max, float) and raw_max.is_integer():
            max_tokens = int(raw_max)
        elif isinstance(raw_max, str) and raw_max.strip().isdigit():
            max_tokens = int(raw_max)
        if max_tokens is None or not 1 <= max_tokens <= MAX_ANTHROPIC_MAX_TOKENS:
            raise ValueError(
                f"model_roles.{role}.max_tokens must be an integer from 1 to {MAX_ANTHROPIC_MAX_TOKENS}, got {raw_max!r}"
            )
        resolved["max_tokens"] = max_tokens
        resolved["effort"] = entry.get("effort")
        resolved["cache_system"] = bool(entry.get("cache_system", True))
        stream = entry.get("stream")
        if stream is not None and not isinstance(stream, bool):
            raise ValueError(f"model_roles.{role}.stream must be true, false, or omitted (auto), got {stream!r}")
        resolved["stream"] = stream
        thinking = entry.get("thinking")
        if thinking is not None and thinking not in ANTHROPIC_THINKING_MODES:
            raise ValueError(
                f"model_roles.{role}.thinking must be 'adaptive' or omitted (Claude thinking cannot be disabled), got {thinking!r}"
            )
        resolved["thinking"] = thinking
    else:
        raise ValueError(f"Unsupported provider {provider!r} for model role {role!r} (use 'ollama' or 'anthropic')")
    if guidance is not None:
        resolved["guidance"] = guidance
    return resolved


def build_model_client(
    config: Any,
    role: str,
    *,
    anthropic_client: Any = None,
    run_id: str | None = None,
    budget_guard: BudgetGuard | None = None,
) -> ModelClient:
    """Create the ModelClient for a role ("author", "validator", "eval", or "chat") from a config dict or ReportSettings.

    anthropic_client lets tests inject a fake SDK client; real runs build anthropic.Anthropic().
    Claude clients always get a BudgetGuard: budget_guard when given, else one built from the
    config's anthropic_* budget keys (caps ON by default) for run_id ("other" when omitted; use
    the report run id, "replay", "compare", or "chat"). Ollama clients are never budgeted.
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
            stream=spec.get("stream"),
            thinking=spec.get("thinking"),
            budget_guard=budget_guard or guard_from_settings(config, run_id),
            role=role,
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
        if isinstance(status, int) and (status == 429 or status >= 500):
            return True
        # An error event inside a stream arrives on an HTTP 200 response; its body names the error type.
        body = getattr(exc, "body", None)
        error = body.get("error") if isinstance(body, dict) else None
        return isinstance(error, dict) and error.get("type") in {"overloaded_error", "api_error", "rate_limit_error"}
    return False
