"""
anthropic_provider.py

Anthropic (Claude) chat completion provider for LLMSwitch.

Prompt-caching strategy
-----------------------
Anthropic supports two caching mechanisms:

1. **Explicit block-level caching** (default, used here):
   ``cache_control`` markers are injected on the last system content block and
   the last tool definition.  This caches the static prefix (system prompt +
   tool schemas) independently from the growing conversation history.

2. **Automatic caching** (top-level ``cache_control`` param):
   Pass ``cache_control={"type": "ephemeral"}`` as a keyword argument to
   ``client.chat.completions.create(...)`` and the provider will forward it
   as a top-level field to ``messages.create``.  Anthropic then automatically
   caches up to the last cacheable block.  Best for multi-turn conversations
   where the full history should be cached automatically.

When ``enable_prompt_caching=True`` (default), the provider injects
``cache_control`` on system blocks and tool definitions (Method 1).  You can
also pass ``cache_control`` in ``create()`` kwargs to activate Method 2.

The cache TTL is 5 minutes by default (``{"type": "ephemeral"}``) or 1 hour
when ``"ttl": "1h"`` is included.

Minimum cacheable tokens (requests below the threshold silently skip caching)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Claude Opus 4.5 / 4.6 / 4.7: ≥ 4 096 tokens
- Claude Sonnet 4.6:            ≥ 2 048 tokens
- Claude Sonnet 4.5, Opus 4 / 4.1, Sonnet 4 / 3.7: ≥ 1 024 tokens
- Claude Haiku 4.5:             ≥ 4 096 tokens
- Claude Haiku 3.5:             ≥ 2 048 tokens

Usage reporting
~~~~~~~~~~~~~~~
``CompletionUsage.cache_creation_input_tokens`` — tokens written to cache.
``CompletionUsage.cache_read_input_tokens``     — tokens served from cache.
``CompletionUsage.cached_tokens``               — convenience property (same as
                                                  ``cache_read_input_tokens``).

Classes:
    AnthropicChatProvider: Sync chat provider for Anthropic Claude models.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal, NoReturn

import anthropic  # type: ignore[import-not-found]

from ..base import BaseChatProvider
from ..constants import DEFAULT_ANTHROPIC_MODEL
from ..errors import (
    APIConnectionError,
    AuthenticationError,
    ChatCompletionError,
    RateLimitExceededError,
    TimeOutError,
)
from ..models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkDelta,
    CompletionUsage,
    Function,
)
from ..structured import (
    is_pydantic_model_class,
    populate_parsed,
    provider_extensions_for,
    provider_extensions_from_raw,
    to_jsonable,
)
from ._responses_emulation import AsyncEmulatedResponsesProvider, SyncEmulatedResponsesProvider
from ._shared import merge_client_options, normalize_token_limit

logger = logging.getLogger(__name__)

_CACHE_CONTROL_EPHEMERAL: dict[str, Any] = {"type": "ephemeral"}
_UNSUPPORTED_OPENAI_PARAMS: tuple[str, ...] = (
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stream_options",
    "top_logprobs",
    "user",
    "store",
    "reasoning_effort",
    "prediction",
    "modalities",
    "audio",
    "parallel_tool_calls",
)
_CHUNK_FINISH_REASON = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
    "finishreason.stop",
    "max_completion_tokens",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_cache_control_system(
        system_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Attach ``cache_control`` to the last system content block so that the
    entire system prompt is stored in the Anthropic prompt cache.

    Args:
        system_blocks: List of Anthropic system content dicts.

    Returns:
        Same list with ``cache_control`` added to the last block.
    """
    if not system_blocks:
        return system_blocks
    result = [dict(b) for b in system_blocks]
    result[-1]["cache_control"] = _CACHE_CONTROL_EPHEMERAL
    return result


def _inject_cache_control_tools(
        tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Attach ``cache_control`` to the last tool definition so that the entire
    tool-schema block is stored in the Anthropic prompt cache.

    Args:
        tools: List of Anthropic tool dicts.

    Returns:
        Same list with ``cache_control`` added to the last tool.
    """
    if not tools:
        return tools
    result = [dict(t) for t in tools]
    result[-1]["cache_control"] = _CACHE_CONTROL_EPHEMERAL
    return result


def _openai_messages_to_anthropic(
        messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Convert OpenAI-style messages into Anthropic ``system`` blocks and
    ``messages`` list.

    OpenAI ``system`` → Anthropic ``system`` (list of text blocks).
    OpenAI ``assistant`` with ``tool_calls`` → Anthropic ``tool_use`` blocks.
    OpenAI ``tool`` (tool results) → Anthropic ``tool_result`` blocks.

    Args:
        messages: List of OpenAI-format message dicts.

    Returns:
        Tuple of (system_blocks, anthropic_messages).
    """
    system_blocks: list[dict[str, Any]] = []
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role in {"system", "developer"}:
            text = _content_to_text(content)
            if text:
                system_blocks.append({"type": "text", "text": text})
            continue

        if role == "user":
            parts = _content_to_anthropic_parts(content)
            anthropic_messages.append({"role": "user", "content": parts})
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            if content:
                text = _content_to_text(content)
                if text:
                    parts.append({"type": "text", "text": text})
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                tc_data = tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else tc
                fn = tc_data.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    input_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    input_dict = {"value": raw_args}
                parts.append({
                    "type": "tool_use",
                    "id": tc_data.get("id", "call_0"),
                    "name": fn.get("name", ""),
                    "input": input_dict,
                })
            if parts:
                anthropic_messages.append({"role": "assistant", "content": parts})
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "call_0")
            tool_result_content = _content_to_text(content) if content else ""
            # Append to last user turn if it contains tool_results, else create new turn
            tool_result_block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": tool_result_content,
            }
            if (
                    anthropic_messages
                    and anthropic_messages[-1]["role"] == "user"
                    and isinstance(anthropic_messages[-1]["content"], list)
                    and any(b.get("type") == "tool_result" for b in anthropic_messages[-1]["content"])
            ):
                anthropic_messages[-1]["content"].append(tool_result_block)
            else:
                anthropic_messages.append({"role": "user", "content": [tool_result_block]})
            continue

        logger.warning("Anthropic provider: unknown message role '%s', skipping.", role)

    return system_blocks, anthropic_messages


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)


def _content_to_anthropic_parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for p in content:
            if not isinstance(p, dict):
                logger.warning("Anthropic: dropping unsupported content part of type %s", type(p).__name__)
                continue
            if p.get("type") == "text":
                parts.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") == "image_url":
                url = p.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    header, data = url.split(",", 1)
                    media_type = header[5:].split(";", 1)[0] or "image/jpeg"
                    parts.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    })
                else:
                    parts.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
            else:
                logger.warning("Anthropic: dropping unsupported content part of type %s", p.get("type"))
        return parts or [{"type": "text", "text": ""}]
    return [{"type": "text", "text": str(content)}]


def _convert_tools_to_anthropic(tools: list[Any]) -> list[dict[str, Any]]:
    """
    Convert OpenAI-format tool dicts to Anthropic tool dicts.

    Args:
        tools: List of OpenAI tool dicts (type=function).

    Returns:
        List of Anthropic tool definition dicts.
    """
    result: list[dict[str, Any]] = []
    for tool in tools:
        tool_data = tool.model_dump(exclude_none=True) if hasattr(tool, "model_dump") else tool
        if not isinstance(tool_data, dict) or tool_data.get("type") != "function":
            continue
        fn = tool_data.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return result


def _apply_parse_model(kwargs: dict[str, Any]) -> Any:
    """
    Translate ``response_format=MyModel`` (.parse() style) into Anthropic's
    structured-output mechanism: a forced tool whose ``input_schema`` is the
    model's JSON schema.

    Anthropic has no native parse and ignores ``response_format``; forcing a
    single tool is the documented way to guarantee a schema-conforming object,
    which is then returned as the tool's ``input``. Mutates ``kwargs`` in place
    and returns the model class (or ``None`` when not applicable) so the caller
    can deserialize ``message.parsed`` afterwards.
    """
    parse_model = kwargs.get("response_format")
    if not is_pydantic_model_class(parse_model):
        return None
    schema_tool = {
        "type": "function",
        "function": {
            "name": parse_model.__name__,
            "description": f"Return a JSON object that conforms to the {parse_model.__name__} schema.",
            "parameters": parse_model.model_json_schema(),
        },
    }
    tools = list(kwargs.get("tools") or [])
    tools.append(schema_tool)
    kwargs["tools"] = tools
    kwargs["tool_choice"] = {"function": {"name": parse_model.__name__}}
    kwargs.pop("response_format", None)
    return parse_model


def _build_usage(usage: Any) -> CompletionUsage:
    """Convert Anthropic ``Usage`` object to ``CompletionUsage``."""
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    prompt_tokens = input_tokens + cache_creation + cache_read

    output_details = getattr(usage, "output_tokens_details", None)
    thinking_tokens = getattr(output_details, "thinking_tokens", 0) or 0

    cache_creation_obj = getattr(usage, "cache_creation", None)
    cache_5m = getattr(cache_creation_obj, "ephemeral_5m_input_tokens", 0) or 0
    cache_1h = getattr(cache_creation_obj, "ephemeral_1h_input_tokens", 0) or 0

    server_tool_use = getattr(usage, "server_tool_use", None)
    web_search_requests = getattr(server_tool_use, "web_search_requests", 0) or 0
    web_fetch_requests = getattr(server_tool_use, "web_fetch_requests", 0) or 0

    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        thinking_tokens=thinking_tokens,
        service_tier=getattr(usage, "service_tier", None),
        cache_creation_5m_tokens=cache_5m,
        cache_creation_1h_tokens=cache_1h,
        web_search_requests=web_search_requests,
        web_fetch_requests=web_fetch_requests,
    )


def _map_tool_calls(content_blocks: list[Any]) -> list[ChatCompletionMessageToolCall] | None:
    calls = [
        ChatCompletionMessageToolCall(
            id=b.id,
            type="function",
            function=Function(
                name=b.name,
                arguments=json.dumps(b.input) if isinstance(b.input, dict) else str(b.input),
            ),
        )
        for b in content_blocks
        if getattr(b, "type", None) == "tool_use"
    ]
    return calls or None


def _extract_message_parts(
        content_blocks: list[Any],
) -> tuple[str | None, list[ChatCompletionMessageToolCall] | None, str | None, str | None, list[Any] | None]:
    """Returns (text, tool_calls, thinking, thinking_signature, citations)."""
    text = ""
    thinking_parts: list[str] = []
    thinking_signature: str | None = None
    citations: list[Any] = []

    for block in content_blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text += getattr(block, "text", "") or ""
            block_citations = getattr(block, "citations", None)
            if block_citations:
                citations.extend(block_citations)
        elif block_type == "thinking":
            thinking_parts.append(getattr(block, "thinking", "") or "")
            if thinking_signature is None:
                thinking_signature = getattr(block, "signature", None)
        # redacted_thinking blocks are encrypted — skip

    tool_calls = _map_tool_calls(content_blocks)
    return (
        text or None,
        tool_calls,
        "\n".join(filter(None, thinking_parts)) or None,
        thinking_signature,
        to_jsonable(citations) if citations else None,
    )


def _to_chat_completion_response(provider: "AnthropicChatProvider", response: Any) -> ChatCompletion:
    text, tool_calls, thinking, thinking_signature, citations = _extract_message_parts(response.content)
    finish_reason = provider._map_stop_reason(response.stop_reason, bool(tool_calls))
    extensions = provider_extensions_from_raw(
        "anthropic",
        response,
        {"id", "type", "role", "content", "model", "stop_reason", "stop_sequence", "usage"},
    )
    return ChatCompletion(
        id=response.id,
        object="chat.completion",
        created=int(time.time()),
        model=response.model,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=text,
                    tool_calls=tool_calls,
                    thinking=thinking,
                    thinking_signature=thinking_signature,
                    citations=citations,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=_build_usage(response.usage),
        stop_sequence=getattr(response, "stop_sequence", None),
        extensions=extensions,
    )


def _retry_after_from_exc(exc: Exception, default: int = 60) -> int:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = headers.get("retry-after") if hasattr(headers, "get") else None
    if value is None:
        return default
    try:
        seconds = float(value)
        if math.isfinite(seconds):
            return max(0, math.ceil(seconds))
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0, math.ceil((date - datetime.now(timezone.utc)).total_seconds()))
    except (TypeError, ValueError, OverflowError):
        return default


def _raise_mapped_anthropic_error(
    exc: Exception, *, context: str, provider_name: str = "anthropic", timeout_seconds: float | None = None
) -> NoReturn:
    if isinstance(exc, anthropic.RateLimitError):
        raise RateLimitExceededError(_retry_after_from_exc(exc)) from exc
    if isinstance(exc, anthropic.AuthenticationError):
        raise AuthenticationError(provider=provider_name) from exc
    if isinstance(exc, anthropic.APITimeoutError):
        timeout_value = 60 if timeout_seconds is None or timeout_seconds <= 0 else max(1, int(math.ceil(timeout_seconds)))
        raise TimeOutError(timeout_value) from exc
    if isinstance(exc, anthropic.APIConnectionError):
        raise APIConnectionError(provider_name) from exc
    if isinstance(exc, anthropic.BadRequestError):
        raise ChatCompletionError(str(exc), status_code=400, provider=provider_name) from exc
    if isinstance(exc, ChatCompletionError):
        raise exc

    status_code = getattr(exc, "status_code", 500)
    logger.error("%s | %s", context, exc)
    raise ChatCompletionError(str(exc), status_code=status_code, provider=provider_name) from exc


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class AnthropicChatProvider(BaseChatProvider):
    """
    Anthropic (Claude) ChatCompletion Provider.

    Supports two prompt-caching modes:

    1. **Explicit block-level caching** (default when ``enable_prompt_caching=True``):
       Injects ``cache_control: {type: ephemeral}`` on the last system content
       block and the last tool definition so the static prefix is cached
       independently from the conversation history.

    2. **Automatic caching** (pass ``cache_control={"type": "ephemeral"}`` as a
       kwarg to ``create()``):
       Forwarded as a top-level field to ``messages.create``.  Anthropic
       automatically caches up to the last cacheable block and advances the
       breakpoint as the conversation grows.

    Prompt-caching notes
    --------------------
    * ``cache_creation_input_tokens`` — charged once when the cache is written
      (1.25× base price for 5-min TTL, 2× for 1-hour TTL).
    * ``cache_read_input_tokens``     — charged at ~10% cost on every cache hit.
    * Both are exposed on ``CompletionUsage`` via ``cached_tokens`` /
      ``cache_savings_pct`` convenience properties.
    * Minimum cacheable tokens per model:
      - Opus 4.5/4.6/4.7, Haiku 4.5: ≥ 4 096 tokens
      - Sonnet 4.6, Haiku 3.5:        ≥ 2 048 tokens
      - Sonnet 4.5/4/3.7, Opus 4/4.1: ≥ 1 024 tokens
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the AnthropicChatProvider.

        Args:
            api_key (str): Anthropic API key.
            deployment_name (str | None): Preferred Anthropic deployment/model identifier.
            model (str): Claude model ID fallback (e.g. ``claude-opus-4-5``).
            endpoint (str | None): Optional custom endpoint URL (proxy / gateway).
            timeout (float | None): Request timeout in seconds.
            enable_prompt_caching (bool): Whether to auto-inject ``cache_control``
                on system messages and tool definitions. Defaults to ``True``.
        """
        self.api_key: str = kwargs.get("api_key", "")
        self.deployment_name: str | None = kwargs.get("deployment_name") or None
        self.model: str = (
                kwargs.get("deployment_name")
                or kwargs.get("model")
                or DEFAULT_ANTHROPIC_MODEL
        )
        self.base_url: str | None = kwargs.get("endpoint") or kwargs.get("base_url")
        self.timeout: float | None = kwargs.get("timeout")
        self.enable_prompt_caching: bool = kwargs.get("enable_prompt_caching", True)

        client_kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout

        self.client = anthropic.Anthropic(**merge_client_options(client_kwargs, kwargs.get("client_options")))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create(self, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
        """
        Dispatch to streaming or non-streaming chat completion.

        Args:
            messages (list[dict]): OpenAI-format message list.
            stream (bool): Whether to stream the response.
            **kwargs: Extra parameters (temperature, max_tokens, tools, etc.).

        Returns:
            ChatCompletion or generator of ChatCompletionChunk.
        """
        parse_model = _apply_parse_model(kwargs)
        if stream:
            return self._streaming_create(messages, **kwargs)
        result = self._non_streaming_create(messages, **kwargs)
        if parse_model is not None:
            populate_parsed(result, parse_model)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_request(
            self, messages: list[dict[str, Any]], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Build the Anthropic ``messages.create`` kwargs from an OpenAI-style
        input, injecting ``cache_control`` where appropriate.

        Args:
            messages: OpenAI-format messages.
            kwargs: Extra call parameters.

        Returns:
            Dict of keyword arguments ready for ``self.client.messages.create``.
        """
        system_blocks, anthropic_messages = _openai_messages_to_anthropic(messages)

        # Top-level cache_control means "automatic caching" mode for this call.
        # To avoid mixed-mode conflicts (e.g., TTL mismatch between top-level and
        # block-level markers), explicit injection is disabled when this is set.
        top_cache_control = kwargs.pop("cache_control", None)

        raw_tools = kwargs.pop("tools", None) or []
        anthropic_tools = _convert_tools_to_anthropic(raw_tools)

        # Auto-inject cache_control on static prefix
        if self.enable_prompt_caching and top_cache_control is None:
            if system_blocks:
                system_blocks = _inject_cache_control_system(system_blocks)
            if anthropic_tools:
                anthropic_tools = _inject_cache_control_tools(anthropic_tools)

        # A configured deployment is a routing contract and always wins. Otherwise a
        # per-call model overrides the configured one, as it does for the other providers.
        call_model = kwargs.pop("model", None)
        kwargs.pop("deployment_name", None)
        request_model = self.deployment_name or call_model or self.model
        normalize_token_limit(kwargs, "max_tokens")
        req: dict[str, Any] = {
            "model": request_model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", None) or 4096,
        }
        if system_blocks:
            req["system"] = system_blocks
        if anthropic_tools:
            req["tools"] = anthropic_tools

        # Top-level cache_control → automatic caching (Method 2).
        if top_cache_control is not None:
            req["cache_control"] = top_cache_control

        # Map tool_choice
        tool_choice = kwargs.pop("tool_choice", None)
        if tool_choice and anthropic_tools:
            if tool_choice == "required":
                req["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                req["tool_choice"] = {"type": "none"}
            elif isinstance(tool_choice, dict) and tool_choice.get("function", {}).get("name"):
                req["tool_choice"] = {
                    "type": "tool",
                    "name": tool_choice["function"]["name"],
                }
            else:
                req["tool_choice"] = {"type": "auto"}

        # Pass through supported scalar params
        for key in ("temperature", "top_p", "top_k", "stop_sequences"):
            val = kwargs.pop(key, None)
            if val is not None:
                req[key] = val

        # stop → stop_sequences
        stop = kwargs.pop("stop", None)
        if stop is not None and "stop_sequences" not in req:
            req["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        # Silently discard unsupported OpenAI params
        for unsupported in _UNSUPPORTED_OPENAI_PARAMS:
            kwargs.pop(unsupported, None)

        # Any remaining kwargs are passed through
        req.update(kwargs)
        return req

    def _non_streaming_create(
            self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatCompletion:
        """
        Non-streaming chat completion.

        Args:
            messages: OpenAI-format messages.
            **kwargs: Extra parameters.

        Returns:
            ChatCompletion with usage including cache fields.
        """
        try:
            req = self._prepare_request(messages, kwargs)
            response = self.client.messages.create(**req)
            return _to_chat_completion_response(self, response)

        except Exception as e:
            _raise_mapped_anthropic_error(e, context="Anthropic chat completion error", timeout_seconds=self.timeout)

    def _streaming_create(
            self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Generator[ChatCompletionChunk, None, None]:
        """
        Streaming chat completion.  Yields ``ChatCompletionChunk`` objects and
        a final usage chunk.

        Args:
            messages: OpenAI-format messages.
            **kwargs: Extra parameters.

        Yields:
            ChatCompletionChunk objects.
        """
        try:
            req = self._prepare_request(messages, kwargs)

            def _generate() -> Generator[ChatCompletionChunk, None, None]:
                chunk_id = f"anthropic-{int(time.time())}"
                model = req.get("model", self.model)
                created = int(time.time())

                with self.client.messages.stream(**req) as stream:
                    # Stream text and thinking incrementally as deltas arrive, rather
                    # than collecting thinking only from the final message. Iterating
                    # raw events exposes thinking_delta / signature_delta, which
                    # stream.text_stream (text-only) hides.
                    for event in stream:
                        if getattr(event, "type", None) != "content_block_delta":
                            continue
                        delta = event.delta
                        delta_type = getattr(delta, "type", None)
                        if delta_type == "text_delta":
                            chunk_delta = ChoiceDelta(role="assistant", content=delta.text)
                        elif delta_type == "thinking_delta":
                            chunk_delta = ChoiceDelta(role="assistant", thinking=delta.thinking)
                        elif delta_type == "signature_delta":
                            chunk_delta = ChoiceDelta(role="assistant", thinking_signature=delta.signature)
                        else:
                            continue
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            object="chat.completion.chunk",
                            created=created,
                            model=model,
                            choices=[ChunkDelta(index=0, delta=chunk_delta, finish_reason=None)],
                            usage=None,
                            extensions=provider_extensions_for("anthropic"),
                        )

                    # Final message for tool calls, citations, and usage. Thinking and
                    # signature already streamed above, so they are not re-emitted here.
                    final = stream.get_final_message()
                    _, tool_calls, _, _, citations = _extract_message_parts(final.content)
                    finish_reason = self._map_stop_reason(final.stop_reason, bool(tool_calls))

                    if tool_calls:
                        delta_tool_calls = [
                            ChoiceDeltaToolCall(
                                index=i,
                                id=tc.id,
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name=tc.function.name,
                                    arguments=tc.function.arguments,
                                ),
                            )
                            for i, tc in enumerate(tool_calls)
                        ]
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            object="chat.completion.chunk",
                            created=created,
                            model=model,
                            choices=[
                                ChunkDelta(
                                    index=0,
                                    delta=ChoiceDelta(
                                        role="assistant",
                                        tool_calls=delta_tool_calls,
                                        citations=citations,
                                    ),
                                    finish_reason=finish_reason,
                                )
                            ],
                            usage=None,
                            extensions=provider_extensions_for("anthropic"),
                        )
                    else:
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            object="chat.completion.chunk",
                            created=created,
                            model=model,
                            choices=[
                                ChunkDelta(
                                    index=0,
                                    delta=ChoiceDelta(
                                        role="assistant",
                                        citations=citations,
                                    ),
                                    finish_reason=finish_reason,
                                )
                            ],
                            usage=None,
                            extensions=provider_extensions_for("anthropic"),
                        )

                    yield ChatCompletionChunk(
                        id=chunk_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model,
                        choices=[],
                        usage=_build_usage(final.usage),
                        extensions=provider_extensions_for("anthropic"),
                    )

            def _wrapped() -> Generator[ChatCompletionChunk, None, None]:
                try:
                    yield from _generate()
                except Exception as e:  # noqa: BLE001 - re-mapped to library errors
                    _raise_mapped_anthropic_error(e, context="Anthropic streaming error", timeout_seconds=self.timeout)

            return _wrapped()

        except Exception as e:
            _raise_mapped_anthropic_error(e, context="Anthropic streaming error", timeout_seconds=self.timeout)

    @staticmethod
    def _map_stop_reason(stop_reason: str | None, has_tool_calls: bool) -> _CHUNK_FINISH_REASON:
        if has_tool_calls:
            return "tool_calls"
        mapping: dict[str, _CHUNK_FINISH_REASON] = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        return mapping.get(stop_reason or "", "stop")


class AsyncAnthropicChatProvider(AnthropicChatProvider):
    """
    Async Anthropic (Claude) ChatCompletion provider.

    Reuses AnthropicChatProvider request preparation and response mapping, while
    issuing network I/O through ``anthropic.AsyncAnthropic``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.api_key: str = kwargs.get("api_key", "")
        self.deployment_name: str | None = kwargs.get("deployment_name") or None
        self.model: str = (
                kwargs.get("deployment_name")
                or kwargs.get("model")
                or "claude-opus-4-5"
        )
        self.base_url: str | None = kwargs.get("endpoint") or kwargs.get("base_url")
        self.timeout: float | None = kwargs.get("timeout")
        self.enable_prompt_caching: bool = kwargs.get("enable_prompt_caching", True)

        client_kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout

        self.client = anthropic.AsyncAnthropic(**merge_client_options(client_kwargs, kwargs.get("client_options")))

    async def create(self, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
        parse_model = _apply_parse_model(kwargs)
        if stream:
            return self._streaming_create(messages, **kwargs)
        result = await self._non_streaming_create(messages, **kwargs)
        if parse_model is not None:
            populate_parsed(result, parse_model)
        return result

    async def _non_streaming_create(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatCompletion:
        try:
            req = self._prepare_request(messages, kwargs)
            response = await self.client.messages.create(**req)
            return _to_chat_completion_response(self, response)

        except Exception as e:
            _raise_mapped_anthropic_error(e, context="Async Anthropic chat completion error", timeout_seconds=self.timeout)

    async def _streaming_create(
            self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        try:
            req = self._prepare_request(messages, kwargs)

            chunk_id = f"anthropic-{int(time.time())}"
            model = req.get("model", self.model)
            created = int(time.time())

            async with self.client.messages.stream(**req) as stream:
                # Stream text and thinking incrementally as deltas arrive (see the
                # sync provider for rationale): raw events expose thinking_delta /
                # signature_delta that stream.text_stream hides.
                async for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = event.delta
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        chunk_delta = ChoiceDelta(role="assistant", content=delta.text)
                    elif delta_type == "thinking_delta":
                        chunk_delta = ChoiceDelta(role="assistant", thinking=delta.thinking)
                    elif delta_type == "signature_delta":
                        chunk_delta = ChoiceDelta(role="assistant", thinking_signature=delta.signature)
                    else:
                        continue
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model,
                        choices=[ChunkDelta(index=0, delta=chunk_delta, finish_reason=None)],
                        usage=None,
                        extensions=provider_extensions_for("anthropic"),
                    )

                # Final message for tool calls, citations, and usage. Thinking and
                # signature already streamed above, so they are not re-emitted here.
                final = await stream.get_final_message()
                _, tool_calls, _, _, citations = _extract_message_parts(final.content)
                finish_reason = self._map_stop_reason(final.stop_reason, bool(tool_calls))

                if tool_calls:
                    delta_tool_calls = [
                        ChoiceDeltaToolCall(
                            index=i,
                            id=tc.id,
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                        for i, tc in enumerate(tool_calls)
                    ]
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model,
                        choices=[
                            ChunkDelta(
                                index=0,
                                delta=ChoiceDelta(
                                    role="assistant",
                                    tool_calls=delta_tool_calls,
                                    citations=citations,
                                ),
                                finish_reason=finish_reason,
                            )
                        ],
                        usage=None,
                        extensions=provider_extensions_for("anthropic"),
                    )
                else:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        object="chat.completion.chunk",
                        created=created,
                        model=model,
                        choices=[
                            ChunkDelta(
                                index=0,
                                delta=ChoiceDelta(
                                    role="assistant",
                                    citations=citations,
                                ),
                                finish_reason=finish_reason,
                            )
                        ],
                        usage=None,
                        extensions=provider_extensions_for("anthropic"),
                    )

                yield ChatCompletionChunk(
                    id=chunk_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=model,
                    choices=[],
                    usage=_build_usage(final.usage),
                    extensions=provider_extensions_for("anthropic"),
                )

        except Exception as e:
            _raise_mapped_anthropic_error(e, context="Async Anthropic streaming error", timeout_seconds=self.timeout)


class AnthropicResponsesProvider(SyncEmulatedResponsesProvider):
    """Responses API emulated on top of Anthropic chat completions."""

    chat_cls = AnthropicChatProvider
    provider_name = "anthropic"


class AsyncAnthropicResponsesProvider(AsyncEmulatedResponsesProvider):
    """Async Responses API emulated on top of Anthropic chat completions."""

    chat_cls = AsyncAnthropicChatProvider
    provider_name = "anthropic"
