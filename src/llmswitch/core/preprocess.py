"""
core.preprocess

Shared request-preparation building blocks for the chat handlers:

* :func:`prepare_messages` — normalize Pydantic message objects to dicts, count
  tokens, and optionally truncate text content (multimodal-aware); shared by the
  sync and async chat handlers.
* :func:`prepare_response_input` — the same for a Responses API ``input``.
* :class:`ProviderCache` — provider-client instantiation + caching + close
  lifecycle, shared by the sync chat, async chat, and embedding handlers.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel

from ..errors import ContextLengthExceededError
from ..ratelimit import RedisRateLimiter

# Opt-in process-wide provider/client cache (``LLMSwitch(..., share_clients=True)``).
# Bounded LRU so it cannot grow without limit in multi-tenant processes; evicted
# clients are closed best-effort. Default behaviour keeps the per-switch cache.
_SHARED_CLIENT_CAP = 64
_NON_TEXT_PART_TOKEN_FLOOR = 1024


def _non_text_part_tokens(value: Any, rate_limiter: RedisRateLimiter, model_name: Any) -> int:
    if isinstance(value, list):
        return sum(_non_text_part_tokens(item, rate_limiter, model_name) for item in value)
    if not isinstance(value, dict):
        return 0
    kind = value.get("type")
    if isinstance(kind, str) and {"image", "audio", "video", "file"}.intersection(kind.split("_")):
        serialized = json.dumps(value, default=str, separators=(",", ":"))
        return max(
            _NON_TEXT_PART_TOKEN_FLOOR,
            rate_limiter.count_tokens(serialized, model_name=model_name),
        )
    if "content" in value:
        return _non_text_part_tokens(value["content"], rate_limiter, model_name)
    if kind is not None and not any(key in value for key in ("text", "output", "arguments", "transcript")):
        serialized = json.dumps(value, default=str, separators=(",", ":"))
        return max(
            _NON_TEXT_PART_TOKEN_FLOOR,
            rate_limiter.count_tokens(serialized, model_name=model_name),
        )
    return 0


@dataclass
class _SharedProviderEntry:
    provider: Any
    leases: int = 0
    evicted: bool = False


_shared_clients: "OrderedDict[Any, _SharedProviderEntry]" = OrderedDict()
_shared_clients_lock = threading.Lock()


def apply_model_priority(config: Any, kwargs: dict[str, Any]) -> Any:
    """Apply unified model routing: deployment name, request model, config model.

    A configured deployment is an endpoint routing contract and therefore cannot
    be replaced by a per-call model alias. Without a deployment, callers may
    override the configured model per request.
    """
    deployment_name = getattr(config, "deployment_name", None)
    if deployment_name:
        kwargs["model"] = deployment_name
        return deployment_name
    return kwargs.get("model") or getattr(config, "model", None)


def _run_close_coro(coro) -> None:
    """Drive a client close-coroutine to completion (eviction may run on or off a loop)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        # Async-switch path runs on the loop thread — fire-and-forget the close.
        loop.create_task(coro)
    else:
        try:
            asyncio.run(coro)
        except Exception:  # noqa: BLE001
            coro.close()


def _close_provider_client(provider: Any) -> None:
    """Best-effort close of a provider's client on shared-cache eviction.

    Prefers ``aclose`` for async clients (driving the coroutine to completion, or
    scheduling it when a loop is running) and falls back to ``close`` for sync
    clients, so async connection pools are actually released rather than leaked.
    """
    for target in (provider, getattr(provider, "client", None)):
        if target is None:
            continue
        for attr in ("aclose", "close"):
            fn = getattr(target, attr, None)
            if not callable(fn):
                continue
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    _run_close_coro(result)
            except Exception:  # noqa: BLE001 - eviction close is best-effort
                pass
            return


def _release_shared(entry: _SharedProviderEntry) -> None:
    close_provider = None
    with _shared_clients_lock:
        entry.leases -= 1
        if entry.leases == 0 and entry.evicted:
            close_provider = entry.provider
    if close_provider is not None:
        _close_provider_client(close_provider)


def _leased_iterator(result: Iterator[Any], entry: _SharedProviderEntry) -> Iterator[Any]:
    try:
        for item in result:
            yield item
    finally:
        try:
            close = getattr(result, "close", None)
            if callable(close):
                close()
        finally:
            _release_shared(entry)


async def _leased_async_iterator(result: AsyncIterator[Any], entry: _SharedProviderEntry) -> AsyncIterator[Any]:
    try:
        async for item in result:
            yield item
    finally:
        try:
            close = getattr(result, "aclose", None) or getattr(result, "close", None)
            if callable(close):
                closing = close()
                if inspect.isawaitable(closing):
                    await closing
        finally:
            _release_shared(entry)


class _SharedProviderProxy:
    """Acquire a shared-provider lease for each provider operation."""

    def __init__(self, entry: _SharedProviderEntry, key: Any, factory: Any) -> None:
        self._entry = entry
        self._key = key
        self._factory = factory

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._entry.provider, name)
        if not callable(target):
            if self._entry.evicted:
                return getattr(_get_or_create_shared(self._key, self._factory), name)
            return target

        def invoke(*args: Any, **kwargs: Any) -> Any:
            with _shared_clients_lock:
                entry = self._entry
                refresh = entry.evicted
                if not refresh:
                    entry.leases += 1
            if refresh:
                return getattr(_get_or_create_shared(self._key, self._factory), name)(*args, **kwargs)
            try:
                result = getattr(entry.provider, name)(*args, **kwargs)
            except BaseException:
                _release_shared(entry)
                raise
            if inspect.isawaitable(result):
                async def await_result() -> Any:
                    try:
                        resolved = await result
                    except BaseException:
                        _release_shared(entry)
                        raise
                    if isinstance(resolved, AsyncIterator):
                        return _leased_async_iterator(resolved, entry)
                    _release_shared(entry)
                    return resolved
                return await_result()
            if isinstance(result, Iterator):
                return _leased_iterator(result, entry)
            _release_shared(entry)
            return result

        return invoke


def _get_or_create_shared(key: Any, factory) -> Any:
    """Return the process-shared provider for ``key``, constructing it on miss."""
    with _shared_clients_lock:
        entry = _shared_clients.get(key)
        if entry is not None:
            _shared_clients.move_to_end(key)
            return _SharedProviderProxy(entry, key, factory)
    provider = factory()  # construct outside the lock (may do real work)
    evicted: list[_SharedProviderEntry] = []
    loser: Any = None
    with _shared_clients_lock:
        existing = _shared_clients.get(key)
        if existing is not None:
            # Another thread built this key first; keep theirs and close ours below.
            _shared_clients.move_to_end(key)
            loser = provider
            entry = existing
        else:
            entry = _SharedProviderEntry(provider=provider)
            _shared_clients[key] = entry
            _shared_clients.move_to_end(key)
            while len(_shared_clients) > _SHARED_CLIENT_CAP:
                _, evicted_entry = _shared_clients.popitem(last=False)
                evicted_entry.evicted = True
                if evicted_entry.leases == 0:
                    evicted.append(evicted_entry)
    if loser is not None:
        _close_provider_client(loser)
    for ev in evicted:
        _close_provider_client(ev.provider)
    return _SharedProviderProxy(entry, key, factory)


def prepare_messages(
        messages: list[dict[str, Any]],
        rate_limiter: "RedisRateLimiter",
        completion_context: int,
        truncate: bool,
        model_name: Any,
        reserved_output_tokens: int = 0,
        overhead_tokens: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """
    Normalize and (optionally) truncate messages, returning
    ``(prepared_messages, total_prompt_tokens)``.

    The ``content`` field may be a string or a list of multimodal content parts;
    non-text parts receive a conservative estimate, while truncation only
    rewrites text (image parts are preserved).
    """
    prepared: list[dict[str, Any]] = []
    message_texts: list[str] = []
    message_tokens: list[int] = []
    message_overhead = 0
    for msg in messages:
        # Normalize Pydantic message objects (e.g. ChatCompletionMessage) to plain dicts.
        dump_fn = getattr(msg, "model_dump", None)
        if callable(dump_fn):
            msg = dump_fn(exclude_none=True)
        content = msg.get("content", "")
        # Handle multimodal (list) content — join text parts for token counting only.
        if isinstance(content, list):
            text_for_tokens = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
            )
            message_overhead += _non_text_part_tokens(content, rate_limiter, model_name)
        else:
            text_for_tokens = content if isinstance(content, str) else ""
        prepared.append(msg)
        message_texts.append(text_for_tokens)
        message_tokens.append(rate_limiter.count_tokens(
            text_for_tokens, model_name=model_name
        ))
        metadata = {key: value for key, value in msg.items() if key not in {"role", "content"}}
        if metadata:
            message_overhead += rate_limiter.count_tokens(
                json.dumps(metadata, default=str, separators=(",", ":")),
                model_name=model_name,
            )

    overhead_tokens += message_overhead
    input_budget = completion_context - reserved_output_tokens - overhead_tokens
    total_tokens = sum(message_tokens) + overhead_tokens
    if input_budget < 0 or total_tokens > completion_context - reserved_output_tokens:
        if not truncate:
            raise ContextLengthExceededError(
                input_tokens=total_tokens,
                input_budget=max(0, input_budget),
                context_window=completion_context,
                reserved_output_tokens=reserved_output_tokens,
                overhead_tokens=overhead_tokens,
            )
        if input_budget < 0:
            raise ContextLengthExceededError(
                input_tokens=total_tokens,
                input_budget=0,
                context_window=completion_context,
                reserved_output_tokens=reserved_output_tokens,
                overhead_tokens=overhead_tokens,
            )

        # Preserve system/developer instructions first, then retain the newest
        # eligible messages. Multimodal blocks are never rewritten because joining
        # or reordering their parts can change semantics.
        remaining = input_budget
        allocations = [0] * len(prepared)
        priority = [
            index for index, message in enumerate(prepared)
            if message.get("role") in {"system", "developer"}
        ]
        priority_indexes = set(priority)
        priority.extend(index for index in range(len(prepared) - 1, -1, -1) if index not in priority_indexes)
        for index in priority:
            tokens = message_tokens[index]
            content = prepared[index].get("content", "")
            if isinstance(content, list) and tokens > remaining:
                raise ContextLengthExceededError(
                    input_tokens=total_tokens,
                    input_budget=input_budget,
                    context_window=completion_context,
                    reserved_output_tokens=reserved_output_tokens,
                    overhead_tokens=overhead_tokens,
                )
            allocation = min(tokens, remaining)
            allocations[index] = allocation
            remaining -= allocation

        for index, allocation in enumerate(allocations):
            content = prepared[index].get("content", "")
            if isinstance(content, str) and allocation < message_tokens[index]:
                truncated_content = rate_limiter.truncate_message(
                    message_texts[index], allocation, truncate=True,
                    model_name=model_name,
                )
                prepared[index] = {**prepared[index], "content": truncated_content}
                if isinstance(truncated_content, str) and truncated_content != content:
                    message_tokens[index] = rate_limiter.count_tokens(
                        truncated_content, model_name=model_name,
                    )

        total_tokens = sum(message_tokens) + overhead_tokens
        if total_tokens > completion_context - reserved_output_tokens:
            raise ContextLengthExceededError(
                input_tokens=total_tokens,
                input_budget=input_budget,
                context_window=completion_context,
                reserved_output_tokens=reserved_output_tokens,
                overhead_tokens=overhead_tokens,
            )

    return prepared, total_tokens


def _response_input_text(value: Any) -> str:
    """Collect every text fragment from a Responses ``input`` item, for token counting."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_response_input_text(v) for v in value)
    dump_fn = getattr(value, "model_dump", None)
    if callable(dump_fn):
        value = dump_fn(exclude_none=True)
    if isinstance(value, dict):
        parts = [str(value[key]) for key in ("text", "output", "arguments", "transcript")
                 if isinstance(value.get(key), str)]
        content = value.get("content")
        if content is not None:
            parts.append(_response_input_text(content))
        return " ".join(p for p in parts if p)
    return ""


def estimate_request_overhead(
        rate_limiter: "RedisRateLimiter",
        model_name: Any,
        request_kwargs: dict[str, Any],
        *,
        message_count: int = 0,
        configured_overhead_tokens: int = 0,
) -> int:
    """Estimate direct prompt overhead from message framing and schema-bearing fields."""
    prompt_fields = {}
    for key, value in request_kwargs.items():
        if key not in {
            "instructions", "tools", "tool_choice", "response_format",
            "text_format", "text", "reasoning", "prompt", "context",
        } or value is None:
            continue
        prompt_fields[key] = (
            value.model_json_schema()
            if isinstance(value, type) and issubclass(value, BaseModel)
            else value
        )
    serialized = json.dumps(prompt_fields, default=str, separators=(",", ":")) if prompt_fields else ""
    schema_tokens = rate_limiter.count_tokens(serialized, model_name=model_name)
    return configured_overhead_tokens + schema_tokens + (4 * message_count if message_count else 0)


def resolve_reserved_output_tokens(request_kwargs: dict[str, Any], configured_reserve: int = 0) -> int:
    """Return the greater of configured reserve and an explicit generation limit."""
    requested = 0
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        value = request_kwargs.get(key)
        if value is not None:
            try:
                requested = max(requested, int(value))
            except (TypeError, ValueError):
                continue
    return max(configured_reserve, requested)


def prepare_response_input(
        input: Any,  # noqa: A002
        rate_limiter: "RedisRateLimiter",
        completion_context: int,
        truncate: bool,
        model_name: Any,
        reserved_output_tokens: int = 0,
        overhead_tokens: int = 0,
) -> tuple[Any, int]:
    """
    Count (and optionally truncate) a Responses API ``input``, returning
    ``(prepared_input, total_prompt_tokens)``.

    ``input`` may be a plain string or a list of input items with arbitrary
    nesting. Truncation only rewrites a plain-string input — item lists carry
    tool calls and multimodal parts that cannot be safely rewritten, so they are
    counted and passed through unchanged.
    """
    input_budget = completion_context - reserved_output_tokens - overhead_tokens
    if input is None:
        if input_budget < 0:
            raise ContextLengthExceededError(
                input_tokens=overhead_tokens,
                input_budget=0,
                context_window=completion_context,
                reserved_output_tokens=reserved_output_tokens,
                overhead_tokens=overhead_tokens,
            )
        return None, overhead_tokens

    prepared: Any = input
    if isinstance(input, str):
        truncated = rate_limiter.truncate_message(
            input, max(0, input_budget), truncate=truncate,
            model_name=model_name)
        if truncated is not None:
            prepared = truncated
        text = prepared
    else:
        items = input if isinstance(input, list) else [input]
        prepared = [item.model_dump(exclude_none=True) if callable(getattr(item, "model_dump", None)) else item
                    for item in items]
        text = _response_input_text(prepared)

    input_tokens = rate_limiter.count_tokens(text, model_name=model_name)
    if not isinstance(input, str):
        input_tokens += _non_text_part_tokens(prepared, rate_limiter, model_name)
    total_tokens = input_tokens + overhead_tokens
    if input_budget < 0 or input_tokens > input_budget:
        raise ContextLengthExceededError(
            input_tokens=total_tokens,
            input_budget=max(0, input_budget),
            context_window=completion_context,
            reserved_output_tokens=reserved_output_tokens,
            overhead_tokens=overhead_tokens,
        )
    return prepared, total_tokens


def prepare_target_messages(
    config: Any,
    messages: list[dict[str, Any]],
    rate_limiter: "RedisRateLimiter",
    truncate: bool,
    request_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    model = getattr(config, "deployment_name", None) or request_kwargs.get("model") or getattr(config, "model", None)
    reserve = resolve_reserved_output_tokens(request_kwargs, config.context_output_reserve_tokens)
    overhead = estimate_request_overhead(
        rate_limiter, model, request_kwargs,
        message_count=len(messages), configured_overhead_tokens=config.context_overhead_tokens,
    )
    return prepare_messages(
        messages, rate_limiter, config.completion_context_window, truncate, model,
        reserve, overhead,
    )


def prepare_target_response_input(
    config: Any,
    input: Any,
    rate_limiter: "RedisRateLimiter",
    truncate: bool,
    request_kwargs: dict[str, Any],
) -> tuple[Any, int]:
    model = getattr(config, "deployment_name", None) or request_kwargs.get("model") or getattr(config, "model", None)
    reserve = resolve_reserved_output_tokens(request_kwargs, config.context_output_reserve_tokens)
    overhead = estimate_request_overhead(
        rate_limiter, model, request_kwargs,
        configured_overhead_tokens=config.context_overhead_tokens,
    )
    return prepare_response_input(
        input, rate_limiter, config.completion_context_window, truncate, model,
        reserve, overhead,
    )


class ProviderCache:
    """
    Provider-client instantiation, caching, and close lifecycle.

    Caches one provider client per distinct resolved config so repeated requests
    on the same switch reuse a single client (and its underlying HTTP connection
    pool) instead of constructing a new one per call. A single ``_instantiate_provider``
    passes the union of every provider's accepted kwargs (providers read what they
    need via ``kwargs.get`` and ignore the rest), so it serves chat, async-chat,
    and embedding providers alike.
    """

    _provider_cache: dict[tuple[type, tuple[Any, ...]], Any]
    _on_event: Optional[Callable[[dict[str, Any]], None]] = None  # set by the client when on_event is given
    _share_clients: bool = False  # set True by the switch when share_clients=True
    _breaker_enabled: bool = False  # set True by the switch when circuit_breaker is enabled
    _breaker_config: dict[str, Any]  # per-instance breaker thresholds (assigned by the switch)

    @staticmethod
    def _provider_cache_key(config: Any, provider_cls: type) -> tuple[type, tuple[Any, ...]]:
        return provider_cls, (
            getattr(config, "provider", None),
            getattr(config, "api_key", None),
            getattr(config, "endpoint", None),
            getattr(config, "api_version", None),
            getattr(config, "responses_api_version", None),
            getattr(config, "model", None),
            getattr(config, "location", None),
            getattr(config, "credentials_path", None),
            tuple(getattr(config, "scopes", None) or []),
            getattr(config, "organization", None),
            getattr(config, "project", None),
            getattr(config, "timeout", None),
            getattr(config, "enable_prompt_caching", None),
        )

    def _build_provider(self, config: Any, provider_cls: type) -> Any:
        """Construct a provider client, passing the union of all providers' kwargs."""
        endpoint = getattr(config, "endpoint", None)
        deployment_name = getattr(config, "deployment_name", None)
        return provider_cls(
            api_key=getattr(config, "api_key", None),
            endpoint=endpoint,
            azure_endpoint=endpoint,
            api_version=getattr(config, "api_version", None),
            responses_api_version=getattr(config, "responses_api_version", None),
            deployment_name=deployment_name,
            azure_deployment=deployment_name,
            model=getattr(config, "model", None),
            location=getattr(config, "location", None),
            credentials_path=getattr(config, "credentials_path", None),
            scopes=getattr(config, "scopes", None),
            organization=getattr(config, "organization", None),
            project=getattr(config, "project", None),
            timeout=getattr(config, "timeout", None),
            enable_prompt_caching=getattr(config, "enable_prompt_caching", None),
            client_options=getattr(config, "client_options", None),
        )

    def _instantiate_provider(self, config: Any, provider_cls: type) -> Any:
        """Instantiate (and cache) a provider client from a config object.

        With ``share_clients=True`` the client is taken from / stored in a bounded
        process-wide cache so switches sharing a config reuse one client (and its
        HTTP connection pool); otherwise it is cached on this handler instance.
        """
        cache_key = self._provider_cache_key(config, provider_cls)
        if self._share_clients:
            return _get_or_create_shared(cache_key, lambda: self._build_provider(config, provider_cls))
        provider = self._provider_cache.get(cache_key)
        if provider is not None:
            return provider
        provider = self._build_provider(config, provider_cls)
        self._provider_cache[cache_key] = provider
        return provider

    def close(self) -> None:
        """Close cached provider clients, if they expose close hooks."""
        for provider in self._provider_cache.values():
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                close_fn()
                continue
            client = getattr(provider, "client", None)
            client_close = getattr(client, "close", None)
            if callable(client_close):
                client_close()
        self._provider_cache.clear()

    async def aclose(self) -> None:
        """Close cached async provider clients, awaiting coroutine close hooks."""
        for provider in self._provider_cache.values():
            aclose_fn = getattr(provider, "aclose", None)
            if callable(aclose_fn):
                maybe_coroutine = aclose_fn()
                if asyncio.iscoroutine(maybe_coroutine):
                    await maybe_coroutine
                continue
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                maybe_coroutine = close_fn()
                if asyncio.iscoroutine(maybe_coroutine):
                    await maybe_coroutine
                continue
            client = getattr(provider, "client", None)
            client_close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if callable(client_close):
                maybe_coroutine = client_close()
                if asyncio.iscoroutine(maybe_coroutine):
                    await maybe_coroutine
        self._provider_cache.clear()
