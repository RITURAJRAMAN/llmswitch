"""
core.async_client

Asynchronous chat handler and the public ``AsyncLLMSwitch`` entry point.
Chat completions use the provider's native async client; embeddings are exposed
via ``.embeddings``: Azure AI Foundry uses its native async client, and the other providers
run their synchronous embedding provider in a worker thread.

Non-blocking I/O
----------------
The pre-flight work — message normalization + token counting (tiktoken, CPU)
and the Redis rate-limit gate (network I/O) — is synchronous. To keep the event
loop free it is run via :func:`asyncio.to_thread`, and the final token-usage
write is offloaded the same way. The actual model call already uses the
provider's native async client, and the rate-limit sleep uses ``asyncio.sleep``.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
from typing import Any

from ..constants import (
    DEFAULT_COMPLETION_CONTEXT_WINDOW,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_DELAY,
    DEFAULT_RPM,
    DEFAULT_TPM,
    DEFAULT_WINDOW_SECONDS,
)
from ..errors import (
    APIConnectionError,
    ChatCompletionError,
    ConfigurationError,
    EmbeddingGenerationError,
    LLMSwitchError,
    RateLimitExceededError,
    ResponsesError,
    TimeOutError,
)
from ..providers.capabilities import require_operation
from ..ratelimit import BatchUtility, RedisRateLimiter
from ..ratelimit.limiter import rate_limit_scope
from .circuit_breaker import get_breaker, provider_call
from .common import (
    build_config,
    normalize_embedding_input,
    provider_class,
    raise_chat_error,
    raise_embedding_error,
    responses_provider_class,
    validate_messages,
)
from .embedding_flow import EmbeddingAccumulator
from .events import emit, traced
from .execution import areserve_rate_limit, build_execution_plan
from .failover import (
    build_backup_execution_plan,
    prepare_backup_provider,
    validate_backup_config,
    validate_backup_for_registry,
)
from .planning import plan_chat, plan_responses
from .preprocess import (
    ProviderCache,
    apply_model_priority,
    prepare_messages,
    prepare_response_input,
    prepare_target_messages,
    prepare_target_response_input,
)
from .redaction import redact_sensitive
from .registry import (
    ASYNC_CHAT_PROVIDER_REGISTRY,
    ASYNC_EMBEDDING_PROVIDER_REGISTRY,
    ASYNC_RESPONSES_PROVIDER_REGISTRY,
    EMBEDDING_PROVIDER_REGISTRY,
)
from .responses_capabilities import (
    is_side_effecting_response_request,
    normalize_response_tool_definitions,
    resolve_response_capabilities,
    validate_responses_create_request,
)
from .responses_management import prepare_management_call, raise_response_error
from .retry import _async_retry_call, _is_retryable, _retry_call
from .streaming import (
    AsyncStreamSession,
    StreamHandle,
    _additional_stream_tokens,
    chat_chunk_has_visible_output,
    response_event_has_visible_output,
    response_event_terminal_status,
)

logger = logging.getLogger(__name__)


class _AsyncChatCompletions(ProviderCache):
    """
    Async chat completions handler. Applies rate limiting and message truncation
    off the event loop, then delegates to the async provider. Supports
    non-streaming (await) and streaming (async generator) modes.
    """

    def __init__(self, config: Any, rate_limiter: "RedisRateLimiter") -> None:
        self.config = config
        self.rate_limiter = rate_limiter
        self.rpm = getattr(config, "rpm", DEFAULT_RPM)
        self.tpm = getattr(config, "tpm", DEFAULT_TPM)
        self.completion_context = getattr(config, "completion_context_window", DEFAULT_COMPLETION_CONTEXT_WINDOW)
        self.window_seconds = DEFAULT_WINDOW_SECONDS
        self._provider_cache: dict[tuple[type, tuple[Any, ...]], Any] = {}

    async def _wrap_streaming(self, primary_factory, user_key: str, pre_count_tokens: int, *,
                              backup_factory=None, max_retries: int = 0, base_delay: float = 0.0):
        async def on_complete(last_usage, limiter_scope, prompt_tokens):
            actual = _additional_stream_tokens(last_usage, prompt_tokens, pre_count_tokens)
            await asyncio.to_thread(
                self.rate_limiter.update_token_usage,
                user_key,
                actual,
                self.window_seconds,
                "completion",
                limiter_scope,
            )

        return AsyncStreamSession(
            primary_factory=primary_factory,
            backup_factory=backup_factory,
            is_output_visible=chat_chunk_has_visible_output,
            on_complete=on_complete,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
        )

    def _preflight(self, messages: list[dict[str, Any]], truncate: bool, model_name: Any,
                   reserved_output_tokens: int, overhead_tokens: int):
        """
        Synchronous pre-flight: normalize/truncate messages, count tokens, and
        apply the rate-limit gate. Runs in a worker thread so the blocking
        tiktoken + Redis calls never touch the event loop.

        Returns ``(prepared_messages, total_tokens)``.
        """
        prepared, total_tokens = prepare_messages(
            messages, self.rate_limiter, self.completion_context, truncate, model_name,
            reserved_output_tokens, overhead_tokens,
        )
        return prepared, total_tokens

    @traced("chat.completions")
    async def create(self, messages: list[dict[str, Any]], stream: bool = False,
                     truncate: bool = False, **kwargs: Any) -> Any:
        """
        Async chat completion. See ``_ChatCompletions.create`` for full parameter
        documentation. For o1/o3/o4-mini reasoning models pass
        ``max_completion_tokens`` instead of ``max_tokens``.
        """
        validate_messages(messages)
        live_logs = kwargs.pop("live_logs", False)
        _max_retries: int = kwargs.pop("max_retries", getattr(self.config, "max_retries", DEFAULT_MAX_RETRIES))
        _base_delay: float = getattr(self.config, "retry_base_delay", DEFAULT_RETRY_BASE_DELAY)

        provider_cls = provider_class(self.config.provider, ASYNC_CHAT_PROVIDER_REGISTRY, "async chat completion")
        require_operation(self.config.provider, "chat.async")
        validate_backup_for_registry(self.config, ASYNC_CHAT_PROVIDER_REGISTRY)

        user_key = getattr(self.config, "user_id", "global")
        if live_logs:
            logger.info(json.dumps(redact_sensitive(
                {"provider": self.config.provider, "messages": "<hidden>", "stream": stream, "kwargs": kwargs}),
                default=str))
        chat_plan = plan_chat(self.config, self.rate_limiter, messages, kwargs, stream)
        plan, model_name = chat_plan.plan, chat_plan.model_name
        reserved_output_tokens, overhead_tokens = chat_plan.reserved_output_tokens, chat_plan.overhead_tokens
        limiter_scope = plan.target.limiter_scope_key

        # Off-load blocking tokenization + Redis gate to a worker thread.
        truncated_messages, total_tokens = await asyncio.to_thread(
            self._preflight, messages, truncate, model_name, reserved_output_tokens, overhead_tokens,
        )
        request_kwargs = dict(plan.provider_request)
        await areserve_rate_limit(
            self.rate_limiter, user_key, self.rpm, self.tpm,
            self.window_seconds, total_tokens, scope=limiter_scope,
        )

        breaker = (get_breaker(plan.target.breaker_key, **getattr(self, "_breaker_config", {}))
                   if getattr(self, "_breaker_enabled", False) else None)
        if stream:
            async def primary_stream_factory():
                try:
                    with provider_call(self.config.provider, model_name, breaker=breaker, record_success=False):
                        result = await self._instantiate_provider(self.config, provider_cls).create(
                            messages=truncated_messages, stream=True, **request_kwargs
                        )
                    return StreamHandle(
                        result, breaker=breaker, limiter_scope=limiter_scope, prompt_tokens=total_tokens,
                    )
                except Exception as exc:
                    raise_chat_error(self.config, exc, user_key)

            backup_stream_factory = None
            if getattr(self.config, "backup", None):
                async def backup_stream_factory():
                    backup_execution = build_backup_execution_plan(
                        self.config,
                        operation="chat.completions",
                        normalized_request={"messages": messages, **kwargs},
                    )
                    if backup_execution is None:
                        raise LLMSwitchError("Backup provider configuration is missing")
                    backup_config, backup_plan = backup_execution
                    backup_limiter_scope = backup_plan.target.limiter_scope_key
                    backup_request_kwargs_payload = dict(backup_plan.provider_request)
                    backup_messages, backup_tokens = await asyncio.to_thread(
                        prepare_target_messages, backup_config, messages, self.rate_limiter,
                        truncate, backup_request_kwargs_payload,
                    )
                    await areserve_rate_limit(
                        self.rate_limiter, user_key, backup_config.rpm, backup_config.tpm,
                        self.window_seconds, backup_tokens, scope=backup_limiter_scope,
                    )
                    backup_provider = prepare_backup_provider(self.config, ASYNC_CHAT_PROVIDER_REGISTRY,
                                                              self._instantiate_provider)
                    backup_breaker = (
                        get_breaker(backup_plan.target.breaker_key, **getattr(self, "_breaker_config", {}))
                        if getattr(self, "_breaker_enabled", False) else None
                    )
                    try:
                        with provider_call(
                            backup_config.provider,
                            backup_plan.target.resolved_model,
                            breaker=backup_breaker,
                            record_success=False,
                        ):
                            result = await backup_provider.create(
                                messages=backup_messages, stream=True, **backup_request_kwargs_payload,
                            )
                        return StreamHandle(
                            result, breaker=backup_breaker, limiter_scope=backup_limiter_scope,
                            prompt_tokens=backup_tokens,
                        )
                    except Exception as exc:
                        raise_chat_error(backup_config, exc, user_key)

            return await self._wrap_streaming(
                primary_stream_factory,
                user_key,
                total_tokens,
                backup_factory=backup_stream_factory,
                max_retries=_max_retries,
                base_delay=_base_delay,
            )

        active_limiter_scope = limiter_scope
        try:
            with provider_call(self.config.provider, model_name, breaker=breaker):
                result = await _async_retry_call(
                    self._instantiate_provider(self.config, provider_cls).create,
                    max_retries=_max_retries, base_delay=_base_delay, max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
                    messages=truncated_messages, stream=stream, **request_kwargs
                )
        except (ChatCompletionError, APIConnectionError, RateLimitExceededError, TimeOutError) as primary_err:
            if not getattr(self.config, "backup", None) or not _is_retryable(primary_err):
                raise
            logger.warning("Primary async provider failed (%s), failing over to backup.", primary_err)
            emit("failover", from_provider=self.config.provider,
                 to_provider=(getattr(self.config, "backup", None) or {}).get("provider"),
                 error=type(primary_err).__name__, status_code=getattr(primary_err, "status_code", None))
            try:
                backup_execution = build_backup_execution_plan(
                    self.config,
                    operation="chat.completions",
                    normalized_request={"messages": messages, **kwargs},
                )
                if backup_execution is None:
                    raise LLMSwitchError("Backup provider configuration is missing")
                backup_config, backup_plan = backup_execution
                backup_limiter_scope = backup_plan.target.limiter_scope_key
                backup_request_kwargs_payload = dict(backup_plan.provider_request)
                backup_messages, backup_tokens = await asyncio.to_thread(
                    prepare_target_messages, backup_config, messages, self.rate_limiter,
                    truncate, backup_request_kwargs_payload,
                )
                await areserve_rate_limit(
                    self.rate_limiter, user_key, backup_config.rpm, backup_config.tpm,
                    self.window_seconds, backup_tokens, scope=backup_limiter_scope,
                )
                backup_provider = prepare_backup_provider(self.config, ASYNC_CHAT_PROVIDER_REGISTRY,
                                                          self._instantiate_provider)
                backup_breaker = (
                    get_breaker(backup_plan.target.breaker_key, **getattr(self, "_breaker_config", {}))
                    if getattr(self, "_breaker_enabled", False) else None
                )
                with provider_call(backup_config.provider, backup_plan.target.resolved_model,
                                   breaker=backup_breaker):
                    result = await _async_retry_call(
                        backup_provider.create,
                        max_retries=_max_retries, base_delay=_base_delay, max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
                        messages=backup_messages, stream=stream, **backup_request_kwargs_payload
                    )
                active_limiter_scope = backup_limiter_scope
                total_tokens = backup_tokens
            except Exception as backup_err:
                logger.error("Async backup provider also failed: %s", backup_err)
                raise primary_err from backup_err
        except Exception as e:
            raise_chat_error(self.config, e, user_key)

        openai_tokens = getattr(getattr(result, "usage", None), "total_tokens", 0)
        actual = max(0, openai_tokens - total_tokens)
        await asyncio.to_thread(
            self.rate_limiter.update_token_usage,
            user_key,
            actual,
            self.window_seconds,
            "completion",
            active_limiter_scope,
        )
        return result

    async def parse(self, messages: list[dict[str, Any]], response_format: Any, truncate: bool = False,
                    **kwargs: Any) -> Any:
        """
        Async structured-output helper mirroring
        ``client.beta.chat.completions.parse``. Always non-streaming.
        """
        return await self.create(messages, stream=False, truncate=truncate,
                                 response_format=response_format, **kwargs)


class _AsyncResponseInputItems:
    """``switch.responses.input_items`` sub-namespace (async)."""

    def __init__(self, responses: "_AsyncResponses") -> None:
        self._responses = responses

    async def list(self, response_id: str, **kwargs: Any) -> Any:
        """List the input items of a response (cursor page)."""
        return await self._responses._management_call("list_input_items", response_id, **kwargs)


class _AsyncResponseInputTokens:
    """``switch.responses.input_tokens`` sub-namespace (async)."""

    def __init__(self, responses: "_AsyncResponses") -> None:
        self._responses = responses

    async def count(self, **kwargs: Any) -> Any:
        """Count the input tokens a Responses request would consume."""
        return await self._responses._management_call("count_input_tokens", **kwargs)


class _AsyncConversations:
    """``switch.responses.conversations`` resource namespace (async)."""

    def __init__(self, responses: "_AsyncResponses") -> None:
        self._responses = responses

    async def create(self, items: Any = None, **kwargs: Any) -> Any:
        return await self._responses._management_call("create_conversation", items, **kwargs)

    async def retrieve(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._responses._management_call("retrieve_conversation", conversation_id, **kwargs)

    async def update(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._responses._management_call("update_conversation", conversation_id, **kwargs)

    async def delete(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._responses._management_call("delete_conversation", conversation_id, **kwargs)

    async def list_items(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._responses._management_call("list_conversation_items", conversation_id, **kwargs)

    async def retrieve_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return await self._responses._management_call("retrieve_conversation_item", conversation_id, item_id, **kwargs)

    async def delete_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return await self._responses._management_call("delete_conversation_item", conversation_id, item_id, **kwargs)


class _AsyncResponses(ProviderCache):
    """
    Async Responses API handler (``switch.responses.*``).

    Mirrors :class:`llmswitch.core.sync_client._Responses`: ``create`` runs the
    full pipeline (truncation, rate limiting, retries, circuit breaker, backup
    failover, token accounting) with pre-flight work off-loaded to a worker
    thread; the management verbs are retried and error-translated but not metered.
    """

    def __init__(self, config: Any, rate_limiter: "RedisRateLimiter") -> None:
        self.config = config
        self.rate_limiter = rate_limiter
        self.rpm = getattr(config, "rpm", DEFAULT_RPM)
        self.tpm = getattr(config, "tpm", DEFAULT_TPM)
        self.completion_context = getattr(config, "completion_context_window", DEFAULT_COMPLETION_CONTEXT_WINDOW)
        self.window_seconds = DEFAULT_WINDOW_SECONDS
        self._provider_cache: dict[tuple[type, tuple[Any, ...]], Any] = {}
        self.input_items = _AsyncResponseInputItems(self)
        self.input_tokens = _AsyncResponseInputTokens(self)
        self.conversations = _AsyncConversations(self)

    def capabilities(self) -> Any:
        """Return the resolved Responses capability contract for this target."""
        return resolve_response_capabilities(self.config)

    def _provider_class(self) -> type:
        return responses_provider_class(self.config, ASYNC_RESPONSES_PROVIDER_REGISTRY)

    @traced("responses.management", detail=lambda args, kwargs: {"verb": args[0] if args else None})
    async def _management_call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        """Run a non-generating Responses verb through retry + error translation."""
        provider_cls = self._provider_class()
        call_args, provider_kwargs, _max_retries, _ = prepare_management_call(
            self.config, operation, args, kwargs,
        )
        user_key: str = getattr(self.config, "user_id", "global")
        try:
            provider = self._instantiate_provider(self.config, provider_cls)
            return await _async_retry_call(
                getattr(provider, operation), *call_args,
                max_retries=_max_retries,
                base_delay=getattr(self.config, "retry_base_delay", DEFAULT_RETRY_BASE_DELAY),
                max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
                **provider_kwargs,
            )
        except Exception as e:
            raise_response_error(self.config, e, user_key, operation)

    async def _wrap_streaming(self, primary_factory, user_key: str, pre_count_tokens: int, *,
                              backup_factory=None, max_retries: int = 0, base_delay: float = 0.0):
        async def on_complete(last_usage, limiter_scope, prompt_tokens):
            actual = _additional_stream_tokens(last_usage, prompt_tokens, pre_count_tokens)
            await asyncio.to_thread(
                self.rate_limiter.update_token_usage,
                user_key,
                actual,
                self.window_seconds,
                "completion",
                limiter_scope,
            )

        return AsyncStreamSession(
            primary_factory=primary_factory,
            backup_factory=backup_factory,
            is_output_visible=response_event_has_visible_output,
            terminal_status=response_event_terminal_status,
            on_complete=on_complete,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
        )

    def _preflight(self, input: Any, truncate: bool, model_name: Any,
                   reserved_output_tokens: int, overhead_tokens: int):  # noqa: A002
        """Blocking pre-flight (tokenize + rate-limit gate), run in a worker thread."""
        prepared, total_tokens = prepare_response_input(
            input, self.rate_limiter, self.completion_context, truncate, model_name,
            reserved_output_tokens, overhead_tokens,
        )
        return prepared, total_tokens

    @traced("responses")
    async def create(self, input: Any = None, stream: bool = False, truncate: bool = False,  # noqa: A002
                     **kwargs: Any) -> Any:
        """Async Responses create. See ``_Responses.create`` for full parameter documentation."""
        live_logs = kwargs.pop("live_logs", False)
        _max_retries: int = kwargs.pop("max_retries", getattr(self.config, "max_retries", DEFAULT_MAX_RETRIES))
        _base_delay: float = getattr(self.config, "retry_base_delay", DEFAULT_RETRY_BASE_DELAY)
        provider_cls = self._provider_class()
        normalize_response_tool_definitions(kwargs)
        validate_responses_create_request(self.config, input, kwargs)
        side_effecting_request = is_side_effecting_response_request(kwargs, input)
        stateful_request = kwargs.get("previous_response_id") is not None or kwargs.get("conversation") is not None
        if side_effecting_request:
            _max_retries = 0
        if not side_effecting_request and not stateful_request:
            validate_backup_for_registry(self.config, ASYNC_RESPONSES_PROVIDER_REGISTRY)

        user_key = getattr(self.config, "user_id", "global")
        if live_logs:
            logger.info("Responses request: %s", json.dumps(redact_sensitive(
                {"provider": self.config.provider, "input": "<hidden>", "stream": stream, "kwargs": kwargs}),
                default=str))
        response_plan = plan_responses(self.config, self.rate_limiter, input, kwargs)
        plan, model_name = response_plan.plan, response_plan.model_name
        reserved_output_tokens, overhead_tokens = response_plan.reserved_output_tokens, response_plan.overhead_tokens
        limiter_scope = plan.target.limiter_scope_key

        prepared_input, total_tokens = await asyncio.to_thread(
            self._preflight, input, truncate, model_name, reserved_output_tokens, overhead_tokens,
        )
        request_kwargs = dict(plan.provider_request)
        await areserve_rate_limit(
            self.rate_limiter, user_key, self.rpm, self.tpm,
            self.window_seconds, total_tokens, scope=limiter_scope,
        )

        breaker = (get_breaker(plan.target.breaker_key, **getattr(self, "_breaker_config", {}))
                   if getattr(self, "_breaker_enabled", False) else None)
        if stream:
            async def primary_stream_factory():
                try:
                    with provider_call(self.config.provider, model_name, breaker=breaker, record_success=False):
                        result = await self._instantiate_provider(self.config, provider_cls).create(
                            input=prepared_input, stream=True, **request_kwargs
                        )
                    return StreamHandle(
                        result, breaker=breaker, limiter_scope=limiter_scope, prompt_tokens=total_tokens,
                    )
                except Exception as exc:
                    raise_response_error(self.config, exc, user_key, "create")

            backup_stream_factory = None
            if not side_effecting_request and not stateful_request and getattr(self.config, "backup", None):
                async def backup_stream_factory():
                    backup_execution = build_backup_execution_plan(
                        self.config,
                        operation="responses",
                        normalized_request={"input": input, **kwargs},
                    )
                    if backup_execution is None:
                        raise LLMSwitchError("Backup provider configuration is missing")
                    backup_config, backup_plan = backup_execution
                    backup_limiter_scope = backup_plan.target.limiter_scope_key
                    backup_request_kwargs_payload = dict(backup_plan.provider_request)
                    validate_responses_create_request(backup_config, input, backup_request_kwargs_payload)
                    backup_input, backup_tokens = await asyncio.to_thread(
                        prepare_target_response_input, backup_config, input, self.rate_limiter,
                        truncate, backup_request_kwargs_payload,
                    )
                    await areserve_rate_limit(
                        self.rate_limiter, user_key, backup_config.rpm, backup_config.tpm,
                        self.window_seconds, backup_tokens, scope=backup_limiter_scope,
                    )
                    backup_provider = prepare_backup_provider(self.config, ASYNC_RESPONSES_PROVIDER_REGISTRY,
                                                              self._instantiate_provider)
                    backup_breaker = (
                        get_breaker(backup_plan.target.breaker_key, **getattr(self, "_breaker_config", {}))
                        if getattr(self, "_breaker_enabled", False) else None
                    )
                    try:
                        with provider_call(
                            backup_config.provider,
                            backup_plan.target.resolved_model,
                            breaker=backup_breaker,
                            record_success=False,
                        ):
                            result = await backup_provider.create(
                                input=backup_input, stream=True, **backup_request_kwargs_payload,
                            )
                        return StreamHandle(
                            result, breaker=backup_breaker, limiter_scope=backup_limiter_scope,
                            prompt_tokens=backup_tokens,
                        )
                    except Exception as exc:
                        raise_response_error(self.config, exc, user_key, "create")

            return await self._wrap_streaming(
                primary_stream_factory,
                user_key,
                total_tokens,
                backup_factory=backup_stream_factory,
                max_retries=_max_retries,
                base_delay=_base_delay,
            )

        active_limiter_scope = limiter_scope
        try:
            with provider_call(self.config.provider, model_name, breaker=breaker):
                result = await _async_retry_call(
                    self._instantiate_provider(self.config, provider_cls).create,
                    max_retries=_max_retries, base_delay=_base_delay, max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
                    input=prepared_input, stream=stream, **request_kwargs
                )
        except (ResponsesError, APIConnectionError, RateLimitExceededError, TimeOutError) as primary_err:
            if (side_effecting_request or stateful_request or not getattr(self.config, "backup", None)
                    or not _is_retryable(primary_err)):
                raise
            logger.warning("Primary async provider failed (%s), failing over to backup.", primary_err)
            emit("failover", from_provider=self.config.provider,
                 to_provider=(getattr(self.config, "backup", None) or {}).get("provider"),
                 error=type(primary_err).__name__, status_code=getattr(primary_err, "status_code", None))
            try:
                backup_execution = build_backup_execution_plan(
                    self.config,
                    operation="responses",
                    normalized_request={"input": input, **kwargs},
                )
                if backup_execution is None:
                    raise LLMSwitchError("Backup provider configuration is missing")
                backup_config, backup_plan = backup_execution
                backup_limiter_scope = backup_plan.target.limiter_scope_key
                backup_request_kwargs_payload = dict(backup_plan.provider_request)
                validate_responses_create_request(backup_config, input, backup_request_kwargs_payload)
                backup_input, backup_tokens = await asyncio.to_thread(
                    prepare_target_response_input, backup_config, input, self.rate_limiter,
                    truncate, backup_request_kwargs_payload,
                )
                await areserve_rate_limit(
                    self.rate_limiter, user_key, backup_config.rpm, backup_config.tpm,
                    self.window_seconds, backup_tokens, scope=backup_limiter_scope,
                )
                backup_provider = prepare_backup_provider(self.config, ASYNC_RESPONSES_PROVIDER_REGISTRY,
                                                          self._instantiate_provider)
                backup_breaker = (
                    get_breaker(backup_plan.target.breaker_key, **getattr(self, "_breaker_config", {}))
                    if getattr(self, "_breaker_enabled", False) else None
                )
                with provider_call(backup_config.provider, backup_plan.target.resolved_model,
                                   breaker=backup_breaker):
                    result = await _async_retry_call(
                        backup_provider.create,
                        max_retries=_max_retries, base_delay=_base_delay, max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
                        input=backup_input, stream=stream, **backup_request_kwargs_payload
                    )
                active_limiter_scope = backup_limiter_scope
                total_tokens = backup_tokens
            except Exception as backup_err:
                logger.error("Async backup provider also failed: %s", backup_err)
                raise primary_err from backup_err
        except Exception as e:
            raise_response_error(self.config, e, user_key, "create")

        provider_tokens = getattr(getattr(result, "usage", None), "total_tokens", 0)
        await asyncio.to_thread(
            self.rate_limiter.update_token_usage,
            user_key,
            max(0, provider_tokens - total_tokens),
            self.window_seconds,
            "completion",
            active_limiter_scope,
        )
        return result

    async def stream(self, input: Any = None, truncate: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        """Streaming shorthand for ``create(..., stream=True)``."""
        return await self.create(input, stream=True, truncate=truncate, **kwargs)

    async def create_background(self, input: Any = None, truncate: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        """Create a stored background response suitable for retrieval and cancellation."""
        kwargs["background"] = True
        kwargs.setdefault("store", True)
        return await self.create(input, stream=False, truncate=truncate, **kwargs)

    async def submit_mcp_approval(self, previous_response_id: str, approval: Any, **kwargs: Any) -> Any:
        """Continue a response after explicit application approval or rejection of an MCP call."""
        item = approval.to_input_item() if hasattr(approval, "to_input_item") else dict(approval)
        return await self.create([item], previous_response_id=previous_response_id, **kwargs)

    async def wait(self, response_id: str, *, timeout: float = 600.0, poll_interval: float = 1.0,
                   max_poll_interval: float = 10.0, cancel_on_timeout: bool = False,
                   on_poll: Any = None, **kwargs: Any) -> Any:
        """Asynchronously poll a background response until it reaches a terminal state."""
        if timeout <= 0 or poll_interval <= 0 or max_poll_interval <= 0:
            raise ValueError("timeout, poll_interval, and max_poll_interval must be positive")
        deadline = asyncio.get_running_loop().time() + timeout
        delay = poll_interval
        attempts = 0
        terminal = {"completed", "failed", "incomplete", "cancelled"}
        while True:
            response = await self.retrieve(response_id, **kwargs)
            attempts += 1
            if callable(on_poll):
                result = on_poll(response, attempts)
                if inspect.isawaitable(result):
                    await result
            if response.status in terminal:
                return response
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                if cancel_on_timeout:
                    try:
                        await self.cancel(response_id)
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to cancel timed-out background response", exc_info=True)
                raise TimeOutError(int(timeout))
            await asyncio.sleep(min(remaining, delay))
            delay = min(max_poll_interval, max(poll_interval, random.uniform(0, delay * 2)))

    async def parse(self, input: Any = None, text_format: Any = None, truncate: bool = False,  # noqa: A002
                    **kwargs: Any) -> Any:
        """Async structured-output helper mirroring ``client.responses.parse``. Always non-streaming."""
        return await self.create(input, stream=False, truncate=truncate, text_format=text_format, **kwargs)

    async def retrieve(self, response_id: str, **kwargs: Any) -> Any:
        """Retrieve a previously created response by id."""
        return await self._management_call("retrieve", response_id, **kwargs)

    async def delete(self, response_id: str, **kwargs: Any) -> Any:
        """Delete a stored response by id."""
        return await self._management_call("delete", response_id, **kwargs)

    async def cancel(self, response_id: str, **kwargs: Any) -> Any:
        """Cancel an in-progress background response."""
        return await self._management_call("cancel", response_id, **kwargs)

    async def compact(self, **kwargs: Any) -> Any:
        """Compact a response's conversation history into a shorter one."""
        return await self._management_call("compact", **kwargs)


class _AsyncChat:
    """Wrapper for async chat completions, used by AsyncLLMSwitch."""

    def __init__(self, config: Any, rate_limiter: "RedisRateLimiter") -> None:
        self.completions = _AsyncChatCompletions(config, rate_limiter)


class _AsyncEmbeddings(ProviderCache):
    """
    Async embeddings facade with the same pipeline as ``_Embeddings``: rate limiting,
    batching, retries, and backup failover. Providers with a native async embedding
    class (see ``ASYNC_EMBEDDING_PROVIDER_REGISTRY``) are awaited directly; the others
    run their sync provider in a worker thread. Blocking tokenization and limiter
    calls always run off the event loop.
    """

    def __init__(self, config: Any, rate_limiter: "RedisRateLimiter") -> None:
        self.config = config
        self.rate_limiter = rate_limiter
        self.batch_utility = BatchUtility()
        self.window_seconds = DEFAULT_WINDOW_SECONDS
        self._provider_cache: dict[tuple[type, tuple[Any, ...]], Any] = {}

    @staticmethod
    def _target_class(provider: str) -> tuple[type, bool]:
        """Return the embedding provider class and whether it is natively async."""
        if provider in ASYNC_EMBEDDING_PROVIDER_REGISTRY:
            return ASYNC_EMBEDDING_PROVIDER_REGISTRY[provider], True
        return provider_class(provider, EMBEDDING_PROVIDER_REGISTRY, "embeddings"), False

    @traced("embeddings")
    async def create(self, input: str | list[str], batching: bool = False, batchsize: int = 32,  # noqa: A002
                     **kwargs: Any):
        """Create embeddings. See ``_Embeddings.create`` for the parameters."""
        input = normalize_embedding_input(input)
        if batchsize <= 0:
            raise ValueError("batchsize must be positive")
        live_logs = kwargs.pop("live_logs", False)
        retry_override: int | None = kwargs.pop("max_retries", None)
        restore_order = kwargs.pop("restore_order", True)
        provider_class(self.config.provider, EMBEDDING_PROVIDER_REGISTRY, "embeddings")
        require_operation(self.config.provider, "embeddings")
        validate_backup_for_registry(self.config, EMBEDDING_PROVIDER_REGISTRY)
        if live_logs:
            logger.info("Embedding request: %s", json.dumps(redact_sensitive(
                {"provider": self.config.provider, "input": "<hidden>", "kwargs": kwargs})))
        apply_model_priority(self.config, kwargs)
        plan = build_execution_plan(
            self.config, operation="embeddings", normalized_request={"embedding_input": input, **kwargs},
        )
        user_key = getattr(self.config, "user_id", "global")

        async def run_target(target_config: Any, target_plan: Any) -> Any:
            model = target_plan.target.resolved_model
            provider_cls, native = self._target_class(target_config.provider)

            def count_all() -> list[int]:
                counts: dict[str, int] = {}
                for text in input:
                    if text not in counts:
                        counts[text] = self.rate_limiter.count_tokens(text, model_name=model)
                return [counts[text] for text in input]

            token_counts = await asyncio.to_thread(count_all)
            batches = self.batch_utility.plan_batches(
                token_counts, batchsize, target_config.embedding_tpm,
                batching=batching, restore_order=restore_order,
            )
            acc = EmbeddingAccumulator(target_config.provider, model, restore_order)
            request_kwargs = dict(target_plan.provider_request)
            for batch_idx, indices in enumerate(batches):
                if batching:
                    self.batch_utility.print_progress_bar(batch_idx + 1, len(batches))
                await areserve_rate_limit(
                    self.rate_limiter, user_key, target_config.embedding_rpm, target_config.embedding_tpm,
                    self.window_seconds, sum(token_counts[i] for i in indices),
                    request_type="embedding", suppress_log=batching, scope=target_plan.target.limiter_scope_key,
                )
                try:
                    provider = self._instantiate_provider(target_config, provider_cls)
                    texts = [input[i] for i in indices]
                    retry_options = {
                        "max_retries": retry_override if retry_override is not None else target_config.max_retries,
                        "base_delay": target_config.retry_base_delay,
                        "max_delay": DEFAULT_RETRY_MAX_DELAY, "deadline": self.config.retry_deadline,
                    }
                    if native:
                        result = await _async_retry_call(provider.create, texts, **retry_options, **request_kwargs)
                    else:
                        result = await asyncio.to_thread(
                            _retry_call, provider.create, texts, **retry_options, **request_kwargs)
                    acc.add(result, indices)
                except Exception as exc:
                    raise_embedding_error(target_config, exc)
            await asyncio.to_thread(
                self.rate_limiter.update_token_usage,
                user_key, max(0, acc.total_tokens - sum(token_counts)), self.window_seconds,
                "embedding", target_plan.target.limiter_scope_key,
            )
            return acc.build()

        try:
            return await run_target(self.config, plan)
        except (EmbeddingGenerationError, APIConnectionError, RateLimitExceededError, TimeOutError) as primary_err:
            if not getattr(self.config, "backup", None):
                raise
            logger.warning("Primary embedding provider failed (%s), failing over to backup.", primary_err)
            emit("failover", from_provider=self.config.provider,
                 to_provider=(getattr(self.config, "backup", None) or {}).get("provider"),
                 error=type(primary_err).__name__, status_code=getattr(primary_err, "status_code", None))
            try:
                backup_execution = build_backup_execution_plan(
                    self.config, operation="embeddings", normalized_request={"embedding_input": input, **kwargs},
                )
                if backup_execution is None:
                    raise LLMSwitchError("Backup provider configuration is missing")
                backup_config, backup_plan = backup_execution
                require_operation(backup_config.provider, "embeddings")
                return await run_target(backup_config, backup_plan)
            except Exception as backup_err:
                logger.error("Backup embedding provider also failed: %s", backup_err)
                raise primary_err from backup_err


class AsyncLLMSwitch:
    """
    Async entry point mimicking the OpenAI async client. Chat completions use the
    provider's native async client. ``.embeddings`` uses the native async client for
    azure_foundry and a worker thread for the other providers, so the event loop stays free.

    Providers with native async chat: azure_openai, azure_foundry, openai, anthropic, vertexai.

    Usage::

        switch = AsyncLLMSwitch(provider="openai", api_key="...", model="gpt-4o")
        # Non-streaming
        response = await switch.chat.completions.create(messages=[...])
        # Streaming
        async for chunk in await switch.chat.completions.create(messages=[...], stream=True):
            print(chunk.choices[0].delta.content, end="")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize AsyncLLMSwitch. Accepts the same kwargs as LLMSwitch."""
        token_counter = kwargs.pop("token_counter", None)
        on_event = kwargs.pop("on_event", None)
        if on_event is not None and not callable(on_event):
            raise ConfigurationError("on_event must be a callable that takes one dict")
        share_clients = kwargs.pop("share_clients", False)
        circuit_breaker = kwargs.pop("circuit_breaker", False)
        self.config = build_config(kwargs)
        try:
            validate_backup_config(self.config)
        except ValueError as e:
            raise ConfigurationError("Invalid backup configuration", details=e.errors() if hasattr(e, "errors") else str(e)) from e

        redis_url = getattr(self.config, "redis_url", None)
        self.rate_limiter = RedisRateLimiter(
            redis_url,
            token_counter=token_counter,
            failure_policy=self.config.rate_limit_failure_policy,
            connect_timeout=self.config.redis_connect_timeout,
            on_limit=self.config.on_rate_limit,
            max_wait=self.config.rate_limit_max_wait,
            scope=rate_limit_scope(
                self.config.provider,
                getattr(self.config, "endpoint", None),
                getattr(self.config, "deployment_name", None),
                self.config.model,
            ),
        )
        self.chat = _AsyncChat(self.config, self.rate_limiter)
        self.embeddings = _AsyncEmbeddings(self.config, self.rate_limiter)
        self.responses = _AsyncResponses(self.config, self.rate_limiter)
        self.chat.completions._on_event = on_event
        self.embeddings._on_event = on_event
        self.responses._on_event = on_event
        if share_clients:
            self.chat.completions._share_clients = True
            self.embeddings._share_clients = True
            self.responses._share_clients = True
        if circuit_breaker:
            self.chat.completions._breaker_enabled = True
            self.responses._breaker_enabled = True
            if isinstance(circuit_breaker, dict):
                breaker_config = {
                    k: circuit_breaker[k] for k in ("failure_threshold", "reset_timeout")
                    if k in circuit_breaker
                }
                self.chat.completions._breaker_config = breaker_config
                self.responses._breaker_config = dict(breaker_config)

    def __repr__(self) -> str:
        return (
            f"AsyncLLMSwitch(provider={self.config.provider!r}, "
            f"model={self.config.model!r}, "
            f"user_id={self.config.user_id!r})"
        )

    async def __aenter__(self) -> "AsyncLLMSwitch":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close async resources held by the switch."""
        await self.chat.completions.aclose()
        await self.embeddings.aclose()
        await self.responses.aclose()
        self.rate_limiter.close()
