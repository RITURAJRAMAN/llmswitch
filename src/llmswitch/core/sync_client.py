"""
core.sync_client

Synchronous chat + embedding + Responses handlers and the public ``LLMSwitch``
entry point (mimics the OpenAI client surface). Provider selection, rate
limiting, retries, truncation, and backup failover are delegated to the shared
``core`` helpers.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from ..constants import (
    DEFAULT_COMPLETION_CONTEXT_WINDOW,
    DEFAULT_EMBEDDING_RPM,
    DEFAULT_EMBEDDING_TPM,
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
from ..models import CreateEmbeddingResponse, ProviderExtensions, Usage
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
from .events import emit, traced
from .execution import build_execution_plan, reserve_rate_limit
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
    CHAT_PROVIDER_REGISTRY,
    EMBEDDING_PROVIDER_REGISTRY,
    RESPONSES_PROVIDER_REGISTRY,
)
from .responses_capabilities import (
    is_side_effecting_response_request,
    normalize_response_tool_definitions,
    resolve_response_capabilities,
    validate_responses_create_request,
)
from .responses_management import prepare_management_call, raise_response_error
from .retry import _is_retryable, _retry_call
from .streaming import (
    StreamHandle,
    SyncStreamSession,
    _additional_stream_tokens,
    chat_chunk_has_visible_output,
    response_event_has_visible_output,
    response_event_terminal_status,
)

logger = logging.getLogger(__name__)


class _ChatCompletions(ProviderCache):
    """Handles chat completion requests, including rate limiting and message truncation."""

    def __init__(self, config: Any, rate_limiter: "RedisRateLimiter") -> None:
        self.config = config
        self.rate_limiter = rate_limiter
        # Requests per minute (RPM) and tokens per minute (TPM) limits
        self.rpm = getattr(config, "rpm", DEFAULT_RPM)
        self.tpm = getattr(config, "tpm", DEFAULT_TPM)
        self.completion_context = getattr(config, "completion_context_window", DEFAULT_COMPLETION_CONTEXT_WINDOW)
        self.window_seconds = DEFAULT_WINDOW_SECONDS
        self.live_logs: bool = False
        self._provider_cache: dict[tuple[type, tuple[Any, ...]], Any] = {}

    def _wrap_streaming(self, primary_factory, user_key: str, pre_count_tokens: int, *,
                        backup_factory=None, max_retries: int = 0, base_delay: float = 0.0):
        def on_complete(last_usage, limiter_scope, prompt_tokens):
            actual = _additional_stream_tokens(last_usage, prompt_tokens, pre_count_tokens)
            self.rate_limiter.update_token_usage(
                user_key,
                actual,
                self.window_seconds,
                request_type="completion",
                scope=limiter_scope,
            )

        return SyncStreamSession(
            primary_factory=primary_factory,
            backup_factory=backup_factory,
            is_output_visible=chat_chunk_has_visible_output,
            on_complete=on_complete,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
        )

    @traced("chat.completions")
    def create(self, messages: list[dict[str, Any]], stream: bool = False, truncate: bool = False,
               **kwargs: Any) -> Any:
        """
        Create a chat completion, applying rate limiting and truncation as needed.

        Args:
            messages (list[dict]): List of message dicts. The 'content' field can be a string
                or a list of content parts (multimodal / vision).
            stream (bool): Whether to use streaming completion.
            truncate (bool): Whether to truncate messages to fit token limits.
            **kwargs: Additional pass-through parameters for the provider (response_format,
                tools, tool_choice, stop, seed, etc.). ``live_logs`` (default False) toggles
                INFO logging; ``max_retries`` overrides the configured retry count.

        Returns:
            ChatCompletion or generator of ChatCompletionChunk.
        """
        validate_messages(messages)
        self.live_logs = kwargs.pop("live_logs", False)
        _max_retries: int = kwargs.pop("max_retries", getattr(self.config, "max_retries", DEFAULT_MAX_RETRIES))
        _base_delay: float = getattr(self.config, "retry_base_delay", DEFAULT_RETRY_BASE_DELAY)
        provider_cls = provider_class(self.config.provider, CHAT_PROVIDER_REGISTRY, "chat completion")
        require_operation(self.config.provider, "chat.sync")
        validate_backup_for_registry(self.config, CHAT_PROVIDER_REGISTRY)
        if self.live_logs:
            logger.info(
                json.dumps(redact_sensitive(
                    {"provider": self.config.provider, "messages": "<hidden>", "stream": stream, "kwargs": kwargs}),
                    default=str))
        chat_plan = plan_chat(self.config, self.rate_limiter, messages, kwargs, stream)
        plan, model_name = chat_plan.plan, chat_plan.model_name
        reserved_output_tokens, overhead_tokens = chat_plan.reserved_output_tokens, chat_plan.overhead_tokens
        truncated_messages, total_tokens = prepare_messages(
            messages, self.rate_limiter, self.completion_context, truncate, model_name,
            reserved_output_tokens, overhead_tokens,
        )
        request_kwargs = dict(plan.provider_request)
        limiter_scope = plan.target.limiter_scope_key

        user_key: str = getattr(self.config, "user_id", "global")
        reserve_rate_limit(
            self.rate_limiter, user_key, self.rpm, self.tpm,
            self.window_seconds, total_tokens, scope=limiter_scope,
        )

        breaker = (get_breaker(plan.target.breaker_key, **getattr(self, "_breaker_config", {}))
                   if getattr(self, "_breaker_enabled", False) else None)
        if stream:
            def primary_stream_factory():
                try:
                    with provider_call(self.config.provider, model_name, breaker=breaker, record_success=False):
                        result = self._instantiate_provider(self.config, provider_cls).create(
                            messages=truncated_messages, stream=True, **request_kwargs
                        )
                    return StreamHandle(
                        result, breaker=breaker, limiter_scope=limiter_scope, prompt_tokens=total_tokens,
                    )
                except Exception as exc:
                    raise_chat_error(self.config, exc, user_key)

            backup_stream_factory = None
            if getattr(self.config, "backup", None):
                def backup_stream_factory():
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
                    backup_messages, backup_tokens = prepare_target_messages(
                        backup_config, messages, self.rate_limiter, truncate, backup_request_kwargs_payload,
                    )
                    reserve_rate_limit(
                        self.rate_limiter, user_key, backup_config.rpm, backup_config.tpm,
                        self.window_seconds, backup_tokens, scope=backup_limiter_scope,
                    )
                    backup_provider = prepare_backup_provider(self.config, CHAT_PROVIDER_REGISTRY,
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
                            result = backup_provider.create(
                                messages=backup_messages, stream=True, **backup_request_kwargs_payload,
                            )
                        return StreamHandle(
                            result, breaker=backup_breaker, limiter_scope=backup_limiter_scope,
                            prompt_tokens=backup_tokens,
                        )
                    except Exception as exc:
                        raise_chat_error(backup_config, exc, user_key)

            if self.live_logs:
                logger.info(
                    "Chat Response: %s",
                    json.dumps(redact_sensitive({"provider": self.config.provider,
                                                 "response": "<Object(ChatCompletionChunk.Response)>"}))
                )
            return self._wrap_streaming(
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
                result = _retry_call(
                    self._instantiate_provider(self.config, provider_cls).create,
                    max_retries=_max_retries, base_delay=_base_delay, max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
                    messages=truncated_messages, stream=stream, **request_kwargs
                )
        except (ChatCompletionError, APIConnectionError, RateLimitExceededError, TimeOutError) as primary_err:
            if not getattr(self.config, "backup", None) or not _is_retryable(primary_err):
                raise
            logger.warning("Primary provider failed (%s), failing over to backup.", primary_err)
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
                backup_messages, backup_tokens = prepare_target_messages(
                    backup_config, messages, self.rate_limiter, truncate, backup_request_kwargs_payload,
                )
                reserve_rate_limit(
                    self.rate_limiter, user_key, backup_config.rpm, backup_config.tpm,
                    self.window_seconds, backup_tokens, scope=backup_limiter_scope,
                )
                backup_provider = prepare_backup_provider(self.config, CHAT_PROVIDER_REGISTRY,
                                                          self._instantiate_provider)
                backup_breaker = (
                    get_breaker(backup_plan.target.breaker_key, **getattr(self, "_breaker_config", {}))
                    if getattr(self, "_breaker_enabled", False) else None
                )
                with provider_call(backup_config.provider, backup_plan.target.resolved_model,
                                   breaker=backup_breaker):
                    result = _retry_call(
                        backup_provider.create,
                        max_retries=_max_retries, base_delay=_base_delay, max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
                        messages=backup_messages, stream=stream, **backup_request_kwargs_payload
                    )
                active_limiter_scope = backup_limiter_scope
                total_tokens = backup_tokens
            except Exception as backup_err:
                logger.error("Backup provider also failed: %s", backup_err)
                raise primary_err from backup_err
        except Exception as e:
            raise_chat_error(self.config, e, user_key)

        if self.live_logs:
            logger.info(
                "Chat Response: %s",
                json.dumps(redact_sensitive({"provider": self.config.provider,
                                             "response": "<Object(ChatCompletionChunk.Response)>" if stream else "<Object(ChatCompletion.Response)>"}))
            )

        openai_tokens = getattr(getattr(result, "usage", None), "total_tokens", 0)
        actual_tokens_used = max(0, openai_tokens - total_tokens)
        self.rate_limiter.update_token_usage(
            user_key,
            actual_tokens_used,
            self.window_seconds,
            request_type="completion",
            scope=active_limiter_scope,
        )
        return result

    def parse(self, messages: list[dict[str, Any]], response_format: Any, truncate: bool = False,
              **kwargs: Any) -> Any:
        """
        Structured-output helper mirroring ``client.beta.chat.completions.parse``.

        Pass a Pydantic model *class* as ``response_format``; the deserialized
        instance is returned on ``choice.message.parsed`` for every provider.
        Always non-streaming.
        """
        return self.create(messages, stream=False, truncate=truncate,
                           response_format=response_format, **kwargs)


class _Embeddings(ProviderCache):
    """Handles embedding requests, including rate limiting and batching."""

    def __init__(self, config: Any, rate_limiter: "RedisRateLimiter") -> None:
        self.live_logs: bool = False
        self.config = config
        self.rate_limiter = rate_limiter
        self.batch_utility = BatchUtility()
        # Embedding requests per minute and tokens per minute
        self.rpm = getattr(config, "embedding_rpm", DEFAULT_EMBEDDING_RPM)
        self.tpm = getattr(config, "embedding_tpm", DEFAULT_EMBEDDING_TPM)
        self.window_seconds = DEFAULT_WINDOW_SECONDS
        self._provider_cache: dict[tuple[type, tuple[Any, ...]], Any] = {}

    @traced("embeddings")
    def create(self, input: str | list[str], batching: bool = False, batchsize: int = 32, **kwargs: Any):  # noqa: A002
        input = normalize_embedding_input(input)
        """
        Create embeddings, applying rate limiting, batching, and truncation as needed.

        Args:
            input (list[str]): List of input strings to embed.
            batching (bool): Whether to use token-aware batching.
            batchsize (int): Maximum number of texts per batch.
            **kwargs: Additional parameters for provider and batching (model, dimensions, etc.).

        Returns:
            Aggregated embedding response object.
        """
        if isinstance(input, str):
            input = [input]
        if not input:
            raise ValueError("Embedding input must contain at least one string")
        if batchsize <= 0:
            raise ValueError("batchsize must be positive")
        self.live_logs = kwargs.pop("live_logs", False)
        retry_override: int | None = kwargs.pop("max_retries", None)
        _base_delay: float = getattr(self.config, "retry_base_delay", DEFAULT_RETRY_BASE_DELAY)
        provider_cls = provider_class(self.config.provider, EMBEDDING_PROVIDER_REGISTRY, "embeddings")
        require_operation(self.config.provider, "embeddings")
        validate_backup_for_registry(self.config, EMBEDDING_PROVIDER_REGISTRY)
        if self.live_logs:
            logger.info(
                "Embedding request: %s",
                json.dumps(redact_sensitive({"provider": self.config.provider, "input": "<hidden>", "kwargs": kwargs})))
        restore_order = kwargs.pop("restore_order", True)

        # Memoizing tokenizer: each unique text is encoded once, then reused for
        # the input total, batching (FFD), and per-batch totals below.
        apply_model_priority(self.config, kwargs)
        plan = build_execution_plan(
            self.config,
            operation="embeddings",
            normalized_request={"embedding_input": input, **kwargs},
        )
        user_key = getattr(self.config, "user_id", "global")

        def run_target(target_config: Any, target_plan: Any, target_cls: type) -> CreateEmbeddingResponse:
            model = target_plan.target.resolved_model
            counts: dict[str, int] = {}

            def count(text: str) -> int:
                if text not in counts:
                    counts[text] = self.rate_limiter.count_tokens(
                        text, model_name=model,
                    )
                return counts[text]

            token_counts = [count(text) for text in input]
            batches = self.batch_utility.plan_batches(
                token_counts, batchsize, target_config.embedding_tpm,
                batching=batching, restore_order=restore_order,
            )
            all_results = []
            usages = []
            extensions: list[tuple[list[int], ProviderExtensions]] = []
            response_model = target_plan.target.resolved_model
            request_kwargs = dict(target_plan.provider_request)
            for batch_idx, indices in enumerate(batches):
                if batching:
                    self.batch_utility.print_progress_bar(batch_idx + 1, len(batches))
                reserve_rate_limit(
                    self.rate_limiter, user_key, target_config.embedding_rpm, target_config.embedding_tpm,
                    self.window_seconds, sum(token_counts[i] for i in indices),
                    request_type="embedding", suppress_log=batching,
                    scope=target_plan.target.limiter_scope_key,
                )
                try:
                    provider = self._instantiate_provider(target_config, target_cls)
                    result = _retry_call(
                        provider.create, [input[i] for i in indices],
                        max_retries=retry_override if retry_override is not None else target_config.max_retries,
                        base_delay=target_config.retry_base_delay,
                        max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline, **request_kwargs,
                    )
                    data = getattr(result, "data", result)
                    if len(data) != len(indices):
                        raise EmbeddingGenerationError(
                            f"Provider returned {len(data)} embeddings for {len(indices)} inputs"
                        )
                    for index, embedding in zip(indices, data, strict=True):
                        embedding.index = index
                        all_results.append(embedding)
                    usage = getattr(result, "usage", None)
                    if usage is not None:
                        usages.append(usage)
                    extension = getattr(result, "extensions", None)
                    if extension is not None:
                        extensions.append((indices, extension))
                    response_model = getattr(result, "model", None) or response_model
                except Exception as exc:
                    raise_embedding_error(target_config, exc)

            if restore_order:
                all_results.sort(key=lambda item: item.index)
            prompt_tokens = sum(getattr(usage, "prompt_tokens", 0) for usage in usages)
            total_tokens = sum(getattr(usage, "total_tokens", 0) for usage in usages)
            self.rate_limiter.update_token_usage(
                user_key, max(0, total_tokens - sum(token_counts)), self.window_seconds,
                request_type="embedding", scope=target_plan.target.limiter_scope_key,
            )
            aggregated_extensions = None
            if len(extensions) == 1:
                aggregated_extensions = extensions[0][1]
            elif extensions:
                aggregated_extensions = ProviderExtensions(
                    provider=target_config.provider,
                    fields={"batches": [
                        {"input_indices": indices, "fields": extension.fields}
                        for indices, extension in extensions
                    ]},
                )
            return CreateEmbeddingResponse(
                data=all_results,
                model=response_model,
                object="list",
                usage=Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
                extensions=aggregated_extensions,
                created=int(time.time()),
            )

        try:
            return run_target(self.config, plan, provider_cls)
        except (EmbeddingGenerationError, APIConnectionError, RateLimitExceededError, TimeOutError) as primary_err:
            if not getattr(self.config, "backup", None):
                raise
            logger.warning("Primary embedding provider failed (%s), failing over to backup.", primary_err)
            emit("failover", from_provider=self.config.provider,
                 to_provider=(getattr(self.config, "backup", None) or {}).get("provider"),
                 error=type(primary_err).__name__, status_code=getattr(primary_err, "status_code", None))
            try:
                backup_execution = build_backup_execution_plan(
                    self.config, operation="embeddings",
                    normalized_request={"embedding_input": input, **kwargs},
                )
                if backup_execution is None:
                    raise LLMSwitchError("Backup provider configuration is missing")
                backup_config, backup_plan = backup_execution
                require_operation(backup_config.provider, "embeddings")
                backup_cls = EMBEDDING_PROVIDER_REGISTRY[backup_config.provider]
                return run_target(backup_config, backup_plan, backup_cls)
            except Exception as backup_err:
                logger.error("Backup embedding provider also failed: %s", backup_err)
                raise primary_err from backup_err


class _ResponseInputItems:
    """``switch.responses.input_items`` sub-namespace."""

    def __init__(self, responses: "_Responses") -> None:
        self._responses = responses

    def list(self, response_id: str, **kwargs: Any) -> Any:
        """List the input items of a response (cursor page)."""
        return self._responses._management_call("list_input_items", response_id, **kwargs)


class _ResponseInputTokens:
    """``switch.responses.input_tokens`` sub-namespace."""

    def __init__(self, responses: "_Responses") -> None:
        self._responses = responses

    def count(self, **kwargs: Any) -> Any:
        """Count the input tokens a Responses request would consume."""
        return self._responses._management_call("count_input_tokens", **kwargs)


class _Conversations:
    """``switch.responses.conversations`` resource namespace."""

    def __init__(self, responses: "_Responses") -> None:
        self._responses = responses

    def create(self, items: Any = None, **kwargs: Any) -> Any:
        return self._responses._management_call("create_conversation", items, **kwargs)

    def retrieve(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._responses._management_call("retrieve_conversation", conversation_id, **kwargs)

    def update(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._responses._management_call("update_conversation", conversation_id, **kwargs)

    def delete(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._responses._management_call("delete_conversation", conversation_id, **kwargs)

    def list_items(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._responses._management_call("list_conversation_items", conversation_id, **kwargs)

    def retrieve_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return self._responses._management_call("retrieve_conversation_item", conversation_id, item_id, **kwargs)

    def delete_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return self._responses._management_call("delete_conversation_item", conversation_id, item_id, **kwargs)


class _Responses(ProviderCache):
    """
    Handles Responses API requests (``switch.responses.*``).

    ``create`` runs the same pipeline as chat completions — truncation, rate
    limiting, retries, circuit breaker, backup failover, token accounting — and
    shares the completion RPM/TPM budget, since both hit the same deployment
    quota. The management verbs (retrieve / delete / cancel / compact /
    input_items.list / input_tokens.count) are retried and error-translated but
    are not metered.
    """

    def __init__(self, config: Any, rate_limiter: "RedisRateLimiter") -> None:
        self.config = config
        self.rate_limiter = rate_limiter
        self.rpm = getattr(config, "rpm", DEFAULT_RPM)
        self.tpm = getattr(config, "tpm", DEFAULT_TPM)
        self.completion_context = getattr(config, "completion_context_window", DEFAULT_COMPLETION_CONTEXT_WINDOW)
        self.window_seconds = DEFAULT_WINDOW_SECONDS
        self.live_logs: bool = False
        self._provider_cache: dict[tuple[type, tuple[Any, ...]], Any] = {}
        self.input_items = _ResponseInputItems(self)
        self.input_tokens = _ResponseInputTokens(self)
        self.conversations = _Conversations(self)

    def capabilities(self) -> Any:
        """Return the resolved Responses capability contract for this target."""
        return resolve_response_capabilities(self.config)

    def _provider_class(self) -> type:
        return responses_provider_class(self.config, RESPONSES_PROVIDER_REGISTRY)

    @traced("responses.management", detail=lambda args, kwargs: {"verb": args[0] if args else None})
    def _management_call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        """Run a non-generating Responses verb through retry + error translation."""
        provider_cls = self._provider_class()
        call_args, provider_kwargs, _max_retries, live_logs = prepare_management_call(
            self.config, operation, args, kwargs,
        )
        user_key: str = getattr(self.config, "user_id", "global")
        if live_logs:
            logger.info("Responses %s request: %s", operation,
                        json.dumps(redact_sensitive({"provider": self.config.provider, "kwargs": provider_kwargs}),
                                   default=str))
        try:
            provider = self._instantiate_provider(self.config, provider_cls)
            return _retry_call(
                getattr(provider, operation), *call_args,
                max_retries=_max_retries,
                base_delay=getattr(self.config, "retry_base_delay", DEFAULT_RETRY_BASE_DELAY),
                max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
                **provider_kwargs,
            )
        except Exception as e:
            raise_response_error(self.config, e, user_key, operation)

    def _wrap_streaming(self, primary_factory, user_key: str, pre_count_tokens: int, *,
                        backup_factory=None, max_retries: int = 0, base_delay: float = 0.0):
        def on_complete(last_usage, limiter_scope, prompt_tokens):
            actual = _additional_stream_tokens(last_usage, prompt_tokens, pre_count_tokens)
            self.rate_limiter.update_token_usage(
                user_key,
                actual,
                self.window_seconds,
                request_type="completion",
                scope=limiter_scope,
            )

        return SyncStreamSession(
            primary_factory=primary_factory,
            backup_factory=backup_factory,
            is_output_visible=response_event_has_visible_output,
            terminal_status=response_event_terminal_status,
            on_complete=on_complete,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
        )

    @traced("responses")
    def create(self, input: Any = None, stream: bool = False, truncate: bool = False,  # noqa: A002
               **kwargs: Any) -> Any:
        """
        Create a model response via the Responses API.

        Args:
            input: A string prompt, or a list of Responses input items.
            stream (bool): Whether to stream server-sent events.
            truncate (bool): Whether to truncate a string input to fit the context window.
            **kwargs: Pass-through parameters (``instructions``, ``tools``, ``text``,
                ``previous_response_id``, ``conversation``, ``background``, …).
                ``live_logs`` (default False) toggles INFO logging; ``max_retries``
                overrides the configured retry count.

        Returns:
            Response, or a generator of ResponseStreamEvent when streaming.
        """
        self.live_logs = kwargs.pop("live_logs", False)
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
            validate_backup_for_registry(self.config, RESPONSES_PROVIDER_REGISTRY)
        if self.live_logs:
            logger.info(
                "Responses request: %s",
                json.dumps(redact_sensitive(
                    {"provider": self.config.provider, "input": "<hidden>", "stream": stream, "kwargs": kwargs}),
                    default=str))

        response_plan = plan_responses(self.config, self.rate_limiter, input, kwargs)
        plan, model_name = response_plan.plan, response_plan.model_name
        reserved_output_tokens, overhead_tokens = response_plan.reserved_output_tokens, response_plan.overhead_tokens
        prepared_input, total_tokens = prepare_response_input(
            input, self.rate_limiter, self.completion_context, truncate, model_name,
            reserved_output_tokens, overhead_tokens,
        )
        request_kwargs = dict(plan.provider_request)
        limiter_scope = plan.target.limiter_scope_key

        user_key: str = getattr(self.config, "user_id", "global")
        reserve_rate_limit(
            self.rate_limiter, user_key, self.rpm, self.tpm,
            self.window_seconds, total_tokens, scope=limiter_scope,
        )

        breaker = (get_breaker(plan.target.breaker_key, **getattr(self, "_breaker_config", {}))
                   if getattr(self, "_breaker_enabled", False) else None)
        if stream:
            def primary_stream_factory():
                try:
                    with provider_call(self.config.provider, model_name, breaker=breaker, record_success=False):
                        result = self._instantiate_provider(self.config, provider_cls).create(
                            input=prepared_input, stream=True, **request_kwargs
                        )
                    return StreamHandle(
                        result, breaker=breaker, limiter_scope=limiter_scope, prompt_tokens=total_tokens,
                    )
                except Exception as exc:
                    raise_response_error(self.config, exc, user_key, "create")

            backup_stream_factory = None
            if not side_effecting_request and not stateful_request and getattr(self.config, "backup", None):
                def backup_stream_factory():
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
                    backup_input, backup_tokens = prepare_target_response_input(
                        backup_config, input, self.rate_limiter, truncate, backup_request_kwargs_payload,
                    )
                    reserve_rate_limit(
                        self.rate_limiter, user_key, backup_config.rpm, backup_config.tpm,
                        self.window_seconds, backup_tokens, scope=backup_limiter_scope,
                    )
                    backup_provider = prepare_backup_provider(self.config, RESPONSES_PROVIDER_REGISTRY,
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
                            result = backup_provider.create(
                                input=backup_input, stream=True, **backup_request_kwargs_payload,
                            )
                        return StreamHandle(
                            result, breaker=backup_breaker, limiter_scope=backup_limiter_scope,
                            prompt_tokens=backup_tokens,
                        )
                    except Exception as exc:
                        raise_response_error(self.config, exc, user_key, "create")

            if self.live_logs:
                logger.info(
                    "Responses Response: %s",
                    json.dumps(redact_sensitive({
                        "provider": self.config.provider,
                        "response": "<Object(ResponseStreamEvent)>"}))
                )
            return self._wrap_streaming(
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
                result = _retry_call(
                    self._instantiate_provider(self.config, provider_cls).create,
                    max_retries=_max_retries, base_delay=_base_delay, max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
                    input=prepared_input, stream=stream, **request_kwargs
                )
        except (ResponsesError, APIConnectionError, RateLimitExceededError, TimeOutError) as primary_err:
            if (side_effecting_request or stateful_request or not getattr(self.config, "backup", None)
                    or not _is_retryable(primary_err)):
                raise
            logger.warning("Primary provider failed (%s), failing over to backup.", primary_err)
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
                backup_input, backup_tokens = prepare_target_response_input(
                    backup_config, input, self.rate_limiter, truncate, backup_request_kwargs_payload,
                )
                reserve_rate_limit(
                    self.rate_limiter, user_key, backup_config.rpm, backup_config.tpm,
                    self.window_seconds, backup_tokens, scope=backup_limiter_scope,
                )
                backup_provider = prepare_backup_provider(self.config, RESPONSES_PROVIDER_REGISTRY,
                                                          self._instantiate_provider)
                backup_breaker = (
                    get_breaker(backup_plan.target.breaker_key, **getattr(self, "_breaker_config", {}))
                    if getattr(self, "_breaker_enabled", False) else None
                )
                with provider_call(backup_config.provider, backup_plan.target.resolved_model,
                                   breaker=backup_breaker):
                    result = _retry_call(
                        backup_provider.create,
                        max_retries=_max_retries, base_delay=_base_delay, max_delay=DEFAULT_RETRY_MAX_DELAY, deadline=self.config.retry_deadline,
                        input=backup_input, stream=stream, **backup_request_kwargs_payload
                    )
                active_limiter_scope = backup_limiter_scope
                total_tokens = backup_tokens
            except Exception as backup_err:
                logger.error("Backup provider also failed: %s", backup_err)
                raise primary_err from backup_err
        except Exception as e:
            raise_response_error(self.config, e, user_key, "create")

        if self.live_logs:
            logger.info(
                "Responses Response: %s",
                json.dumps(redact_sensitive({
                    "provider": self.config.provider,
                    "response": "<Object(ResponseStreamEvent)>" if stream else "<Object(Response)>"}))
            )

        provider_tokens = getattr(getattr(result, "usage", None), "total_tokens", 0)
        self.rate_limiter.update_token_usage(
            user_key,
            max(0, provider_tokens - total_tokens),
            self.window_seconds,
            request_type="completion",
            scope=active_limiter_scope,
        )
        return result

    def stream(self, input: Any = None, truncate: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        """Streaming shorthand for ``create(..., stream=True)``."""
        return self.create(input, stream=True, truncate=truncate, **kwargs)

    def create_background(self, input: Any = None, truncate: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        """Create a stored background response suitable for retrieval and cancellation."""
        kwargs["background"] = True
        kwargs.setdefault("store", True)
        return self.create(input, stream=False, truncate=truncate, **kwargs)

    def submit_mcp_approval(self, previous_response_id: str, approval: Any, **kwargs: Any) -> Any:
        """Continue a response after explicit application approval or rejection of an MCP call."""
        item = approval.to_input_item() if hasattr(approval, "to_input_item") else dict(approval)
        return self.create([item], previous_response_id=previous_response_id, **kwargs)

    def wait(self, response_id: str, *, timeout: float = 600.0, poll_interval: float = 1.0,
             max_poll_interval: float = 10.0, cancel_on_timeout: bool = False,
             on_poll: Any = None, **kwargs: Any) -> Any:
        """Poll a background response until it reaches a terminal state."""
        if timeout <= 0 or poll_interval <= 0 or max_poll_interval <= 0:
            raise ValueError("timeout, poll_interval, and max_poll_interval must be positive")
        deadline = time.monotonic() + timeout
        delay = poll_interval
        attempts = 0
        terminal = {"completed", "failed", "incomplete", "cancelled"}
        while True:
            response = self.retrieve(response_id, **kwargs)
            attempts += 1
            if callable(on_poll):
                on_poll(response, attempts)
            if response.status in terminal:
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if cancel_on_timeout:
                    try:
                        self.cancel(response_id)
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to cancel timed-out background response", exc_info=True)
                raise TimeOutError(int(timeout))
            time.sleep(min(remaining, delay))
            delay = min(max_poll_interval, max(poll_interval, random.uniform(0, delay * 2)))

    def parse(self, input: Any = None, text_format: Any = None, truncate: bool = False,  # noqa: A002
              **kwargs: Any) -> Any:
        """
        Structured-output helper mirroring ``client.responses.parse``.

        Pass a Pydantic model *class* as ``text_format``; the deserialized
        instance is returned on ``response.output_parsed``. Always non-streaming.
        """
        return self.create(input, stream=False, truncate=truncate, text_format=text_format, **kwargs)

    def retrieve(self, response_id: str, **kwargs: Any) -> Any:
        """Retrieve a previously created response by id."""
        return self._management_call("retrieve", response_id, **kwargs)

    def delete(self, response_id: str, **kwargs: Any) -> Any:
        """Delete a stored response by id."""
        return self._management_call("delete", response_id, **kwargs)

    def cancel(self, response_id: str, **kwargs: Any) -> Any:
        """Cancel an in-progress background response."""
        return self._management_call("cancel", response_id, **kwargs)

    def compact(self, **kwargs: Any) -> Any:
        """Compact a response's conversation history into a shorter one."""
        return self._management_call("compact", **kwargs)


class _Chat:
    """Wrapper for chat completions, used by LLMSwitch."""

    def __init__(self, config: Any, rate_limiter: "RedisRateLimiter") -> None:
        self.completions = _ChatCompletions(config, rate_limiter)

class LLMSwitch:
    """
    Unified entry point for Chat + Embeddings (mimics OpenAI client).
    Handles configuration, rate limiting, and provider selection.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the LLMSwitch with provider configuration and rate limiter.

        Args:
            provider (str): Name of the LLM provider (e.g., 'azure_openai').
            api_key (str): API key of the model.
            endpoint (str): API endpoint of the model.
            deployment_name (str): Deployment name of the model.
            api_version (str): API version of the model.
            model (str): Model name.
            mcp_policy (dict, optional): MCP policy configuration.
            redis_url (str, optional): Redis URL for rate limiting.
            user_id (str, optional): User identifier of the model.
            rpm / tpm / embedding_rpm / embedding_tpm (int, optional): Rate limits.
            completion_context_window (int, optional): Context window size for completion models.
        """
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
        # Set up Redis-based rate limiter
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
        # Set up chat and embedding interfaces
        self.chat = _Chat(self.config, self.rate_limiter)
        self.embeddings = _Embeddings(self.config, self.rate_limiter)
        self.responses = _Responses(self.config, self.rate_limiter)
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
            f"LLMSwitch(provider={self.config.provider!r}, "
            f"model={self.config.model!r}, "
            f"user_id={self.config.user_id!r})"
        )

    def __enter__(self) -> "LLMSwitch":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close any resources held by the switch."""
        self.chat.completions.close()
        self.embeddings.close()
        self.responses.close()
        self.rate_limiter.close()
