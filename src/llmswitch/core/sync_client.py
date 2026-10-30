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
from typing import Any, Unpack

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
from ..models import CreateEmbeddingResponse
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
from .execution import build_execution_plan, reserve_rate_limit
from .failover import (
    build_backup_execution_plan,
    prepare_backup_provider,
    validate_backup_config,
    validate_backup_for_registry,
)
from .interfaces import ChatAPI, EmbeddingsAPI, ResponsesAPI
from .options import LLMSwitchOptions
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
from .retry import _fails_over, _retry_call
from .streaming import (
    StreamHandle,
    SyncStreamSession,
    chat_chunk_has_visible_output,
    response_event_has_visible_output,
    response_event_terminal_status,
    stream_completion_tokens,
)

logger = logging.getLogger(__name__)


class _ChatCompletions(ProviderCache):
    """Chat completions: ``client.chat.completions.create`` and ``client.chat.completions.parse``.

    Each call applies message truncation, rate limiting, retries, the circuit breaker and backup failover, and
    counts its tokens against the chat limits (``rpm`` and ``tpm``).
    """

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
        def on_complete(last_usage, limiter_scope, prompt_tokens, output_text):
            actual = stream_completion_tokens(last_usage, prompt_tokens, pre_count_tokens,
                                                  output_text, getattr(self.config, "model", None))
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
        """Create a chat completion.

        Messages are truncated first when ``truncate`` is set, then the call waits for the rate limit window if needed.

        Args:
            messages (list[dict]): The conversation, as dicts with ``role`` and ``content``. ``content`` can be a
                string or a list of content parts for images (multimodal input).
            stream (bool): When True, return an iterator of chunks instead of one reply. Defaults to False.
            truncate (bool): When True, drop the oldest messages so the input fits ``completion_context_window``.
                Defaults to False.
            **kwargs: Passed to the provider unchanged. Common ones are ``model`` (a per-call model, unless a
                ``deployment_name`` is configured), ``max_tokens`` or ``max_completion_tokens``, ``temperature``,
                ``tools``, ``tool_choice``, ``response_format`` and ``stop``. The library also reads ``live_logs``
                (default False, logs requests at INFO level) and ``max_retries`` (overrides the configured retries).

        Returns:
            ChatCompletion when ``stream`` is False. An iterator of ChatCompletionChunk when ``stream`` is True.

        Raises:
            RateLimitExceededError: The rate limit window stayed full beyond ``rate_limit_max_wait``, or
                ``on_rate_limit`` is ``"raise"``.
            AuthenticationError: The provider rejected the credentials.
            ChatCompletionError: The provider failed. ``status_code`` tells whether it was a client or server error.
            ContentFilterError: The provider blocked the prompt or the reply.
            LLMSwitchError: Any other failure. Every error raised by this library is a subclass of it.
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
                    with provider_call(self.config.provider, model_name, breaker=breaker, record_success=False) as probe:
                        result = self._instantiate_provider(self.config, provider_cls).create(
                            messages=truncated_messages, stream=True, **request_kwargs
                        )
                    return StreamHandle(
                        result, breaker=breaker, limiter_scope=limiter_scope, prompt_tokens=total_tokens, probe=probe,
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
                        ) as probe:
                            result = backup_provider.create(
                                messages=backup_messages, stream=True, **backup_request_kwargs_payload,
                            )
                        return StreamHandle(
                            result, breaker=backup_breaker, limiter_scope=backup_limiter_scope,
                            prompt_tokens=backup_tokens, probe=probe,
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
            if not getattr(self.config, "backup", None) or not _fails_over(primary_err):
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
        """Structured output: return the reply parsed into a Pydantic model.

        Pass a Pydantic model class as ``response_format``. The deserialized instance is returned on
        ``choice.message.parsed``, for every provider. The call is never streamed.

        Args:
            messages (list[dict]): The conversation, as in ``create``.
            response_format (type): A Pydantic model class that describes the reply.
            truncate (bool): Truncate the messages to fit the context window. Defaults to False.
            **kwargs: Passed to the provider, as in ``create``.

        Returns:
            ChatCompletion. Its ``choices[0].message.parsed`` holds an instance of ``response_format``.

        Raises:
            The same errors as ``create``.
        """
        return self.create(messages, stream=False, truncate=truncate,
                           response_format=response_format, **kwargs)


class _Embeddings(ProviderCache):
    """Embeddings: ``client.embeddings.create``.

    Calls count against the embedding limits (``embedding_rpm`` and ``embedding_tpm``), not the chat limits.
    """

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
        """Create embedding vectors for one text or a list of texts.

        Args:
            input (str or list[str]): The text or texts to embed.
            batching (bool): Split a long list into batches of ``batchsize`` texts, sent one batch at a time.
                Defaults to False.
            batchsize (int): Texts per batch when ``batching`` is set. Defaults to 32.
            **kwargs: Passed to the provider unchanged, for example ``model`` and ``dimensions``. The library also
                reads ``restore_order`` (default True), which returns the vectors in the order of ``input`` when
                batches finish out of order.

        Returns:
            CreateEmbeddingResponse. Its ``data`` list holds one Embedding per input text, each with ``index`` and
            ``embedding`` (a list of floats).

        Raises:
            RateLimitExceededError: The embedding rate limit stayed full beyond ``rate_limit_max_wait``.
            EmbeddingGenerationError: The provider failed.
            LLMSwitchError: Any other failure.
        """
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
            acc = EmbeddingAccumulator(target_config.provider, target_plan.target.resolved_model, restore_order)
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
                    acc.add(result, indices)
                except Exception as exc:
                    raise_embedding_error(target_config, exc)

            self.rate_limiter.update_token_usage(
                user_key, max(0, acc.total_tokens - sum(token_counts)), self.window_seconds,
                request_type="embedding", scope=target_plan.target.limiter_scope_key,
            )
            return acc.build()

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
    """The input items of a response: ``client.responses.input_items``."""

    def __init__(self, responses: "_Responses") -> None:
        self._responses = responses

    def list(self, response_id: str, **kwargs: Any) -> Any:
        """List the input items of a stored response, one page at a time.

        Args:
            response_id (str): The id of the response.
            **kwargs: Passed to the provider, for example paging options.

        Returns:
            ResponseItemList: One page of items.
        """
        return self._responses._management_call("list_input_items", response_id, **kwargs)


class _ResponseInputTokens:
    """Token counting for Responses requests: ``client.responses.input_tokens``."""

    def __init__(self, responses: "_Responses") -> None:
        self._responses = responses

    def count(self, **kwargs: Any) -> Any:
        """Count the input tokens a Responses request would use, without running it.

        Args:
            **kwargs: The request to count, as for ``create`` (``input``, ``model``, ``instructions`` and so on).

        Returns:
            InputTokenCountResponse, with the number of input tokens.
        """
        return self._responses._management_call("count_input_tokens", **kwargs)


class _Conversations:
    """Conversations stored with the provider: ``client.responses.conversations``."""

    def __init__(self, responses: "_Responses") -> None:
        self._responses = responses

    def create(self, items: Any = None, **kwargs: Any) -> Any:
        """Create a conversation, optionally seeded with items.

        Args:
            items (list, optional): Items to start the conversation with.
            **kwargs: Passed to the provider, for example metadata.

        Returns:
            Conversation.
        """
        return self._responses._management_call("create_conversation", items, **kwargs)

    def retrieve(self, conversation_id: str, **kwargs: Any) -> Any:
        """Retrieve a conversation by its id.

        Args:
            conversation_id (str): The id of the conversation.
            **kwargs: Passed to the provider.

        Returns:
            Conversation.
        """
        return self._responses._management_call("retrieve_conversation", conversation_id, **kwargs)

    def update(self, conversation_id: str, **kwargs: Any) -> Any:
        """Change a conversation, for example its metadata.

        Args:
            conversation_id (str): The id of the conversation.
            **kwargs: The fields to change, as the provider accepts them.

        Returns:
            Conversation, after the change.
        """
        return self._responses._management_call("update_conversation", conversation_id, **kwargs)

    def delete(self, conversation_id: str, **kwargs: Any) -> Any:
        """Delete a conversation by its id.

        Args:
            conversation_id (str): The id of the conversation to delete.
            **kwargs: Passed to the provider.

        Returns:
            ConversationDeleted: Confirms the deletion.
        """
        return self._responses._management_call("delete_conversation", conversation_id, **kwargs)

    def list_items(self, conversation_id: str, **kwargs: Any) -> Any:
        """List the items of a conversation, one page at a time.

        Args:
            conversation_id (str): The id of the conversation.
            **kwargs: Passed to the provider, for example paging options.

        Returns:
            ResponseItemList: One page of items.
        """
        return self._responses._management_call("list_conversation_items", conversation_id, **kwargs)

    def retrieve_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        """Retrieve one item of a conversation.

        Args:
            conversation_id (str): The id of the conversation.
            item_id (str): The id of the item.
            **kwargs: Passed to the provider.

        Returns:
            ResponseItem.
        """
        return self._responses._management_call("retrieve_conversation_item", conversation_id, item_id, **kwargs)

    def delete_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        """Delete one item from a conversation.

        Args:
            conversation_id (str): The id of the conversation.
            item_id (str): The id of the item to delete.
            **kwargs: Passed to the provider.

        Returns:
            The provider's confirmation of the deletion.
        """
        return self._responses._management_call("delete_conversation_item", conversation_id, item_id, **kwargs)


class _Responses(ProviderCache):
    """The Responses API: ``client.responses``.

    ``create`` runs the same pipeline as chat completions: truncation, rate limiting, retries, the circuit breaker,
    backup failover and token accounting. It shares the chat limits, because both reach the same deployment quota.
    The management calls (``retrieve``, ``delete``, ``cancel``, ``compact``, ``input_items.list``,
    ``input_tokens.count`` and ``conversations``) are retried and their errors translated, but they are not
    metered.

    Native for ``openai`` and ``azure_openai``. For ``anthropic`` and ``vertexai`` the calls are emulated on top of
    chat completions, and server-side state or hosted tools raise ``UnsupportedCapabilityError``. ``azure_foundry``
    has no Responses API.
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
        """Return the Responses capabilities that apply to this client.

        Returns:
            ResponseCapabilities: For each Responses feature, whether this provider and model support it, and why not
            when it does not.
        """
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
        def on_complete(last_usage, limiter_scope, prompt_tokens, output_text):
            actual = stream_completion_tokens(last_usage, prompt_tokens, pre_count_tokens,
                                                  output_text, getattr(self.config, "model", None))
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
        """Create a model response.

        Args:
            input (str or list): A string prompt, or a list of Responses input items, such as messages, function call
                outputs or images.
            stream (bool): When True, return an iterator of server-sent events instead of one response. Defaults to
                False.
            truncate (bool): When True, truncate a string input so it fits the context window. Defaults to False.
            **kwargs: Passed to the provider. Common ones are ``model``, ``instructions``, ``tools``, ``text``,
                ``previous_response_id``, ``conversation``, ``background``, ``max_output_tokens`` and ``store``. The
                library also reads ``live_logs`` and ``max_retries``, as in chat completions.

        Returns:
            Response when ``stream`` is False. An iterator of ResponseStreamEvent when ``stream`` is True.

        Raises:
            UnsupportedCapabilityError: The provider or model does not support a requested feature.
            InvalidResponseRequestError: The request has an option that the Responses API does not accept.
            ResponsesError: The provider failed.
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
                    with provider_call(self.config.provider, model_name, breaker=breaker, record_success=False) as probe:
                        result = self._instantiate_provider(self.config, provider_cls).create(
                            input=prepared_input, stream=True, **request_kwargs
                        )
                    return StreamHandle(
                        result, breaker=breaker, limiter_scope=limiter_scope, prompt_tokens=total_tokens, probe=probe,
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
                        ) as probe:
                            result = backup_provider.create(
                                input=backup_input, stream=True, **backup_request_kwargs_payload,
                            )
                        return StreamHandle(
                            result, breaker=backup_breaker, limiter_scope=backup_limiter_scope,
                            prompt_tokens=backup_tokens, probe=probe,
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
                    or not _fails_over(primary_err)):
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
        """Stream a model response. This is ``create`` with ``stream=True``.

        Args:
            input (str or list): The prompt or input items, as in ``create``.
            truncate (bool): Truncate a string input to fit the context window. Defaults to False.
            **kwargs: Passed to the provider, as in ``create``.

        Returns:
            An iterator of ResponseStreamEvent.
        """
        return self.create(input, stream=True, truncate=truncate, **kwargs)

    def create_background(self, input: Any = None, truncate: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        """Create a stored background response, which can be polled, retrieved or cancelled later.

        Args:
            input (str or list): The prompt or input items, as in ``create``.
            truncate (bool): Truncate a string input to fit the context window. Defaults to False.
            **kwargs: Passed to the provider, as in ``create``.

        Returns:
            Response, usually still in progress. Use ``wait`` to block until it finishes, or ``retrieve`` to check it.
        """
        kwargs["background"] = True
        kwargs.setdefault("store", True)
        return self.create(input, stream=False, truncate=truncate, **kwargs)

    def submit_mcp_approval(self, previous_response_id: str, approval: Any, **kwargs: Any) -> Any:
        """Continue a response after the application approves or rejects an MCP tool call.

        Args:
            previous_response_id (str): The id of the response that asked for approval.
            approval (McpApprovalDecision): The approval or rejection for the pending MCP call.
            **kwargs: Passed to the provider, as in ``create``.

        Returns:
            Response, the continuation of the earlier one.
        """
        item = approval.to_input_item() if hasattr(approval, "to_input_item") else dict(approval)
        return self.create([item], previous_response_id=previous_response_id, **kwargs)

    def wait(self, response_id: str, *, timeout: float = 600.0, poll_interval: float = 1.0,
             max_poll_interval: float = 10.0, cancel_on_timeout: bool = False,
             on_poll: Any = None, **kwargs: Any) -> Any:
        """Poll a background response until it reaches a final state.

        The final states are ``completed``, ``failed``, ``incomplete`` and ``cancelled``. The wait between polls starts
            at
        ``poll_interval`` and grows, up to ``max_poll_interval``.

        Args:
            response_id (str): The id of the response to wait for.
            timeout (float): The most seconds to wait. Defaults to 600.
            poll_interval (float): Seconds between the first polls. Defaults to 1.0.
            max_poll_interval (float): The longest wait between polls. Defaults to 10.0.
            cancel_on_timeout (bool): When True, cancel the response if the timeout passes. Defaults to False.
            on_poll (callable, optional): Called after each poll as ``on_poll(response, attempts)``, where
        ``attempts`` counts the polls so far.
            **kwargs: Passed to the retrieve call on each poll.

        Returns:
            Response in a final state.

        Raises:
            ValueError: A timeout or an interval is zero or negative.
            TimeOutError: The timeout passed before the response reached a final state.
        """
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
        """Structured output for the Responses API.

        Pass a Pydantic model class as ``text_format``. The deserialized instance is returned on
        ``response.output_parsed``. The call is never streamed.

        Args:
            input (str or list): The prompt or input items, as in ``create``.
            text_format (type): A Pydantic model class that describes the output.
            truncate (bool): Truncate a string input to fit the context window. Defaults to False.
            **kwargs: Passed to the provider, as in ``create``.

        Returns:
            Response. Its ``output_parsed`` holds an instance of ``text_format``.
        """
        return self.create(input, stream=False, truncate=truncate, text_format=text_format, **kwargs)

    def retrieve(self, response_id: str, **kwargs: Any) -> Any:
        """Retrieve a stored response by its id.

        Args:
            response_id (str): The id of the response.
            **kwargs: Passed to the provider.

        Returns:
            Response.
        """
        return self._management_call("retrieve", response_id, **kwargs)

    def delete(self, response_id: str, **kwargs: Any) -> Any:
        """Delete a stored response by its id.

        Args:
            response_id (str): The id of the response to delete.
            **kwargs: Passed to the provider.

        Returns:
            ResponseDeleted: Confirms the deletion.
        """
        return self._management_call("delete", response_id, **kwargs)

    def cancel(self, response_id: str, **kwargs: Any) -> Any:
        """Cancel a background response that is still in progress.

        Args:
            response_id (str): The id of the response to cancel.
            **kwargs: Passed to the provider.

        Returns:
            Response, with its status after the cancellation.
        """
        return self._management_call("cancel", response_id, **kwargs)

    def compact(self, **kwargs: Any) -> Any:
        """Compact a conversation's history into a shorter form that keeps its meaning.

        Args:
            **kwargs: Passed to the provider, for example the model and the history to compact.

        Returns:
            CompactedResponse.
        """
        return self._management_call("compact", **kwargs)


class _Chat:
    """Chat namespace of ``LLMSwitch``. Use ``chat.completions``.

    Attributes:
        completions: The chat completions resource, with ``create`` and ``parse``.
    """

    def __init__(self, config: Any, rate_limiter: "RedisRateLimiter") -> None:
        self.completions = _ChatCompletions(config, rate_limiter)

class LLMSwitch:
    """Synchronous client for chat completions, embeddings and the Responses API.

    One interface covers OpenAI, Azure OpenAI, Azure AI Foundry, Anthropic and Vertex AI. Switching provider
    changes the constructor arguments, not the calls. Every call goes through the same pipeline: message
    truncation, rate limiting per ``user_id``, retries, the circuit breaker and backup failover.

    Attributes:
        chat: Chat completions. Use ``chat.completions.create`` and ``chat.completions.parse``.
        embeddings: Embeddings. Use ``embeddings.create``.
        responses: The Responses API. Use ``responses.create``, ``responses.stream``, ``responses.parse`` and the
            background, retrieval and conversation helpers.

    Example:
        with LLMSwitch(provider="openai", api_key="sk-...", model="gpt-4o-mini") as client:
            reply = client.chat.completions.create(messages=[{"role": "user", "content": "Hello"}])
            print(reply.choices[0].message.content)

    Use it as a context manager, or call ``close()`` when you are done, to release the provider connections.
    """

    # Typed for editors and mypy only. Each is set in __init__; the annotations add no runtime attribute.
    chat: ChatAPI
    embeddings: EmbeddingsAPI
    responses: ResponsesAPI

    def __init__(self, **kwargs: Unpack[LLMSwitchOptions]) -> None:
        """Create a client for one provider.

        Options that no provider defines are rejected. Options that another provider uses are accepted and ignored.

        Args:
            provider (str): ``openai``, ``azure_openai``, ``azure_foundry``, ``anthropic`` or ``vertexai``.
            model (str, optional): Model name. A ``deployment_name`` takes precedence when it is set.
            deployment_name (str, optional): Azure deployment name. Always used as the model when set.
            api_key (str, optional): Provider API key. Pass it from your own configuration; the library does not read
                environment variables or ``.env`` files.
            endpoint (str, optional): Custom endpoint or gateway URL.
            api_version (str, optional): API version for the Azure providers.
            responses_api_version (str, optional): API version used for Responses calls only.
            location (str, optional): Vertex AI location, for example ``us-central1``.
            credentials_path (str, optional): Path to a Vertex AI service account JSON file.
            scopes (list[str], optional): OAuth scopes for Vertex AI.
            organization (str, optional): OpenAI organization ID.
            project (str, optional): OpenAI project ID.
            timeout (float, optional): Request timeout in seconds. Defaults to the SDK's timeout.
            enable_prompt_caching (bool): Anthropic only. Adds cache markers to system messages and tool definitions.
                Defaults to True.
            client_options (dict, optional): Extra keyword arguments for the native SDK client, for example
                ``{"max_retries": 3}``. Unified options such as ``api_key`` or ``timeout`` cannot be set here.
            user_id (str): The key that rate limits are counted against. Defaults to "global".
            rpm (int): Chat requests per minute. Defaults to 10000.
            tpm (int): Chat tokens per minute. Defaults to 1000000.
            embedding_rpm (int): Embedding requests per minute. Defaults to 10000.
            embedding_tpm (int): Embedding tokens per minute. Defaults to 1000000.
            on_rate_limit (str): ``"wait"`` (default) to wait for the window to reopen, or ``"raise"`` to fail at once.
            rate_limit_max_wait (float, optional): The longest wait, in seconds, before a rate limit error is raised.
            redis_url (str, optional): Redis URL for limits shared across processes. Needs the ``redis`` extra.
            redis_connect_timeout (float): Seconds to wait for Redis. Defaults to 1.0.
            rate_limit_failure_policy (str): What to do when Redis is unavailable: ``"memory_fallback"`` (default),
                ``"fail_open"`` or ``"fail_closed"``.
            max_retries (int): Retries for transient errors. Defaults to 3.
            retry_base_delay (float): Seconds before the first retry. The delay doubles on each retry. Defaults to 1.0.
            retry_deadline (float, optional): The most total seconds one call may keep retrying.
            circuit_breaker (bool or dict): ``True`` for the defaults, or a dict with ``failure_threshold`` (default 5)
                and ``reset_timeout`` (default 30 seconds).
            backup (dict, optional): Settings for a backup provider, in the same form as the primary, used on failover.
            completion_context_window (int): Token budget for input and output together. Defaults to 123000.
            context_output_reserve_tokens (int): Output tokens kept free in every request. Defaults to 0.
            context_overhead_tokens (int): Extra protocol tokens to count. Defaults to 0.
            token_counter (callable, optional): A function ``fn(text, model_name) -> int`` that counts tokens.
            mcp_policy (dict, optional): Allow-list for remote MCP servers and connectors.
            responses_capabilities (dict, optional): Turns Responses features on or off, for example
                ``{"background": True}``.
            on_event (callable, optional): Called with one dict per event: requests, retries, failovers and usage.
            share_clients (bool): Reuse SDK clients across clients with the same configuration. Defaults to False.

        Raises:
            ConfigurationError: An option is not defined by any provider. Reserved unified options set in
        ``client_options`` are rejected too.
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
        # Set up chat and embedding interfaces. The implementations are kept under private names, because the
        # public attributes are typed as the public interfaces, which hide their helper methods.
        self._chat_impl = _Chat(self.config, self.rate_limiter)
        self._embeddings_impl = _Embeddings(self.config, self.rate_limiter)
        self._responses_impl = _Responses(self.config, self.rate_limiter)
        self.chat = self._chat_impl
        self.embeddings = self._embeddings_impl
        self.responses = self._responses_impl
        self._chat_impl.completions._on_event = on_event
        self._embeddings_impl._on_event = on_event
        self._responses_impl._on_event = on_event
        if share_clients:
            self._chat_impl.completions._share_clients = True
            self._embeddings_impl._share_clients = True
            self._responses_impl._share_clients = True
        if circuit_breaker:
            self._chat_impl.completions._breaker_enabled = True
            self._responses_impl._breaker_enabled = True
            if isinstance(circuit_breaker, dict):
                breaker_config: dict[str, Any] = {
                    k: circuit_breaker[k] for k in ("failure_threshold", "reset_timeout")
                    if k in circuit_breaker
                }
                self._chat_impl.completions._breaker_config = breaker_config
                self._responses_impl._breaker_config = dict(breaker_config)

    def __repr__(self) -> str:
        return (
            f"LLMSwitch(provider={self.config.provider!r}, "
            f"model={self.config.model!r}, "
            f"user_id={self.config.user_id!r})"
        )

    def __enter__(self) -> "LLMSwitch":
        """Return the client, so that it can be used in a ``with`` block.

        Returns:
            LLMSwitch: This client.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the client when a ``with`` block ends.

        Exceptions raised inside the block are not suppressed.

        Args:
            exc_type: The exception class, or None when the block ended normally.
            exc_val: The exception instance, or None.
            exc_tb: The traceback, or None.

        Returns:
            None
        """
        self.close()

    def close(self) -> None:
        """Close the provider connections held by this client.

        Returns:
            None
        """
        self._chat_impl.completions.close()
        self._embeddings_impl.close()
        self._responses_impl.close()
        self.rate_limiter.close()
