"""
azureopenai_provider.py

Azure OpenAI providers for chat completions and embeddings. The chat provider
supports streaming and non-streaming completions and converts Azure OpenAI
responses into the unified LLM Switch format. Response mapping shared with the
native OpenAI provider lives in :mod:`llmswitch.providers._shared`; Azure-only
fields (content filtering, prompt filter results, message ``context``) are
handled here.

Classes:
    AzureOpenAIChatProvider: Chat provider for Azure OpenAI models.
    AzureOpenAIEmbeddingProvider: Embedding provider for Azure OpenAI models.
    AsyncAzureOpenAIChatProvider: Async chat provider for Azure OpenAI models.
"""
import logging
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any

import openai
from openai import AsyncAzureOpenAI, AzureOpenAI

from ..base import BaseChatProvider, BaseEmbeddingProvider
from ..errors import (
    APIConnectionError,
    AuthenticationError,
    ChatCompletionError,
    EmbeddingGenerationError,
    InvalidResponseError,
    RateLimitExceededError,
    TimeOutError,
    detect_content_filter_error,
)
from ..models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkDelta,
    CreateEmbeddingResponse,
    Embedding,
    Usage,
)
from ..structured import (
    is_pydantic_model_class,
    provider_extensions_for,
    provider_extensions_from_raw,
    pydantic_to_response_format,
    to_jsonable,
    unmapped_provider_fields,
)
from ._responses_base import AsyncResponsesProviderMixin, SyncResponsesProviderMixin
from ._shared import (
    audio_from,
    azure_token_limit_name,
    build_openai_usage,
    close_async_stream,
    close_stream,
    configured_timeout_seconds,
    get_retry_after_seconds,
    map_delta_logprobs,
    map_function_call,
    map_function_call_delta,
    map_message_logprobs,
    map_tool_call_deltas,
    map_tool_calls,
    merge_client_options,
    normalize_token_limit,
)

# Name resolves to "llmswitch.providers.azureopenai_provider"
logger = logging.getLogger(__name__)


def _raise_if_content_filtered(exc: Exception) -> None:
    content_filter_error = detect_content_filter_error(exc, provider="azure_openai")
    if content_filter_error:
        logger.error("Azure OpenAI content filter error | Exception: %s", exc)
        raise content_filter_error from exc


def _sanitize_completion_kwargs(kwargs: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    """Drop only invalid top-level kwargs while preserving schema-level strictness."""
    sanitized = dict(kwargs)

    response_format = sanitized.get("response_format")
    # A Pydantic model *class* (response_format=MyModel) is consumed directly by
    # .parse(); leave it untouched rather than calling its unbound model_dump.
    if is_pydantic_model_class(response_format):
        sanitized.pop("strict", None)
        return sanitized
    model_dump = getattr(response_format, "model_dump", None)
    if callable(model_dump):
        response_format = model_dump(by_alias=True, exclude_none=True)

    if isinstance(response_format, dict):
        sanitized["response_format"] = dict(response_format)

    sanitized.pop("strict", None)
    return sanitized


class AzureOpenAIChatProvider(BaseChatProvider):
    """
    Azure OpenAI ChatCompletion Provider.
    Adapts Azure-hosted OpenAI models into the unified LLM Switch response format.
    Handles both streaming and non-streaming chat completions.
    """

    def __init__(self, **kwargs):
        """
        Initialize the AzureOpenAIChatProvider with API credentials and deployment info.
        Args:
            **kwargs: Should include 'api_key', 'endpoint', 'api_version', 'deployment_name', and 'model'.
        """
        self.api_key = kwargs.get("api_key")
        self.endpoint = kwargs.get("endpoint")
        self.api_version = kwargs.get("api_version")
        self._token_limit = azure_token_limit_name(self.api_version)
        self.deployment_name = kwargs.get("deployment_name")
        self.model = kwargs.get("model")
        self.timeout = kwargs.get("timeout")
        self.client = AzureOpenAI(**merge_client_options({
            "api_key": self.api_key,
            "api_version": self.api_version,
            "azure_endpoint": self.endpoint,
            "azure_deployment": self.deployment_name,
            "max_retries": 0,
            "timeout": kwargs.get("timeout"),
        }, kwargs.get("client_options")))

    def create(self, messages: list, stream: bool = False, **kwargs):
        """Dispatch to streaming or non-streaming chat completion."""
        kwargs = _sanitize_completion_kwargs(kwargs, self.model)
        normalize_token_limit(kwargs, self._token_limit)
        if stream:
            # Streaming has no native .parse() helper — request the schema instead.
            if is_pydantic_model_class(kwargs.get("response_format")):
                kwargs["response_format"] = pydantic_to_response_format(kwargs["response_format"])
            return self._streaming_create(messages, **kwargs)
        return self._non_streaming_create(messages, **kwargs)

    def _non_streaming_create(self, messages: list, **kwargs) -> ChatCompletion:
        """
        Handle non-streaming chat completion.

        When ``response_format`` is a Pydantic model class, routes to the SDK's
        ``beta.chat.completions.parse`` so ``message.parsed`` is populated.
        """
        try:
            if is_pydantic_model_class(kwargs.get("response_format")):
                response = self.client.beta.chat.completions.parse(
                    model=kwargs.pop("model", self.model),
                    messages=messages,
                    **kwargs,
                )
            else:
                response = self.client.chat.completions.create(
                    model=kwargs.pop("model", self.model),
                    messages=messages,
                    stream=False,
                    **kwargs,
                )

            choices = [
                Choice(
                    index=ch.index,
                    message=ChatCompletionMessage(
                        role=ch.message.role,
                        content=ch.message.content,
                        tool_calls=map_tool_calls(ch.message),
                        function_call=map_function_call(ch.message.function_call),
                        refusal=ch.message.refusal,
                        audio=audio_from(getattr(ch.message, "audio", None)),
                        annotations=getattr(ch.message, "annotations", None),
                        parsed=getattr(ch.message, "parsed", None),
                        context=to_jsonable(getattr(ch.message, "context", None)),
                        extensions=provider_extensions_for("azure_openai", unmapped_provider_fields(ch.message, {
                            "role", "content", "tool_calls", "function_call", "refusal", "audio",
                            "annotations", "parsed", "context",
                        })),
                    ),
                    finish_reason=ch.finish_reason,
                    logprobs=map_message_logprobs(ch.logprobs),
                    content_filter_results=getattr(ch, "content_filter_results", None),
                    extensions=provider_extensions_for("azure_openai", unmapped_provider_fields(ch, {
                        "index", "message", "finish_reason", "logprobs", "content_filter_results",
                    })),
                )
                for ch in response.choices
            ]

            extensions = provider_extensions_from_raw(
                "azure_openai",
                response,
                {
                    "id", "object", "created", "model", "choices", "usage", "system_fingerprint",
                    "service_tier", "store", "metadata", "prompt_filter_results", "moderation",
                },
            )

            return ChatCompletion(
                id=response.id,
                object=response.object,
                created=response.created,
                model=response.model,
                choices=choices,
                usage=build_openai_usage(response.usage),
                system_fingerprint=response.system_fingerprint,
                service_tier=response.service_tier,
                store=getattr(response, "store", None),
                metadata=getattr(response, "metadata", None),
                prompt_filter_results=getattr(response, "prompt_filter_results", None),
                moderation=getattr(response, "moderation", None),
                extensions=extensions,
            )
        except openai.RateLimitError as e:
            retry_after = get_retry_after_seconds(e)
            logger.error("Azure OpenAI rate limit error | Exception: %s", e)
            raise RateLimitExceededError(retry_after) from e
        except openai.AuthenticationError as e:
            logger.error("Azure OpenAI authentication error | Exception: %s", e)
            raise AuthenticationError(provider="azure_openai") from e
        except openai.APITimeoutError as e:
            logger.error("Azure OpenAI timeout error | Exception: %s", e)
            raise TimeOutError(configured_timeout_seconds(self.timeout)) from e
        except openai.BadRequestError as e:
            _raise_if_content_filtered(e)
            logger.error("Azure OpenAI bad request error | Exception: %s", e)
            raise ChatCompletionError(str(e), status_code=400) from e
        except ConnectionError as e:
            logger.error("Connection error in chat completion: %s | Exception: %s", self.endpoint, e)
            raise APIConnectionError(self.endpoint) from e
        except PermissionError as e:
            logger.error("Authentication error in chat completion | Exception: %s", e)
            raise AuthenticationError() from e
        except ValueError as e:
            logger.error("Invalid response in chat completion | Exception: %s", e)
            raise InvalidResponseError(str(e)) from e
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            logger.error("Chat completion error | Exception: %s", e)
            raise ChatCompletionError(str(e), status_code=status_code) from e

    def _streaming_create(self, messages: list, **kwargs) -> Generator[ChatCompletionChunk, None, None]:
        """Handle streaming chat completion; yields ChatCompletionChunk objects."""
        stream = None
        try:
            stream = self.client.chat.completions.create(
                model=kwargs.pop("model", self.model),
                messages=messages,
                stream=True,
                **kwargs,
            )

            for event in stream:
                # If the event contains usage info, yield a usage-only chunk.
                if hasattr(event, "usage") and event.usage:
                    extensions = provider_extensions_from_raw(
                        "azure_openai",
                        event,
                        {
                            "id", "object", "created", "model", "system_fingerprint", "choices", "service_tier", "usage",
                        },
                    )
                    yield ChatCompletionChunk(
                        id=event.id,
                        object=event.object,
                        created=event.created,
                        model=event.model,
                        system_fingerprint=event.system_fingerprint,
                        choices=[],
                        service_tier=event.service_tier,
                        usage=build_openai_usage(event.usage),
                        extensions=extensions,
                    )
                    continue

                # Otherwise, yield content chunks.
                for ch in event.choices:
                    audio_delta = getattr(ch.delta, "audio", None)
                    delta = ChoiceDelta(
                        role=ch.delta.role,
                        content=ch.delta.content if not audio_delta else getattr(audio_delta, "transcript", None),
                        tool_calls=map_tool_call_deltas(ch.delta),
                        function_call=map_function_call_delta(getattr(ch.delta, "function_call", None)),
                        refusal=ch.delta.refusal,
                        extensions=provider_extensions_for("azure_openai", unmapped_provider_fields(ch.delta, {
                            "role", "content", "tool_calls", "function_call", "refusal", "audio",
                        })),
                    )
                    extensions = provider_extensions_from_raw(
                        "azure_openai",
                        event,
                        {
                            "id", "object", "created", "model", "system_fingerprint", "choices", "service_tier", "usage",
                        },
                    )
                    yield ChatCompletionChunk(
                        id=event.id,
                        object=event.object,
                        created=event.created,
                        model=event.model,
                        system_fingerprint=event.system_fingerprint,
                        choices=[ChunkDelta(
                            index=ch.index,
                            delta=delta,
                            finish_reason=ch.finish_reason,
                            logprobs=map_delta_logprobs(ch.logprobs),
                            content_filter_results=getattr(ch, "content_filter_results", None),
                            extensions=provider_extensions_for("azure_openai", unmapped_provider_fields(ch, {
                                "index", "delta", "finish_reason", "logprobs", "content_filter_results",
                            })),
                        )],
                        service_tier=event.service_tier,
                        usage=None,
                        extensions=extensions,
                    )

        except openai.RateLimitError as e:
            retry_after = get_retry_after_seconds(e)
            logger.error("Azure OpenAI streaming rate limit error | Exception: %s", e)
            raise RateLimitExceededError(retry_after) from e
        except openai.AuthenticationError as e:
            logger.error("Azure OpenAI streaming authentication error | Exception: %s", e)
            raise AuthenticationError(provider="azure_openai") from e
        except openai.APITimeoutError as e:
            logger.error("Azure OpenAI streaming timeout error | Exception: %s", e)
            raise TimeOutError(configured_timeout_seconds(self.timeout)) from e
        except openai.BadRequestError as e:
            _raise_if_content_filtered(e)
            logger.error("Azure OpenAI streaming bad request error | Exception: %s", e)
            raise ChatCompletionError(str(e), status_code=400) from e
        except ConnectionError as e:
            logger.error("Azure OpenAI streaming connection error: %s", e)
            raise APIConnectionError(self.endpoint) from e
        except PermissionError as e:
            logger.error("Azure OpenAI streaming authentication error: %s", e)
            raise AuthenticationError() from e
        except ValueError as e:
            logger.error("Azure OpenAI streaming invalid response: %s", e)
            raise InvalidResponseError(str(e)) from e
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            logger.error("Azure OpenAI streaming error: %s", e)
            raise ChatCompletionError(str(e), status_code=status_code) from e
        finally:
            if stream is not None:
                close_stream(stream)


class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    Azure OpenAI Embedding Provider.
    Adapts Azure-hosted OpenAI embedding models into the unified LLM Switch response format.
    """

    def __init__(self, **kwargs):
        """
        Initialize the AzureOpenAIEmbeddingProvider with API credentials and deployment info.
        Args:
            **kwargs: Should include 'api_key', 'api_version', 'azure_endpoint', 'azure_deployment', and 'model'.
        """
        self.endpoint = kwargs.get("azure_endpoint")
        self.client = AzureOpenAI(**merge_client_options({
            "api_key": kwargs.get("api_key"),
            "api_version": kwargs.get("api_version"),
            "azure_endpoint": self.endpoint,
            "azure_deployment": kwargs.get("azure_deployment"),
            "max_retries": 0,
            "timeout": kwargs.get("timeout"),
        }, kwargs.get("client_options")))
        self.model = kwargs.get("model")

    def create(self, input: list[str], **kwargs) -> CreateEmbeddingResponse:  # noqa: A002
        """Create embeddings for the given input using Azure OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=kwargs.pop("model", self.model),
                input=input,
                **kwargs,
            )
            data = [
                Embedding(index=item.index, embedding=item.embedding, object=item.object)
                for item in response.data
            ]

            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
            )

            extensions = provider_extensions_from_raw(
                "azure_openai",
                response,
                {"object", "model", "data", "usage"},
            )

            return CreateEmbeddingResponse(
                object=response.object,
                created=int(time.time()),
                model=response.model,
                data=data,
                usage=usage,
                extensions=extensions,
            )
        except ConnectionError as e:
            logger.error("Connection error in embedding creation: %s | Exception: %s", self.endpoint, e)
            raise APIConnectionError(self.endpoint) from e
        except PermissionError as e:
            logger.error("Authentication error in embedding creation | Exception: %s", e)
            raise AuthenticationError() from e
        except ValueError as e:
            logger.error("Invalid response in embedding creation | Exception: %s", e)
            raise InvalidResponseError(str(e)) from e
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            logger.error("Embedding generation error | Exception: %s", e)
            raise EmbeddingGenerationError(str(e), status_code=status_code) from e


class AsyncAzureOpenAIChatProvider(BaseChatProvider):
    """
    Async Azure OpenAI ChatCompletion Provider.
    Uses openai.AsyncAzureOpenAI for non-blocking chat completions.
    Supports both non-streaming and streaming (async generator) modes.
    """

    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.endpoint = kwargs.get("endpoint")
        self.api_version = kwargs.get("api_version")
        self._token_limit = azure_token_limit_name(self.api_version)
        self.deployment_name = kwargs.get("deployment_name")
        self.model = kwargs.get("model")
        self.timeout = kwargs.get("timeout")
        self.client = AsyncAzureOpenAI(**merge_client_options({
            "api_key": self.api_key,
            "api_version": self.api_version,
            "azure_endpoint": self.endpoint,
            "azure_deployment": self.deployment_name,
            "max_retries": 0,
            "timeout": kwargs.get("timeout"),
        }, kwargs.get("client_options")))

    async def create(self, messages: list, stream: bool = False, **kwargs):
        """Dispatch to async streaming or non-streaming completion."""
        kwargs = _sanitize_completion_kwargs(kwargs, self.model)
        normalize_token_limit(kwargs, self._token_limit)
        if stream:
            if is_pydantic_model_class(kwargs.get("response_format")):
                kwargs["response_format"] = pydantic_to_response_format(kwargs["response_format"])
            return self._streaming_create(messages, **kwargs)
        return await self._non_streaming_create(messages, **kwargs)

    async def _non_streaming_create(self, messages: list, **kwargs) -> ChatCompletion:
        try:
            if is_pydantic_model_class(kwargs.get("response_format")):
                response = await self.client.beta.chat.completions.parse(
                    model=kwargs.pop("model", self.model),
                    messages=messages,
                    **kwargs,
                )
            else:
                response = await self.client.chat.completions.create(
                    model=kwargs.pop("model", self.model),
                    messages=messages,
                    stream=False,
                    **kwargs,
                )
            choices = [
                Choice(
                    index=ch.index,
                    message=ChatCompletionMessage(
                        role=ch.message.role,
                        content=ch.message.content,
                        tool_calls=map_tool_calls(ch.message),
                        function_call=map_function_call(ch.message.function_call),
                        refusal=ch.message.refusal,
                        audio=audio_from(getattr(ch.message, "audio", None)),
                        annotations=getattr(ch.message, "annotations", None),
                        parsed=getattr(ch.message, "parsed", None),
                        context=to_jsonable(getattr(ch.message, "context", None)),
                        extensions=provider_extensions_for("azure_openai", unmapped_provider_fields(ch.message, {
                            "role", "content", "tool_calls", "function_call", "refusal", "audio",
                            "annotations", "parsed", "context",
                        })),
                    ),
                    finish_reason=ch.finish_reason,
                    logprobs=map_message_logprobs(ch.logprobs),
                    content_filter_results=getattr(ch, "content_filter_results", None),
                    extensions=provider_extensions_for("azure_openai", unmapped_provider_fields(ch, {
                        "index", "message", "finish_reason", "logprobs", "content_filter_results",
                    })),
                )
                for ch in response.choices
            ]
            extensions = provider_extensions_from_raw(
                "azure_openai",
                response,
                {
                    "id", "object", "created", "model", "choices", "usage", "system_fingerprint",
                    "service_tier", "store", "metadata", "prompt_filter_results", "moderation",
                },
            )
            return ChatCompletion(
                id=response.id, object=response.object, created=response.created,
                model=response.model, choices=choices, usage=build_openai_usage(response.usage),
                system_fingerprint=response.system_fingerprint, service_tier=response.service_tier,
                store=getattr(response, "store", None), metadata=getattr(response, "metadata", None),
                prompt_filter_results=getattr(response, "prompt_filter_results", None),
                moderation=getattr(response, "moderation", None),
                extensions=extensions,
            )
        except openai.RateLimitError as e:
            retry_after = get_retry_after_seconds(e)
            raise RateLimitExceededError(retry_after) from e
        except openai.AuthenticationError as e:
            raise AuthenticationError(provider="azure_openai") from e
        except openai.APITimeoutError as e:
            raise TimeOutError(configured_timeout_seconds(self.timeout)) from e
        except openai.BadRequestError as e:
            _raise_if_content_filtered(e)
            raise ChatCompletionError(str(e), status_code=400) from e
        except Exception as e:
            raise ChatCompletionError(str(e), status_code=getattr(e, "status_code", 500)) from e

    async def _streaming_create(self, messages: list, **kwargs) -> AsyncGenerator[ChatCompletionChunk, None]:
        stream = None
        try:
            stream = await self.client.chat.completions.create(
                model=kwargs.pop("model", self.model),
                messages=messages,
                stream=True,
                **kwargs,
            )
            async for event in stream:
                if hasattr(event, "usage") and event.usage:
                    extensions = provider_extensions_from_raw(
                        "azure_openai",
                        event,
                        {
                            "id", "object", "created", "model", "system_fingerprint", "choices", "service_tier", "usage",
                        },
                    )
                    yield ChatCompletionChunk(
                        id=event.id, object=event.object, created=event.created, model=event.model,
                        system_fingerprint=event.system_fingerprint, choices=[], service_tier=event.service_tier,
                        usage=build_openai_usage(event.usage),
                        extensions=extensions,
                    )
                    continue
                for ch in event.choices:
                    audio_delta = getattr(ch.delta, "audio", None)
                    extensions = provider_extensions_from_raw(
                        "azure_openai",
                        event,
                        {
                            "id", "object", "created", "model", "system_fingerprint", "choices", "service_tier", "usage",
                        },
                    )
                    yield ChatCompletionChunk(
                        id=event.id, object=event.object, created=event.created, model=event.model,
                        system_fingerprint=event.system_fingerprint, service_tier=event.service_tier, usage=None,
                        choices=[ChunkDelta(
                            index=ch.index,
                            delta=ChoiceDelta(
                                role=ch.delta.role,
                                content=ch.delta.content if not audio_delta else getattr(audio_delta, "transcript",
                                                                                         None),
                                tool_calls=map_tool_call_deltas(ch.delta),
                                function_call=map_function_call_delta(getattr(ch.delta, "function_call", None)),
                                refusal=ch.delta.refusal,
                                extensions=provider_extensions_for("azure_openai", unmapped_provider_fields(ch.delta, {
                                    "role", "content", "tool_calls", "function_call", "refusal", "audio",
                                })),
                            ),
                            finish_reason=ch.finish_reason,
                            logprobs=map_delta_logprobs(ch.logprobs),
                            content_filter_results=getattr(ch, "content_filter_results", None),
                            extensions=provider_extensions_for("azure_openai", unmapped_provider_fields(ch, {
                                "index", "delta", "finish_reason", "logprobs", "content_filter_results",
                            })),
                        )],
                        extensions=extensions,
                    )
        except openai.RateLimitError as e:
            retry_after = get_retry_after_seconds(e)
            raise RateLimitExceededError(retry_after) from e
        except openai.AuthenticationError as e:
            raise AuthenticationError(provider="azure_openai") from e
        except openai.APITimeoutError as e:
            raise TimeOutError(configured_timeout_seconds(self.timeout)) from e
        except openai.BadRequestError as e:
            _raise_if_content_filtered(e)
            raise ChatCompletionError(str(e), status_code=400) from e
        except Exception as e:
            raise ChatCompletionError(str(e), status_code=getattr(e, "status_code", 500)) from e
        finally:
            if stream is not None:
                await close_async_stream(stream)


# Azure exposes /responses only from this API version onward; older versions 404.
MIN_RESPONSES_API_VERSION = "2025-03-01-preview"


def _resolve_responses_api_version(kwargs: dict) -> str:
    """
    Pick the api-version for the Responses client.

    ``responses_api_version`` wins when set, so a client can keep an older
    api-version for chat while still reaching /responses. Otherwise the
    configured api-version is used, upgraded to the minimum supported one when
    it predates the Responses API.
    """
    override = kwargs.get("responses_api_version")
    if override:
        return override
    api_version = kwargs.get("api_version")
    if not api_version:
        return MIN_RESPONSES_API_VERSION
    if api_version.replace("-preview", "") < MIN_RESPONSES_API_VERSION.replace("-preview", ""):
        logger.warning(
            "api_version %s predates the Azure Responses API; using %s for responses.* calls. "
            "Set responses_api_version to override.", api_version, MIN_RESPONSES_API_VERSION)
        return MIN_RESPONSES_API_VERSION
    return api_version


class AzureOpenAIResponsesProvider(SyncResponsesProviderMixin):
    """Azure OpenAI Responses API provider (``client.responses.*``)."""

    provider_name = "azure_openai"

    def __init__(self, **kwargs) -> None:
        self.endpoint = kwargs.get("endpoint")
        self.deployment_name = kwargs.get("deployment_name")
        self.api_version = _resolve_responses_api_version(kwargs)
        # Azure routes by deployment, so the deployment name is the effective model id.
        self.model = self.deployment_name or kwargs.get("model")
        self.timeout = kwargs.get("timeout")
        self.client = AzureOpenAI(**merge_client_options({
            "api_key": kwargs.get("api_key"),
            "api_version": self.api_version,
            "azure_endpoint": self.endpoint,
            "azure_deployment": self.deployment_name,
            "max_retries": 0,
            "timeout": kwargs.get("timeout"),
        }, kwargs.get("client_options")))

    def _build_kwargs(self, input: Any, kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: A002
        payload = super()._build_kwargs(input, kwargs)
        payload["model"] = self.deployment_name or self.model
        return payload


class AsyncAzureOpenAIResponsesProvider(AsyncResponsesProviderMixin):
    """Async Azure OpenAI Responses API provider (``client.responses.*``)."""

    provider_name = "azure_openai"

    def __init__(self, **kwargs) -> None:
        self.endpoint = kwargs.get("endpoint")
        self.deployment_name = kwargs.get("deployment_name")
        self.api_version = _resolve_responses_api_version(kwargs)
        self.model = self.deployment_name or kwargs.get("model")
        self.timeout = kwargs.get("timeout")
        self.client = AsyncAzureOpenAI(**merge_client_options({
            "api_key": kwargs.get("api_key"),
            "api_version": self.api_version,
            "azure_endpoint": self.endpoint,
            "azure_deployment": self.deployment_name,
            "max_retries": 0,
            "timeout": kwargs.get("timeout"),
        }, kwargs.get("client_options")))

    def _build_kwargs(self, input: Any, kwargs: dict[str, Any]) -> dict[str, Any]:  # noqa: A002
        payload = super()._build_kwargs(input, kwargs)
        payload["model"] = self.deployment_name or self.model
        return payload
