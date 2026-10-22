"""
openai_provider.py

Native OpenAI providers for chat completions and embeddings. The chat provider
supports streaming and non-streaming completions and converts OpenAI responses
into the unified LLM Switch format. Response mapping shared with the Azure
OpenAI provider lives in :mod:`llmswitch.providers._shared`.

Classes:
    OpenAIChatProvider: Chat provider for OpenAI models (streaming and non-streaming).
    OpenAIEmbeddingProvider: Embedding provider for OpenAI models.
    AsyncOpenAIChatProvider: Async chat provider for OpenAI models.
"""

import logging
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any, Optional

import openai
from openai import AsyncOpenAI, OpenAI

from ..base import BaseChatProvider, BaseEmbeddingProvider
from ..errors import (
    APIConnectionError,
    AuthenticationError,
    ChatCompletionError,
    EmbeddingGenerationError,
    InvalidResponseError,
    RateLimitExceededError,
    TimeOutError,
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
    unmapped_provider_fields,
)
from ._shared import (
    audio_from,
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
)

# Name resolves to "llmswitch.providers.openai_provider"
logger = logging.getLogger(__name__)


class OpenAIChatProvider(BaseChatProvider):
    """
    Native OpenAI ChatCompletion Provider.
    Adapts OpenAI models into the unified LLM Switch response format.
    Supports both streaming and non-streaming chat completions.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the OpenAIChatProvider with API credentials and optional configuration.

        Args:
            **kwargs: Should include 'api_key' and optionally 'endpoint' (base_url),
                      'model', 'organization', and 'timeout'.
        """
        self.api_key: str = kwargs.get("api_key")
        self.endpoint: Optional[str] = kwargs.get("endpoint")
        self.model: Optional[str] = kwargs.get("deployment_name") or kwargs.get("model")
        self.organization: Optional[str] = kwargs.get("organization")
        self.project: Optional[str] = kwargs.get("project")
        self.timeout: Optional[float] = kwargs.get("timeout")

        self.client: OpenAI = OpenAI(**merge_client_options(_openai_client_kwargs(kwargs), kwargs.get("client_options")))

    def create(self, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
        """
        Dispatch to streaming or non-streaming chat completion.

        Returns:
            ChatCompletion or generator of ChatCompletionChunk.
        """
        if stream:
            # Streaming has no native .parse() helper — request the schema instead.
            if is_pydantic_model_class(kwargs.get("response_format")):
                kwargs["response_format"] = pydantic_to_response_format(kwargs["response_format"])
            return self._streaming_create(messages, **kwargs)
        return self._non_streaming_create(messages, **kwargs)

    def _non_streaming_create(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatCompletion:
        """
        Handle non-streaming chat completion.

        When ``response_format`` is a Pydantic model class, routes to the SDK's
        ``beta.chat.completions.parse`` so the deserialized object lands in
        ``message.parsed`` (OpenAI ``.parse()`` semantics).
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
                        extensions=provider_extensions_for("openai", unmapped_provider_fields(ch.message, {
                            "role", "content", "tool_calls", "function_call", "refusal", "audio", "annotations", "parsed",
                        })),
                    ),
                    finish_reason=ch.finish_reason,
                    logprobs=map_message_logprobs(ch.logprobs),
                    extensions=provider_extensions_for("openai", unmapped_provider_fields(ch, {
                        "index", "message", "finish_reason", "logprobs",
                    })),
                )
                for ch in response.choices
            ]

            extensions = provider_extensions_from_raw(
                "openai",
                response,
                {
                    "id", "object", "created", "model", "choices", "usage", "system_fingerprint",
                    "service_tier", "store", "metadata", "moderation",
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
                service_tier=getattr(response, "service_tier", None),
                store=getattr(response, "store", None),
                metadata=getattr(response, "metadata", None),
                moderation=getattr(response, "moderation", None),
                extensions=extensions,
            )
        except openai.RateLimitError as e:
            retry_after = get_retry_after_seconds(e)
            logger.error("OpenAI rate limit error | Exception: %s", e)
            raise RateLimitExceededError(retry_after) from e
        except openai.AuthenticationError as e:
            logger.error("OpenAI authentication error | Exception: %s", e)
            raise AuthenticationError(provider="openai") from e
        except openai.APITimeoutError as e:
            logger.error("OpenAI timeout error | Exception: %s", e)
            raise TimeOutError(configured_timeout_seconds(self.timeout)) from e
        except openai.BadRequestError as e:
            logger.error("OpenAI bad request error | Exception: %s", e)
            raise ChatCompletionError(str(e), status_code=400) from e
        except ConnectionError as e:
            logger.error("Connection error in chat completion: %s | Exception: %s", self.endpoint, e)
            raise APIConnectionError(self.endpoint or "https://api.openai.com") from e
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

    def _streaming_create(self, messages: list[dict[str, Any]], **kwargs: Any) -> Generator[
        ChatCompletionChunk, None, None]:
        """Handle streaming chat completion; yields ChatCompletionChunk objects."""
        stream = None
        try:
            stream = self.client.chat.completions.create(
                model=kwargs.pop("model", self.model),
                messages=messages,
                stream=True,
                **kwargs,
            )

            last_stream_usage = None
            last_event_with_metadata = None

            for event in stream:
                # Capture usage if present (may arrive on any chunk or a dedicated final chunk).
                if hasattr(event, "usage") and event.usage:
                    last_stream_usage = build_openai_usage(event.usage)

                # Skip chunks with no choices (usage-only or empty intermediate chunks).
                if not event.choices:
                    continue

                last_event_with_metadata = event

                # Yield content chunks with usage=None (unified with Azure OpenAI output).
                for ch in event.choices:
                    audio_delta = getattr(ch.delta, "audio", None)
                    delta = ChoiceDelta(
                        role=ch.delta.role,
                        content=ch.delta.content if not audio_delta else getattr(audio_delta, "transcript", None),
                        tool_calls=map_tool_call_deltas(ch.delta),
                        function_call=map_function_call_delta(getattr(ch.delta, "function_call", None)),
                        refusal=ch.delta.refusal,
                        extensions=provider_extensions_for("openai", unmapped_provider_fields(ch.delta, {
                            "role", "content", "tool_calls", "function_call", "refusal", "audio",
                        })),
                    )
                    extensions = provider_extensions_from_raw(
                        "openai",
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
                            extensions=provider_extensions_for("openai", unmapped_provider_fields(ch, {
                                "index", "delta", "finish_reason", "logprobs",
                            })),
                        )],
                        service_tier=event.service_tier,
                        usage=None,
                        extensions=extensions,
                    )

            # After the loop, yield a final usage-only chunk (unified with Azure OpenAI output).
            if last_stream_usage:
                ref = last_event_with_metadata
                extensions = provider_extensions_for("openai")
                if ref is not None:
                    extensions = provider_extensions_from_raw(
                        "openai",
                        ref,
                        {
                            "id", "object", "created", "model", "system_fingerprint", "choices", "service_tier", "usage",
                        },
                    )
                yield ChatCompletionChunk(
                    id=getattr(ref, "id", "openai-response") if ref else "openai-response",
                    object="chat.completion.chunk",
                    created=getattr(ref, "created", int(time.time())) if ref else int(time.time()),
                    model=getattr(ref, "model", self.model) if ref else self.model,
                    system_fingerprint=getattr(ref, "system_fingerprint", None) if ref else None,
                    choices=[],
                    service_tier=getattr(ref, "service_tier", None) if ref else None,
                    usage=last_stream_usage,
                    extensions=extensions,
                )

        except openai.RateLimitError as e:
            retry_after = get_retry_after_seconds(e)
            logger.error("OpenAI streaming rate limit error | Exception: %s", e)
            raise RateLimitExceededError(retry_after) from e
        except openai.AuthenticationError as e:
            logger.error("OpenAI streaming authentication error | Exception: %s", e)
            raise AuthenticationError(provider="openai") from e
        except openai.APITimeoutError as e:
            logger.error("OpenAI streaming timeout error | Exception: %s", e)
            raise TimeOutError(configured_timeout_seconds(self.timeout)) from e
        except openai.BadRequestError as e:
            logger.error("OpenAI streaming bad request error | Exception: %s", e)
            raise ChatCompletionError(str(e), status_code=400) from e
        except ConnectionError as e:
            logger.error("OpenAI streaming connection error: %s", e)
            raise APIConnectionError(self.endpoint or "https://api.openai.com") from e
        except PermissionError as e:
            logger.error("OpenAI streaming authentication error: %s", e)
            raise AuthenticationError() from e
        except ValueError as e:
            logger.error("OpenAI streaming invalid response: %s", e)
            raise InvalidResponseError(str(e)) from e
        except Exception as e:
            status_code = getattr(e, "status_code", 500)
            logger.error("OpenAI streaming error: %s", e)
            raise ChatCompletionError(str(e), status_code=status_code) from e
        finally:
            if stream is not None:
                close_stream(stream)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    Native OpenAI Embedding Provider.
    Adapts OpenAI embedding models into the unified LLM Switch response format.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the OpenAIEmbeddingProvider with API credentials and optional configuration.

        Args:
            **kwargs: Should include 'api_key' and optionally 'endpoint' (base_url),
                      'model', 'organization', 'project', and 'timeout'.
        """
        self.endpoint: Optional[str] = kwargs.get("endpoint")
        self.model: Optional[str] = kwargs.get("deployment_name") or kwargs.get("model")
        self.organization: Optional[str] = kwargs.get("organization")
        self.project: Optional[str] = kwargs.get("project")
        self.timeout: Optional[float] = kwargs.get("timeout")

        self.client: OpenAI = OpenAI(**merge_client_options(_openai_client_kwargs(kwargs), kwargs.get("client_options")))

    def create(self, input: list[str], **kwargs: Any) -> CreateEmbeddingResponse:  # noqa: A002
        """Create embeddings for the given input using the OpenAI API."""
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
                "openai",
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
            raise APIConnectionError(self.endpoint or "https://api.openai.com") from e
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


class AsyncOpenAIChatProvider(BaseChatProvider):
    """
    Async native OpenAI ChatCompletion Provider.
    Uses openai.AsyncOpenAI for non-blocking chat completions.
    Supports non-streaming (awaitable) and streaming (async generator) modes.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.api_key: str = kwargs.get("api_key")
        self.endpoint: Optional[str] = kwargs.get("endpoint")
        self.model: Optional[str] = kwargs.get("deployment_name") or kwargs.get("model")
        self.organization: Optional[str] = kwargs.get("organization")
        self.project: Optional[str] = kwargs.get("project")
        self.timeout: Optional[float] = kwargs.get("timeout")

        self.client: AsyncOpenAI = AsyncOpenAI(**merge_client_options(_openai_client_kwargs(kwargs), kwargs.get("client_options")))

    async def create(self, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
        """Dispatch to async streaming or non-streaming completion."""
        if stream:
            if is_pydantic_model_class(kwargs.get("response_format")):
                kwargs["response_format"] = pydantic_to_response_format(kwargs["response_format"])
            return self._streaming_create(messages, **kwargs)
        return await self._non_streaming_create(messages, **kwargs)

    async def _non_streaming_create(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatCompletion:
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
                        extensions=provider_extensions_for("openai", unmapped_provider_fields(ch.message, {
                            "role", "content", "tool_calls", "function_call", "refusal", "audio", "annotations", "parsed",
                        })),
                    ),
                    finish_reason=ch.finish_reason,
                    logprobs=map_message_logprobs(ch.logprobs),
                    extensions=provider_extensions_for("openai", unmapped_provider_fields(ch, {
                        "index", "message", "finish_reason", "logprobs",
                    })),
                )
                for ch in response.choices
            ]
            extensions = provider_extensions_from_raw(
                "openai",
                response,
                {
                    "id", "object", "created", "model", "choices", "usage", "system_fingerprint",
                    "service_tier", "store", "metadata", "moderation",
                },
            )
            return ChatCompletion(
                id=response.id, object=response.object, created=response.created,
                model=response.model, choices=choices, usage=build_openai_usage(response.usage),
                system_fingerprint=response.system_fingerprint,
                service_tier=getattr(response, "service_tier", None),
                store=getattr(response, "store", None),
                metadata=getattr(response, "metadata", None),
                moderation=getattr(response, "moderation", None),
                extensions=extensions,
            )
        except openai.RateLimitError as e:
            retry_after = get_retry_after_seconds(e)
            raise RateLimitExceededError(retry_after) from e
        except openai.AuthenticationError as e:
            raise AuthenticationError(provider="openai") from e
        except openai.APITimeoutError as e:
            raise TimeOutError(configured_timeout_seconds(self.timeout)) from e
        except openai.BadRequestError as e:
            raise ChatCompletionError(str(e), status_code=400) from e
        except Exception as e:
            raise ChatCompletionError(str(e), status_code=getattr(e, "status_code", 500)) from e

    async def _streaming_create(self, messages: list[dict[str, Any]], **kwargs: Any) -> AsyncGenerator[
        ChatCompletionChunk, None]:
        stream = None
        try:
            stream = await self.client.chat.completions.create(
                model=kwargs.pop("model", self.model),
                messages=messages,
                stream=True,
                **kwargs,
            )
            last_stream_usage = None
            last_event_with_metadata = None

            async for event in stream:
                if hasattr(event, "usage") and event.usage:
                    last_stream_usage = build_openai_usage(event.usage)
                if not event.choices:
                    continue
                last_event_with_metadata = event
                for ch in event.choices:
                    audio_delta = getattr(ch.delta, "audio", None)
                    extensions = provider_extensions_from_raw(
                        "openai",
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
                                extensions=provider_extensions_for("openai", unmapped_provider_fields(ch.delta, {
                                    "role", "content", "tool_calls", "function_call", "refusal", "audio",
                                })),
                            ),
                            finish_reason=ch.finish_reason, logprobs=map_delta_logprobs(ch.logprobs),
                            extensions=provider_extensions_for("openai", unmapped_provider_fields(ch, {
                                "index", "delta", "finish_reason", "logprobs",
                            })),
                        )],
                        extensions=extensions,
                    )

            if last_stream_usage:
                ref = last_event_with_metadata
                extensions = provider_extensions_for("openai")
                if ref is not None:
                    extensions = provider_extensions_from_raw(
                        "openai",
                        ref,
                        {
                            "id", "object", "created", "model", "system_fingerprint", "choices", "service_tier", "usage",
                        },
                    )
                yield ChatCompletionChunk(
                    id=getattr(ref, "id", "openai-response") if ref else "openai-response",
                    object="chat.completion.chunk",
                    created=getattr(ref, "created", int(time.time())) if ref else int(time.time()),
                    model=getattr(ref, "model", self.model) if ref else self.model,
                    system_fingerprint=getattr(ref, "system_fingerprint", None) if ref else None,
                    choices=[], service_tier=getattr(ref, "service_tier", None) if ref else None,
                    usage=last_stream_usage,
                    extensions=extensions,
                )
        except openai.RateLimitError as e:
            retry_after = get_retry_after_seconds(e)
            raise RateLimitExceededError(retry_after) from e
        except openai.AuthenticationError as e:
            raise AuthenticationError(provider="openai") from e
        except openai.APITimeoutError as e:
            raise TimeOutError(configured_timeout_seconds(self.timeout)) from e
        except openai.BadRequestError as e:
            raise ChatCompletionError(str(e), status_code=400) from e
        except Exception as e:
            raise ChatCompletionError(str(e), status_code=getattr(e, "status_code", 500)) from e
        finally:
            if stream is not None:
                await close_async_stream(stream)


def _openai_client_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build ``OpenAI``/``AsyncOpenAI`` constructor kwargs from client config kwargs."""
    client_kwargs: dict[str, Any] = {"api_key": kwargs.get("api_key"), "max_retries": 0}
    for source, target in (("endpoint", "base_url"), ("organization", "organization"), ("project", "project")):
        if kwargs.get(source):
            client_kwargs[target] = kwargs[source]
    if kwargs.get("timeout") is not None:
        client_kwargs["timeout"] = kwargs["timeout"]
    return client_kwargs
