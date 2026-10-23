"""
azurefoundry_provider.py

This module implements Azure Foundry providers for chat completions and embeddings.
It provides sync and async chat providers that support both streaming and non-streaming
completions, converting OpenAI-style messages to Azure Foundry format and handling
error translation.

Classes:
    AzureFoundryChatProvider: Sync chat provider for Azure Foundry models.
    AsyncAzureFoundryChatProvider: Async chat provider for Azure Foundry models.
    AzureFoundryEmbeddingProvider: Embedding provider for Azure Foundry models.
"""

import logging
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any

from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
from azure.ai.inference.aio import ChatCompletionsClient as AsyncChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletionsToolCall,
    FunctionCall,
    ImageContentItem,
    JsonSchemaFormat,
    SystemMessage,
    TextContentItem,
    ToolMessage,
    UserMessage,
)
from azure.ai.inference.models import (
    ImageUrl as AzureImageUrl,
)
from azure.core.credentials import AzureKeyCredential

from ..base import BaseChatProvider, BaseEmbeddingProvider
from ..errors import (
    APIConnectionError,
    AuthenticationError,
    ChatCompletionError,
    EmbeddingGenerationError,
    InvalidResponseError,
    detect_content_filter_error,
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
    CreateEmbeddingResponse,
    Embedding,
    Function,
    Usage,
)
from ..structured import (
    is_pydantic_model_class,
    populate_parsed,
    provider_extensions_for,
    provider_extensions_from_raw,
    pydantic_to_response_format,
)
from ._shared import close_async_stream, close_stream, merge_client_options, normalize_token_limit

# Initialize logger for this module
# Name resolves to "llmswitch.providers.azurefoundry_provider"
logger = logging.getLogger(__name__)


def _coerce_response_format(kwargs: dict[str, Any]) -> Any:
    """
    If ``response_format`` is a Pydantic model class (.parse() style), replace it
    in ``kwargs`` with the SDK's ``JsonSchemaFormat`` and return the model class
    so the caller can deserialize ``message.parsed`` afterwards. Azure Foundry
    has no native parse, so structured output is requested via the JSON schema.

    The schema is built strict-compliant (additionalProperties:false, every
    property required) so the service does not reject it under strict mode.
    """
    response_format = kwargs.get("response_format")
    if not is_pydantic_model_class(response_format):
        return None
    json_schema = pydantic_to_response_format(response_format)["json_schema"]
    kwargs["response_format"] = JsonSchemaFormat(
        name=json_schema["name"],
        schema=json_schema["schema"],
        strict=bool(json_schema.get("strict", False)),
    )
    return response_format


def _usage_field(usage_obj: Any, key: str, default: Any = 0) -> Any:
    """Read usage values from both SDK attributes and dict-like payloads."""
    if usage_obj is None:
        return default
    attr_value = getattr(usage_obj, key, None)
    if attr_value is not None:
        return attr_value
    if hasattr(usage_obj, "get"):
        return usage_obj.get(key, default)
    return default


class _AzureFoundryMessageConverterMixin:
    def _sanitize_request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Remove unified kwargs that Azure AI Inference does not accept."""
        request_kwargs = dict(kwargs)
        request_kwargs.pop("parallel_tool_calls", None)
        request_kwargs.pop("stream_options", None)
        # Azure AI Inference uses max_tokens
        normalize_token_limit(request_kwargs, "max_tokens")
        return request_kwargs

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        """
        Convert OpenAI-style messages to Azure Foundry format.
        Supports system, user (text + multimodal), assistant (with optional tool_calls), and tool role messages.

        Args:
            messages (list[dict]): List of message dicts with 'role' and 'content'.

        Returns:
            list: List of Azure Foundry message objects.
        """
        converted: list[Any] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role in {"system", "developer"}:
                converted.append(SystemMessage(content=content))
            elif role == "user":
                if isinstance(content, list):
                    # Multimodal content: build Azure Foundry multi-part content
                    parts = []
                    for part in content:
                        if part.get("type") == "text":
                            parts.append(TextContentItem(text=part["text"]))
                        elif part.get("type") == "image_url":
                            img = part.get("image_url", {})
                            parts.append(ImageContentItem(
                                image_url=AzureImageUrl(url=img.get("url", ""))
                            ))
                    converted.append(UserMessage(content=parts))
                else:
                    converted.append(UserMessage(content=content))
            elif role == "assistant":
                raw_tool_calls = msg.get("tool_calls")
                if raw_tool_calls:
                    # Convert OpenAI-format tool_calls to Azure Foundry format using FunctionCall
                    azure_tool_calls = []
                    for tc in raw_tool_calls:
                        if isinstance(tc, dict):
                            tc_id = tc.get("id")
                            fn = tc.get("function", {})
                            fn_name = fn.get("name", "")
                            fn_args = fn.get("arguments", "{}")
                        else:
                            # already a Pydantic / SDK object
                            tc_id = getattr(tc, "id", None)
                            fn_obj = getattr(tc, "function", None)
                            fn_name = getattr(fn_obj, "name", "") if fn_obj else ""
                            fn_args = getattr(fn_obj, "arguments", "{}") if fn_obj else "{}"
                        azure_tool_calls.append(
                            ChatCompletionsToolCall(
                                id=tc_id,
                                function=FunctionCall(name=fn_name, arguments=fn_args),
                            )
                        )
                    converted.append(AssistantMessage(content=content or "", tool_calls=azure_tool_calls))
                else:
                    converted.append(AssistantMessage(content=content))
            elif role == "tool":
                # Tool result message
                converted.append(ToolMessage(content=content or "", tool_call_id=msg.get("tool_call_id", "")))
            else:
                logger.warning("Unknown message role: %s", role)
        return converted


class AzureFoundryChatProvider(_AzureFoundryMessageConverterMixin, BaseChatProvider):
    """
    Chat provider for Azure Foundry models. Handles both streaming and non-streaming chat completions.
    Converts OpenAI/Haystack-style messages to Azure Foundry format and dispatches requests.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the AzureFoundryChatProvider with endpoint, model, and API credentials.
        Args:
            **kwargs: Should include 'endpoint', 'model', 'api_key', and optionally 'api_version'.
        """
        self.endpoint: str = kwargs.get("endpoint")
        self.model: str = kwargs.get("deployment_name") or kwargs.get("model")
        self.client: ChatCompletionsClient = ChatCompletionsClient(**merge_client_options({
            "endpoint": self.endpoint,
            "credential": AzureKeyCredential(kwargs.get("api_key")),
            "api_version": kwargs.get("api_version", "2024-05-01-preview"),
            "retry_total": 0,
            "connection_timeout": kwargs.get("timeout"),
            "read_timeout": kwargs.get("timeout"),
        }, kwargs.get("client_options")))

    def create(self, messages: list[dict], stream: bool = False, **kwargs) -> Generator[
                                                                                  ChatCompletionChunk, Any, None] | ChatCompletion:
        """
        Dispatch to streaming or non-streaming chat completion.

        Args:
            messages (list[dict]): List of message dicts.
            stream (bool): Whether to use streaming completion.
            **kwargs: Additional parameters for completion.

        Returns:
            ChatCompletion or generator of ChatCompletionChunk.
        """
        azure_messages = self._convert_messages(messages)
        request_kwargs = self._sanitize_request_kwargs(kwargs)
        parse_model = _coerce_response_format(request_kwargs)
        if stream:
            return self._create_streaming(azure_messages, **request_kwargs)
        result = self._create_non_streaming(azure_messages, **request_kwargs)
        if parse_model is not None:
            populate_parsed(result, parse_model)
        return result

    def _create_non_streaming(self, azure_messages: list, **kwargs) -> ChatCompletion:
        """
        Handle non-streaming chat completion.

        Args:
            azure_messages (list): List of Azure Foundry message objects.
            **kwargs: Additional parameters for completion.

        Returns:
            ChatCompletion: The completion result.
        """
        try:
            response = self.client.complete(model=kwargs.pop("model", self.model), messages=azure_messages, **kwargs)

            choices = []
            for ch in response.choices:
                # Map tool_calls from Azure Foundry format to unified model
                raw_tool_calls = (ch.message.get("tool_calls")
                                  if hasattr(ch.message, "get")
                                  else getattr(ch.message, "tool_calls", None))
                mapped_tool_calls = None
                if raw_tool_calls:
                    mapped_tool_calls = [
                        ChatCompletionMessageToolCall(
                            id=getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None),
                            type="function",
                            function=Function(
                                name=(tc.get("function", {}).get("name") if isinstance(tc, dict)
                                      else getattr(getattr(tc, "function", None), "name", "")),
                                arguments=(tc.get("function", {}).get("arguments", "{}") if isinstance(tc, dict)
                                           else getattr(getattr(tc, "function", None), "arguments", "{}")),
                            ),
                        )
                        for tc in raw_tool_calls
                    ]

                choices.append(Choice(
                    index=ch.index,
                    message=ChatCompletionMessage(
                        role=ch.message.role,
                        content=ch.message.content,
                        tool_calls=mapped_tool_calls,
                        reasoning_content=(ch.message.get("reasoning_content")
                                           if hasattr(ch.message, "get")
                                           else getattr(ch.message, "reasoning_content", None)),
                    ),
                    finish_reason=ch.finish_reason,
                    content_filter_results=getattr(ch, "content_filter_results", None),
                ))

            usage = CompletionUsage(
                completion_tokens=_usage_field(response.usage, "completion_tokens", 0),
                prompt_tokens=_usage_field(response.usage, "prompt_tokens", 0),
                total_tokens=_usage_field(response.usage, "total_tokens", 0),
                prompt_tokens_details=_usage_field(response.usage, "prompt_tokens_details", None),
                audio_prompt_tokens=_usage_field(response.usage, "audio_prompt_tokens", 0),
                latency_checkpoint=_usage_field(response.usage, "latency_checkpoint", None),
            )
            extensions = provider_extensions_from_raw(
                "azure_foundry",
                response,
                {"id", "object", "created", "model", "choices", "usage", "prompt_filter_results", "moderation"},
            )
            return ChatCompletion(
                id=response.id,
                created=int(response.created.timestamp()),
                model=response.model,
                object=response.get("object", "chat.completion"),
                choices=choices,
                usage=usage,
                prompt_filter_results=response.get("prompt_filter_results", None),
                moderation=getattr(response, "moderation", None),
                extensions=extensions,
            )
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
            content_filter_error = detect_content_filter_error(e, provider="azure_foundry")
            if content_filter_error:
                raise content_filter_error from e
            status_code = getattr(e, 'status_code', 500)
            logger.error("Error in AzureFoundryChatProvider._create_non_streaming | Exception: %s", e)
            raise ChatCompletionError(str(e), status_code=status_code) from e

    def _create_streaming(self, azure_messages: list, **kwargs) -> Generator[ChatCompletionChunk, Any, None]:
        """
        Handle streaming chat completion.

        Args:
            azure_messages (list): List of Azure Foundry message objects.
            **kwargs: Additional parameters for completion.

        Yields:
            ChatCompletionChunk: Streaming completion chunks.
        """
        stream = None
        try:
            kwargs.pop("stream_options", None)  # Azure AI Inference SDK does not support stream_options
            stream = self.client.complete(model=kwargs.pop("model", self.model), messages=azure_messages, stream=True,
                                          **kwargs)
            last_chunk_with_metadata = None
            last_usage = None
            usage_pending = False

            for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    last_usage = CompletionUsage(
                        completion_tokens=_usage_field(chunk.usage, "completion_tokens", 0),
                        prompt_tokens=_usage_field(chunk.usage, "prompt_tokens", 0),
                        total_tokens=_usage_field(chunk.usage, "total_tokens", 0),
                        prompt_tokens_details=_usage_field(chunk.usage, "prompt_tokens_details", None),
                        audio_prompt_tokens=_usage_field(chunk.usage, "audio_prompt_tokens", 0),
                    )
                    usage_pending = True

                if not chunk.choices:
                    continue

                last_chunk_with_metadata = chunk

                chunk_choices = []
                for choice in chunk.choices:
                    delta = getattr(choice, "delta", None)
                    # Map tool_call deltas if present
                    raw_tc_deltas = getattr(delta, "tool_calls", None) if delta else None
                    mapped_tc_deltas = None
                    if raw_tc_deltas:
                        mapped_tc_deltas = [
                            ChoiceDeltaToolCall(
                                index=getattr(tc, "index", idx),
                                id=getattr(tc, "id", None),
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name=getattr(getattr(tc, "function", None), "name", None),
                                    arguments=getattr(getattr(tc, "function", None), "arguments", None),
                                ),
                            )
                            for idx, tc in enumerate(raw_tc_deltas)
                        ]
                    chunk_choices.append(
                        ChunkDelta(
                            index=choice.index,
                            delta=ChoiceDelta(
                                content=getattr(delta, "content", None) if delta else None,
                                tool_calls=mapped_tc_deltas,
                            ),
                            finish_reason=getattr(choice, "finish_reason", None),
                        )
                    )

                chunk_usage = last_usage if getattr(chunk, "usage", None) else None
                yield ChatCompletionChunk(
                    id=getattr(chunk, "id", None),
                    object=getattr(chunk, "object", "chat.completion.chunk"),
                    created=int(chunk.created.timestamp()) if hasattr(chunk, "created") and chunk.created else None,
                    model=getattr(chunk, "model", None),
                    choices=chunk_choices,
                    usage=chunk_usage,
                    extensions=provider_extensions_for("azure_foundry"),
                )
                if chunk_usage is not None:
                    usage_pending = False

            if usage_pending:
                yield ChatCompletionChunk(
                    id=getattr(last_chunk_with_metadata, "id", "default_id"),
                    object="chat.completion.chunk",
                    created=int(last_chunk_with_metadata.created.timestamp())
                    if (last_chunk_with_metadata is not None and getattr(last_chunk_with_metadata, "created",
                                                                         None)) else 0,
                    model=getattr(last_chunk_with_metadata, "model", "default_model"),
                    choices=[],
                    usage=last_usage,
                    extensions=provider_extensions_for("azure_foundry"),
                )
        except ConnectionError as e:
            logger.error("Connection error in streaming chat completion: %s | Exception: %s", self.endpoint, e)
            raise APIConnectionError(self.endpoint) from e
        except PermissionError as e:
            logger.error("Authentication error in streaming chat completion | Exception: %s", e)
            raise AuthenticationError() from e
        except ValueError as e:
            logger.error("Invalid response in streaming chat completion | Exception: %s", e)
            raise InvalidResponseError(str(e)) from e
        except Exception as e:
            content_filter_error = detect_content_filter_error(e, provider="azure_foundry")
            if content_filter_error:
                raise content_filter_error from e
            status_code = getattr(e, 'status_code', 500)
            logger.error("Error in AzureFoundryChatProvider._create_streaming | Exception: %s", e)
            raise ChatCompletionError(str(e), status_code=status_code) from e
        finally:
            if stream is not None:
                close_stream(stream)


class AsyncAzureFoundryChatProvider(_AzureFoundryMessageConverterMixin, BaseChatProvider):
    """
    Async Azure Foundry chat provider.
    Reuses the same OpenAI-style to Foundry message conversion as the sync provider.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.endpoint: str = kwargs.get("endpoint")
        self.model: str = kwargs.get("deployment_name") or kwargs.get("model")
        self.client: AsyncChatCompletionsClient = AsyncChatCompletionsClient(**merge_client_options({
            "endpoint": self.endpoint,
            "credential": AzureKeyCredential(kwargs.get("api_key")),
            "api_version": kwargs.get("api_version", "2024-05-01-preview"),
            "retry_total": 0,
            "connection_timeout": kwargs.get("timeout"),
            "read_timeout": kwargs.get("timeout"),
        }, kwargs.get("client_options")))

    async def create(self, messages: list[dict], stream: bool = False, **kwargs: Any) -> Any:
        """Dispatch to async streaming or non-streaming chat completion."""
        azure_messages = self._convert_messages(messages)
        request_kwargs = self._sanitize_request_kwargs(kwargs)
        parse_model = _coerce_response_format(request_kwargs)
        if stream:
            return self._create_streaming(azure_messages, **request_kwargs)
        result = await self._create_non_streaming(azure_messages, **request_kwargs)
        if parse_model is not None:
            populate_parsed(result, parse_model)
        return result

    async def _create_non_streaming(self, azure_messages: list, **kwargs: Any) -> ChatCompletion:
        try:
            response = await self.client.complete(
                model=kwargs.pop("model", self.model),
                messages=azure_messages,
                **kwargs,
            )

            choices = []
            for ch in response.choices:
                raw_tool_calls = (
                    ch.message.get("tool_calls")
                    if hasattr(ch.message, "get")
                    else getattr(ch.message, "tool_calls", None)
                )
                mapped_tool_calls = None
                if raw_tool_calls:
                    mapped_tool_calls = [
                        ChatCompletionMessageToolCall(
                            id=getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None),
                            type="function",
                            function=Function(
                                name=(
                                    tc.get("function", {}).get("name")
                                    if isinstance(tc, dict)
                                    else getattr(getattr(tc, "function", None), "name", "")
                                ),
                                arguments=(
                                    tc.get("function", {}).get("arguments", "{}")
                                    if isinstance(tc, dict)
                                    else getattr(getattr(tc, "function", None), "arguments", "{}")
                                ),
                            ),
                        )
                        for tc in raw_tool_calls
                    ]

                choices.append(Choice(
                    index=ch.index,
                    message=ChatCompletionMessage(
                        role=ch.message.role,
                        content=ch.message.content,
                        tool_calls=mapped_tool_calls,
                        reasoning_content=(
                            ch.message.get("reasoning_content")
                            if hasattr(ch.message, "get")
                            else getattr(ch.message, "reasoning_content", None)
                        ),
                    ),
                    finish_reason=ch.finish_reason,
                    content_filter_results=getattr(ch, "content_filter_results", None),
                ))

            usage = CompletionUsage(
                completion_tokens=_usage_field(response.usage, "completion_tokens", 0),
                prompt_tokens=_usage_field(response.usage, "prompt_tokens", 0),
                total_tokens=_usage_field(response.usage, "total_tokens", 0),
                prompt_tokens_details=_usage_field(response.usage, "prompt_tokens_details", None),
                audio_prompt_tokens=_usage_field(response.usage, "audio_prompt_tokens", 0),
                latency_checkpoint=_usage_field(response.usage, "latency_checkpoint", None),
            )
            extensions = provider_extensions_from_raw(
                "azure_foundry",
                response,
                {"id", "object", "created", "model", "choices", "usage", "prompt_filter_results", "moderation"},
            )
            return ChatCompletion(
                id=response.id,
                created=int(response.created.timestamp()),
                model=response.model,
                object=response.get("object", "chat.completion"),
                choices=choices,
                usage=usage,
                prompt_filter_results=response.get("prompt_filter_results", None),
                moderation=getattr(response, "moderation", None),
                extensions=extensions,
            )
        except ConnectionError as e:
            logger.error("Connection error in async chat completion: %s | Exception: %s", self.endpoint, e)
            raise APIConnectionError(self.endpoint) from e
        except PermissionError as e:
            logger.error("Authentication error in async chat completion | Exception: %s", e)
            raise AuthenticationError() from e
        except ValueError as e:
            logger.error("Invalid response in async chat completion | Exception: %s", e)
            raise InvalidResponseError(str(e)) from e
        except Exception as e:
            content_filter_error = detect_content_filter_error(e, provider="azure_foundry")
            if content_filter_error:
                raise content_filter_error from e
            status_code = getattr(e, "status_code", 500)
            logger.error("Error in AsyncAzureFoundryChatProvider._create_non_streaming | Exception: %s", e)
            raise ChatCompletionError(str(e), status_code=status_code) from e

    async def _create_streaming(self, azure_messages: list, **kwargs: Any) -> AsyncGenerator[ChatCompletionChunk, None]:
        stream = None
        try:
            kwargs.pop("stream_options", None)  # Azure AI Inference SDK does not support stream_options
            stream = await self.client.complete(
                model=kwargs.pop("model", self.model),
                messages=azure_messages,
                stream=True,
                **kwargs,
            )
            last_chunk_with_metadata = None
            last_usage = None
            usage_pending = False

            async for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    last_usage = CompletionUsage(
                        completion_tokens=_usage_field(chunk.usage, "completion_tokens", 0),
                        prompt_tokens=_usage_field(chunk.usage, "prompt_tokens", 0),
                        total_tokens=_usage_field(chunk.usage, "total_tokens", 0),
                        prompt_tokens_details=_usage_field(chunk.usage, "prompt_tokens_details", None),
                        audio_prompt_tokens=_usage_field(chunk.usage, "audio_prompt_tokens", 0),
                    )
                    usage_pending = True

                if not chunk.choices:
                    continue

                last_chunk_with_metadata = chunk

                chunk_choices = []
                for choice in chunk.choices:
                    delta = getattr(choice, "delta", None)
                    raw_tc_deltas = getattr(delta, "tool_calls", None) if delta else None
                    mapped_tc_deltas = None
                    if raw_tc_deltas:
                        mapped_tc_deltas = [
                            ChoiceDeltaToolCall(
                                index=getattr(tc, "index", idx),
                                id=getattr(tc, "id", None),
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name=getattr(getattr(tc, "function", None), "name", None),
                                    arguments=getattr(getattr(tc, "function", None), "arguments", None),
                                ),
                            )
                            for idx, tc in enumerate(raw_tc_deltas)
                        ]
                    chunk_choices.append(
                        ChunkDelta(
                            index=choice.index,
                            delta=ChoiceDelta(
                                content=getattr(delta, "content", None) if delta else None,
                                tool_calls=mapped_tc_deltas,
                            ),
                            finish_reason=getattr(choice, "finish_reason", None),
                        )
                    )

                chunk_usage = last_usage if getattr(chunk, "usage", None) else None
                yield ChatCompletionChunk(
                    id=getattr(chunk, "id", None),
                    object=getattr(chunk, "object", "chat.completion.chunk"),
                    created=int(chunk.created.timestamp()) if hasattr(chunk, "created") and chunk.created else None,
                    model=getattr(chunk, "model", None),
                    choices=chunk_choices,
                    usage=chunk_usage,
                    extensions=provider_extensions_for("azure_foundry"),
                )
                if chunk_usage is not None:
                    usage_pending = False

            if usage_pending:
                yield ChatCompletionChunk(
                    id=getattr(last_chunk_with_metadata, "id", "default_id"),
                    object="chat.completion.chunk",
                    created=int(last_chunk_with_metadata.created.timestamp())
                    if (last_chunk_with_metadata is not None and getattr(last_chunk_with_metadata, "created",
                                                                         None)) else 0,
                    model=getattr(last_chunk_with_metadata, "model", "default_model"),
                    choices=[],
                    usage=last_usage,
                    extensions=provider_extensions_for("azure_foundry"),
                )
        except ConnectionError as e:
            logger.error("Connection error in async streaming chat completion: %s | Exception: %s", self.endpoint, e)
            raise APIConnectionError(self.endpoint) from e
        except PermissionError as e:
            logger.error("Authentication error in async streaming chat completion | Exception: %s", e)
            raise AuthenticationError() from e
        except ValueError as e:
            logger.error("Invalid response in async streaming chat completion | Exception: %s", e)
            raise InvalidResponseError(str(e)) from e
        except Exception as e:
            content_filter_error = detect_content_filter_error(e, provider="azure_foundry")
            if content_filter_error:
                raise content_filter_error from e
            status_code = getattr(e, "status_code", 500)
            logger.error("Error in AsyncAzureFoundryChatProvider._create_streaming | Exception: %s", e)
            raise ChatCompletionError(str(e), status_code=status_code) from e
        finally:
            if stream is not None:
                await close_async_stream(stream)


class AzureFoundryEmbeddingProvider(BaseEmbeddingProvider):
    """
    Azure Foundry Embedding Provider.
    Adapts Azure-hosted OpenAI embedding models into unified LLM Switch response format.
    """

    def __init__(self, **kwargs):
        """
        Initialize the AzureFoundryEmbeddingProvider with API credentials and deployment info.
        Args:
            **kwargs: Should include 'api_key', 'api_version', 'endpoint', and 'model'.
        """
        self.endpoint: str = kwargs.get("endpoint")
        self.model: str = kwargs.get("deployment_name") or kwargs.get("model")
        self.client: EmbeddingsClient = EmbeddingsClient(**merge_client_options({
            "endpoint": self.endpoint,
            "credential": AzureKeyCredential(kwargs.get("api_key")),
            "api_version": kwargs.get("api_version", "2024-05-01-preview"),
            "retry_total": 0,
            "connection_timeout": kwargs.get("timeout"),
            "read_timeout": kwargs.get("timeout"),
        }, kwargs.get("client_options")))

    def create(self, input: list[str], **kwargs) -> CreateEmbeddingResponse:  # noqa: A002
        """
        Create embeddings for the given input using Azure OpenAI.
        Args:
            input (list[str]): List of input strings to embed.
            **kwargs: Additional parameters for embedding creation.
        Returns:
            CreateEmbeddingResponse: The embedding response object.
        """
        try:
            response = self.client.embed(
                model=kwargs.pop("model", self.model),
                input=input,
                **kwargs,
            )

            # Map response to unified model
            data = [
                Embedding(
                    index=getattr(item, "index", None),
                    embedding=getattr(item, "embedding", None),
                    object=getattr(item, "object", "embedding")
                )
                for item in response.data
            ]

            usage = Usage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
                total_tokens=getattr(response.usage, "total_tokens", 0),
                completion_tokens=getattr(response.usage, "completion_tokens", 0)
            )
            extensions = provider_extensions_from_raw(
                "azure_foundry",
                response,
                {"object", "model", "data", "usage"},
            )

            return CreateEmbeddingResponse(
                object=getattr(response, "object", "list"),
                created=int(time.time()),
                model=getattr(response, "model", None),
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
