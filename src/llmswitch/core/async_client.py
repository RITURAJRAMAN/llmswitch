"""
core.async_client

Asynchronous ``AsyncLLMSwitch`` client. Chat completions use each provider's
native async client. Embedding providers are synchronous, so they run in a
worker thread to keep the event loop free.

Native async chat is available for azure_openai, azure_foundry, openai and
anthropic. vertexai is sync-only in this release.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..errors import LLMSwitchError
from .common import (
    build_config,
    map_chat_error,
    map_embedding_error,
    map_responses_error,
    normalize_embedding_input,
    prepare_responses_create,
    prepare_responses_management,
    provider_kwargs,
    resolve_model,
    responses_provider_class,
    validate_messages,
)
from .registry import (
    ASYNC_CHAT_PROVIDER_REGISTRY,
    ASYNC_RESPONSES_PROVIDER_REGISTRY,
    EMBEDDING_PROVIDER_REGISTRY,
)

logger = logging.getLogger(__name__)


class _AsyncChatCompletions:
    def __init__(self, client: "AsyncLLMSwitch") -> None:
        self._client = client

    async def create(self, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
        """Async counterpart of ``LLMSwitch.chat.completions.create``.

        With ``stream=True`` the awaited result is an async generator of chunks.
        """
        config = self._client.config
        validate_messages(messages)
        provider_cls = ASYNC_CHAT_PROVIDER_REGISTRY.get(config.provider)
        if provider_cls is None:
            raise LLMSwitchError(f"Unsupported provider for async chat completion: {config.provider}",
                                 provider=config.provider)
        if stream:
            kwargs.setdefault("stream_options", {"include_usage": True})
        resolve_model(config, kwargs)
        provider = self._client._provider(provider_cls)
        try:
            return await provider.create(messages=messages, stream=stream, **kwargs)
        except Exception as exc:
            raise map_chat_error(exc, config) from exc

    async def parse(self, messages: list[dict[str, Any]], response_format: Any, **kwargs: Any) -> Any:
        """Async structured output. See ``LLMSwitch.chat.completions.parse``."""
        return await self.create(messages, stream=False, response_format=response_format, **kwargs)


class _AsyncEmbeddings:
    def __init__(self, client: "AsyncLLMSwitch") -> None:
        self._client = client

    async def create(self, input: str | list[str], **kwargs: Any) -> Any:  # noqa: A002
        """Create embeddings without blocking the event loop."""
        config = self._client.config
        inputs = normalize_embedding_input(input)
        provider_cls = EMBEDDING_PROVIDER_REGISTRY.get(config.provider)
        if provider_cls is None:
            raise LLMSwitchError(f"Unsupported provider for embeddings: {config.provider}",
                                 provider=config.provider)
        resolve_model(config, kwargs)
        provider = self._client._provider(provider_cls)
        try:
            return await asyncio.to_thread(provider.create, inputs, **kwargs)
        except Exception as exc:
            raise map_embedding_error(exc, config) from exc


class AsyncLLMSwitch:
    """
    Unified asynchronous client, mirroring ``LLMSwitch``.

    Usage::

        client = AsyncLLMSwitch(provider="openai", api_key="...", model="gpt-4o")
        reply = await client.chat.completions.create(messages=[{"role": "user", "content": "Hi"}])
        vectors = await client.embeddings.create(["hello", "world"])
        await client.aclose()
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = build_config(kwargs)
        self._providers: dict[type, Any] = {}
        self.chat = _AsyncChat(self)
        self.embeddings = _AsyncEmbeddings(self)
        self.responses = _AsyncResponses(self)

    def __repr__(self) -> str:
        return f"AsyncLLMSwitch(provider={self.config.provider!r}, model={self.config.model!r})"

    async def __aenter__(self) -> "AsyncLLMSwitch":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def _provider(self, provider_cls: type) -> Any:
        """Return the cached provider client for ``provider_cls``, building it on first use."""
        provider = self._providers.get(provider_cls)
        if provider is None:
            provider = provider_cls(**provider_kwargs(self.config))
            self._providers[provider_cls] = provider
        return provider

    async def aclose(self) -> None:
        """Close the provider clients this client created, awaiting async close hooks."""
        for provider in self._providers.values():
            close_fn = getattr(provider, "aclose", None) or getattr(provider, "close", None)
            if close_fn is None:
                close_fn = getattr(getattr(provider, "client", None), "aclose", None) or getattr(
                    getattr(provider, "client", None), "close", None)
            if callable(close_fn):
                result = close_fn()
                if asyncio.iscoroutine(result):
                    await result
        self._providers.clear()


class _AsyncResponseInputItems:
    def __init__(self, responses: "_AsyncResponses") -> None:
        self._responses = responses

    async def list(self, response_id: str, **kwargs: Any) -> Any:
        return await self._responses._call("list_input_items", response_id, **kwargs)


class _AsyncResponseInputTokens:
    def __init__(self, responses: "_AsyncResponses") -> None:
        self._responses = responses

    async def count(self, **kwargs: Any) -> Any:
        return await self._responses._call("count_input_tokens", **kwargs)


class _AsyncConversations:
    def __init__(self, responses: "_AsyncResponses") -> None:
        self._responses = responses

    async def create(self, items: Any = None, **kwargs: Any) -> Any:
        return await self._responses._call("create_conversation", items, **kwargs)

    async def retrieve(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._responses._call("retrieve_conversation", conversation_id, **kwargs)

    async def update(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._responses._call("update_conversation", conversation_id, **kwargs)

    async def delete(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._responses._call("delete_conversation", conversation_id, **kwargs)

    async def list_items(self, conversation_id: str, **kwargs: Any) -> Any:
        return await self._responses._call("list_conversation_items", conversation_id, **kwargs)

    async def retrieve_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return await self._responses._call("retrieve_conversation_item", conversation_id, item_id, **kwargs)

    async def delete_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return await self._responses._call("delete_conversation_item", conversation_id, item_id, **kwargs)


class _AsyncResponses:
    """Async OpenAI-style ``responses`` namespace. Supported for openai and azure_openai."""

    def __init__(self, client: "AsyncLLMSwitch") -> None:
        self._client = client
        self.input_items = _AsyncResponseInputItems(self)
        self.input_tokens = _AsyncResponseInputTokens(self)
        self.conversations = _AsyncConversations(self)

    async def create(self, input: Any = None, stream: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        """Create a model response. With ``stream=True`` returns a generator of response events."""
        config = self._client.config
        provider_cls = responses_provider_class(config, ASYNC_RESPONSES_PROVIDER_REGISTRY)
        prepare_responses_create(config, input, kwargs)
        provider = self._client._provider(provider_cls)
        try:
            return await provider.create(input, stream=stream, **kwargs)
        except Exception as exc:
            raise map_responses_error(exc, config, "create") from exc

    async def stream(self, input: Any = None, **kwargs: Any) -> Any:  # noqa: A002
        return await self.create(input, stream=True, **kwargs)

    async def parse(self, input: Any = None, text_format: Any = None, **kwargs: Any) -> Any:  # noqa: A002
        """Structured output. Pass a Pydantic model class as ``text_format``."""
        return await self.create(input, stream=False, text_format=text_format, **kwargs)

    async def retrieve(self, response_id: str, **kwargs: Any) -> Any:
        return await self._call("retrieve", response_id, **kwargs)

    async def delete(self, response_id: str, **kwargs: Any) -> Any:
        return await self._call("delete", response_id, **kwargs)

    async def cancel(self, response_id: str, **kwargs: Any) -> Any:
        return await self._call("cancel", response_id, **kwargs)

    async def compact(self, **kwargs: Any) -> Any:
        return await self._call("compact", **kwargs)

    async def _call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        config = self._client.config
        provider_cls = responses_provider_class(config, ASYNC_RESPONSES_PROVIDER_REGISTRY)
        prepare_responses_management(config, operation, kwargs)
        provider = self._client._provider(provider_cls)
        try:
            return await getattr(provider, operation)(*args, **kwargs)
        except Exception as exc:
            raise map_responses_error(exc, config, operation) from exc


class _AsyncChat:
    def __init__(self, client: AsyncLLMSwitch) -> None:
        self.completions = _AsyncChatCompletions(client)
