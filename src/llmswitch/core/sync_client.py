"""
core.sync_client

Synchronous ``LLMSwitch`` client with an OpenAI-style ``chat.completions`` and
``embeddings`` surface.
"""
from __future__ import annotations

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
from .registry import CHAT_PROVIDER_REGISTRY, EMBEDDING_PROVIDER_REGISTRY, RESPONSES_PROVIDER_REGISTRY

logger = logging.getLogger(__name__)


class _ChatCompletions:
    def __init__(self, client: "LLMSwitch") -> None:
        self._client = client

    def create(self, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any) -> Any:
        """
        Create a chat completion.

        Args:
            messages: Message dicts. ``content`` may be a string or a list of content parts.
            stream: Return a generator of ``ChatCompletionChunk`` instead of a ``ChatCompletion``.
            **kwargs: Pass-through provider parameters (``response_format``, ``tools``,
                ``temperature``, ``max_tokens``, ...). ``model`` overrides the configured model
                unless a deployment name is configured.
        """
        config = self._client.config
        validate_messages(messages)
        provider_cls = CHAT_PROVIDER_REGISTRY.get(config.provider)
        if provider_cls is None:
            raise LLMSwitchError(f"Unsupported provider for chat completion: {config.provider}",
                                 provider=config.provider)
        if stream:
            kwargs.setdefault("stream_options", {"include_usage": True})
        resolve_model(config, kwargs)
        provider = self._client._provider(provider_cls)
        try:
            return provider.create(messages=messages, stream=stream, **kwargs)
        except Exception as exc:
            raise map_chat_error(exc, config) from exc

    def parse(self, messages: list[dict[str, Any]], response_format: Any, **kwargs: Any) -> Any:
        """
        Structured output. Pass a Pydantic model class as ``response_format``; the parsed
        instance is available on ``choice.message.parsed``. Always non-streaming.
        """
        return self.create(messages, stream=False, response_format=response_format, **kwargs)


class _Embeddings:
    def __init__(self, client: "LLMSwitch") -> None:
        self._client = client

    def create(self, input: str | list[str], **kwargs: Any) -> Any:  # noqa: A002
        """Create embeddings for one string or a list of strings."""
        config = self._client.config
        inputs = normalize_embedding_input(input)
        provider_cls = EMBEDDING_PROVIDER_REGISTRY.get(config.provider)
        if provider_cls is None:
            raise LLMSwitchError(f"Unsupported provider for embeddings: {config.provider}",
                                 provider=config.provider)
        resolve_model(config, kwargs)
        provider = self._client._provider(provider_cls)
        try:
            return provider.create(inputs, **kwargs)
        except Exception as exc:
            raise map_embedding_error(exc, config) from exc


class LLMSwitch:
    """
    Unified synchronous client. Construct it with a ``provider`` name plus that
    provider's credentials and settings; the same call shape works across providers.

    Usage::

        client = LLMSwitch(provider="openai", api_key="...", model="gpt-4o")
        reply = client.chat.completions.create(messages=[{"role": "user", "content": "Hi"}])
        vectors = client.embeddings.create(["hello", "world"])
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = build_config(kwargs)
        self._providers: dict[type, Any] = {}
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)
        self.responses = _Responses(self)

    def __repr__(self) -> str:
        return f"LLMSwitch(provider={self.config.provider!r}, model={self.config.model!r})"

    def __enter__(self) -> "LLMSwitch":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _provider(self, provider_cls: type) -> Any:
        """Return the cached provider client for ``provider_cls``, building it on first use."""
        provider = self._providers.get(provider_cls)
        if provider is None:
            provider = provider_cls(**provider_kwargs(self.config))
            self._providers[provider_cls] = provider
        return provider

    def close(self) -> None:
        """Close the provider clients this client created."""
        for provider in self._providers.values():
            close_fn = getattr(provider, "close", None) or getattr(getattr(provider, "client", None), "close", None)
            if callable(close_fn):
                close_fn()
        self._providers.clear()


class _ResponseInputItems:
    def __init__(self, responses: "_Responses") -> None:
        self._responses = responses

    def list(self, response_id: str, **kwargs: Any) -> Any:
        return self._responses._call("list_input_items", response_id, **kwargs)


class _ResponseInputTokens:
    def __init__(self, responses: "_Responses") -> None:
        self._responses = responses

    def count(self, **kwargs: Any) -> Any:
        return self._responses._call("count_input_tokens", **kwargs)


class _Conversations:
    def __init__(self, responses: "_Responses") -> None:
        self._responses = responses

    def create(self, items: Any = None, **kwargs: Any) -> Any:
        return self._responses._call("create_conversation", items, **kwargs)

    def retrieve(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._responses._call("retrieve_conversation", conversation_id, **kwargs)

    def update(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._responses._call("update_conversation", conversation_id, **kwargs)

    def delete(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._responses._call("delete_conversation", conversation_id, **kwargs)

    def list_items(self, conversation_id: str, **kwargs: Any) -> Any:
        return self._responses._call("list_conversation_items", conversation_id, **kwargs)

    def retrieve_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return self._responses._call("retrieve_conversation_item", conversation_id, item_id, **kwargs)

    def delete_item(self, conversation_id: str, item_id: str, **kwargs: Any) -> Any:
        return self._responses._call("delete_conversation_item", conversation_id, item_id, **kwargs)


class _Responses:
    """OpenAI-style ``responses`` namespace. Supported for openai and azure_openai."""

    def __init__(self, client: "LLMSwitch") -> None:
        self._client = client
        self.input_items = _ResponseInputItems(self)
        self.input_tokens = _ResponseInputTokens(self)
        self.conversations = _Conversations(self)

    def create(self, input: Any = None, stream: bool = False, **kwargs: Any) -> Any:  # noqa: A002
        """Create a model response. With ``stream=True`` returns a generator of response events."""
        config = self._client.config
        provider_cls = responses_provider_class(config, RESPONSES_PROVIDER_REGISTRY)
        prepare_responses_create(config, input, kwargs)
        provider = self._client._provider(provider_cls)
        try:
            return provider.create(input, stream=stream, **kwargs)
        except Exception as exc:
            raise map_responses_error(exc, config, "create") from exc

    def stream(self, input: Any = None, **kwargs: Any) -> Any:  # noqa: A002
        return self.create(input, stream=True, **kwargs)

    def parse(self, input: Any = None, text_format: Any = None, **kwargs: Any) -> Any:  # noqa: A002
        """Structured output. Pass a Pydantic model class as ``text_format``."""
        return self.create(input, stream=False, text_format=text_format, **kwargs)

    def retrieve(self, response_id: str, **kwargs: Any) -> Any:
        return self._call("retrieve", response_id, **kwargs)

    def delete(self, response_id: str, **kwargs: Any) -> Any:
        return self._call("delete", response_id, **kwargs)

    def cancel(self, response_id: str, **kwargs: Any) -> Any:
        return self._call("cancel", response_id, **kwargs)

    def compact(self, **kwargs: Any) -> Any:
        return self._call("compact", **kwargs)

    def _call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        config = self._client.config
        provider_cls = responses_provider_class(config, RESPONSES_PROVIDER_REGISTRY)
        prepare_responses_management(config, operation, kwargs)
        provider = self._client._provider(provider_cls)
        try:
            return getattr(provider, operation)(*args, **kwargs)
        except Exception as exc:
            raise map_responses_error(exc, config, operation) from exc


class _Chat:
    def __init__(self, client: LLMSwitch) -> None:
        self.completions = _ChatCompletions(client)
