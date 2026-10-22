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
    normalize_embedding_input,
    provider_kwargs,
    resolve_model,
    validate_messages,
)
from .registry import CHAT_PROVIDER_REGISTRY, EMBEDDING_PROVIDER_REGISTRY

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


class _Chat:
    def __init__(self, client: LLMSwitch) -> None:
        self.completions = _ChatCompletions(client)
