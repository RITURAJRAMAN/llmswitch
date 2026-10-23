"""
core.registry

Maps provider names to their chat, async chat, embedding and Responses classes.

Provider SDKs are heavy to import, so a provider's module is imported only when
its entry is first read. If the import fails, the reason is stored in
``UNAVAILABLE_PROVIDERS`` and the entry reads as missing, so the client can say why.
"""
from __future__ import annotations

import importlib
from collections.abc import Iterator, MutableMapping

UNAVAILABLE_PROVIDERS: dict[str, str] = {}


class _LazyRegistry(MutableMapping):
    """A provider-name mapping that imports a provider module on first access."""

    def __init__(self, entries: dict[str, tuple[str, str]]) -> None:
        self._entries = dict(entries)  # provider name -> (module, class name)
        self._loaded: dict[str, type] = {}

    def __getitem__(self, name: str) -> type:
        if name in self._loaded:
            return self._loaded[name]
        module_name, class_name = self._entries[name]
        try:
            module = importlib.import_module(f"..providers.{module_name}", package=__package__)
        except ImportError as exc:
            UNAVAILABLE_PROVIDERS[name] = str(exc)
            del self._entries[name]
            raise KeyError(name) from exc
        self._loaded[name] = getattr(module, class_name)
        return self._loaded[name]

    def __setitem__(self, name: str, provider_cls: type) -> None:
        self._loaded[name] = provider_cls
        self._entries.setdefault(name, ("", ""))

    def __delitem__(self, name: str) -> None:
        if name not in self._entries and name not in self._loaded:
            raise KeyError(name)
        self._entries.pop(name, None)
        self._loaded.pop(name, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({sorted(self._entries)})"


CHAT_PROVIDER_REGISTRY = _LazyRegistry({
    "openai": ("openai_provider", "OpenAIChatProvider"),
    "azure_openai": ("azureopenai_provider", "AzureOpenAIChatProvider"),
    "azure_foundry": ("azurefoundry_provider", "AzureFoundryChatProvider"),
    "anthropic": ("anthropic_provider", "AnthropicChatProvider"),
    "vertexai": ("vertexai_provider", "VertexAIChatProvider"),
})
ASYNC_CHAT_PROVIDER_REGISTRY = _LazyRegistry({
    "openai": ("openai_provider", "AsyncOpenAIChatProvider"),
    "azure_openai": ("azureopenai_provider", "AsyncAzureOpenAIChatProvider"),
    "azure_foundry": ("azurefoundry_provider", "AsyncAzureFoundryChatProvider"),
    "anthropic": ("anthropic_provider", "AsyncAnthropicChatProvider"),
})
EMBEDDING_PROVIDER_REGISTRY = _LazyRegistry({
    "openai": ("openai_provider", "OpenAIEmbeddingProvider"),
    "azure_openai": ("azureopenai_provider", "AzureOpenAIEmbeddingProvider"),
    "azure_foundry": ("azurefoundry_provider", "AzureFoundryEmbeddingProvider"),
})
RESPONSES_PROVIDER_REGISTRY = _LazyRegistry({
    "openai": ("openai_provider", "OpenAIResponsesProvider"),
    "azure_openai": ("azureopenai_provider", "AzureOpenAIResponsesProvider"),
})
ASYNC_RESPONSES_PROVIDER_REGISTRY = _LazyRegistry({
    "openai": ("openai_provider", "AsyncOpenAIResponsesProvider"),
    "azure_openai": ("azureopenai_provider", "AsyncAzureOpenAIResponsesProvider"),
})
