"""
core.registry

Maps provider names to their chat, async chat, embedding and Responses classes.
Every provider SDK is a core dependency, but each import is still guarded so a broken
installation only disables that provider. The reason is kept in UNAVAILABLE_PROVIDERS
so the client can report it.
"""
from __future__ import annotations

CHAT_PROVIDER_REGISTRY: dict[str, type] = {}
ASYNC_CHAT_PROVIDER_REGISTRY: dict[str, type] = {}
EMBEDDING_PROVIDER_REGISTRY: dict[str, type] = {}
RESPONSES_PROVIDER_REGISTRY: dict[str, type] = {}
ASYNC_RESPONSES_PROVIDER_REGISTRY: dict[str, type] = {}
UNAVAILABLE_PROVIDERS: dict[str, str] = {}

try:
    from ..providers.openai_provider import (
        AsyncOpenAIChatProvider,
        AsyncOpenAIResponsesProvider,
        OpenAIChatProvider,
        OpenAIEmbeddingProvider,
        OpenAIResponsesProvider,
    )

    CHAT_PROVIDER_REGISTRY["openai"] = OpenAIChatProvider
    ASYNC_CHAT_PROVIDER_REGISTRY["openai"] = AsyncOpenAIChatProvider
    EMBEDDING_PROVIDER_REGISTRY["openai"] = OpenAIEmbeddingProvider
    RESPONSES_PROVIDER_REGISTRY["openai"] = OpenAIResponsesProvider
    ASYNC_RESPONSES_PROVIDER_REGISTRY["openai"] = AsyncOpenAIResponsesProvider
except ImportError as exc:
    UNAVAILABLE_PROVIDERS["openai"] = str(exc)

try:
    from ..providers.azureopenai_provider import (
        AsyncAzureOpenAIChatProvider,
        AsyncAzureOpenAIResponsesProvider,
        AzureOpenAIChatProvider,
        AzureOpenAIEmbeddingProvider,
        AzureOpenAIResponsesProvider,
    )

    CHAT_PROVIDER_REGISTRY["azure_openai"] = AzureOpenAIChatProvider
    ASYNC_CHAT_PROVIDER_REGISTRY["azure_openai"] = AsyncAzureOpenAIChatProvider
    EMBEDDING_PROVIDER_REGISTRY["azure_openai"] = AzureOpenAIEmbeddingProvider
    RESPONSES_PROVIDER_REGISTRY["azure_openai"] = AzureOpenAIResponsesProvider
    ASYNC_RESPONSES_PROVIDER_REGISTRY["azure_openai"] = AsyncAzureOpenAIResponsesProvider
except ImportError as exc:
    UNAVAILABLE_PROVIDERS["azure_openai"] = str(exc)

try:
    from ..providers.azurefoundry_provider import (
        AsyncAzureFoundryChatProvider,
        AzureFoundryChatProvider,
        AzureFoundryEmbeddingProvider,
    )

    CHAT_PROVIDER_REGISTRY["azure_foundry"] = AzureFoundryChatProvider
    ASYNC_CHAT_PROVIDER_REGISTRY["azure_foundry"] = AsyncAzureFoundryChatProvider
    EMBEDDING_PROVIDER_REGISTRY["azure_foundry"] = AzureFoundryEmbeddingProvider
except ImportError as exc:
    UNAVAILABLE_PROVIDERS["azure_foundry"] = str(exc)

try:
    from ..providers.anthropic_provider import AnthropicChatProvider, AsyncAnthropicChatProvider

    CHAT_PROVIDER_REGISTRY["anthropic"] = AnthropicChatProvider
    ASYNC_CHAT_PROVIDER_REGISTRY["anthropic"] = AsyncAnthropicChatProvider
except ImportError as exc:
    UNAVAILABLE_PROVIDERS["anthropic"] = str(exc)

try:
    from ..providers.vertexai_provider import VertexAIChatProvider

    CHAT_PROVIDER_REGISTRY["vertexai"] = VertexAIChatProvider
except ImportError as exc:
    UNAVAILABLE_PROVIDERS["vertexai"] = str(exc)
