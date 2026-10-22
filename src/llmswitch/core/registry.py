"""
core.registry

Maps provider names to their chat, async chat and embedding provider classes.
Each provider SDK is an optional extra, so every import is guarded: a missing
SDK leaves that provider out of the registry instead of breaking the package.
"""
from __future__ import annotations

CHAT_PROVIDER_REGISTRY: dict[str, type] = {}
ASYNC_CHAT_PROVIDER_REGISTRY: dict[str, type] = {}
EMBEDDING_PROVIDER_REGISTRY: dict[str, type] = {}
RESPONSES_PROVIDER_REGISTRY: dict[str, type] = {}
ASYNC_RESPONSES_PROVIDER_REGISTRY: dict[str, type] = {}

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
except ImportError:
    pass

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
except ImportError:
    pass

try:
    from ..providers.azurefoundry_provider import (
        AsyncAzureFoundryChatProvider,
        AzureFoundryChatProvider,
        AzureFoundryEmbeddingProvider,
    )

    CHAT_PROVIDER_REGISTRY["azure_foundry"] = AzureFoundryChatProvider
    ASYNC_CHAT_PROVIDER_REGISTRY["azure_foundry"] = AsyncAzureFoundryChatProvider
    EMBEDDING_PROVIDER_REGISTRY["azure_foundry"] = AzureFoundryEmbeddingProvider
except ImportError:
    pass

try:
    from ..providers.anthropic_provider import AnthropicChatProvider, AsyncAnthropicChatProvider

    CHAT_PROVIDER_REGISTRY["anthropic"] = AnthropicChatProvider
    ASYNC_CHAT_PROVIDER_REGISTRY["anthropic"] = AsyncAnthropicChatProvider
except ImportError:
    pass

try:
    from ..providers.vertexai_provider import VertexAIChatProvider

    CHAT_PROVIDER_REGISTRY["vertexai"] = VertexAIChatProvider
except ImportError:
    pass
