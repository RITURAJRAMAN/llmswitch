"""Contract tests: every provider maps its own SDK's response and errors the same way.

Each case mocks only the provider's network client and returns a real SDK response
object, so the provider's mapping code runs for real.
"""
from unittest.mock import AsyncMock, MagicMock

import anthropic.types
import openai.types
import openai.types.chat
import pytest
from azure.ai.inference.models import ChatCompletions, EmbeddingsResult
from conftest import provider_for
from google.genai import types as genai_types

from llmswitch import AsyncLLMSwitch, ChatCompletionError, LLMSwitch
from llmswitch.providers import vertexai_provider
from llmswitch.providers.anthropic_provider import AnthropicChatProvider, AsyncAnthropicChatProvider
from llmswitch.providers.azurefoundry_provider import (
    AsyncAzureFoundryChatProvider,
    AzureFoundryChatProvider,
    AzureFoundryEmbeddingProvider,
)
from llmswitch.providers.azureopenai_provider import (
    AsyncAzureOpenAIChatProvider,
    AzureOpenAIChatProvider,
    AzureOpenAIEmbeddingProvider,
)
from llmswitch.providers.openai_provider import AsyncOpenAIChatProvider, OpenAIChatProvider, OpenAIEmbeddingProvider

TEXT = "hello from the provider"
MESSAGES = [{"role": "user", "content": "Hi"}]


class UpstreamError(Exception):
    status_code = 503


def openai_completion():
    return openai.types.chat.ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 1730000000, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": TEXT}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    })


def anthropic_message():
    return anthropic.types.Message.model_validate({
        "id": "m1", "type": "message", "role": "assistant", "model": "claude-opus-4-5",
        "content": [{"type": "text", "text": TEXT}], "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 3},
    })


def foundry_completion():
    return ChatCompletions({
        "id": "f1", "created": 1730000000, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": TEXT}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    })


def vertex_response():
    return genai_types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": TEXT}]}, "finish_reason": "STOP", "index": 0}],
        "usage_metadata": {"prompt_token_count": 4, "candidates_token_count": 3, "total_token_count": 7},
        "model_version": "gemini-2.5-pro",
    })


AZURE = dict(provider="azure_openai", api_key="k", endpoint="https://x.openai.azure.com/",
             api_version="2025-04-01-preview", deployment_name="prod")
FOUNDRY = dict(provider="azure_foundry", api_key="k", endpoint="https://x.services.ai.azure.com/models",
               api_version="2024-05-01-preview", deployment_name="m")


def sync_case(name, monkeypatch):
    """Return (client, install) where install(response, error) mocks the provider's SDK call."""
    if name == "openai":
        client = LLMSwitch(provider="openai", api_key="k", model="gpt-4o", max_retries=0, user_id="contract-openai")
        provider = provider_for(client, OpenAIChatProvider)
        target = provider.client.chat.completions
        return client, lambda response, error: setattr(target, "create", MagicMock(return_value=response, side_effect=error))
    if name == "azure_openai":
        client = LLMSwitch(**AZURE, max_retries=0, user_id="contract-azure")
        provider = provider_for(client, AzureOpenAIChatProvider)
        target = provider.client.chat.completions
        return client, lambda response, error: setattr(target, "create", MagicMock(return_value=response, side_effect=error))
    if name == "anthropic":
        client = LLMSwitch(provider="anthropic", api_key="k", model="claude-opus-4-5", max_retries=0,
                           user_id="contract-anthropic")
        provider = provider_for(client, AnthropicChatProvider)
        target = provider.client.messages
        return client, lambda response, error: setattr(target, "create", MagicMock(return_value=response, side_effect=error))
    if name == "azure_foundry":
        client = LLMSwitch(**FOUNDRY, max_retries=0, user_id="contract-foundry")
        provider = provider_for(client, AzureFoundryChatProvider)
        return client, lambda response, error: setattr(
            provider.client, "complete", MagicMock(return_value=response, side_effect=error))
    if name == "vertexai":
        fake = MagicMock()
        monkeypatch.setattr(vertexai_provider, "load_credentials_from_file", lambda path, scopes=None: (object(), "project"))
        monkeypatch.setattr(vertexai_provider.genai, "Client", lambda **kwargs: fake)
        client = LLMSwitch(provider="vertexai", location="us-central1", credentials_path="/not/used.json",
                           model="gemini-2.5-pro", max_retries=0, user_id="contract-vertex")
        return client, lambda response, error: setattr(
            fake.models, "generate_content", MagicMock(return_value=response, side_effect=error))
    raise AssertionError(name)


RESPONSES = {"openai": openai_completion, "azure_openai": openai_completion, "anthropic": anthropic_message,
             "azure_foundry": foundry_completion, "vertexai": vertex_response}
PROVIDERS = list(RESPONSES)


@pytest.mark.parametrize("name", PROVIDERS)
def test_chat_returns_the_same_shape_for_every_provider(name, monkeypatch):
    client, install = sync_case(name, monkeypatch)
    install(RESPONSES[name](), None)

    result = client.chat.completions.create(messages=MESSAGES)

    assert result.choices[0].message.content == TEXT
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (4, 3, 7)


@pytest.mark.parametrize("name", PROVIDERS)
def test_a_provider_failure_becomes_a_chat_completion_error(name, monkeypatch):
    client, install = sync_case(name, monkeypatch)
    install(None, UpstreamError("upstream is down"))

    with pytest.raises(ChatCompletionError) as info:
        client.chat.completions.create(messages=MESSAGES)

    assert info.value.status_code == 503
    assert "upstream is down" in str(info.value)


@pytest.mark.parametrize("name", PROVIDERS)
def test_closing_the_client_releases_the_provider(name, monkeypatch):
    client, install = sync_case(name, monkeypatch)
    install(RESPONSES[name](), None)
    client.chat.completions.create(messages=MESSAGES)
    client.close()
    assert client.chat.completions._provider_cache == {}


def embedding_case(name):
    if name == "openai":
        client = LLMSwitch(provider="openai", api_key="k", model="text-embedding-3-small", user_id="contract-e-openai")
        provider = provider_for(client, OpenAIEmbeddingProvider)
        response = openai.types.CreateEmbeddingResponse.model_validate({
            "object": "list", "model": "m", "usage": {"prompt_tokens": 2, "total_tokens": 2},
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]})
        provider.client.embeddings.create = MagicMock(return_value=response)
    elif name == "azure_openai":
        client = LLMSwitch(**AZURE, user_id="contract-e-azure")
        provider = provider_for(client, AzureOpenAIEmbeddingProvider)
        response = openai.types.CreateEmbeddingResponse.model_validate({
            "object": "list", "model": "m", "usage": {"prompt_tokens": 2, "total_tokens": 2},
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]})
        provider.client.embeddings.create = MagicMock(return_value=response)
    else:
        client = LLMSwitch(**FOUNDRY, user_id="contract-e-foundry")
        provider = provider_for(client, AzureFoundryEmbeddingProvider)
        provider.client.embed = MagicMock(return_value=EmbeddingsResult({
            "id": "e", "model": "m", "usage": {"prompt_tokens": 2, "total_tokens": 2},
            "data": [{"index": 0, "embedding": [0.1, 0.2]}]}))
    return client


@pytest.mark.parametrize("name", ["openai", "azure_openai", "azure_foundry"])
def test_embeddings_return_the_same_shape_for_every_provider(name):
    result = embedding_case(name).embeddings.create("hello")
    assert [item.embedding for item in result.data] == [[0.1, 0.2]]
    assert [item.index for item in result.data] == [0]
    assert result.usage.total_tokens == 2


def async_case(name):
    if name == "openai":
        client = AsyncLLMSwitch(provider="openai", api_key="k", model="gpt-4o", max_retries=0, user_id="contract-a-openai")
        provider = provider_for(client, AsyncOpenAIChatProvider)
        return client, lambda r, e: setattr(provider.client.chat.completions, "create", AsyncMock(return_value=r, side_effect=e))
    if name == "azure_openai":
        client = AsyncLLMSwitch(**AZURE, max_retries=0, user_id="contract-a-azure")
        provider = provider_for(client, AsyncAzureOpenAIChatProvider)
        return client, lambda r, e: setattr(provider.client.chat.completions, "create", AsyncMock(return_value=r, side_effect=e))
    if name == "anthropic":
        client = AsyncLLMSwitch(provider="anthropic", api_key="k", model="claude-opus-4-5", max_retries=0,
                                user_id="contract-a-anthropic")
        provider = provider_for(client, AsyncAnthropicChatProvider)
        return client, lambda r, e: setattr(provider.client.messages, "create", AsyncMock(return_value=r, side_effect=e))
    client = AsyncLLMSwitch(**FOUNDRY, max_retries=0, user_id="contract-a-foundry")
    provider = provider_for(client, AsyncAzureFoundryChatProvider)
    return client, lambda r, e: setattr(provider.client, "complete", AsyncMock(return_value=r, side_effect=e))


ASYNC_PROVIDERS = ["openai", "azure_openai", "anthropic", "azure_foundry"]


@pytest.mark.parametrize("name", ASYNC_PROVIDERS)
async def test_async_chat_returns_the_same_shape_for_every_provider(name):
    client, install = async_case(name)
    install(RESPONSES[name](), None)

    result = await client.chat.completions.create(messages=MESSAGES)

    assert result.choices[0].message.content == TEXT
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (4, 3, 7)


@pytest.mark.parametrize("name", ASYNC_PROVIDERS)
async def test_async_provider_failures_become_chat_completion_errors(name):
    client, install = async_case(name)
    install(None, UpstreamError("upstream is down"))

    with pytest.raises(ChatCompletionError) as info:
        await client.chat.completions.create(messages=MESSAGES)

    assert info.value.status_code == 503
