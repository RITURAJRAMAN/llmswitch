"""Native async embeddings: the Foundry provider and the async pipeline around it."""
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import StubEmbeddings, provider_for
from test_contract import FOUNDRY, EmbeddingsResult

from llmswitch import APIConnectionError, AsyncLLMSwitch, EmbeddingGenerationError
from llmswitch.providers.azurefoundry_provider import AsyncAzureFoundryEmbeddingProvider, AzureFoundryEmbeddingProvider
from llmswitch.providers.azureopenai_provider import AsyncAzureOpenAIEmbeddingProvider
from llmswitch.providers.openai_provider import AsyncOpenAIEmbeddingProvider


def sdk_result(count=1):
    return EmbeddingsResult({
        "id": "e", "model": "m", "usage": {"prompt_tokens": 2, "total_tokens": 2},
        "data": [{"index": i, "embedding": [0.1 * (i + 1)]} for i in range(count)]})


def foundry_client(user_id, **options):
    client = AsyncLLMSwitch(**FOUNDRY, max_retries=0, user_id=user_id, **options)
    provider = provider_for(client, AsyncAzureFoundryEmbeddingProvider)
    return client, provider


async def test_foundry_embeddings_use_the_native_async_client():
    client, provider = foundry_client("aemb-native")
    provider.client.embed = AsyncMock(return_value=sdk_result())
    result = await client.embeddings.create("hello")
    assert [item.embedding for item in result.data] == [[0.1]]
    assert result.usage.total_tokens == 2
    provider.client.embed.assert_awaited_once()


async def test_foundry_embeddings_do_not_start_a_worker_thread_for_the_provider(monkeypatch):
    client, provider = foundry_client("aemb-nothread")
    threads = []

    async def embed(**kwargs):
        threads.append(threading.current_thread() is threading.main_thread())
        return sdk_result()

    provider.client.embed = embed
    await client.embeddings.create("hello")
    assert threads == [True]


async def test_foundry_batches_keep_input_order():
    client, provider = foundry_client("aemb-batches")
    provider.client.embed = AsyncMock(side_effect=lambda **kw: sdk_result(len(kw["input"])))
    result = await client.embeddings.create(["a", "b", "c"], batching=True, batchsize=2)
    assert [item.index for item in result.data] == [0, 1, 2]
    assert provider.client.embed.await_count == 2


async def test_foundry_errors_are_mapped():
    client, provider = foundry_client("aemb-error")
    provider.client.embed = AsyncMock(side_effect=RuntimeError("upstream down"))
    with pytest.raises(EmbeddingGenerationError, match="upstream down"):
        await client.embeddings.create("hello")


async def test_foundry_connection_errors_are_retried_then_raised(monkeypatch):
    client, provider = foundry_client("aemb-conn")
    provider.client.embed = AsyncMock(side_effect=ConnectionError("down"))
    with pytest.raises(APIConnectionError):
        await client.embeddings.create("hello")


async def test_failover_goes_from_a_native_async_primary_to_a_sync_backup(stubs, monkeypatch):
    client, provider = foundry_client(
        "aemb-failover", backup={"provider": "openai", "api_key": "k2", "model": "text-embedding-3-small"})
    provider.client.embed = AsyncMock(side_effect=ConnectionError("down"))
    result = await client.embeddings.create(["a", "bb"])
    assert [item.embedding for item in result.data] == [[1.0], [2.0]]
    assert StubEmbeddings.calls


async def test_async_embeddings_report_events_for_the_native_path():
    events = []
    client, provider = foundry_client("aemb-events", on_event=events.append)
    provider.client.embed = AsyncMock(return_value=sdk_result())
    await client.embeddings.create("hello")
    assert [e["event"] for e in events] == ["request", "response"]
    assert events[0]["operation"] == "embeddings"


async def test_aclose_closes_the_async_embedding_client():
    client, provider = foundry_client("aemb-close")
    provider.client.close = AsyncMock()
    await client.aclose()
    provider.client.close.assert_awaited_once()


def test_the_sync_provider_still_maps_through_the_shared_mixin():
    from llmswitch import LLMSwitch
    client = LLMSwitch(**FOUNDRY, user_id="aemb-sync")
    provider = provider_for(client, AzureFoundryEmbeddingProvider)
    provider.client.embed = MagicMock(return_value=sdk_result())
    assert client.embeddings.create("hello").data[0].embedding == [0.1]


def openai_style_reply(count, model="text-embedding-3-small"):
    """A reply shaped like the OpenAI and Azure OpenAI SDKs return for embeddings."""
    return SimpleNamespace(
        object="list", model=model, usage=SimpleNamespace(prompt_tokens=2, total_tokens=2),
        data=[SimpleNamespace(index=i, embedding=[0.1 * (i + 1)], object="embedding") for i in range(count)])


async def test_openai_embeddings_use_the_native_async_client():
    client = AsyncLLMSwitch(provider="openai", api_key="k", model="text-embedding-3-small", user_id="aemb-openai")
    provider = provider_for(client, AsyncOpenAIEmbeddingProvider)
    provider.client.embeddings.create = AsyncMock(side_effect=lambda **kw: openai_style_reply(len(kw["input"])))
    result = await client.embeddings.create(["a", "bb"])
    assert [item.embedding for item in result.data] == [[0.1], [0.2]]
    assert result.usage.total_tokens == 2
    provider.client.embeddings.create.assert_awaited_once()


async def test_azure_openai_embeddings_use_the_native_async_client():
    client = AsyncLLMSwitch(provider="azure_openai", api_key="k", endpoint="https://x.openai.azure.com",
                            api_version="2024-10-21", deployment_name="emb", model="text-embedding-3-small",
                            user_id="aemb-azure-openai")
    provider = provider_for(client, AsyncAzureOpenAIEmbeddingProvider)
    provider.client.embeddings.create = AsyncMock(side_effect=lambda **kw: openai_style_reply(len(kw["input"])))
    result = await client.embeddings.create(["a", "bb"])
    assert [item.index for item in result.data] == [0, 1]
    provider.client.embeddings.create.assert_awaited_once()


async def test_openai_async_close_closes_the_native_client():
    client = AsyncLLMSwitch(provider="openai", api_key="k", model="text-embedding-3-small", user_id="aemb-openai-close")
    provider = provider_for(client, AsyncOpenAIEmbeddingProvider)
    provider.client.close = AsyncMock()
    await client.aclose()
    provider.client.close.assert_awaited_once()
