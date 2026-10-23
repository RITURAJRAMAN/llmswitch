"""Native async embeddings: the Foundry provider and the async pipeline around it."""
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import StubEmbeddings, provider_for
from test_contract import FOUNDRY, EmbeddingsResult

from llmswitch import APIConnectionError, AsyncLLMSwitch, EmbeddingGenerationError
from llmswitch.core.registry import ASYNC_EMBEDDING_PROVIDER_REGISTRY
from llmswitch.providers.azurefoundry_provider import AsyncAzureFoundryEmbeddingProvider, AzureFoundryEmbeddingProvider


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


async def test_a_provider_without_an_async_class_still_works_in_a_thread(stubs):
    assert "openai" not in ASYNC_EMBEDDING_PROVIDER_REGISTRY
    client = AsyncLLMSwitch(provider="openai", api_key="k", model="gpt-4o", user_id="aemb-fallback")
    result = await client.embeddings.create(["a", "bb"])
    assert [item.embedding for item in result.data] == [[1.0], [2.0]]


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
