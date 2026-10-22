import threading

import pytest

from llmswitch import AsyncLLMSwitch, ChatCompletionError, EmbeddingGenerationError, LLMSwitchError

MESSAGES = [{"role": "user", "content": "Hello"}]


def openai_client():
    return AsyncLLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o")


async def test_async_chat_returns_awaited_result(stubs):
    StubChat, _ = stubs
    result = await openai_client().chat.completions.create(messages=MESSAGES)
    assert result == {"ok": True, "stream": False}
    assert StubChat.calls[-1]["model"] == "gpt-4o"


async def test_async_streaming_returns_an_async_generator(stubs):
    client = openai_client()
    stream = await client.chat.completions.create(messages=MESSAGES, stream=True)
    chunks = [chunk async for chunk in stream]
    assert chunks == [{"chunk": 1}, {"chunk": 2}]


async def test_async_parse_passes_response_format(stubs):
    StubChat, _ = stubs

    class Answer:
        pass

    await openai_client().chat.completions.parse(MESSAGES, response_format=Answer)
    assert StubChat.calls[-1]["response_format"] is Answer


async def test_async_invalid_messages_are_rejected(stubs):
    with pytest.raises(ValueError):
        await openai_client().chat.completions.create(messages=[])


async def test_async_chat_is_unsupported_for_vertexai():
    client = AsyncLLMSwitch(provider="vertexai", location="us-central1", credentials_path="/tmp/sa.json")
    with pytest.raises(LLMSwitchError, match="Unsupported provider for async chat completion: vertexai"):
        await client.chat.completions.create(messages=MESSAGES)


async def test_async_chat_errors_are_mapped(stubs):
    StubChat, _ = stubs
    StubChat.error = ValueError("bad request")
    with pytest.raises(ChatCompletionError) as info:
        await openai_client().chat.completions.create(messages=MESSAGES)
    assert info.value.status_code == 400


async def test_async_embeddings_run_the_sync_provider_off_the_event_loop(stubs):
    _, StubEmbeddings = stubs
    seen_threads = []
    original_create = StubEmbeddings.create

    def tracking_create(self, input, **kwargs):  # noqa: A002
        seen_threads.append(threading.current_thread() is not threading.main_thread())
        return original_create(self, input, **kwargs)

    StubEmbeddings.create = tracking_create
    try:
        result = await openai_client().embeddings.create(["a", "bb"])
    finally:
        StubEmbeddings.create = original_create
    assert result["data"] == [[1.0], [2.0]]
    assert seen_threads == [True]


async def test_async_embedding_errors_are_mapped(stubs):
    _, StubEmbeddings = stubs
    StubEmbeddings.error = RuntimeError("upstream down")
    with pytest.raises(EmbeddingGenerationError):
        await openai_client().embeddings.create("hello")


async def test_async_embeddings_are_unsupported_for_anthropic():
    client = AsyncLLMSwitch(provider="anthropic", api_key="k")
    with pytest.raises(LLMSwitchError, match="Unsupported provider for embeddings: anthropic"):
        await client.embeddings.create("hello")


async def test_aclose_awaits_async_close_hooks(stubs):
    StubChat, _ = stubs
    async with openai_client() as client:
        await client.chat.completions.create(messages=MESSAGES)
    assert StubChat.closed is True
    assert client._providers == {}


def test_async_client_construction_is_sync_safe():
    client = openai_client()
    assert client.config.provider == "openai"
    assert repr(client).startswith("AsyncLLMSwitch(")
