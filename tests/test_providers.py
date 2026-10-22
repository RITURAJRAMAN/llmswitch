"""
Provider-level tests. The SDK response objects are real SDK types built from
literal dicts; only the network client is replaced with a mock, so no call
leaves the machine and no credentials are needed.
"""
from unittest.mock import AsyncMock, MagicMock

import anthropic.types
import openai.types
import openai.types.chat
import pytest
from pydantic import BaseModel

from llmswitch import AsyncLLMSwitch, ChatCompletionError, LLMSwitch
from llmswitch.providers.anthropic_provider import AnthropicChatProvider
from llmswitch.providers.azureopenai_provider import AzureOpenAIChatProvider
from llmswitch.providers.openai_provider import AsyncOpenAIChatProvider, OpenAIChatProvider, OpenAIEmbeddingProvider

MESSAGES = [{"role": "user", "content": "Say hi"}]

OPENAI_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1730000000,
    "model": "gpt-4o",
    "choices": [{
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "hi there"},
    }],
    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
}

OPENAI_EMBEDDINGS = {
    "object": "list",
    "model": "text-embedding-3-small",
    "data": [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
    ],
    "usage": {"prompt_tokens": 3, "total_tokens": 3},
}

ANTHROPIC_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-5",
    "content": [{"type": "text", "text": "hi from claude"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 3},
}


def test_openai_chat_maps_the_sdk_response():
    client = LLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o")
    provider = client._provider(OpenAIChatProvider)
    provider.client.chat.completions.create = MagicMock(
        return_value=openai.types.chat.ChatCompletion.model_validate(OPENAI_COMPLETION))

    result = client.chat.completions.create(messages=MESSAGES)

    assert result.choices[0].message.content == "hi there"
    assert result.usage.total_tokens == 6
    sent = provider.client.chat.completions.create.call_args.kwargs
    assert sent["model"] == "gpt-4o"
    assert sent["messages"] == MESSAGES


async def test_openai_async_chat_maps_the_sdk_response():
    client = AsyncLLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o")
    provider = client._provider(AsyncOpenAIChatProvider)
    provider.client.chat.completions.create = AsyncMock(
        return_value=openai.types.chat.ChatCompletion.model_validate(OPENAI_COMPLETION))

    result = await client.chat.completions.create(messages=MESSAGES)

    assert result.choices[0].message.content == "hi there"


def test_openai_embeddings_map_the_sdk_response_in_order():
    client = LLMSwitch(provider="openai", api_key="sk-test")
    provider = client._provider(OpenAIEmbeddingProvider)
    provider.client.embeddings.create = MagicMock(
        return_value=openai.types.CreateEmbeddingResponse.model_validate(OPENAI_EMBEDDINGS))

    result = client.embeddings.create(["first", "second"], model="text-embedding-3-small")

    assert [item.embedding for item in result.data] == [[0.1, 0.2], [0.3, 0.4]]
    assert [item.index for item in result.data] == [0, 1]
    sent = provider.client.embeddings.create.call_args.kwargs
    assert sent["input"] == ["first", "second"]
    assert sent["model"] == "text-embedding-3-small"


def test_azure_openai_chat_uses_the_deployment_as_model():
    client = LLMSwitch(provider="azure_openai", api_key="k", endpoint="https://x.openai.azure.com/",
                       api_version="2024-10-21", deployment_name="prod-gpt4o")
    provider = client._provider(AzureOpenAIChatProvider)
    provider.client.chat.completions.create = MagicMock(
        return_value=openai.types.chat.ChatCompletion.model_validate(OPENAI_COMPLETION))

    result = client.chat.completions.create(messages=MESSAGES)

    assert result.choices[0].message.content == "hi there"
    assert provider.client.chat.completions.create.call_args.kwargs["model"] == "prod-gpt4o"


def test_anthropic_chat_maps_the_sdk_message_and_separates_system_prompt():
    client = LLMSwitch(provider="anthropic", api_key="sk-ant-test", model="claude-opus-4-5")
    provider = client._provider(AnthropicChatProvider)
    provider.client.messages.create = MagicMock(
        return_value=anthropic.types.Message.model_validate(ANTHROPIC_MESSAGE))

    result = client.chat.completions.create(messages=[
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Say hi"},
    ])

    assert result.choices[0].message.content == "hi from claude"
    sent = provider.client.messages.create.call_args.kwargs
    assert sent["model"] == "claude-opus-4-5"
    assert sent["system"]
    assert all(message["role"] != "system" for message in sent["messages"])


def test_provider_sdk_errors_are_translated_to_switch_errors():
    client = LLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o")
    provider = client._provider(OpenAIChatProvider)
    provider.client.chat.completions.create = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(ChatCompletionError):
        client.chat.completions.create(messages=MESSAGES)


OPENAI_CHUNKS = [
    {"id": "c1", "object": "chat.completion.chunk", "created": 1730000000, "model": "gpt-4o",
     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}]},
    {"id": "c1", "object": "chat.completion.chunk", "created": 1730000000, "model": "gpt-4o",
     "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
]


def test_openai_streaming_yields_mapped_chunks():
    client = LLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o")
    provider = client._provider(OpenAIChatProvider)
    raw_chunks = [openai.types.chat.ChatCompletionChunk.model_validate(chunk) for chunk in OPENAI_CHUNKS]
    provider.client.chat.completions.create = MagicMock(return_value=iter(raw_chunks))

    stream = client.chat.completions.create(messages=MESSAGES, stream=True)
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)

    assert text == "Hello"
    assert provider.client.chat.completions.create.call_args.kwargs["stream"] is True
    assert provider.client.chat.completions.create.call_args.kwargs["stream_options"] == {"include_usage": True}


async def test_openai_async_streaming_yields_mapped_chunks():
    client = AsyncLLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o")
    provider = client._provider(AsyncOpenAIChatProvider)

    async def raw_stream():
        for chunk in OPENAI_CHUNKS:
            yield openai.types.chat.ChatCompletionChunk.model_validate(chunk)

    provider.client.chat.completions.create = AsyncMock(return_value=raw_stream())

    stream = await client.chat.completions.create(messages=MESSAGES, stream=True)
    text = ""
    async for chunk in stream:
        text += chunk.choices[0].delta.content or ""

    assert text == "Hello"


class City(BaseModel):
    name: str
    country: str


def test_openai_structured_output_populates_parsed_model():
    client = LLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o")
    provider = client._provider(OpenAIChatProvider)
    body = {
        **OPENAI_COMPLETION,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"name": "Tokyo", "country": "Japan"}',
                        "parsed": {"name": "Tokyo", "country": "Japan"}},
        }],
    }
    provider.client.beta.chat.completions.parse = MagicMock(
        return_value=openai.types.chat.ParsedChatCompletion[City].model_validate(body))

    result = client.chat.completions.parse(MESSAGES, response_format=City)

    assert result.choices[0].message.parsed == City(name="Tokyo", country="Japan")
