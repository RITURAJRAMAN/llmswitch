"""Async Vertex AI chat: the same mapping as the sync provider, on the google-genai aio client."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors as google_errors
from google.genai import types as genai_types
from test_vertex import MESSAGES, response

from llmswitch import AsyncLLMSwitch, AuthenticationError, ChatCompletionError, RateLimitExceededError
from llmswitch.providers import vertexai_provider


@pytest.fixture
def vertex(monkeypatch):
    fake = MagicMock()
    fake.aio.models.generate_content = AsyncMock()
    fake.aio.models.generate_content_stream = AsyncMock()
    fake.aio.aclose = AsyncMock()
    monkeypatch.setattr(vertexai_provider, "load_credentials_from_file", lambda path, scopes=None: (object(), "project"))
    monkeypatch.setattr(vertexai_provider.genai, "Client", lambda **kwargs: fake)
    client = AsyncLLMSwitch(provider="vertexai", location="us-central1", credentials_path="/not/used.json",
                            model="gemini-2.5-pro", max_retries=0, user_id="vertex-async-tests")
    return client, fake


async def agen(items):
    for item in items:
        yield item


async def test_async_chat_maps_the_response(vertex):
    client, fake = vertex
    moment = datetime(2024, 10, 27, 3, 33, 20, tzinfo=timezone.utc)
    fake.aio.models.generate_content.return_value = response(create_time=moment, response_id="r-1")

    result = await client.chat.completions.create(messages=MESSAGES, max_completion_tokens=50)

    assert result.choices[0].message.content == "hi"
    assert (result.id, result.created, result.usage.total_tokens) == ("r-1", 1730000000, 7)
    kwargs = fake.aio.models.generate_content.await_args.kwargs
    assert kwargs["model"] == "gemini-2.5-pro"
    assert kwargs["config"].max_output_tokens == 50
    fake.models.generate_content.assert_not_called()


async def test_a_sparse_response_still_maps(vertex):
    client, fake = vertex
    fake.aio.models.generate_content.return_value = response()
    result = await client.chat.completions.create(messages=MESSAGES)
    assert (result.id, result.model) == ("vertexai-response", "gemini-2.5-pro")


async def test_async_tool_calls_are_mapped(vertex):
    client, fake = vertex
    fake.aio.models.generate_content.return_value = genai_types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [
            {"function_call": {"name": "get_weather", "args": {"city": "Paris"}}}]},
            "finish_reason": "STOP", "index": 0}],
    })
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    result = await client.chat.completions.create(messages=MESSAGES, tools=tools)
    call = result.choices[0].message.tool_calls[0]
    assert (call.function.name, call.function.arguments) == ("get_weather", '{"city": "Paris"}')
    assert result.choices[0].finish_reason == "tool_calls"


async def test_async_structured_output_is_parsed(vertex):
    from pydantic import BaseModel

    class Answer(BaseModel):
        city: str

    client, fake = vertex
    fake.aio.models.generate_content.return_value = response('{"city": "Paris"}')
    result = await client.chat.completions.parse(MESSAGES, response_format=Answer)
    assert result.choices[0].message.parsed == Answer(city="Paris")


@pytest.mark.parametrize("code, expected", [(429, RateLimitExceededError), (401, AuthenticationError),
                                            (400, ChatCompletionError)])
async def test_async_client_errors_map_to_library_errors(vertex, code, expected):
    client, fake = vertex
    fake.aio.models.generate_content.side_effect = google_errors.ClientError(code, {"error": {"message": "nope"}})
    with pytest.raises(expected):
        await client.chat.completions.create(messages=MESSAGES)


async def test_async_server_errors_become_503(vertex):
    client, fake = vertex
    fake.aio.models.generate_content.side_effect = google_errors.ServerError(500, {"error": {"message": "down"}})
    with pytest.raises(ChatCompletionError) as info:
        await client.chat.completions.create(messages=MESSAGES)
    assert info.value.status_code == 503


async def test_async_streaming_yields_the_text_in_order_with_usage(vertex):
    client, fake = vertex
    fake.aio.models.generate_content_stream.return_value = agen([response("Hel"), response("lo")])

    stream = await client.chat.completions.create(messages=MESSAGES, stream=True)
    chunks = [chunk async for chunk in stream]

    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Hello"
    assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens == 7


async def test_async_stream_errors_are_mapped(vertex):
    client, fake = vertex

    async def broken():
        yield response("Hel")
        raise google_errors.ServerError(500, {"error": {"message": "dropped"}})

    fake.aio.models.generate_content_stream.return_value = broken()
    stream = await client.chat.completions.create(messages=MESSAGES, stream=True)
    with pytest.raises(ChatCompletionError) as info:
        async for _ in stream:
            pass
    assert info.value.status_code == 503


async def test_a_failure_opening_the_stream_is_mapped(vertex):
    client, fake = vertex
    fake.aio.models.generate_content_stream.side_effect = google_errors.ClientError(
        401, {"error": {"message": "nope"}})
    with pytest.raises(AuthenticationError):
        stream = await client.chat.completions.create(messages=MESSAGES, stream=True)
        async for _ in stream:
            pass


async def test_aclose_closes_the_aio_client(vertex):
    client, fake = vertex
    fake.aio.models.generate_content.return_value = response()
    await client.chat.completions.create(messages=MESSAGES)
    await client.aclose()
    fake.aio.aclose.assert_awaited()


async def test_async_events_are_reported(vertex):
    events = []
    client, fake = vertex
    client.chat.completions._on_event = events.append
    fake.aio.models.generate_content.return_value = response()
    await client.chat.completions.create(messages=MESSAGES)
    assert [e["event"] for e in events] == ["request", "response"]
    assert events[0]["provider"] == "vertexai"


async def test_an_async_blocked_prompt_is_a_content_filter_error(vertex):
    from llmswitch import ContentFilterError
    client, fake = vertex
    fake.aio.models.generate_content.return_value = genai_types.GenerateContentResponse.model_validate(
        {"prompt_feedback": {"block_reason": "SAFETY"}})
    with pytest.raises(ContentFilterError):
        await client.chat.completions.create(messages=MESSAGES)
