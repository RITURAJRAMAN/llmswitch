"""Vertex AI provider: Google SDK errors, sparse responses and streaming, with only the SDK client mocked."""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from google.genai import errors as google_errors
from google.genai import types as genai_types

from llmswitch import AuthenticationError, ChatCompletionError, LLMSwitch, RateLimitExceededError
from llmswitch.providers import vertexai_provider

MESSAGES = [{"role": "user", "content": "Hi"}]


def response(text="hi", **extra):
    return genai_types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finish_reason": "STOP", "index": 0}],
        "usage_metadata": {"prompt_token_count": 4, "candidates_token_count": 3, "total_token_count": 7},
        **extra,
    })


@pytest.fixture
def vertex(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(vertexai_provider, "load_credentials_from_file", lambda path, scopes=None: (object(), "project"))
    monkeypatch.setattr(vertexai_provider.genai, "Client", lambda **kwargs: fake)
    client = LLMSwitch(provider="vertexai", location="us-central1", credentials_path="/not/used.json",
                       model="gemini-2.5-pro", max_retries=0, user_id="vertex-tests")
    return client, fake


def test_a_response_without_id_model_or_time_still_maps(vertex):
    client, fake = vertex
    fake.models.generate_content.return_value = response()  # response_id, model_version, create_time are all None

    result = client.chat.completions.create(messages=MESSAGES)

    assert result.choices[0].message.content == "hi"
    assert result.id == "vertexai-response"
    assert result.model == "gemini-2.5-pro"
    assert result.created > 0


def test_the_created_time_is_the_utc_timestamp_of_the_response(vertex):
    client, fake = vertex
    moment = datetime(2024, 10, 27, 3, 33, 20, tzinfo=timezone.utc)
    fake.models.generate_content.return_value = response(create_time=moment, response_id="r-1",
                                                         model_version="gemini-x")

    result = client.chat.completions.create(messages=MESSAGES)

    assert result.created == 1730000000
    assert (result.id, result.model) == ("r-1", "gemini-x")


@pytest.mark.parametrize("code, expected", [(429, RateLimitExceededError), (401, AuthenticationError),
                                            (403, AuthenticationError), (400, ChatCompletionError)])
def test_google_client_errors_map_to_library_errors(vertex, code, expected):
    client, fake = vertex
    fake.models.generate_content.side_effect = google_errors.ClientError(code, {"error": {"message": "nope"}})
    with pytest.raises(expected) as info:
        client.chat.completions.create(messages=MESSAGES)
    if expected is ChatCompletionError:
        assert info.value.status_code == code


def test_google_server_errors_become_503(vertex):
    client, fake = vertex
    fake.models.generate_content.side_effect = google_errors.ServerError(500, {"error": {"message": "down"}})
    with pytest.raises(ChatCompletionError) as info:
        client.chat.completions.create(messages=MESSAGES)
    assert info.value.status_code == 503


def test_streaming_yields_the_text_in_order(vertex):
    client, fake = vertex
    fake.models.generate_content_stream.return_value = iter([response("Hel"), response("lo")])

    chunks = list(client.chat.completions.create(messages=MESSAGES, stream=True))

    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert text == "Hello"
