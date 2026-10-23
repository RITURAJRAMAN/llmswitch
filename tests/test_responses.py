"""Responses API: sync and async create, streaming, structured output and management verbs.

The SDK response objects are real SDK types built from literal dicts; only the
network client is mocked.
"""
from unittest.mock import AsyncMock, MagicMock

import openai.types.responses as sdk
import pytest
from conftest import provider_for
from pydantic import BaseModel

from llmswitch import (
    AsyncLLMSwitch,
    InvalidResponseRequestError,
    LLMSwitch,
    LLMSwitchError,
    ResponsesError,
    UnsupportedCapabilityError,
)
from llmswitch.providers.azureopenai_provider import (
    MIN_RESPONSES_API_VERSION,
    AsyncAzureOpenAIResponsesProvider,
    AzureOpenAIResponsesProvider,
    _resolve_responses_api_version,
)
from llmswitch.providers.openai_provider import AsyncOpenAIResponsesProvider, OpenAIResponsesProvider

RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1730000000,
    "model": "gpt-4o",
    "status": "completed",
    "output": [{
        "id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "hi there", "annotations": []}],
    }],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}


class City(BaseModel):
    name: str
    country: str


def openai_client():
    return LLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o")


def azure_client(**overrides):
    kwargs = dict(provider="azure_openai", api_key="k", endpoint="https://x.openai.azure.com/",
                  api_version="2025-04-01-preview", deployment_name="prod-gpt5")
    kwargs.update(overrides)
    return LLMSwitch(**kwargs)


def test_create_returns_a_mapped_response_and_forwards_parameters():
    client = openai_client()
    provider = provider_for(client, OpenAIResponsesProvider)
    provider.client.responses.create = MagicMock(return_value=sdk.Response.model_validate(RESPONSE))

    result = client.responses.create(input="hello", temperature=0.2)

    assert result.output_text == "hi there"
    sent = provider.client.responses.create.call_args.kwargs
    assert sent["model"] == "gpt-4o"
    assert sent["temperature"] == 0.2


async def test_async_create_returns_a_mapped_response():
    client = AsyncLLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o")
    provider = provider_for(client, AsyncOpenAIResponsesProvider)
    provider.client.responses.create = AsyncMock(return_value=sdk.Response.model_validate(RESPONSE))

    result = await client.responses.create(input="hello")

    assert result.output_text == "hi there"


def test_parse_sends_the_pydantic_schema_as_text_format():
    client = openai_client()
    provider = provider_for(client, OpenAIResponsesProvider)
    provider.client.responses.parse = MagicMock(return_value=sdk.Response.model_validate(RESPONSE))

    client.responses.parse(input="Largest city in Japan?", text_format=City)

    assert provider.client.responses.parse.call_args.kwargs["text_format"] is City


def test_model_override_is_ignored_when_a_deployment_is_configured():
    client = azure_client()
    provider = provider_for(client, AzureOpenAIResponsesProvider)
    provider.client.responses.create = MagicMock(return_value=sdk.Response.model_validate(RESPONSE))

    client.responses.create(input="hello", model="other")

    assert provider.client.responses.create.call_args.kwargs["model"] == "prod-gpt5"


def test_retrieve_and_delete_use_the_provider_verbs():
    client = openai_client()
    provider = provider_for(client, OpenAIResponsesProvider)
    provider.client.responses.retrieve = MagicMock(return_value=sdk.Response.model_validate(RESPONSE))
    provider.client.responses.delete = MagicMock(return_value=MagicMock(id="resp_1", deleted=True, object="response"))

    assert client.responses.retrieve("resp_1").id == "resp_1"
    client.responses.delete("resp_1")
    provider.client.responses.delete.assert_called_once()


def test_streaming_yields_mapped_events():
    client = openai_client()
    provider = provider_for(client, OpenAIResponsesProvider)
    event = sdk.ResponseTextDeltaEvent.model_validate({
        "type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
        "content_index": 0, "delta": "hi", "sequence_number": 1, "logprobs": [],
    })
    provider.client.responses.create = MagicMock(return_value=iter([event]))

    events = list(client.responses.stream(input="hello"))

    assert len(events) == 1
    assert events[0].delta == "hi"


def test_unsupported_provider_raises_501():
    client = LLMSwitch(provider="anthropic", api_key="sk-ant-test", model="claude-opus-4-5")
    with pytest.raises(ResponsesError) as info:
        client.responses.create(input="hello")
    assert info.value.status_code == 501


def test_remote_mcp_tool_requires_an_explicit_policy():
    client = openai_client()
    tool = {"type": "mcp", "server_label": "docs", "server_url": "https://example.com/mcp"}
    with pytest.raises(InvalidResponseRequestError, match="mcp_policy"):
        client.responses.create(input="hello", tools=[tool])


def test_remote_mcp_tool_is_allowed_by_policy():
    client = LLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o",
                       mcp_policy={"enabled": True, "allowed_server_labels": ["docs"]})
    provider = provider_for(client, OpenAIResponsesProvider)
    provider.client.responses.create = MagicMock(return_value=sdk.Response.model_validate(RESPONSE))
    tool = {"type": "mcp", "server_label": "docs", "server_url": "https://example.com/mcp"}

    client.responses.create(input="hello", tools=[tool])

    assert provider.client.responses.create.call_args.kwargs["tools"][0]["server_label"] == "docs"


def test_unknown_request_field_is_rejected_before_dispatch():
    client = openai_client()
    with pytest.raises(InvalidResponseRequestError):
        client.responses.create(input="hello", not_a_real_field=1)


def test_responses_capability_override_rejects_input_token_counting():
    client = LLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o",
                       responses_capabilities={"input_tokens.count": False})
    with pytest.raises(UnsupportedCapabilityError):
        client.responses.input_tokens.count(input="hello")


def test_responses_errors_are_llmswitch_errors():
    assert issubclass(ResponsesError, LLMSwitchError)


@pytest.mark.parametrize("api_version, expected", [
    ("2024-10-21", MIN_RESPONSES_API_VERSION),
    ("2025-04-01-preview", "2025-04-01-preview"),
])
def test_azure_responses_api_version_is_raised_to_the_minimum(api_version, expected):
    assert _resolve_responses_api_version({"api_version": api_version}) == expected


def test_azure_responses_override_wins():
    assert _resolve_responses_api_version({"api_version": "2024-10-21",
                                           "responses_api_version": "2025-06-01"}) == "2025-06-01"


def test_azure_responses_client_uses_the_responses_version():
    client = azure_client(api_version="2024-10-21")
    sdk_client = provider_for(client, AzureOpenAIResponsesProvider).client
    assert sdk_client._api_version == MIN_RESPONSES_API_VERSION


async def test_async_azure_responses_provider_is_registered():
    client = AsyncLLMSwitch(provider="azure_openai", api_key="k", endpoint="https://x.openai.azure.com/",
                            api_version="2025-04-01-preview", deployment_name="prod-gpt5")
    assert isinstance(provider_for(client, AsyncAzureOpenAIResponsesProvider), AsyncAzureOpenAIResponsesProvider)
