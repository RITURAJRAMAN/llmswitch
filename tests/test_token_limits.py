"""max_tokens, max_completion_tokens and max_output_tokens are interchangeable on every provider."""
from unittest.mock import MagicMock

import anthropic.types
import openai.types.chat
import pytest
from conftest import provider_for
from test_providers import ANTHROPIC_MESSAGE, OPENAI_COMPLETION

from llmswitch import LLMSwitch
from llmswitch.providers._shared import azure_token_limit_name, normalize_token_limit, openai_token_limit_name
from llmswitch.providers.anthropic_provider import AnthropicChatProvider
from llmswitch.providers.azurefoundry_provider import AzureFoundryChatProvider
from llmswitch.providers.azureopenai_provider import AzureOpenAIChatProvider
from llmswitch.providers.openai_provider import OpenAIChatProvider

MESSAGES = [{"role": "user", "content": "Hi"}]


@pytest.mark.parametrize("given", ["max_tokens", "max_completion_tokens", "max_output_tokens"])
def test_any_name_becomes_the_target_name(given):
    kwargs = {given: 50, "temperature": 0.1}
    normalize_token_limit(kwargs, "max_completion_tokens")
    assert kwargs == {"max_completion_tokens": 50, "temperature": 0.1}


def test_the_target_name_wins_when_several_are_given():
    kwargs = {"max_tokens": 10, "max_output_tokens": 20, "max_completion_tokens": 30}
    normalize_token_limit(kwargs, "max_tokens")
    assert kwargs == {"max_tokens": 10}


def test_without_a_target_the_first_named_limit_is_used():
    kwargs = {"max_tokens": 10, "max_output_tokens": 20}
    normalize_token_limit(kwargs, "max_completion_tokens")
    assert kwargs == {"max_completion_tokens": 20}


def test_no_limit_is_left_untouched():
    kwargs = {"temperature": 0.2}
    normalize_token_limit(kwargs, "max_tokens")
    assert kwargs == {"temperature": 0.2}


@pytest.mark.parametrize("endpoint, expected", [
    (None, "max_completion_tokens"),
    ("https://api.openai.com/v1", "max_completion_tokens"),
    ("https://my-resource.openai.azure.com/openai/v1", "max_completion_tokens"),
    ("https://my-resource.cognitiveservices.azure.com/", "max_completion_tokens"),
    ("http://localhost:8000/v1", "max_tokens"),
    ("https://llm-proxy.example.com/v1", "max_tokens"),
])
def test_openai_token_limit_name_depends_on_the_host(endpoint, expected):
    assert openai_token_limit_name(endpoint) == expected


@pytest.mark.parametrize("api_version, expected", [
    (None, "max_completion_tokens"),
    ("2024-02-01", "max_tokens"),
    ("2024-08-01-preview", "max_tokens"),
    ("2024-10-21", "max_completion_tokens"),
    ("2025-04-01-preview", "max_completion_tokens"),
])
def test_azure_token_limit_name_depends_on_the_api_version(api_version, expected):
    assert azure_token_limit_name(api_version) == expected


def sent_to_openai_sdk(client, provider_cls, **create_kwargs):
    provider = provider_for(client, provider_cls)
    provider.client.chat.completions.create = MagicMock(
        return_value=openai.types.chat.ChatCompletion.model_validate(OPENAI_COMPLETION))
    client.chat.completions.create(messages=MESSAGES, **create_kwargs)
    return provider.client.chat.completions.create.call_args.kwargs


@pytest.mark.parametrize("given", ["max_tokens", "max_output_tokens", "max_completion_tokens"])
def test_openai_sends_max_completion_tokens(given):
    client = LLMSwitch(provider="openai", api_key="k", model="gpt-5", user_id=f"tl-openai-{given}")
    sent = sent_to_openai_sdk(client, OpenAIChatProvider, **{given: 64})
    assert sent["max_completion_tokens"] == 64
    assert "max_tokens" not in sent and "max_output_tokens" not in sent


def test_openai_compatible_servers_keep_max_tokens():
    client = LLMSwitch(provider="openai", api_key="k", model="local", endpoint="http://localhost:8000/v1",
                       user_id="tl-compatible")
    sent = sent_to_openai_sdk(client, OpenAIChatProvider, max_output_tokens=64)
    assert sent["max_tokens"] == 64
    assert "max_completion_tokens" not in sent


@pytest.mark.parametrize("api_version, expected", [("2025-04-01-preview", "max_completion_tokens"),
                                                   ("2024-02-01", "max_tokens")])
def test_azure_openai_follows_the_api_version(api_version, expected):
    client = LLMSwitch(provider="azure_openai", api_key="k", endpoint="https://x.openai.azure.com/",
                       api_version=api_version, deployment_name="custom-name-without-a-model-prefix",
                       user_id=f"tl-azure-{api_version}")
    sent = sent_to_openai_sdk(client, AzureOpenAIChatProvider, max_tokens=64)
    assert sent[expected] == 64
    assert len({"max_tokens", "max_completion_tokens"} & set(sent)) == 1


@pytest.mark.parametrize("given", ["max_tokens", "max_output_tokens", "max_completion_tokens"])
def test_anthropic_sends_max_tokens(given):
    client = LLMSwitch(provider="anthropic", api_key="k", model="claude-opus-4-5", user_id=f"tl-anthropic-{given}")
    provider = provider_for(client, AnthropicChatProvider)
    provider.client.messages.create = MagicMock(return_value=anthropic.types.Message.model_validate(ANTHROPIC_MESSAGE))
    client.chat.completions.create(messages=MESSAGES, **{given: 77})
    sent = provider.client.messages.create.call_args.kwargs
    assert sent["max_tokens"] == 77
    assert "max_output_tokens" not in sent and "max_completion_tokens" not in sent


@pytest.mark.parametrize("given", ["max_tokens", "max_output_tokens", "max_completion_tokens"])
def test_azure_foundry_sends_max_tokens(given):
    client = LLMSwitch(provider="azure_foundry", api_key="k", endpoint="https://x.services.ai.azure.com/models",
                       api_version="2024-05-01-preview", deployment_name="m", user_id=f"tl-foundry-{given}")
    provider = provider_for(client, AzureFoundryChatProvider)
    assert provider._sanitize_request_kwargs({given: 33, "stream_options": {}}) == {"max_tokens": 33}
