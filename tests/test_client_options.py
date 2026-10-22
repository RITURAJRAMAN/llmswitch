"""client_options are passed to the native SDK client constructors."""
import pytest

from llmswitch import AsyncLLMSwitch, ConfigurationError, LLMSwitch
from llmswitch.providers.anthropic_provider import AnthropicChatProvider
from llmswitch.providers.azureopenai_provider import AzureOpenAIChatProvider
from llmswitch.providers.openai_provider import AsyncOpenAIChatProvider, OpenAIChatProvider


def test_openai_client_options_reach_the_sdk_client():
    client = LLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o",
                       client_options={"default_headers": {"X-Team": "llm"}, "max_retries": 7})
    sdk = client._provider(OpenAIChatProvider).client
    assert sdk.default_headers["X-Team"] == "llm"
    assert sdk.max_retries == 7


def test_openai_client_options_default_to_no_sdk_retries():
    client = LLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o")
    assert client._provider(OpenAIChatProvider).client.max_retries == 0


async def test_async_openai_client_options_reach_the_sdk_client():
    client = AsyncLLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o",
                            client_options={"default_headers": {"X-Team": "async"}})
    sdk = client._provider(AsyncOpenAIChatProvider).client
    assert sdk.default_headers["X-Team"] == "async"


def test_azure_openai_client_options_reach_the_sdk_client():
    client = LLMSwitch(provider="azure_openai", api_key="k", endpoint="https://x.openai.azure.com/",
                       api_version="2024-10-21", deployment_name="prod",
                       client_options={"default_headers": {"X-Team": "azure"}})
    sdk = client._provider(AzureOpenAIChatProvider).client
    assert sdk.default_headers["X-Team"] == "azure"


def test_anthropic_client_options_reach_the_sdk_client():
    client = LLMSwitch(provider="anthropic", api_key="sk-ant-test", model="claude-opus-4-5",
                       client_options={"max_retries": 5, "default_headers": {"X-Team": "claude"}})
    sdk = client._provider(AnthropicChatProvider).client
    assert sdk.max_retries == 5
    assert sdk.default_headers["X-Team"] == "claude"


@pytest.mark.parametrize("reserved", ["api_key", "base_url", "api_version", "timeout", "http_options"])
def test_reserved_names_are_rejected(reserved):
    with pytest.raises(ConfigurationError, match=reserved):
        LLMSwitch(provider="openai", api_key="sk-test", client_options={reserved: "x"})


def test_reserved_names_are_rejected_for_async_client():
    with pytest.raises(ConfigurationError, match="api_key"):
        AsyncLLMSwitch(provider="anthropic", api_key="k", client_options={"api_key": "other"})


def test_client_options_must_be_a_mapping():
    with pytest.raises(ConfigurationError):
        LLMSwitch(provider="openai", api_key="sk-test", client_options=["not", "a", "dict"])


def test_unified_fields_still_set_the_endpoint_when_client_options_are_present():
    client = LLMSwitch(provider="openai", api_key="sk-test", endpoint="https://proxy.example/v1",
                       client_options={"max_retries": 1})
    sdk = client._provider(OpenAIChatProvider).client
    assert str(sdk.base_url).startswith("https://proxy.example/v1")
    assert sdk.max_retries == 1
