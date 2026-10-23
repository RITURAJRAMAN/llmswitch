import pytest

from llmswitch import ConfigurationError, LLMSwitch, LLMSwitchError
from llmswitch.config import CONFIG_REGISTRY

AZURE_OPENAI = {"provider": "azure_openai", "api_key": "k", "endpoint": "https://x.openai.azure.com/",
                "api_version": "2024-10-21", "deployment_name": "gpt-4o"}
VERTEXAI = {"provider": "vertexai", "location": "us-central1", "credentials_path": "/tmp/sa.json"}


def test_registry_covers_the_five_providers():
    assert set(CONFIG_REGISTRY) == {"openai", "azure_openai", "azure_foundry", "anthropic", "vertexai"}


def test_openai_requires_api_key():
    with pytest.raises(ConfigurationError):
        LLMSwitch(provider="openai")


def test_azure_openai_requires_endpoint_and_version():
    kwargs = dict(AZURE_OPENAI)
    del kwargs["endpoint"]
    with pytest.raises(ConfigurationError):
        LLMSwitch(**kwargs)


def test_vertexai_requires_location_and_credentials():
    with pytest.raises(ConfigurationError):
        LLMSwitch(provider="vertexai", location="us-central1")


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMSwitchError, match="Unsupported provider config: nope"):
        LLMSwitch(provider="nope", api_key="k")


def test_missing_provider_is_rejected():
    with pytest.raises(LLMSwitchError, match="Unsupported provider config"):
        LLMSwitch(api_key="k")


def test_blank_provider_is_rejected():
    with pytest.raises(LLMSwitchError, match="Unsupported provider config"):
        LLMSwitch(provider="   ", api_key="k")


def test_non_positive_timeout_is_rejected():
    with pytest.raises(ConfigurationError):
        LLMSwitch(provider="openai", api_key="k", timeout=0)


def test_fields_that_belong_to_other_providers_are_accepted_by_the_unified_constructor():
    client = LLMSwitch(provider="openai", api_key="k", model="gpt-4o", location="us-central1", region="eu")
    assert client.config.model == "gpt-4o"


def test_a_misspelled_option_is_rejected_with_a_suggestion():
    with pytest.raises(ConfigurationError, match="'max_retires' \\(did you mean 'max_retries'\\?\\)"):
        LLMSwitch(provider="openai", api_key="k", model="gpt-4o", max_retires=5)


def test_every_unknown_option_is_reported_together():
    with pytest.raises(ConfigurationError) as info:
        LLMSwitch(provider="openai", api_key="k", rmp=10, totally_made_up=1)
    assert info.value.context["details"]["unknown"] == ["rmp", "totally_made_up"]
    assert "client_options" in str(info.value)


def test_config_exposes_deployment_and_model():
    client = LLMSwitch(**AZURE_OPENAI, model="ignored-alias")
    assert client.config.deployment_name == "gpt-4o"
    assert client.config.model == "ignored-alias"


def test_repr_does_not_include_api_key():
    client = LLMSwitch(provider="openai", api_key="sk-secret", model="gpt-4o")
    assert "sk-secret" not in repr(client)
    assert "gpt-4o" in repr(client)
