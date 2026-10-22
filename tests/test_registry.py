import pytest

from llmswitch import LLMSwitch, LLMSwitchError
from llmswitch.core import registry


def test_all_five_providers_are_registered():
    assert set(registry.CHAT_PROVIDER_REGISTRY) == {"openai", "azure_openai", "azure_foundry", "anthropic", "vertexai"}


def test_missing_sdk_reports_the_reason_and_how_to_fix_it(monkeypatch):
    monkeypatch.delitem(registry.CHAT_PROVIDER_REGISTRY, "vertexai")
    monkeypatch.setitem(registry.UNAVAILABLE_PROVIDERS, "vertexai", "No module named 'google'")
    client = LLMSwitch(provider="vertexai", location="us-central1", credentials_path="/tmp/sa.json")

    with pytest.raises(LLMSwitchError, match="could not load its SDK.*No module named 'google'.*pip install --upgrade llmswitch-sdk"):
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])


def test_unknown_provider_still_says_unsupported(monkeypatch):
    monkeypatch.delitem(registry.CHAT_PROVIDER_REGISTRY, "vertexai")
    client = LLMSwitch(provider="vertexai", location="us-central1", credentials_path="/tmp/sa.json")
    with pytest.raises(LLMSwitchError, match="Unsupported provider for chat completion: vertexai"):
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
