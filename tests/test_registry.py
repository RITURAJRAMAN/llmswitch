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


def test_importing_the_package_does_not_load_provider_sdks():
    import subprocess
    import sys

    code = (
        "import sys, llmswitch\n"
        "loaded = [m for m in ('openai', 'anthropic', 'google.genai', 'azure.ai.inference') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_a_provider_module_loads_only_when_it_is_first_used():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from llmswitch.core.registry import CHAT_PROVIDER_REGISTRY as R\n"
        "assert 'anthropic' not in sys.modules\n"
        "R['anthropic']\n"
        "assert 'anthropic' in sys.modules and 'google.genai' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_a_failed_sdk_import_is_recorded_and_reads_as_missing(monkeypatch):
    import importlib

    from llmswitch.core.registry import _LazyRegistry

    def failing_import(name, package=None):
        raise ImportError("No module named 'google'")

    monkeypatch.setattr(importlib, "import_module", failing_import)
    monkeypatch.setattr(registry, "UNAVAILABLE_PROVIDERS", {})
    lazy = _LazyRegistry({"vertexai": ("vertexai_provider", "VertexAIChatProvider")})

    assert lazy.get("vertexai") is None
    assert "vertexai" not in lazy
    assert registry.UNAVAILABLE_PROVIDERS["vertexai"] == "No module named 'google'"


def test_the_client_reports_why_a_provider_could_not_load(monkeypatch):
    monkeypatch.delitem(registry.CHAT_PROVIDER_REGISTRY, "vertexai")
    monkeypatch.setitem(registry.UNAVAILABLE_PROVIDERS, "vertexai", "No module named 'google'")
    client = LLMSwitch(provider="vertexai", location="us-central1", credentials_path="/tmp/sa.json")
    with pytest.raises(LLMSwitchError, match="could not load its SDK.*No module named 'google'"):
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
