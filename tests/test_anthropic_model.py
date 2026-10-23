"""Model selection for Anthropic: a configured deployment wins, otherwise a per-call model overrides the config."""
import uuid

import pytest

from llmswitch import AsyncLLMSwitch, LLMSwitch
from llmswitch.core.registry import ASYNC_CHAT_PROVIDER_REGISTRY, CHAT_PROVIDER_REGISTRY
from llmswitch.providers.anthropic_provider import AnthropicChatProvider, AsyncAnthropicChatProvider

MESSAGES = [{"role": "user", "content": "Hi"}]


class Capture:
    sent: list = []

    def __init__(self, **kwargs):
        self.init = kwargs

    def create(self, messages, stream=False, **kwargs):
        Capture.sent.append(kwargs)
        return {"ok": True}


class AsyncCapture(Capture):
    async def create(self, messages, stream=False, **kwargs):
        Capture.sent.append(kwargs)
        return {"ok": True}


@pytest.fixture(autouse=True)
def capture(monkeypatch):
    Capture.sent = []
    monkeypatch.setitem(CHAT_PROVIDER_REGISTRY, "anthropic", Capture)
    monkeypatch.setitem(ASYNC_CHAT_PROVIDER_REGISTRY, "anthropic", AsyncCapture)


def user():
    return f"anth-{uuid.uuid4().hex}"


def test_a_per_call_model_overrides_the_configured_model_without_a_deployment():
    client = LLMSwitch(provider="anthropic", api_key="k", model="claude-a", user_id=user())
    client.chat.completions.create(messages=MESSAGES, model="claude-b")
    assert Capture.sent[-1]["model"] == "claude-b"


def test_without_a_per_call_model_the_configured_model_is_used():
    client = LLMSwitch(provider="anthropic", api_key="k", model="claude-a", user_id=user())
    client.chat.completions.create(messages=MESSAGES)
    assert Capture.sent[-1]["model"] == "claude-a"


def test_a_configured_deployment_is_not_replaced_by_a_per_call_model():
    client = LLMSwitch(provider="anthropic", api_key="k", model="claude-a", deployment_name="prod-claude",
                       user_id=user())
    client.chat.completions.create(messages=MESSAGES, model="claude-b")
    assert Capture.sent[-1]["model"] == "prod-claude"


async def test_the_async_client_follows_the_same_rule():
    client = AsyncLLMSwitch(provider="anthropic", api_key="k", model="claude-a", user_id=user())
    await client.chat.completions.create(messages=MESSAGES, model="claude-b")
    assert Capture.sent[-1]["model"] == "claude-b"


def test_prepare_request_applies_the_rule_directly():
    provider = AnthropicChatProvider(api_key="k", model="claude-a")
    assert provider._prepare_request(MESSAGES, {"model": "claude-b"})["model"] == "claude-b"
    assert provider._prepare_request(MESSAGES, {})["model"] == "claude-a"
    pinned = AnthropicChatProvider(api_key="k", model="claude-a", deployment_name="prod")
    assert pinned._prepare_request(MESSAGES, {"model": "claude-b"})["model"] == "prod"


def test_the_async_provider_keeps_the_deployment_rule():
    provider = AsyncAnthropicChatProvider(api_key="k", model="claude-a", deployment_name="prod")
    assert provider._prepare_request(MESSAGES, {"model": "claude-b"})["model"] == "prod"
