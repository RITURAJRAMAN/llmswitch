"""on_event callbacks and the default-off request logging."""
import logging
from types import SimpleNamespace

import pytest
from conftest import StubChat

from llmswitch import APIConnectionError, AsyncLLMSwitch, ChatCompletionError, ConfigurationError, LLMSwitch
from llmswitch.core.registry import ASYNC_CHAT_PROVIDER_REGISTRY, CHAT_PROVIDER_REGISTRY

MESSAGES = [{"role": "user", "content": "Hello"}]


class Recorder:
    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)

    @property
    def names(self):
        return [event["event"] for event in self.events]


def client_with(recorder, **overrides):
    return LLMSwitch(provider="openai", api_key="k-events", model="gpt-4o", user_id=f"events-{id(recorder)}",
                     on_event=recorder, **overrides)


def test_a_successful_call_reports_request_then_response(stubs):
    recorder = Recorder()
    client_with(recorder).chat.completions.create(messages=MESSAGES)
    assert recorder.names == ["request", "response"]
    request, response = recorder.events
    assert request["operation"] == "chat.completions"
    assert request["provider"] == "openai"
    assert request["model"] == "gpt-4o"
    assert request["stream"] is False
    assert response["latency"] >= 0


def test_the_response_event_carries_token_usage(stubs, monkeypatch):
    class UsageChat(StubChat):
        def create(self, messages, stream=False, **kwargs):
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5))

    monkeypatch.setitem(CHAT_PROVIDER_REGISTRY, "openai", UsageChat)
    recorder = Recorder()
    client_with(recorder).chat.completions.create(messages=MESSAGES)
    assert recorder.events[-1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_a_failed_call_reports_an_error_event(stubs):
    StubChat, _ = stubs
    StubChat.error = ValueError("bad temperature")
    recorder = Recorder()
    with pytest.raises(ChatCompletionError):
        client_with(recorder).chat.completions.create(messages=MESSAGES)
    assert recorder.names == ["request", "error"]
    assert recorder.events[-1]["error"] == "ChatCompletionError"
    assert recorder.events[-1]["status_code"] == 400


def test_retries_are_reported_between_request_and_response(monkeypatch):
    class Flaky(StubChat):
        failures = 2

        def create(self, messages, stream=False, **kwargs):
            if Flaky.failures > 0:
                Flaky.failures -= 1
                raise APIConnectionError("https://api.example", provider="openai")
            return {"ok": True}

    monkeypatch.setitem(CHAT_PROVIDER_REGISTRY, "openai", Flaky)
    monkeypatch.setattr("llmswitch.core.retry.time.sleep", lambda seconds: None)
    recorder = Recorder()
    client_with(recorder, max_retries=3, retry_base_delay=0).chat.completions.create(messages=MESSAGES)
    assert recorder.names == ["request", "retry", "retry", "response"]
    assert [e["attempt"] for e in recorder.events if e["event"] == "retry"] == [1, 2]
    assert recorder.events[1]["error"] == "APIConnectionError"


def test_failover_is_reported(monkeypatch, stubs):
    class AlwaysDown(StubChat):
        def create(self, messages, stream=False, **kwargs):
            raise APIConnectionError("https://api.example", provider="openai")

    monkeypatch.setitem(CHAT_PROVIDER_REGISTRY, "openai", AlwaysDown)
    recorder = Recorder()
    client_with(recorder, max_retries=0, backup={"provider": "anthropic", "api_key": "k2",
                                                 "model": "claude-opus-4-5"}).chat.completions.create(messages=MESSAGES)
    failover = [e for e in recorder.events if e["event"] == "failover"]
    assert len(failover) == 1
    assert failover[0]["from_provider"] == "openai"
    assert failover[0]["to_provider"] == "anthropic"
    assert recorder.names[-1] == "response"


def test_a_stream_reports_request_and_a_streaming_response(stubs):
    recorder = Recorder()
    list(client_with(recorder).chat.completions.create(messages=MESSAGES, stream=True))
    assert recorder.names == ["request", "response"]
    assert recorder.events[0]["stream"] is True
    assert recorder.events[1]["stream"] is True


def test_embeddings_report_their_own_operation(stubs):
    recorder = Recorder()
    client_with(recorder).embeddings.create("hello")
    assert [(e["event"], e["operation"]) for e in recorder.events] == [
        ("request", "embeddings"), ("response", "embeddings")]


def test_a_callback_that_raises_does_not_break_the_call(stubs, caplog):
    def broken(event):
        raise RuntimeError("callback bug")

    client = LLMSwitch(provider="openai", api_key="k-broken", model="gpt-4o", user_id="events-broken", on_event=broken)
    with caplog.at_level(logging.ERROR, logger="llmswitch.core.events"):
        assert client.chat.completions.create(messages=MESSAGES) == {"ok": True, "stream": False}
    assert "on_event callback raised" in caplog.text


def test_on_event_must_be_callable():
    with pytest.raises(ConfigurationError, match="on_event"):
        LLMSwitch(provider="openai", api_key="k", model="gpt-4o", on_event="not callable")


def test_without_a_callback_no_events_are_needed(stubs):
    client = LLMSwitch(provider="openai", api_key="k-none", model="gpt-4o", user_id="events-none")
    assert client.chat.completions.create(messages=MESSAGES) == {"ok": True, "stream": False}


async def test_async_calls_report_events(stubs):
    recorder = Recorder()
    client = AsyncLLMSwitch(provider="openai", api_key="k-async-events", model="gpt-4o", user_id="events-async",
                            on_event=recorder)
    await client.chat.completions.create(messages=MESSAGES)
    assert recorder.names == ["request", "response"]
    assert recorder.events[0]["operation"] == "chat.completions"


async def test_async_embeddings_report_events(stubs):
    recorder = Recorder()
    client = AsyncLLMSwitch(provider="openai", api_key="k-async-emb", model="gpt-4o", user_id="events-async-emb",
                            on_event=recorder)
    await client.embeddings.create("hello")
    assert recorder.names == ["request", "response"]
    assert recorder.events[0]["operation"] == "embeddings"


async def test_async_failed_calls_report_an_error(stubs, monkeypatch):
    class AsyncDown(StubChat):
        async def create(self, messages, stream=False, **kwargs):
            raise ValueError("bad request")

    monkeypatch.setitem(ASYNC_CHAT_PROVIDER_REGISTRY, "openai", AsyncDown)
    recorder = Recorder()
    client = AsyncLLMSwitch(provider="openai", api_key="k-async-err", model="gpt-4o", user_id="events-async-err",
                            on_event=recorder, max_retries=0)
    with pytest.raises(ChatCompletionError):
        await client.chat.completions.create(messages=MESSAGES)
    assert recorder.names == ["request", "error"]


def test_request_details_are_not_logged_by_default(stubs, caplog):
    client = LLMSwitch(provider="openai", api_key="k-quiet", model="gpt-4o", user_id="events-quiet")
    with caplog.at_level(logging.INFO, logger="llmswitch"):
        client.chat.completions.create(messages=MESSAGES)
    assert [r for r in caplog.records if r.name.startswith("llmswitch.core.sync_client")] == []


def test_live_logs_turns_request_logging_on_for_one_call(stubs, caplog):
    client = LLMSwitch(provider="openai", api_key="k-loud", model="gpt-4o", user_id="events-loud")
    with caplog.at_level(logging.INFO, logger="llmswitch"):
        client.chat.completions.create(messages=MESSAGES, live_logs=True)
    assert any('"provider": "openai"' in r.getMessage() for r in caplog.records)
    assert "k-loud" not in caplog.text


async def test_async_live_logs_matches_sync(stubs, caplog):
    client = AsyncLLMSwitch(provider="openai", api_key="k-aloud", model="gpt-4o", user_id="events-aloud")
    with caplog.at_level(logging.INFO, logger="llmswitch"):
        await client.chat.completions.create(messages=MESSAGES, live_logs=True)
    assert any('"provider": "openai"' in r.getMessage() for r in caplog.records)


def test_management_calls_report_their_verb():
    from unittest.mock import MagicMock

    import openai.types.responses as sdk
    from conftest import provider_for
    from test_responses import RESPONSE

    from llmswitch.providers.openai_provider import OpenAIResponsesProvider

    recorder = Recorder()
    client = LLMSwitch(provider="openai", api_key="k-mgmt", model="gpt-4o", user_id="events-mgmt", on_event=recorder)
    provider = provider_for(client, OpenAIResponsesProvider)
    provider.client.responses.retrieve = MagicMock(return_value=sdk.Response.model_validate(RESPONSE))

    client.responses.retrieve("resp_1")

    assert recorder.names == ["request", "response"]
    assert {(e["operation"], e["verb"]) for e in recorder.events} == {("responses.management", "retrieve")}
