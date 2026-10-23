"""Rate limiting, retries, circuit breaker, backup failover, context truncation and batching."""
import pytest
from conftest import StubChat, provider_for

from llmswitch import (
    APIConnectionError,
    ContextLengthExceededError,
    LLMSwitch,
    LLMSwitchError,
    RateLimitExceededError,
)
from llmswitch.core.registry import CHAT_PROVIDER_REGISTRY

MESSAGES = [{"role": "user", "content": "Hello"}]


class FlakyChat(StubChat):
    """Fails with a retryable connection error for the first ``failures`` calls."""

    failures = 0
    calls = 0

    def create(self, messages, stream=False, **kwargs):
        FlakyChat.calls += 1
        if FlakyChat.failures > 0:
            FlakyChat.failures -= 1
            raise APIConnectionError("https://api.example", provider="openai")
        return {"ok": True}


@pytest.fixture
def flaky(monkeypatch):
    FlakyChat.failures, FlakyChat.calls = 0, 0
    monkeypatch.setitem(CHAT_PROVIDER_REGISTRY, "openai", FlakyChat)
    return FlakyChat


def test_requests_over_the_per_minute_limit_wait_then_fail(stubs, monkeypatch):
    waits = []
    monkeypatch.setattr("llmswitch.core.execution.time.sleep", lambda seconds: waits.append(seconds))
    client = LLMSwitch(provider="openai", api_key="k-rpm", model="gpt-4o", user_id="rpm-user", rpm=1)
    client.chat.completions.create(messages=MESSAGES)
    with pytest.raises(RateLimitExceededError):
        client.chat.completions.create(messages=MESSAGES)
    assert waits and waits[0] > 0


def test_retryable_errors_are_retried_until_success(flaky):
    client = LLMSwitch(provider="openai", api_key="k-retry", model="gpt-4o", user_id="retry-user",
                       max_retries=2, retry_base_delay=0)
    flaky.failures = 2
    assert client.chat.completions.create(messages=MESSAGES) == {"ok": True}
    assert flaky.calls == 3


def test_retries_stop_at_max_retries(flaky):
    client = LLMSwitch(provider="openai", api_key="k-retry-stop", model="gpt-4o", user_id="retry-stop-user",
                       max_retries=1, retry_base_delay=0)
    flaky.failures = 10
    with pytest.raises(APIConnectionError):
        client.chat.completions.create(messages=MESSAGES)
    assert flaky.calls == 2


def test_circuit_breaker_opens_after_repeated_failures(flaky):
    client = LLMSwitch(provider="openai", api_key="k-breaker", model="gpt-4o", user_id="breaker-user",
                       max_retries=0, circuit_breaker={"failure_threshold": 2, "reset_timeout": 60})
    flaky.failures = 100
    for _ in range(2):
        with pytest.raises(APIConnectionError):
            client.chat.completions.create(messages=MESSAGES)
    calls_before = flaky.calls
    with pytest.raises(APIConnectionError):
        client.chat.completions.create(messages=MESSAGES)
    assert flaky.calls == calls_before


def test_backup_provider_answers_when_the_primary_fails(flaky, stubs):
    flaky.failures = 100
    client = LLMSwitch(provider="openai", api_key="k-failover", model="gpt-4o", user_id="failover-user",
                       max_retries=0, retry_base_delay=0,
                       backup={"provider": "anthropic", "api_key": "k-backup", "model": "claude-opus-4-5"})
    result = client.chat.completions.create(messages=MESSAGES)
    assert result == {"ok": True, "stream": False}


def test_long_messages_are_rejected_without_truncate(stubs):
    client = LLMSwitch(provider="openai", api_key="k-ctx", model="gpt-4o", user_id="ctx-user",
                       completion_context_window=30)
    long_messages = [{"role": "user", "content": "word " * 500}]
    with pytest.raises(ContextLengthExceededError):
        client.chat.completions.create(messages=long_messages)


def test_long_messages_are_trimmed_with_truncate(stubs):
    StubChat, _ = stubs
    client = LLMSwitch(provider="openai", api_key="k-ctx-trunc", model="gpt-4o", user_id="ctx-trunc-user",
                       completion_context_window=60)
    long_text = "word " * 500
    client.chat.completions.create(messages=[{"role": "user", "content": long_text}], truncate=True)
    sent = StubChat.calls[-1]["messages"][-1]["content"]
    assert len(sent) < len(long_text)


def test_embedding_batching_splits_inputs_into_batches(stubs):
    _, StubEmbeddings_ = stubs
    client = LLMSwitch(provider="openai", api_key="k-batch", model="text-embedding-3-small", user_id="batch-user")
    result = client.embeddings.create(["a", "b", "c", "d", "e"], batching=True, batchsize=2)
    assert len(StubEmbeddings_.calls) == 3
    assert [item.index for item in result.data] == [0, 1, 2, 3, 4]


def test_share_clients_reuses_one_sdk_client_across_adapters(stubs):
    first = LLMSwitch(provider="openai", api_key="k-shared", model="gpt-4o", user_id="shared-1", share_clients=True)
    second = LLMSwitch(provider="openai", api_key="k-shared", model="gpt-4o", user_id="shared-2", share_clients=True)
    first.chat.completions.create(messages=MESSAGES)
    second.chat.completions.create(messages=MESSAGES)
    assert provider_for(first, StubChat)._entry is provider_for(second, StubChat)._entry


def test_redis_outage_falls_back_to_in_process_limits_by_default(stubs):
    client = LLMSwitch(provider="openai", api_key="k-redis", model="gpt-4o", user_id="redis-user",
                       redis_url="redis://127.0.0.1:1", redis_connect_timeout=0.2)
    assert client.chat.completions.create(messages=MESSAGES) == {"ok": True, "stream": False}


def test_redis_outage_can_fail_closed(stubs):
    client = LLMSwitch(provider="openai", api_key="k-redis-closed", model="gpt-4o", user_id="redis-closed-user",
                       redis_url="redis://127.0.0.1:1", redis_connect_timeout=0.2,
                       rate_limit_failure_policy="fail_closed")
    with pytest.raises(LLMSwitchError):
        client.chat.completions.create(messages=MESSAGES)



class SlowDownChat(StubChat):
    """Always fails with a rate-limit error that asks the caller to wait 30 seconds."""

    calls = 0

    def create(self, messages, stream=False, **kwargs):
        SlowDownChat.calls += 1
        raise RateLimitExceededError(30)


class AsyncSlowDownChat(StubChat):
    calls = 0

    async def create(self, messages, stream=False, **kwargs):
        AsyncSlowDownChat.calls += 1
        raise RateLimitExceededError(30)


def test_on_rate_limit_raise_fails_immediately_without_waiting(stubs, monkeypatch):
    waits = []
    monkeypatch.setattr("llmswitch.core.execution.time.sleep", lambda seconds: waits.append(seconds))
    client = LLMSwitch(provider="openai", api_key="k-raise", model="gpt-4o", user_id="raise-user", rpm=1,
                       on_rate_limit="raise")
    client.chat.completions.create(messages=MESSAGES)
    with pytest.raises(RateLimitExceededError):
        client.chat.completions.create(messages=MESSAGES)
    assert waits == []


def test_a_wait_longer_than_the_maximum_raises_without_sleeping(stubs, monkeypatch):
    waits = []
    monkeypatch.setattr("llmswitch.core.execution.time.sleep", lambda seconds: waits.append(seconds))
    client = LLMSwitch(provider="openai", api_key="k-maxwait", model="gpt-4o", user_id="maxwait-user", rpm=1,
                       rate_limit_max_wait=0.001)
    client.chat.completions.create(messages=MESSAGES)
    with pytest.raises(RateLimitExceededError):
        client.chat.completions.create(messages=MESSAGES)
    assert waits == []


def test_rate_limit_policy_reaches_the_limiter(stubs):
    client = LLMSwitch(provider="openai", api_key="k-policy", model="gpt-4o", on_rate_limit="raise",
                       rate_limit_max_wait=5)
    assert client.rate_limiter.on_limit == "raise"
    assert client.rate_limiter.max_wait == 5


def test_retries_stop_when_the_next_wait_would_pass_the_deadline(monkeypatch):
    SlowDownChat.calls = 0
    monkeypatch.setitem(CHAT_PROVIDER_REGISTRY, "openai", SlowDownChat)
    client = LLMSwitch(provider="openai", api_key="k-deadline", model="gpt-4o", user_id="deadline-user",
                       max_retries=5, retry_base_delay=0, retry_deadline=1)
    with pytest.raises(RateLimitExceededError):
        client.chat.completions.create(messages=MESSAGES)
    assert SlowDownChat.calls == 1


async def test_async_retries_stop_at_the_deadline(monkeypatch):
    from llmswitch import AsyncLLMSwitch
    from llmswitch.core.registry import ASYNC_CHAT_PROVIDER_REGISTRY

    AsyncSlowDownChat.calls = 0
    monkeypatch.setitem(ASYNC_CHAT_PROVIDER_REGISTRY, "openai", AsyncSlowDownChat)
    client = AsyncLLMSwitch(provider="openai", api_key="k-adeadline", model="gpt-4o", user_id="adeadline-user",
                            max_retries=5, retry_base_delay=0, retry_deadline=1)
    with pytest.raises(RateLimitExceededError):
        await client.chat.completions.create(messages=MESSAGES)
    assert AsyncSlowDownChat.calls == 1


def test_without_a_deadline_the_retry_budget_is_used(monkeypatch):
    SlowDownChat.calls = 0
    monkeypatch.setitem(CHAT_PROVIDER_REGISTRY, "openai", SlowDownChat)
    monkeypatch.setattr("llmswitch.core.retry.time.sleep", lambda seconds: None)
    client = LLMSwitch(provider="openai", api_key="k-nodeadline", model="gpt-4o", user_id="nodeadline-user",
                       max_retries=2, retry_base_delay=0)
    with pytest.raises(RateLimitExceededError):
        client.chat.completions.create(messages=MESSAGES)
    assert SlowDownChat.calls == 3


def test_a_custom_token_counter_replaces_the_default_counting(stubs):
    seen = []

    def counter(text, model_name):
        seen.append((text, model_name))
        return 1

    client = LLMSwitch(provider="openai", api_key="k-counter", model="gpt-4o", user_id="counter-user",
                       token_counter=counter)
    assert client.rate_limiter.count_tokens("hello there", "claude-opus-4-5") == 1
    client.chat.completions.create(messages=MESSAGES)
    assert seen[0] == ("hello there", "claude-opus-4-5")
    assert any(text == "Hello" for text, _ in seen)
