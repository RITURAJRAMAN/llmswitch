import pytest

from llmswitch import (
    APIConnectionError,
    AuthenticationError,
    ChatCompletionError,
    EmbeddingGenerationError,
    RateLimitExceededError,
    ResponsesError,
)


def test_every_error_exposes_status_code_provider_and_user_id():
    error = ChatCompletionError("boom", status_code=502, provider="openai", user_id="u1")
    assert (error.status_code, error.provider, error.user_id) == (502, "openai", "u1")


@pytest.mark.parametrize("error", [
    AuthenticationError(provider="anthropic"),
    APIConnectionError("https://api.example", provider="anthropic"),
    EmbeddingGenerationError("x", status_code=500, provider="anthropic"),
    ResponsesError("x", status_code=501, provider="anthropic"),
])
def test_provider_is_available_on_the_typed_errors(error):
    assert error.provider == "anthropic"


def test_attributes_are_none_when_the_context_has_no_value():
    error = ChatCompletionError("boom", status_code=500)
    assert error.provider is None
    assert error.user_id is None
    assert error.retry_after is None


def test_rate_limit_errors_expose_retry_after():
    assert RateLimitExceededError(30).retry_after == 30


def test_attributes_survive_serialization():
    from llmswitch.errors import LLMSwitchError

    restored = LLMSwitchError.from_dict(ChatCompletionError("boom", status_code=502, provider="openai").to_dict())
    assert restored.provider == "openai"
    assert restored.status_code == 502
