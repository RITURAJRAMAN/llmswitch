import pytest

from llmswitch import (
    APIConnectionError,
    AuthenticationError,
    ChatCompletionError,
    EmbeddingGenerationError,
    LLMSwitch,
    LLMSwitchError,
)

MESSAGES = [{"role": "user", "content": "Hello"}]


def openai_client(**overrides):
    return LLMSwitch(provider="openai", api_key="sk-test", model="gpt-4o", **overrides)


def azure_client():
    return LLMSwitch(provider="azure_openai", api_key="k", endpoint="https://x.openai.azure.com/",
                     api_version="2024-10-21", deployment_name="prod-gpt4o")


def test_chat_create_forwards_messages_and_configured_model(stubs):
    StubChat, _ = stubs
    result = openai_client().chat.completions.create(messages=MESSAGES)
    assert result == {"ok": True, "stream": False}
    call = StubChat.calls[-1]
    assert call["messages"] == MESSAGES
    assert call["stream"] is False
    assert call["model"] == "gpt-4o"


def test_per_call_model_overrides_configured_model(stubs):
    StubChat, _ = stubs
    openai_client().chat.completions.create(messages=MESSAGES, model="gpt-4o-mini")
    assert StubChat.calls[-1]["model"] == "gpt-4o-mini"


def test_deployment_name_takes_precedence_over_per_call_model(stubs):
    StubChat, _ = stubs
    azure_client().chat.completions.create(messages=MESSAGES, model="other-model")
    assert StubChat.calls[-1]["model"] == "prod-gpt4o"


def test_streaming_requests_usage_in_the_final_chunk(stubs):
    StubChat, _ = stubs
    result = openai_client().chat.completions.create(messages=MESSAGES, stream=True)
    assert result == {"ok": True, "stream": True}
    assert StubChat.calls[-1]["stream_options"] == {"include_usage": True}


def test_caller_stream_options_are_not_overwritten(stubs):
    StubChat, _ = stubs
    openai_client().chat.completions.create(messages=MESSAGES, stream=True,
                                            stream_options={"include_usage": False})
    assert StubChat.calls[-1]["stream_options"] == {"include_usage": False}


def test_provider_parameters_pass_through(stubs):
    StubChat, _ = stubs
    openai_client().chat.completions.create(messages=MESSAGES, temperature=0.2, max_tokens=50)
    assert StubChat.calls[-1]["temperature"] == 0.2
    assert StubChat.calls[-1]["max_tokens"] == 50


@pytest.mark.parametrize("messages", [[], "hello", None, ["not a dict"]])
def test_invalid_messages_are_rejected_before_dispatch(stubs, messages):
    StubChat, _ = stubs
    with pytest.raises(ValueError):
        openai_client().chat.completions.create(messages=messages)
    assert StubChat.calls == []


def test_parse_passes_response_format(stubs):
    StubChat, _ = stubs

    class Answer:
        pass

    openai_client().chat.completions.parse(MESSAGES, response_format=Answer)
    assert StubChat.calls[-1]["response_format"] is Answer
    assert StubChat.calls[-1]["stream"] is False


def test_chat_on_vertexai_is_dispatched_to_its_provider(stubs):
    StubChat, _ = stubs
    client = LLMSwitch(provider="vertexai", location="us-central1", credentials_path="/tmp/sa.json",
                       model="gemini-2.5-pro")
    client.chat.completions.create(messages=MESSAGES)
    assert StubChat.calls[-1]["model"] == "gemini-2.5-pro"


def test_embeddings_accept_a_single_string(stubs):
    _, StubEmbeddings = stubs
    result = openai_client().embeddings.create("hello")
    assert StubEmbeddings.calls[-1]["input"] == ["hello"]
    assert result["data"] == [[5.0]]


def test_embeddings_preserve_input_order(stubs):
    _, StubEmbeddings = stubs
    result = openai_client().embeddings.create(["a", "bbb", "cc"])
    assert StubEmbeddings.calls[-1]["input"] == ["a", "bbb", "cc"]
    assert result["data"] == [[1.0], [3.0], [2.0]]


def test_embeddings_use_configured_model(stubs):
    _, StubEmbeddings = stubs
    openai_client().embeddings.create(["a"], model="text-embedding-3-small")
    assert StubEmbeddings.calls[-1]["model"] == "text-embedding-3-small"


@pytest.mark.parametrize("value", [[], "", None])
def test_empty_embedding_input_is_rejected(stubs, value):
    _, StubEmbeddings = stubs
    with pytest.raises(ValueError):
        openai_client().embeddings.create(value)
    assert StubEmbeddings.calls == []


def test_non_string_embedding_input_is_rejected(stubs):
    with pytest.raises(ValueError, match="string"):
        openai_client().embeddings.create(["ok", 3])


def test_embeddings_are_unsupported_for_vertexai():
    client = LLMSwitch(provider="vertexai", location="us-central1", credentials_path="/tmp/sa.json")
    with pytest.raises(LLMSwitchError, match="Unsupported provider for embeddings: vertexai"):
        client.embeddings.create("hello")


def test_embeddings_are_unsupported_for_anthropic():
    client = LLMSwitch(provider="anthropic", api_key="k")
    with pytest.raises(LLMSwitchError, match="Unsupported provider for embeddings: anthropic"):
        client.embeddings.create("hello")


def test_value_error_from_provider_maps_to_chat_error_400(stubs):
    StubChat, _ = stubs
    StubChat.error = ValueError("bad temperature")
    with pytest.raises(ChatCompletionError, match="bad temperature") as info:
        openai_client().chat.completions.create(messages=MESSAGES)
    assert info.value.status_code == 400


def test_connection_error_maps_to_api_connection_error(stubs):
    StubChat, _ = stubs
    StubChat.error = ConnectionError("reset")
    with pytest.raises(APIConnectionError):
        openai_client().chat.completions.create(messages=MESSAGES)


def test_permission_error_maps_to_authentication_error(stubs):
    StubChat, _ = stubs
    StubChat.error = PermissionError("denied")
    with pytest.raises(AuthenticationError):
        openai_client().chat.completions.create(messages=MESSAGES)


def test_unknown_exception_keeps_status_code_when_present(stubs):
    StubChat, _ = stubs

    class Upstream(Exception):
        status_code = 429

    StubChat.error = Upstream("slow down")
    with pytest.raises(ChatCompletionError) as info:
        openai_client().chat.completions.create(messages=MESSAGES)
    assert info.value.status_code == 429


def test_library_errors_pass_through_unchanged(stubs):
    StubChat, _ = stubs
    original = AuthenticationError(provider="openai")
    StubChat.error = original
    with pytest.raises(AuthenticationError) as info:
        openai_client().chat.completions.create(messages=MESSAGES)
    assert info.value is original


def test_embedding_errors_map_to_embedding_generation_error(stubs):
    _, StubEmbeddings = stubs
    StubEmbeddings.error = RuntimeError("upstream down")
    with pytest.raises(EmbeddingGenerationError):
        openai_client().embeddings.create("hello")


def test_provider_client_is_built_once_and_reused(stubs):
    client = openai_client()
    client.chat.completions.create(messages=MESSAGES)
    first = client._providers[next(iter(client._providers))]
    client.chat.completions.create(messages=MESSAGES)
    assert len(client._providers) == 1
    assert client._provider(type(first)) is first


def test_close_closes_provider_clients_and_context_manager_closes(stubs):
    StubChat, _ = stubs
    with openai_client() as client:
        client.chat.completions.create(messages=MESSAGES)
    assert StubChat.closed is True
    assert client._providers == {}
