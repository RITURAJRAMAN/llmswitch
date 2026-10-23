"""Responses API emulated on chat completions for Anthropic and Vertex AI; only the chat provider is stubbed."""
import pytest
from pydantic import BaseModel

from llmswitch import (
    APIConnectionError,
    AsyncLLMSwitch,
    InvalidResponseRequestError,
    LLMSwitch,
    RateLimitExceededError,
    ResponsesError,
    UnsupportedCapabilityError,
)
from llmswitch.models import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChunkDelta,
    CompletionUsage,
    Function,
)
from llmswitch.providers import anthropic_provider, vertexai_provider
from llmswitch.providers._responses_emulation import (
    input_to_messages,
    text_to_response_format,
    tool_choice_to_chat,
    tools_to_chat,
)

USAGE = CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
WEATHER = {"type": "function", "name": "get_weather", "description": "Weather",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}


def completion(content="hi", tool_calls=None, finish="stop", parsed=None):
    return ChatCompletion(
        id="c-1", created=1730000000, model="stub-model", usage=USAGE,
        choices=[Choice(index=0, finish_reason=finish, message=ChatCompletionMessage(
            role="assistant", content=content, tool_calls=tool_calls, parsed=parsed))])


def tool_call(call_id="call_1", name="get_weather", arguments='{"city": "Paris"}'):
    return ChatCompletionMessageToolCall(id=call_id, type="function", function=Function(name=name, arguments=arguments))


def chunk(content=None, tool_calls=None, finish=None, usage=None, choices=True):
    delta = ChoiceDelta(role="assistant", content=content, tool_calls=tool_calls)
    return ChatCompletionChunk(id="c-1", created=1730000000, model="stub-model", usage=usage,
                               choices=[ChunkDelta(index=0, delta=delta, finish_reason=finish)] if choices else [])


class ScriptedChat:
    """Stands in for the chat provider under the emulation."""

    result = None
    calls: list = []

    def __init__(self, **kwargs):
        self.model = kwargs.get("model") or "claude-test"
        self.client = object()

    def _run(self, messages, stream, kwargs):
        ScriptedChat.calls.append({"messages": messages, "stream": stream, **kwargs})
        result = ScriptedChat.result
        if isinstance(result, Exception):
            raise result
        return result


class SyncChat(ScriptedChat):
    def create(self, messages, stream=False, **kwargs):
        result = self._run(messages, stream, kwargs)
        return iter(result) if stream else result


class AsyncChat(ScriptedChat):
    async def create(self, messages, stream=False, **kwargs):
        result = self._run(messages, stream, kwargs)
        if not stream:
            return result

        async def agen():
            for item in result:
                yield item
        return agen()

    async def aclose(self):
        ScriptedChat.calls.append("closed")


@pytest.fixture(autouse=True)
def stub_chat(monkeypatch):
    ScriptedChat.result, ScriptedChat.calls = completion(), []
    for module, sync_name, async_name in ((anthropic_provider, "AnthropicResponsesProvider", "AsyncAnthropicResponsesProvider"),
                                          (vertexai_provider, "VertexAIResponsesProvider", "AsyncVertexAIResponsesProvider")):
        monkeypatch.setattr(getattr(module, sync_name), "chat_cls", SyncChat)
        monkeypatch.setattr(getattr(module, async_name), "chat_cls", AsyncChat)


OPTIONS = {
    "anthropic": dict(provider="anthropic", api_key="k", model="claude-opus-4-5"),
    "vertexai": dict(provider="vertexai", location="us-central1", credentials_path="/x.json", model="gemini-2.5-pro"),
}


def sync_client(name, **extra):
    return LLMSwitch(**OPTIONS[name], max_retries=0, user_id=f"emu-{name}-{id(extra)}", **extra)


def async_client(name, **extra):
    return AsyncLLMSwitch(**OPTIONS[name], max_retries=0, user_id=f"aemu-{name}-{id(extra)}", **extra)


PROVIDERS = pytest.mark.parametrize("name", ["anthropic", "vertexai"])


# --- request translation -------------------------------------------------------------------------

def test_a_string_input_and_instructions_become_messages():
    assert input_to_messages("Hi", "Be brief", "anthropic") == [
        {"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hi"}]


def test_input_items_translate_roles_images_and_tool_calls():
    items = [
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": [{"type": "input_text", "text": "look"},
                                     {"type": "input_image", "image_url": "https://x/a.png", "detail": "low"}]},
        {"type": "reasoning", "id": "r1", "summary": []},
        {"role": "assistant", "content": [{"type": "output_text", "text": "calling"}]},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        {"type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
        {"type": "function_call_output", "call_id": "c2", "output": {"t": 1}},
    ]
    messages = input_to_messages(items, None, "vertexai")
    assert messages[0] == {"role": "system", "content": "rules"}
    assert messages[1]["content"] == [{"type": "text", "text": "look"},
                                      {"type": "image_url", "image_url": {"url": "https://x/a.png", "detail": "low"}}]
    assert messages[2]["role"] == "assistant" and messages[2]["content"] == "calling"
    assert [c["id"] for c in messages[2]["tool_calls"]] == ["c1", "c2"]
    assert messages[3:] == [{"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                            {"role": "tool", "tool_call_id": "c2", "content": '{"t": 1}'}]


def test_function_calls_without_a_preceding_message_get_their_own_assistant_message():
    messages = input_to_messages([{"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}],
                                 None, "anthropic")
    assert messages == [{"role": "assistant", "content": None,
                         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]


@pytest.mark.parametrize("bad", [[{"type": "item_reference", "id": "x"}], [{"role": "tool", "content": "x"}],
                                 [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}],
                                 [{"role": "user", "content": [{"type": "input_image", "file_id": "f"}]}], []])
def test_unsupported_input_is_rejected(bad):
    with pytest.raises(InvalidResponseRequestError):
        input_to_messages(bad, None, "anthropic")


def test_function_tools_and_tool_choice_translate_to_the_chat_shape():
    assert tools_to_chat([WEATHER], "anthropic") == [{"type": "function", "function": {
        "name": "get_weather", "description": "Weather", "parameters": WEATHER["parameters"]}}]
    assert tool_choice_to_chat({"type": "function", "name": "get_weather"}, "anthropic") == {
        "type": "function", "function": {"name": "get_weather"}}
    assert tool_choice_to_chat("required", "anthropic") == "required"
    with pytest.raises(InvalidResponseRequestError):
        tool_choice_to_chat({"type": "allowed_tools", "tools": []}, "anthropic")


def test_text_format_translates_to_a_chat_response_format():
    assert text_to_response_format({"format": {"type": "text"}}, "anthropic") is None
    assert text_to_response_format({"format": {"type": "json_object"}}, "anthropic") == {"type": "json_object"}
    assert text_to_response_format({"format": {"type": "json_schema", "name": "n", "schema": {"a": 1}, "strict": True}},
                                   "anthropic") == {"type": "json_schema",
                                                    "json_schema": {"name": "n", "schema": {"a": 1}, "strict": True}}


# --- create ---------------------------------------------------------------------------------------

@PROVIDERS
def test_create_returns_a_response_built_from_the_chat_completion(name):
    response = sync_client(name).responses.create(
        input="Hello", instructions="Be brief", temperature=0.2, max_output_tokens=50, tools=[WEATHER])

    assert response.output_text == "hi"
    assert response.status == "completed"
    assert response.id.startswith("resp_") and response.output[0].id.startswith("msg_")
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (5, 3, 8)
    assert response.model == "stub-model" and response.created_at == 1730000000
    assert response.temperature == 0.2 and response.instructions == "Be brief"
    assert response.provider_extensions.fields["emulated_via"] == "chat.completions"
    sent = ScriptedChat.calls[0]
    assert sent["messages"] == [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hello"}]
    assert sent["temperature"] == 0.2 and sent["max_output_tokens"] == 50 and sent["stream"] is False
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert sent["model"] == OPTIONS[name]["model"]


@PROVIDERS
def test_a_tool_call_becomes_a_function_call_item(name):
    ScriptedChat.result = completion(content=None, tool_calls=[tool_call()], finish="tool_calls")
    response = sync_client(name).responses.create(input="Weather?", tools=[WEATHER])
    assert [item.type for item in response.output] == ["function_call"]
    call = response.output[0]
    assert (call.name, call.call_id, call.arguments) == ("get_weather", "call_1", '{"city": "Paris"}')
    assert response.output_text == ""


@PROVIDERS
def test_a_truncated_completion_is_incomplete(name):
    ScriptedChat.result = completion(finish="length")
    response = sync_client(name).responses.create(input="x")
    assert response.status == "incomplete"
    assert response.incomplete_details.reason == "max_output_tokens"


@PROVIDERS
def test_structured_output_is_parsed_from_text_format(name):
    class Answer(BaseModel):
        city: str

    ScriptedChat.result = completion('{"city": "Paris"}')
    response = sync_client(name).responses.parse("Capital?", text_format=Answer)
    assert response.output_parsed == Answer(city="Paris")
    assert ScriptedChat.calls[0]["response_format"] is Answer


@PROVIDERS
def test_a_json_schema_in_text_is_sent_as_a_response_format(name):
    fmt = {"format": {"type": "json_schema", "name": "n", "schema": {"type": "object"}}}
    sync_client(name).responses.create(input="x", text=fmt)
    assert ScriptedChat.calls[0]["response_format"]["json_schema"]["name"] == "n"


@PROVIDERS
def test_fields_without_a_chat_equivalent_are_accepted_and_not_sent(name):
    sync_client(name).responses.create(input="x", metadata={"a": "b"}, user="u", truncation="auto", store=False)
    assert set(ScriptedChat.calls[0]) == {"messages", "stream", "model"}


# --- streaming ------------------------------------------------------------------------------------

def text_stream():
    return [chunk("Hel"), chunk("lo"), chunk(finish="stop"), chunk(usage=USAGE, choices=False)]


@PROVIDERS
def test_a_text_stream_produces_the_responses_event_sequence(name):
    ScriptedChat.result = text_stream()
    events = list(sync_client(name).responses.create(input="x", stream=True))

    assert [e.type for e in events] == [
        "response.created", "response.in_progress", "response.output_item.added", "response.content_part.added",
        "response.output_text.delta", "response.output_text.delta", "response.output_text.done",
        "response.content_part.done", "response.output_item.done", "response.completed"]
    assert [e.sequence_number for e in events] == list(range(len(events)))
    assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == "Hello"
    done = next(e for e in events if e.type == "response.output_text.done")
    assert done.text == "Hello"
    final = events[-1].response
    assert final.status == "completed" and final.output_text == "Hello" and final.usage.total_tokens == 8
    assert events[0].response.status == "in_progress" and events[0].response.output == []
    assert events[-1].usage.total_tokens == 8


@PROVIDERS
def test_a_tool_call_stream_produces_function_call_events(name):
    first = ChoiceDeltaToolCall(index=0, id="call_9", type="function",
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"city"'))
    second = ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments=': "Paris"}'))
    ScriptedChat.result = [chunk(tool_calls=[first]), chunk(tool_calls=[second]), chunk(finish="tool_calls"),
                           chunk(usage=USAGE, choices=False)]
    events = list(sync_client(name).responses.create(input="Weather?", tools=[WEATHER], stream=True))

    types = [e.type for e in events]
    assert types[2:] == ["response.output_item.added", "response.function_call_arguments.delta",
                         "response.function_call_arguments.delta", "response.function_call_arguments.done",
                         "response.output_item.done", "response.completed"]
    item = events[-1].response.output[0]
    assert (item.type, item.name, item.call_id, item.arguments) == ("function_call", "get_weather", "call_9",
                                                                    '{"city": "Paris"}')
    assert events[2].item["name"] == "get_weather"


@PROVIDERS
def test_a_truncated_stream_ends_with_an_incomplete_event(name):
    ScriptedChat.result = [chunk("Hel"), chunk(finish="length")]
    events = list(sync_client(name).responses.create(input="x", stream=True))
    assert events[-1].type == "response.incomplete"
    assert events[-1].response.incomplete_details.reason == "max_output_tokens"


@PROVIDERS
def test_a_structured_stream_sends_the_schema_as_a_response_format(name):
    class Answer(BaseModel):
        city: str

    ScriptedChat.result = text_stream()
    list(sync_client(name).responses.create(input="x", stream=True, text_format=Answer))
    assert ScriptedChat.calls[0]["response_format"]["type"] == "json_schema"


# --- what is not emulated --------------------------------------------------------------------------

@PROVIDERS
@pytest.mark.parametrize("kwargs, feature", [
    ({"previous_response_id": "resp_1"}, "previous_response_id"),
    ({"conversation": "conv_1"}, "conversation"),
    ({"background": True}, "background"),
    ({"store": True}, "store"),
    ({"prompt": {"id": "pmpt_1"}}, "prompt"),
    ({"tools": [{"type": "web_search"}]}, "tool.web_search"),
    ({"tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]}, "tool.code_interpreter"),
])
def test_server_side_features_are_rejected_before_any_request(name, kwargs, feature):
    with pytest.raises(UnsupportedCapabilityError, match=feature) as info:
        sync_client(name).responses.create(input="x", **kwargs)
    assert "emulated on top of chat completions" in str(info.value)
    assert ScriptedChat.calls == []


@PROVIDERS
def test_stored_object_verbs_are_unsupported(name):
    responses = sync_client(name).responses
    for call in (lambda: responses.retrieve("resp_1"), lambda: responses.delete("resp_1"),
                 lambda: responses.cancel("resp_1"), lambda: responses.compact(input="x"),
                 lambda: responses.input_items.list("resp_1"), lambda: responses.input_tokens.count(input="x"),
                 lambda: responses.conversations.create()):
        with pytest.raises(UnsupportedCapabilityError):
            call()


@PROVIDERS
def test_unknown_fields_are_still_rejected(name):
    with pytest.raises(InvalidResponseRequestError):
        sync_client(name).responses.create(input="x", not_a_field=1)


@PROVIDERS
def test_capabilities_report_the_emulation(name):
    capabilities = sync_client(name).responses.capabilities()
    assert capabilities.emulated is True
    assert capabilities.supports("create") is True and capabilities.supports("tool.function") is True
    assert capabilities.supports("retrieve") is False and capabilities.supports("conversation") is False


# --- errors and resilience -------------------------------------------------------------------------

@PROVIDERS
def test_chat_errors_become_responses_errors_with_their_status(name):
    from llmswitch import ChatCompletionError
    ScriptedChat.result = ChatCompletionError("model overloaded", status_code=529, provider=name)
    with pytest.raises(ResponsesError) as info:
        sync_client(name).responses.create(input="x")
    assert info.value.status_code == 529 and "model overloaded" in info.value.message


@PROVIDERS
def test_typed_errors_pass_through(name):
    ScriptedChat.result = RateLimitExceededError(7)
    with pytest.raises(RateLimitExceededError):
        sync_client(name).responses.create(input="x")


@PROVIDERS
def test_a_stream_error_is_a_responses_error(name):
    from llmswitch import ChatCompletionError

    def broken():
        yield chunk("Hel")
        raise ChatCompletionError("dropped", status_code=503, provider=name)

    class BrokenChat(SyncChat):
        def create(self, messages, stream=False, **kwargs):
            return broken()

    ScriptedChat.result = None
    from llmswitch.core import registry
    cls = registry.RESPONSES_PROVIDER_REGISTRY[name]
    cls.chat_cls = BrokenChat
    try:
        with pytest.raises(ResponsesError) as info:
            list(sync_client(name).responses.create(input="x", stream=True))
    finally:
        cls.chat_cls = SyncChat
    assert info.value.status_code == 503


def test_an_openai_primary_fails_over_to_an_emulated_backup(monkeypatch):

    from llmswitch.core.registry import RESPONSES_PROVIDER_REGISTRY

    class Down:
        def __init__(self, **kwargs):
            pass

        def create(self, input=None, stream=False, **kwargs):  # noqa: A002
            raise APIConnectionError("https://api.example", provider="openai")

    monkeypatch.setitem(RESPONSES_PROVIDER_REGISTRY, "openai", Down)
    events = []
    client = LLMSwitch(provider="openai", api_key="k", model="gpt-4o", max_retries=0, user_id="emu-failover",
                       backup={"provider": "anthropic", "api_key": "k2", "model": "claude-opus-4-5"},
                       on_event=events.append)
    response = client.responses.create(input="Hello")
    assert response.output_text == "hi"
    assert [e["event"] for e in events if e["event"] == "failover"] == ["failover"]


def test_an_emulated_primary_can_fail_over_to_a_native_backup(monkeypatch):
    from llmswitch.core.registry import RESPONSES_PROVIDER_REGISTRY
    from llmswitch.models import Response

    class Native:
        def __init__(self, **kwargs):
            pass

        def create(self, input=None, stream=False, **kwargs):  # noqa: A002
            return Response(id="resp_native", output=[], status="completed")

    monkeypatch.setitem(RESPONSES_PROVIDER_REGISTRY, "openai", Native)
    ScriptedChat.result = APIConnectionError("https://api.example", provider="anthropic")
    client = LLMSwitch(**OPTIONS["anthropic"], max_retries=0, user_id="emu-failover-2",
                       backup={"provider": "openai", "api_key": "k2", "model": "gpt-4o"})
    assert client.responses.create(input="Hello").id == "resp_native"


def test_on_event_reports_the_emulated_call():
    events = []
    sync_client("anthropic", on_event=events.append).responses.create(input="x")
    assert [e["event"] for e in events] == ["request", "response"]
    assert events[0]["operation"] == "responses" and events[0]["provider"] == "anthropic"


# --- async -----------------------------------------------------------------------------------------

@PROVIDERS
async def test_async_create_builds_a_response(name):
    response = await async_client(name).responses.create(input="Hello", instructions="Be brief")
    assert response.output_text == "hi" and response.usage.total_tokens == 8
    assert ScriptedChat.calls[0]["messages"][0] == {"role": "system", "content": "Be brief"}


@PROVIDERS
async def test_async_streaming_produces_the_event_sequence(name):
    ScriptedChat.result = text_stream()
    stream = await async_client(name).responses.create(input="x", stream=True)
    events = [event async for event in stream]
    assert events[0].type == "response.created" and events[-1].type == "response.completed"
    assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == "Hello"
    assert events[-1].usage.total_tokens == 8


@PROVIDERS
async def test_async_rejects_server_side_features(name):
    with pytest.raises(UnsupportedCapabilityError):
        await async_client(name).responses.create(input="x", previous_response_id="resp_1")
    with pytest.raises(UnsupportedCapabilityError):
        await async_client(name).responses.retrieve("resp_1")


@PROVIDERS
async def test_async_errors_are_responses_errors(name):
    from llmswitch import ChatCompletionError
    ScriptedChat.result = ChatCompletionError("down", status_code=503, provider=name)
    with pytest.raises(ResponsesError) as info:
        await async_client(name).responses.create(input="x")
    assert info.value.status_code == 503


async def test_async_aclose_closes_the_chat_provider():
    client = async_client("anthropic")
    await client.responses.create(input="x")
    await client.aclose()
    assert "closed" in ScriptedChat.calls
