"""Responses mode of the chat model (api="responses"): request, reply, streaming and failures."""
import uuid

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from llmswitch import ResponsesError  # noqa: E402
from llmswitch.agentic import ChatLLMSwitch, run_tool_loop  # noqa: E402
from llmswitch.agentic.messages import response_event_to_chunk, to_response_input  # noqa: E402
from llmswitch.core.registry import ASYNC_RESPONSES_PROVIDER_REGISTRY, RESPONSES_PROVIDER_REGISTRY  # noqa: E402
from llmswitch.models import Response, response_stream_event_from_data  # noqa: E402


def response(output, status="completed", usage=True, incomplete=None):
    data = {
        "id": "resp_1", "object": "response", "created_at": 1.0, "model": "gpt-4o", "status": status,
        "output": output,
        "usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13,
                  "input_tokens_details": {"cached_tokens": 2}, "output_tokens_details": {"reasoning_tokens": 1}}
        if usage else None,
    }
    if incomplete:
        data["incomplete_details"] = {"reason": incomplete}
    return Response.model_validate(data)


def text_item(text):
    return {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def call_item(call_id, name, arguments):
    return {"type": "function_call", "id": "fc_1", "status": "completed", "call_id": call_id,
            "name": name, "arguments": arguments}


class Script:
    responses: list = []
    calls: list = []


class ScriptedResponses:
    def __init__(self, **kwargs):
        self.init = kwargs

    def create(self, input=None, stream=False, **kwargs):  # noqa: A002
        Script.calls.append({"input": input, "stream": stream, **kwargs})
        item = Script.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return iter(item) if stream else item


class AsyncScriptedResponses(ScriptedResponses):
    async def create(self, input=None, stream=False, **kwargs):  # noqa: A002
        Script.calls.append({"input": input, "stream": stream, **kwargs})
        item = Script.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if not stream:
            return item

        async def agen():
            for event in item:
                yield event
        return agen()


@pytest.fixture(autouse=True)
def scripted(monkeypatch):
    Script.responses, Script.calls = [], []
    monkeypatch.setitem(RESPONSES_PROVIDER_REGISTRY, "openai", ScriptedResponses)
    monkeypatch.setitem(ASYNC_RESPONSES_PROVIDER_REGISTRY, "openai", AsyncScriptedResponses)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather",
                                             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "sunny"


WEATHER_TOOL = get_weather


def model(**options):
    options.setdefault("provider", "openai")
    options.setdefault("api_key", "k-resp")
    options.setdefault("model", "gpt-4o")
    options.setdefault("max_retries", 0)
    options.setdefault("user_id", f"resp-{uuid.uuid4().hex}")
    return ChatLLMSwitch(api="responses", **options)


# --- configuration ----------------------------------------------------------------------------------

@pytest.mark.parametrize("provider", ["anthropic", "vertexai", "azure_foundry"])
def test_responses_mode_is_rejected_for_providers_without_the_api(provider):
    with pytest.raises(ValueError, match="only for provider='openai' or 'azure_openai'"):
        ChatLLMSwitch(api="responses", provider=provider, api_key="k", model="m", user_id="x-" + provider)


def test_responses_mode_rejects_stop_sequences():
    with pytest.raises(ValueError, match="stop sequences are not supported"):
        ChatLLMSwitch(api="responses", provider="openai", api_key="k", model="m", stop=["END"], user_id="stop-x")


def test_chat_is_the_default_api():
    assert ChatLLMSwitch(provider="openai", api_key="k", model="m", user_id="default-x").api == "chat"


def test_an_unknown_api_name_is_rejected():
    with pytest.raises(ValueError):
        ChatLLMSwitch(api="assistants", provider="openai", api_key="k", model="m", user_id="bad-x")


# --- request ----------------------------------------------------------------------------------------

def test_invoke_uses_the_responses_api_and_maps_the_reply():
    Script.responses = [response([text_item("Hi there")])]
    result = model().invoke("Hello")
    assert isinstance(result, AIMessage)
    assert result.content == "Hi there"
    assert result.usage_metadata == {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13,
                                     "input_token_details": {"cache_read": 2},
                                     "output_token_details": {"reasoning": 1}}
    assert result.response_metadata["id"] == "resp_1"
    assert Script.calls[0]["input"] == [{"role": "user", "content": "Hello"}]
    assert Script.calls[0]["stream"] is False


def test_system_and_history_become_input_items_in_order():
    Script.responses = [response([text_item("ok")])]
    model().invoke([SystemMessage(content="Be brief."), HumanMessage(content="Hi"), AIMessage(content="Hello!")])
    assert Script.calls[0]["input"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_max_tokens_is_sent_as_max_output_tokens():
    Script.responses = [response([text_item("ok")])]
    model(max_tokens=50, temperature=0.3).invoke("x")
    assert Script.calls[0]["max_output_tokens"] == 50
    assert "max_tokens" not in Script.calls[0]
    assert Script.calls[0]["temperature"] == 0.3


def test_tools_are_sent_in_the_flat_responses_shape():
    Script.responses = [response([call_item("c1", "get_weather", '{"city": "Paris"}')])]
    result = model().bind_tools([WEATHER_TOOL]).invoke("Weather?")
    sent = Script.calls[0]["tools"][0]
    assert sent["type"] == "function" and sent["name"] == "get_weather"
    assert sent["parameters"]["required"] == ["city"]
    assert "function" not in sent
    assert result.tool_calls == [{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1", "type": "tool_call"}]


def test_a_forced_tool_choice_is_sent_flat():
    Script.responses = [response([text_item("ok")])]
    model().bind_tools([WEATHER_TOOL], tool_choice="get_weather").invoke("x")
    assert Script.calls[0]["tool_choice"] == {"type": "function", "name": "get_weather"}


def test_the_tool_loop_sends_results_as_function_call_output_items():
    Script.responses = [response([call_item("c9", "get_weather", '{"city": "Rome"}')]),
                        response([text_item("It is sunny in Rome.")])]
    history = run_tool_loop(model(), "Weather in Rome?", [WEATHER_TOOL], max_steps=3)
    assert history[-1].content == "It is sunny in Rome."
    second_input = Script.calls[1]["input"]
    assert {"type": "function_call", "call_id": "c9", "name": "get_weather", "arguments": '{"city": "Rome"}'} in second_input
    assert second_input[-1]["type"] == "function_call_output"
    assert second_input[-1]["call_id"] == "c9"


def test_malformed_function_arguments_are_invalid_tool_calls():
    Script.responses = [response([call_item("c2", "get_weather", '{"city":')])]
    result = model().bind_tools([WEATHER_TOOL]).invoke("x")
    assert result.tool_calls == []
    assert result.invalid_tool_calls[0]["id"] == "c2"


def test_reasoning_summaries_and_hosted_tool_items_are_kept():
    Script.responses = [response([
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Thinking about it."}]},
        {"type": "web_search_call", "id": "ws_1", "status": "completed"},
        text_item("Answer."),
    ])]
    result = model().invoke("x")
    assert result.content == "Answer."
    assert result.additional_kwargs["reasoning_content"] == ["Thinking about it."]
    assert result.additional_kwargs["responses_output_items"][0]["type"] == "web_search_call"


def test_an_incomplete_response_says_why():
    Script.responses = [response([text_item("Partial")], status="incomplete", incomplete="max_output_tokens")]
    result = model().invoke("x")
    assert result.response_metadata["status"] == "incomplete"
    assert result.response_metadata["incomplete_reason"] == "max_output_tokens"


def test_an_image_in_a_system_message_is_rejected_for_responses_too():
    with pytest.raises(ValueError, match="only supported in user messages"):
        to_response_input([SystemMessage(content=[{"type": "image", "url": "https://x/a.png"}])])


def test_an_image_in_a_user_message_is_sent_as_an_input_image():
    items = to_response_input([HumanMessage(content=[{"type": "text", "text": "What is this?"},
                                                      {"type": "image", "url": "https://x/cat.png"}])])
    assert items == [{"role": "user", "content": [
        {"type": "input_text", "text": "What is this?"},
        {"type": "input_image", "image_url": "https://x/cat.png"},
    ]}]


def test_a_tool_message_becomes_a_function_call_output():
    items = to_response_input([ToolMessage(content="Sunny", tool_call_id="c5")])
    assert items == [{"type": "function_call_output", "call_id": "c5", "output": "Sunny"}]


# --- streaming --------------------------------------------------------------------------------------

def events(*payloads):
    return [response_stream_event_from_data(p) for p in payloads]


def text_events():
    return events(
        {"type": "response.created", "response": {"id": "resp_s", "model": "gpt-4o", "status": "in_progress", "output": []}},
        {"type": "response.output_text.delta", "delta": "Hel", "output_index": 0, "content_index": 0},
        {"type": "response.output_text.delta", "delta": "lo", "output_index": 0, "content_index": 0},
        {"type": "response.completed", "response": {
            "id": "resp_s", "model": "gpt-4o", "status": "completed", "output": [],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}},
    )


def test_stream_yields_text_and_the_usage_from_the_final_event():
    Script.responses = [text_events()]
    chunks = list(model().stream("x"))
    assert "".join(c.content for c in chunks) == "Hello"
    assert [c.usage_metadata["total_tokens"] for c in chunks if c.usage_metadata] == [5]


def test_stream_joins_function_call_fragments_into_one_call():
    Script.responses = [events(
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "function_call", "call_id": "c7", "name": "get_weather", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"city"'},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": ': "Paris"}'},
        {"type": "response.completed", "response": {"id": "r", "model": "gpt-4o", "status": "completed", "output": []}},
    )]
    chunks = list(model().bind_tools([WEATHER_TOOL]).stream("x"))
    merged = chunks[0]
    for part in chunks[1:]:
        merged = merged + part
    assert merged.tool_calls == [{"name": "get_weather", "args": {"city": "Paris"}, "id": "c7", "type": "tool_call"}]


def test_a_failed_stream_raises_instead_of_returning_an_empty_reply():
    Script.responses = [events(
        {"type": "response.output_text.delta", "delta": "Hel", "output_index": 0},
        {"type": "response.failed", "response": {"id": "r", "model": "gpt-4o", "status": "failed", "output": [],
                                                 "error": {"code": "server_error", "message": "model crashed"}}},
    )]
    with pytest.raises(ResponsesError, match="model crashed"):
        list(model().stream("x"))


def test_reasoning_deltas_are_kept_on_the_chunk():
    chunk = response_event_to_chunk(response_stream_event_from_data(
        {"type": "response.reasoning_summary_text.delta", "delta": "step"}))
    assert chunk.additional_kwargs["reasoning_delta"] == "step"


def test_other_events_produce_no_chunk():
    assert response_event_to_chunk(response_stream_event_from_data({"type": "response.in_progress"})) is None


# --- async ------------------------------------------------------------------------------------------

async def test_ainvoke_uses_the_responses_api():
    Script.responses = [response([text_item("Async hi")])]
    result = await model().ainvoke("Hello")
    assert result.content == "Async hi"
    assert Script.calls[0]["stream"] is False


async def test_astream_yields_text_from_the_responses_events():
    Script.responses = [text_events()]
    chunks = [c async for c in model().astream("x")]
    assert "".join(c.content for c in chunks) == "Hello"


async def test_async_errors_are_raised():
    Script.responses = [ResponsesError("down", status_code=503, provider="openai")]
    with pytest.raises(ResponsesError):
        await model().ainvoke("x")
