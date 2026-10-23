"""ChatLLMSwitch: a LangChain chat model over the switch. Only the provider SDK is replaced; skipped without langchain-core."""
import uuid

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from llmswitch import ChatCompletionError, LLMSwitchError  # noqa: E402
from llmswitch.core.registry import ASYNC_CHAT_PROVIDER_REGISTRY, CHAT_PROVIDER_REGISTRY  # noqa: E402
from llmswitch.integrations.langchain import ChatLLMSwitch  # noqa: E402
from llmswitch.models import (  # noqa: E402
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


class Script:
    """Responses queued for the next provider calls, and the requests that reached the provider."""

    responses: list = []
    calls: list = []


class ScriptedSync:
    def __init__(self, **kwargs):
        self.init = kwargs

    def create(self, messages, stream=False, **kwargs):
        Script.calls.append({"messages": messages, "stream": stream, **kwargs})
        item = Script.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return iter(item) if stream else item


class ScriptedAsync:
    def __init__(self, **kwargs):
        self.init = kwargs

    async def create(self, messages, stream=False, **kwargs):
        Script.calls.append({"messages": messages, "stream": stream, **kwargs})
        item = Script.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if not stream:
            return item

        async def agen():
            for chunk in item:
                yield chunk
        return agen()


@pytest.fixture(autouse=True)
def scripted_openai(monkeypatch):
    Script.responses, Script.calls = [], []
    monkeypatch.setitem(CHAT_PROVIDER_REGISTRY, "openai", ScriptedSync)
    monkeypatch.setitem(ASYNC_CHAT_PROVIDER_REGISTRY, "openai", ScriptedAsync)


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "sunny"


USAGE = CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)


def completion(content=None, tool_calls=None, finish="stop", usage=USAGE):
    return ChatCompletion(
        id="c-1", created=1, model="gpt-4o", usage=usage,
        choices=[Choice(index=0, finish_reason=finish,
                        message=ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls))])


def call(call_id, name, arguments):
    return ChatCompletionMessageToolCall(id=call_id, type="function", function=Function(name=name, arguments=arguments))


def chunk(content=None, tool_calls=None, finish=None, usage=None, choices=True):
    delta = ChoiceDelta(role="assistant", content=content, tool_calls=tool_calls)
    return ChatCompletionChunk(id="c-1", created=1, model="gpt-4o", usage=usage,
                               choices=[ChunkDelta(index=0, delta=delta, finish_reason=finish)] if choices else [])


def model(**options):
    options.setdefault("provider", "openai")
    options.setdefault("api_key", "k-lc")
    options.setdefault("model", "gpt-4o")
    options.setdefault("max_retries", 0)
    options.setdefault("user_id", f"lc-{uuid.uuid4().hex}")
    return ChatLLMSwitch(**options)


# --- construction ----------------------------------------------------------------------------------

def test_it_is_a_langchain_chat_model():
    assert isinstance(model(), BaseChatModel)


def test_identifying_params_name_the_provider_and_model():
    llm = model()
    assert llm._llm_type == "llmswitch"
    assert llm._identifying_params == {"provider": "openai", "model": "gpt-4o"}


def test_a_model_without_a_provider_is_rejected():
    with pytest.raises(LLMSwitchError):
        ChatLLMSwitch()


def test_switch_options_are_not_treated_as_model_fields():
    llm = model(temperature=0.5, endpoint="https://example.test")
    assert llm.temperature == 0.5
    assert "endpoint" not in type(llm).model_fields


# --- invoke ----------------------------------------------------------------------------------------

def test_invoke_returns_an_ai_message_with_usage():
    Script.responses = [completion("Hi there")]
    result = model().invoke("Hello")
    assert isinstance(result, AIMessage)
    assert result.content == "Hi there"
    assert result.usage_metadata == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
    assert Script.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]
    assert Script.calls[0]["model"] == "gpt-4o"
    assert Script.calls[0]["stream"] is False


def test_the_message_history_is_sent_in_order():
    Script.responses = [completion("ok")]
    model().invoke([SystemMessage(content="Be brief."), HumanMessage(content="Hi"),
                    AIMessage(content="Hello!"), HumanMessage(content="Again")])
    assert [m["role"] for m in Script.calls[0]["messages"]] == ["system", "user", "assistant", "user"]


def test_sampling_options_go_with_every_request():
    Script.responses = [completion("ok")]
    model(temperature=0.2, max_tokens=50, stop=["END"], call_options={"seed": 7}).invoke("x")
    sent = Script.calls[0]
    assert sent["temperature"] == 0.2
    assert sent.get("max_tokens", sent.get("max_completion_tokens")) == 50
    assert sent["stop"] == ["END"]
    assert sent["seed"] == 7


def test_per_call_stop_overrides_the_configured_stop():
    Script.responses = [completion("ok")]
    model(stop=["END"]).invoke("x", stop=["STOP"])
    assert Script.calls[0]["stop"] == ["STOP"]


def test_langchain_internal_keywords_are_not_sent():
    Script.responses = [completion("ok")]
    model().invoke("x", ls_custom_trace="abc", seed=3)
    assert "ls_custom_trace" not in Script.calls[0]
    assert Script.calls[0]["seed"] == 3


def test_a_prompt_chain_composes_with_the_model():
    Script.responses = [completion("Paris")]
    prompt = ChatPromptTemplate.from_messages([("system", "Be brief."), ("user", "{question}")])
    result = (prompt | model()).invoke({"question": "Capital of France?"})
    assert result.content == "Paris"
    assert Script.calls[0]["messages"][1] == {"role": "user", "content": "Capital of France?"}


def test_provider_errors_reach_the_caller_as_library_errors():
    Script.responses = [ChatCompletionError("bad request", status_code=400, provider="openai")]
    with pytest.raises(ChatCompletionError):
        model().invoke("x")


def test_on_event_reports_the_call():
    events = []
    Script.responses = [completion("ok")]
    model(on_event=events.append).invoke("x")
    assert [e["event"] for e in events] == ["request", "response"]


# --- tools -----------------------------------------------------------------------------------------

def test_bind_tools_sends_openai_style_tool_definitions():
    Script.responses = [completion(tool_calls=[call("c1", "get_weather", '{"city": "Paris"}')], finish="tool_calls")]
    result = model().bind_tools([get_weather]).invoke("Weather in Paris?")
    assert result.tool_calls == [{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1", "type": "tool_call"}]
    sent_tool = Script.calls[0]["tools"][0]
    assert sent_tool["type"] == "function"
    assert sent_tool["function"]["name"] == "get_weather"
    assert sent_tool["function"]["parameters"]["required"] == ["city"]


@pytest.mark.parametrize("choice, expected", [
    ("auto", "auto"),
    ("required", "required"),
    ("any", "required"),
    ("none", "none"),
    ("get_weather", {"type": "function", "function": {"name": "get_weather"}}),
])
def test_tool_choice_is_translated(choice, expected):
    Script.responses = [completion("ok")]
    model().bind_tools([get_weather], tool_choice=choice).invoke("x")
    assert Script.calls[0]["tool_choice"] == expected


def test_malformed_tool_arguments_are_reported_not_raised():
    Script.responses = [completion(tool_calls=[call("c2", "get_weather", '{"city":')], finish="tool_calls")]
    result = model().bind_tools([get_weather]).invoke("x")
    assert result.tool_calls == []
    assert result.invalid_tool_calls[0]["id"] == "c2"
    assert result.invalid_tool_calls[0]["error"]


# --- streaming -------------------------------------------------------------------------------------

def test_stream_yields_text_and_a_final_usage_chunk():
    Script.responses = [[chunk("Hel"), chunk("lo"), chunk(finish="stop"), chunk(usage=USAGE, choices=False)]]
    chunks = list(model().stream("x"))
    assert all(isinstance(c, AIMessageChunk) for c in chunks)
    assert "".join(c.content for c in chunks) == "Hello"
    usage = [c.usage_metadata for c in chunks if c.usage_metadata]
    assert usage == [{"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}]


def test_streamed_tool_call_fragments_join_into_one_call():
    first = ChoiceDeltaToolCall(index=0, id="c1", type="function",
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"city"'))
    second = ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments=': "Paris"}'))
    Script.responses = [[chunk(tool_calls=[first]), chunk(tool_calls=[second]), chunk(finish="tool_calls"),
                         chunk(usage=USAGE, choices=False)]]
    chunks = list(model().bind_tools([get_weather]).stream("x"))
    merged = chunks[0]
    for part in chunks[1:]:
        merged = merged + part
    assert merged.tool_calls == [{"name": "get_weather", "args": {"city": "Paris"}, "id": "c1", "type": "tool_call"}]


# --- async -----------------------------------------------------------------------------------------

async def test_ainvoke_returns_an_ai_message():
    Script.responses = [completion("Hi async")]
    result = await model().ainvoke("Hello")
    assert result.content == "Hi async"
    assert Script.calls[0]["stream"] is False


async def test_astream_yields_text_chunks():
    Script.responses = [[chunk("Hel"), chunk("lo"), chunk(usage=USAGE, choices=False)]]
    chunks = [c async for c in model().astream("x")]
    assert "".join(c.content for c in chunks) == "Hello"
    assert [c.usage_metadata["total_tokens"] for c in chunks if c.usage_metadata] == [8]


async def test_async_tool_calls_work():
    Script.responses = [completion(tool_calls=[call("c1", "get_weather", '{"city": "Rome"}')], finish="tool_calls")]
    result = await model().bind_tools([get_weather]).ainvoke("x")
    assert result.tool_calls[0]["args"] == {"city": "Rome"}


async def test_async_provider_errors_reach_the_caller():
    Script.responses = [ChatCompletionError("down", status_code=503, provider="openai")]
    with pytest.raises(ChatCompletionError):
        await model().ainvoke("x")


async def test_the_async_client_is_built_only_when_used():
    llm = model()
    assert llm._async_switch is None
    Script.responses = [completion("ok")]
    await llm.ainvoke("x")
    assert llm._async_switch is not None


def test_close_closes_the_sync_client(monkeypatch):
    llm = model()
    closed = []
    monkeypatch.setattr(llm._switch, "close", lambda: closed.append(True))
    llm.close()
    assert closed == [True]


async def test_aclose_closes_both_clients(monkeypatch):
    llm = model()
    Script.responses = [completion("ok")]
    await llm.ainvoke("x")
    closed = []

    async def fake_aclose():
        closed.append("async")

    monkeypatch.setattr(llm._async_switch, "aclose", fake_aclose)
    monkeypatch.setattr(llm._switch, "close", lambda: closed.append("sync"))
    await llm.aclose()
    assert closed == ["async", "sync"]


def test_tool_choice_false_means_no_tools():
    Script.responses = [completion("ok")]
    model().bind_tools([get_weather], tool_choice=False).invoke("x")
    assert Script.calls[0]["tool_choice"] == "none"


def test_concurrent_first_async_calls_share_one_client():
    import threading

    llm = model()
    built = []
    original = ChatLLMSwitch._async

    results = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(id(original(llm)))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    built.extend(results)
    assert len(set(built)) == 1
