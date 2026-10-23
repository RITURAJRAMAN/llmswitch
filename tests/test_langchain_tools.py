"""The bounded tool loop. Uses a scripted chat model; skipped without langchain-core."""
import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from pydantic import Field  # noqa: E402

from llmswitch import LLMSwitchError  # noqa: E402
from llmswitch.integrations.langchain import ToolLoopLimitError, arun_tool_loop, run_tool_loop  # noqa: E402


class Scripted(BaseChatModel):
    """Returns queued AI messages and records the history it was sent on each call."""

    script: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _next(self, messages):
        self.seen.append(list(messages))
        return self.script.pop(0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def boom(text: str) -> str:
    """Always fails."""
    raise RuntimeError(f"cannot handle {text}")


@tool
def describe(item: dict) -> dict:
    """Return the item as a dict."""
    return {"item": item}


def test_a_model_answer_without_tools_returns_the_history():
    model = Scripted(script=[AIMessage(content="Hello!")])
    result = run_tool_loop(model, "Hi", [add])
    assert [type(m) for m in result] == [HumanMessage, AIMessage]
    assert result[-1].content == "Hello!"


def test_a_tool_result_goes_back_to_the_model():
    model = Scripted(script=[call("add", {"a": 2, "b": 3}, "c1"), AIMessage(content="The sum is 5.")])
    result = run_tool_loop(model, "2 + 3?", [add])
    assert isinstance(result[2], ToolMessage)
    assert (result[2].content, result[2].tool_call_id, result[2].status) == ("5", "c1", "success")
    assert result[-1].content == "The sum is 5."
    assert isinstance(model.seen[1][-1], ToolMessage)


def test_several_calls_in_one_turn_all_run_in_order():
    turn = AIMessage(content="", tool_calls=[
        {"name": "add", "args": {"a": 1, "b": 1}, "id": "a1", "type": "tool_call"},
        {"name": "add", "args": {"a": 2, "b": 2}, "id": "a2", "type": "tool_call"},
    ])
    model = Scripted(script=[turn, AIMessage(content="done")])
    result = run_tool_loop(model, "x", [add])
    assert [m.content for m in result if isinstance(m, ToolMessage)] == ["2", "4"]


def test_an_unknown_tool_name_is_reported_to_the_model_which_can_recover():
    model = Scripted(script=[call("multiply", {}, "m1"), AIMessage(content="I could not multiply.")])
    result = run_tool_loop(model, "x", [add])
    error = result[2]
    assert error.status == "error" and "unknown tool 'multiply'" in error.content
    assert "add" in error.content
    assert result[-1].content == "I could not multiply."


def test_a_failing_tool_is_reported_not_raised():
    model = Scripted(script=[call("boom", {"text": "x"}, "b1"), AIMessage(content="It failed.")])
    result = run_tool_loop(model, "x", [boom])
    assert result[2].status == "error"
    assert result[2].content == "Error: RuntimeError: cannot handle x"


def test_invalid_arguments_for_a_tool_are_reported_by_the_tool_schema():
    model = Scripted(script=[call("add", {"a": "two", "b": 3}, "v1"), AIMessage(content="Fixed.")])
    result = run_tool_loop(model, "x", [add])
    assert result[2].status == "error"
    assert "ValidationError" in result[2].content


def test_malformed_json_arguments_are_reported_back():
    turn = AIMessage(content="", invalid_tool_calls=[
        {"name": "add", "args": '{"a":', "id": "j1", "error": "Expecting value", "type": "invalid_tool_call"}])
    model = Scripted(script=[turn, AIMessage(content="ok")])
    result = run_tool_loop(model, "x", [add])
    assert result[2].status == "error"
    assert result[2].tool_call_id == "j1"
    assert "not valid JSON" in result[2].content


def test_a_dict_result_is_sent_as_json():
    model = Scripted(script=[call("describe", {"item": {"k": 1}}, "d1"), AIMessage(content="ok")])
    result = run_tool_loop(model, "x", [describe])
    assert result[2].content == '{"item": {"k": 1}}'


def test_plain_functions_can_be_tools():
    def shout(text: str) -> str:
        """Upper-case the text."""
        return text.upper()

    model = Scripted(script=[call("shout", {"text": "hi"}, "s1"), AIMessage(content="HI")])
    result = run_tool_loop(model, "x", [shout])
    assert result[2].content == "HI"


def test_the_loop_stops_at_max_steps():
    model = Scripted(script=[call("add", {"a": 1, "b": 1}, f"c{i}") for i in range(3)])
    with pytest.raises(ToolLoopLimitError, match="after 2 model calls") as info:
        run_tool_loop(model, "x", [add], max_steps=2)
    assert info.value.max_steps == 2
    assert isinstance(info.value, LLMSwitchError)
    assert len(model.seen) == 2


def test_the_last_allowed_call_can_still_answer():
    model = Scripted(script=[call("add", {"a": 1, "b": 2}, "c1"), AIMessage(content="3")])
    assert run_tool_loop(model, "x", [add], max_steps=2)[-1].content == "3"


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_max_steps_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="max_steps"):
        run_tool_loop(Scripted(), "x", [add], max_steps=bad)


def test_duplicate_tool_names_are_rejected():
    def add(a: int) -> int:  # shadows the LangChain tool name "add"
        """Shadow."""
        return a

    with pytest.raises(ValueError, match="Duplicate tool name: add"):
        run_tool_loop(Scripted(), "x", [add, globals()["add"]])


# --- async -----------------------------------------------------------------------------------------

async def test_the_async_loop_runs_a_sync_tool():
    model = Scripted(script=[call("add", {"a": 4, "b": 4}, "c1"), AIMessage(content="8")])
    result = await arun_tool_loop(model, "x", [add])
    assert result[2].content == "8"
    assert result[-1].content == "8"


async def test_the_async_loop_awaits_async_plain_functions():
    async def double(n: int) -> int:
        """Double a number."""
        return n * 2

    model = Scripted(script=[call("double", {"n": 21}, "d1"), AIMessage(content="42")])
    result = await arun_tool_loop(model, "x", [double])
    assert result[2].content == "42"


async def test_the_async_loop_reports_failures_and_limits():
    model = Scripted(script=[call("boom", {"text": "y"}, "b1"), AIMessage(content="no")])
    result = await arun_tool_loop(model, "x", [boom])
    assert result[2].status == "error"

    looping = Scripted(script=[call("add", {"a": 1, "b": 1}, "c") for _ in range(2)])
    with pytest.raises(ToolLoopLimitError):
        await arun_tool_loop(looping, "x", [add], max_steps=1)


# --- through the real chat model -------------------------------------------------------------------

def test_the_loop_runs_through_chat_llmswitch(monkeypatch):
    import uuid

    from llmswitch.core.registry import CHAT_PROVIDER_REGISTRY
    from llmswitch.integrations.langchain import ChatLLMSwitch
    from llmswitch.models import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall, Choice, Function

    replies = [
        ChatCompletion(id="r1", created=1, model="gpt-4o", choices=[Choice(index=0, finish_reason="tool_calls", message=ChatCompletionMessage(role="assistant", content=None, tool_calls=[ChatCompletionMessageToolCall(id="t1", type="function", function=Function(name="add", arguments='{"a": 2, "b": 3}'))]))]),
        ChatCompletion(id="r2", created=1, model="gpt-4o", choices=[Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="It is 5."))]),
    ]
    sent = []

    class Provider:
        def __init__(self, **kwargs):
            pass

        def create(self, messages, stream=False, **kwargs):
            sent.append(messages)
            return replies.pop(0)

    monkeypatch.setitem(CHAT_PROVIDER_REGISTRY, "openai", Provider)
    llm = ChatLLMSwitch(provider="openai", api_key="k-loop", model="gpt-4o", max_retries=0,
                        user_id=f"loop-{uuid.uuid4().hex}")
    result = run_tool_loop(llm, "2 + 3?", [add])
    assert result[-1].content == "It is 5."
    assert result[2].content == "5"
    assert sent[1][-1] == {"role": "tool", "tool_call_id": "t1", "content": "5"}
