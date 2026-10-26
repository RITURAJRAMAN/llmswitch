"""The agent loop: answers, reflection, tool steps, and the limits that stop a run."""
import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from pydantic import Field  # noqa: E402

from llmswitch.agentic import Agent, AgentConfig  # noqa: E402


class Scripted(BaseChatModel):
    """Returns queued messages in order and records the input of every call."""

    script: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    bound_with: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def _identifying_params(self) -> dict:
        return {"model": "scripted-1"}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound_with.append(tool_choice)
        return self

    def _next(self, messages):
        self.seen.append(list(messages))
        return self.script.pop(0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])


def verdict(ok, feedback=""):
    """A reflection answer: the JSON the check asks the model to return as its text."""
    import json
    return AIMessage(content=json.dumps({"is_sufficient": ok, "feedback": feedback}))


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def agent(script, **config):
    model = Scripted(script=list(script))
    return model, Agent(model, tools=[add] if config.pop("with_tools", True) else None,
                        config=AgentConfig(**config) if config else None)


# --- configuration ----------------------------------------------------------------------------------

@pytest.mark.parametrize("options", [
    {"max_iterations": 0}, {"max_runtime_seconds": 0}, {"max_input_tokens_per_step": 0},
    {"reflection_max_retries": -1},
])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        AgentConfig(**options)


def test_the_defaults_match_the_reference():
    config = AgentConfig()
    assert (config.max_iterations, config.max_runtime_seconds, config.reflection_enabled,
            config.reflection_max_retries, config.tool_choice) == (12, 180.0, True, 2, "auto")


# --- completing a run -------------------------------------------------------------------------------

def test_a_direct_answer_that_passes_reflection_completes():
    model, runner = agent([AIMessage(content="Paris."), verdict(True)])
    result = runner.invoke("Capital of France?")
    assert result["status"] == "completed"
    assert result["output"] == "Paris."
    assert len(result["steps"]) == 1
    assert result["steps"][0].model_used == "scripted-1"


def test_with_reflection_off_no_review_call_is_made():
    model, runner = agent([AIMessage(content="Paris.")], reflection_enabled=False)
    result = runner.invoke("Capital of France?")
    assert result["status"] == "completed"
    assert len(model.seen) == 1


def test_a_tool_step_feeds_its_result_back_before_the_answer():
    model, runner = agent([call("add", {"a": 2, "b": 3}, "t1"), AIMessage(content="The sum is 5."), verdict(True)])
    result = runner.invoke("2 + 3?")
    assert result["status"] == "completed" and result["output"] == "The sum is 5."
    assert [type(m) for m in result["messages"]] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert result["steps"][0].tool_results[0].content == "5"
    assert len(result["steps"]) == 2


def test_tool_choice_is_passed_when_binding_tools():
    model, runner = agent([AIMessage(content="ok"), verdict(True)], tool_choice="required")
    runner.invoke("x")
    assert model.bound_with[0] == "required"


def test_without_tools_the_answer_is_not_bound_to_any_tool_choice():
    model, runner = agent([AIMessage(content="ok"), verdict(True)], with_tools=False)
    assert runner.invoke("x")["status"] == "completed"
    # Only the reflection check binds (to its own schema); the answer call is unbound.
    assert "auto" not in model.bound_with


# --- reflection -------------------------------------------------------------------------------------

def test_a_rejected_answer_is_revised_with_the_feedback():
    model, runner = agent([AIMessage(content="Lyon."), verdict(False, "Paris is the capital."),
                           AIMessage(content="Paris."), verdict(True)])
    result = runner.invoke("Capital of France?")
    assert result["status"] == "completed" and result["output"] == "Paris."
    revision = [m for m in result["messages"] if isinstance(m, HumanMessage)][-1]
    assert revision.content == "Please revise your answer. Feedback: Paris is the capital."


def test_when_revisions_run_out_the_last_answer_is_returned_with_that_status():
    model, runner = agent([AIMessage(content="a"), verdict(False, "no"),
                           AIMessage(content="b"), verdict(False, "still no")],
                          reflection_max_retries=1)
    result = runner.invoke("x")
    assert result["status"] == "completed_with_reflection_limit"
    assert result["output"] == "b"


def test_a_reflection_without_a_verdict_is_asked_again():
    model, runner = agent([AIMessage(content="ok"), AIMessage(content="not a verdict"), verdict(True)])
    result = runner.invoke("x")
    assert result["status"] == "completed"
    assert len(model.seen) == 3


def test_a_reflection_that_never_gives_a_verdict_is_an_error():
    model, runner = agent([AIMessage(content="ok")] + [AIMessage(content="no verdict")] * 3)
    with pytest.raises(ValueError, match="in 3 attempts"):
        runner.invoke("x")


# --- limits -----------------------------------------------------------------------------------------

def test_max_iterations_stops_a_run_that_keeps_calling_tools():
    model, runner = agent([call("add", {"a": 1, "b": 1}, "a"), call("add", {"a": 2, "b": 2}, "b")],
                          max_iterations=2)
    result = runner.invoke("x")
    assert result["status"] == "stopped" and result["reason"] == "max_iterations_reached"
    assert len(result["steps"]) == 2


def test_a_zero_runtime_budget_stops_before_any_model_call():
    model, runner = agent([AIMessage(content="never")])
    result = runner.invoke("x", max_runtime_seconds=0)
    assert result == {"status": "stopped", "reason": "max_runtime_exceeded",
                      "messages": result["messages"], "steps": []}
    assert model.seen == []


def test_the_per_call_runtime_limit_can_only_tighten_the_configured_one():
    model, runner = agent([AIMessage(content="ok"), verdict(True)], max_runtime_seconds=100)
    assert runner.invoke("x", max_runtime_seconds=0)["reason"] == "max_runtime_exceeded"
    with pytest.raises(ValueError):
        runner.invoke("x", max_runtime_seconds=-1)


def test_an_input_budget_stops_before_a_call_that_is_too_large():
    model, runner = agent([AIMessage(content="never")], max_input_tokens_per_step=1)
    result = runner.invoke("This question is long enough to exceed a one-token budget.")
    assert result["reason"] == "max_input_tokens_per_step_exceeded"
    assert result["input_tokens"] > 1 and result["limit"] == 1
    assert model.seen == []


# --- async ------------------------------------------------------------------------------------------

async def test_ainvoke_completes_with_tools_and_reflection():
    model, runner = agent([call("add", {"a": 4, "b": 4}, "t9"), AIMessage(content="8"), verdict(True)])
    result = await runner.ainvoke("4 + 4?")
    assert result["status"] == "completed" and result["output"] == "8"
    assert result["steps"][0].tool_results[0].content == "8"


async def test_ainvoke_revises_a_rejected_answer():
    model, runner = agent([AIMessage(content="no"), verdict(False, "wrong"), AIMessage(content="yes"), verdict(True)])
    result = await runner.ainvoke("x")
    assert result["output"] == "yes"


async def test_ainvoke_respects_the_iteration_limit():
    model, runner = agent([call("add", {"a": 1, "b": 1}, "x1"), call("add", {"a": 1, "b": 1}, "x2")],
                          max_iterations=1)
    result = await runner.ainvoke("x")
    assert result["reason"] == "max_iterations_reached"


# --- the agent on the Responses API ------------------------------------------------------------------

def test_the_agent_runs_on_the_responses_mode_of_the_chat_model(monkeypatch):
    import uuid

    from llmswitch.agentic import ChatLLMSwitch
    from llmswitch.core.registry import RESPONSES_PROVIDER_REGISTRY
    from llmswitch.models import Response

    def output_text(text, status="completed"):
        return Response.model_validate({"id": "r", "model": "gpt-4o", "status": status, "output": [
            {"type": "message", "id": "m", "status": "completed", "role": "assistant",
             "content": [{"type": "output_text", "text": text, "annotations": []}]}]})

    replies = [output_text("Hello!"), output_text('{"is_sufficient": true, "feedback": ""}')]

    class Provider:
        def __init__(self, **kwargs):
            pass

        def create(self, input=None, stream=False, **kwargs):  # noqa: A002
            return replies.pop(0)

    monkeypatch.setitem(RESPONSES_PROVIDER_REGISTRY, "openai", Provider)
    llm = ChatLLMSwitch(api="responses", provider="openai", api_key="k-agent", model="gpt-4o", max_retries=0,
                        user_id=f"agent-resp-{uuid.uuid4().hex}")
    result = Agent(llm, config=AgentConfig()).invoke("Say hello.")
    assert result["status"] == "completed"
    assert result["output"] == "Hello!"


def test_the_reflection_check_gets_text_not_tool_history():
    model, runner = agent([call("add", {"a": 2, "b": 3}, "t3"), AIMessage(content="5."), verdict(True)])
    runner.invoke("2 + 3?")
    reflection_input = model.seen[-1]
    assert len(reflection_input) == 1 and isinstance(reflection_input[0], HumanMessage)
    text = reflection_input[0].content
    assert "User: 2 + 3?" in text
    assert "Assistant called tool add with {'a': 2, 'b': 3}" in text
    assert "Tool result: 5" in text
    assert "Assistant: 5." in text
    assert text.endswith(__import__("llmswitch.agentic.agent", fromlist=["x"]).REFLECTION_PROMPT)


def test_the_reflection_check_gets_enough_output_tokens_when_the_model_limit_is_small():
    import uuid

    from llmswitch.agentic import ChatLLMSwitch
    from llmswitch.agentic.agent import REFLECTION_MIN_OUTPUT_TOKENS

    small = ChatLLMSwitch(provider="openai", api_key="k", model="gpt-4o", max_tokens=100, user_id=f"tok-{uuid.uuid4().hex}")
    runner = Agent(small)
    bound = runner._reflection_model()
    assert bound.kwargs == {"max_tokens": REFLECTION_MIN_OUTPUT_TOKENS}
    large = ChatLLMSwitch(provider="openai", api_key="k", model="gpt-4o", max_tokens=8000, user_id=f"tok2-{uuid.uuid4().hex}")
    assert Agent(large)._reflection_model() is large
    unlimited = ChatLLMSwitch(provider="openai", api_key="k", model="gpt-4o", user_id=f"tok3-{uuid.uuid4().hex}")
    assert Agent(unlimited)._reflection_model() is unlimited
