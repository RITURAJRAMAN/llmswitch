"""Regression tests for the problems found in review of the 0.1.4 agentic code."""
import asyncio
import time

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from test_agentic_agent import Scripted, add, call, verdict  # noqa: E402
from test_agentic_specialists import (  # noqa: E402
    AsyncSlow,
    Named,  # noqa: E402
    Slow,
)

from llmswitch.agentic import (  # noqa: E402
    Agent,
    AgentConfig,
    CollectingEventHandler,
    InMemoryCheckpointStore,
    RoundRobinRoutingPolicy,
    SharedState,
    idempotent,
    invoke_specialists,
)


def test_a_reflection_check_with_no_readable_verdict_stops_the_run_with_a_result():
    model = Scripted(script=[AIMessage(content="an answer"), AIMessage(content="prose"),
                             AIMessage(content="more prose"), AIMessage(content="still prose")])
    result = Agent(model, config=AgentConfig()).invoke("question")

    assert result["status"] == "stopped"
    assert result["reason"] == "reflection_verdict_unreadable"
    assert len(result["steps"]) == 1
    assert result["steps"][0].assistant_content == "an answer"


async def test_the_async_reflection_check_with_no_readable_verdict_also_returns_a_result():
    model = Scripted(script=[AIMessage(content="an answer"), AIMessage(content="prose"),
                             AIMessage(content="more prose"), AIMessage(content="still prose")])
    result = await Agent(model, config=AgentConfig()).ainvoke("question")

    assert result["status"] == "stopped"
    assert result["reason"] == "reflection_verdict_unreadable"


def test_resuming_an_uncertain_tool_run_with_feedback_returns_the_interrupted_result():
    runs = {"crash": 0}

    @tool
    def finished(x: int) -> str:
        """Finishes."""
        return "done"

    @tool
    def crashes(x: int) -> str:
        """Crashes once, after it has started."""
        runs["crash"] += 1
        if runs["crash"] == 1:
            raise SystemExit("process killed")
        return "recovered"

    store = InMemoryCheckpointStore()
    model = Scripted(script=[AIMessage(content="", tool_calls=[
        {"name": "finished", "args": {"x": 1}, "id": "f1", "type": "tool_call"},
        {"name": "crashes", "args": {"x": 2}, "id": "c1", "type": "tool_call"},
    ])])
    agent = Agent(model, tools=[finished, crashes], config=AgentConfig(parallel_tools=False), checkpoint_store=store)
    with pytest.raises(SystemExit):
        agent.invoke("x")
    checkpoint = next(cid for cid in store._data if store.load(cid)["phase"] == "awaiting_tools")

    result = agent.resume(checkpoint, user_feedback="try another way")

    assert result["status"] == "interrupted"
    assert result["reason"] == "uncertain_tool_execution"
    assert result["checkpoint_id"] == checkpoint


def test_a_chat_model_reports_its_deployment_name_as_the_model_used():
    from llmswitch.agentic import ChatLLMSwitch

    llm = ChatLLMSwitch(provider="azure_openai", api_key="not-a-real-key", endpoint="https://example.invalid",
                        api_version="2024-10-21", deployment_name="my-deployment", model="gpt-4o-mini")
    try:
        assert llm._identifying_params["model"] == "my-deployment"
    finally:
        llm.close()


def test_a_specialist_that_timed_out_does_not_write_to_the_shared_state_later():
    state = SharedState()
    specialists = {
        "slow": Agent(Slow(script=[AIMessage(content="late")], delay=1.0), config=AgentConfig(reflection_enabled=False)),
        "fast": Agent(Scripted(script=[AIMessage(content="on time")]), config=AgentConfig(reflection_enabled=False)),
    }
    rows = invoke_specialists(task="go", specialists=specialists, shared_state=state, max_runtime_seconds=0.3)

    assert {row["specialist"]: row["status"] for row in rows} == {"slow": "timeout", "fast": "completed"}
    time.sleep(1.2)  # long enough for the slow specialist's thread to finish
    assert sorted(state.snapshot()) == ["specialist_outputs.fast"]


async def test_an_async_specialist_that_timed_out_does_not_write_to_the_shared_state():
    from llmswitch.agentic import ainvoke_specialists

    state = SharedState()
    specialists = {
        "slow": Agent(AsyncSlow(script=[AIMessage(content="late")], delay=1.0), config=AgentConfig(reflection_enabled=False)),
        "fast": Agent(Scripted(script=[AIMessage(content="on time")]), config=AgentConfig(reflection_enabled=False)),
    }
    rows = await ainvoke_specialists(task="go", specialists=specialists, shared_state=state, max_runtime_seconds=0.3)
    await asyncio.sleep(0.1)

    assert {row["specialist"]: row["status"] for row in rows} == {"slow": "timeout", "fast": "completed"}
    assert sorted(state.snapshot()) == ["specialist_outputs.fast"]


def test_a_failed_tool_with_no_retries_stops_as_tool_failed():
    @tool
    def broken(x: int) -> str:
        """Breaks."""
        raise RuntimeError("nope")

    model = Scripted(script=[call("broken", {"x": 1}, "b1")])
    result = Agent(model, tools=[broken], config=AgentConfig(escalate_on_retry_exhausted=True)).invoke("x")

    assert result["status"] == "stopped" and result["reason"] == "tool_failed"


def test_a_failed_tool_that_stays_failed_after_its_retries_stops_as_retry_exhausted():
    @idempotent
    @tool
    def broken(x: int) -> str:
        """Breaks."""
        raise RuntimeError("nope")

    model = Scripted(script=[call("broken", {"x": 1}, "b1")])
    config = AgentConfig(escalate_on_retry_exhausted=True, tool_retry_attempts=1)
    result = Agent(model, tools=[broken], config=config).invoke("x")

    assert result["status"] == "stopped" and result["reason"] == "tool_retry_exhausted"


def test_a_refused_tool_call_still_gets_a_start_event_and_an_error_event():
    handler = CollectingEventHandler()
    model = Scripted(script=[call("missing", {"x": 1}, "m1"), AIMessage(content="done"), verdict(True)])
    Agent(model, tools=[add], config=AgentConfig(), event_handler=handler).invoke("x")

    tool_events = [(e.event, e.payload["tool_name"]) for e in handler.events if e.event.startswith("tool_")]
    assert tool_events == [("tool_start", "missing"), ("tool_error", "missing")]
    error = next(e for e in handler.events if e.event == "tool_error")
    assert "not found" in error.payload["error"].lower()


def test_the_per_call_model_is_the_one_reported_for_steps_unless_a_deployment_name_is_set():
    from llmswitch.agentic import ChatLLMSwitch

    per_call = ChatLLMSwitch(provider="openai", api_key="not-a-real-key", model="configured",
                             call_options={"model": "per-call"})
    deployed = ChatLLMSwitch(provider="azure_openai", api_key="not-a-real-key", endpoint="https://example.invalid",
                             api_version="2024-10-21", deployment_name="my-deployment",
                             call_options={"model": "per-call"})
    try:
        assert per_call._identifying_params["model"] == "per-call"
        assert deployed._identifying_params["model"] == "my-deployment"
    finally:
        per_call.close()
        deployed.close()


def test_the_routing_rotation_carries_over_from_one_run_to_the_next():
    primary = Named(name="primary", script=[AIMessage(content="first")])
    fallback = Named(name="fallback", script=[AIMessage(content="second")])
    agent = Agent(primary, fallback_model=fallback, routing_policy=RoundRobinRoutingPolicy(),
                  config=AgentConfig(reflection_enabled=False))

    first = agent.invoke("one")
    second = agent.invoke("two")

    assert first["steps"][0].model_used == "primary"
    assert second["steps"][0].model_used == "fallback"


def test_a_sequential_specialist_still_running_at_the_limit_is_reported_as_a_timeout_on_time():
    slow = Agent(Slow(script=[AIMessage(content="late")], delay=1.0), config=AgentConfig(reflection_enabled=False))

    started = time.monotonic()
    rows = invoke_specialists(task="go", specialists={"slow": slow}, parallel=False, max_runtime_seconds=0.3)

    assert rows[0]["status"] == "timeout"
    assert time.monotonic() - started < 0.9  # the caller stopped waiting at the limit, not after the full second


def test_a_failure_that_was_never_retried_is_failed_not_exhausted():
    @tool
    def broken(x: int) -> str:
        """Breaks, and is not safe to run twice, so it is never retried."""
        raise RuntimeError("nope")

    model = Scripted(script=[call("broken", {"x": 1}, "b1")])
    config = AgentConfig(escalate_on_retry_exhausted=True, tool_retry_attempts=2)
    result = Agent(model, tools=[broken], config=config).invoke("x")

    assert result["status"] == "stopped" and result["reason"] == "tool_failed"
