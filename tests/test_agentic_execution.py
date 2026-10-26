"""Tool execution for the agent: parsing, timeouts, parallelism, retries, events and the failure paths."""
import asyncio
import logging
import time

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from test_agentic_agent import Scripted, call, verdict  # noqa: E402

from llmswitch.agentic import (  # noqa: E402
    Agent,
    AgentConfig,
    CollectingEventHandler,
    LoggingEventHandler,
    ToolExecutionResult,
    idempotent,
)
from llmswitch.agentic.execution import (  # noqa: E402
    aexecute_tool_calls,
    aretry_failed_tool_calls,
    execute_tool_calls,
    is_idempotent,
    parse_arguments,
    retry_failed_tool_calls,
)


def run(tools, calls, **options):
    tool_map = {getattr(t, "name", None) or t.__name__: t for t in tools}
    return execute_tool_calls(step=1, tool_calls=calls, tool_map=tool_map, timeout_s=options.pop("timeout_s", 5.0),
                              **options)


def arun(tools, calls, **options):
    tool_map = {getattr(t, "name", None) or t.__name__: t for t in tools}
    return aexecute_tool_calls(step=1, tool_calls=calls, tool_map=tool_map,
                               timeout_s=options.pop("timeout_s", 5.0), **options)


def c(name, args, call_id):
    return {"name": name, "args": args, "id": call_id}


# --- parsing and the basic outcomes -------------------------------------------------------------

@pytest.mark.parametrize("raw, ok", [
    ({"a": 1}, True), (None, True), ("", True), ('{"a": 1}', True),
    ('{"a":', False), ("[1, 2]", False), (42, False),
])
def test_arguments_are_parsed_or_rejected(raw, ok):
    assert parse_arguments(raw)[0] is ok


def test_a_malformed_argument_string_is_a_parse_failure_and_the_tool_is_not_called():
    calls_made = []

    @tool
    def spy(x: int) -> int:
        """Record a call."""
        calls_made.append(x)
        return x

    result = run([spy], [c("spy", '{"x":', "p1")])[0]
    assert (result.ok, result.error_kind) == (False, "parse")
    assert calls_made == []


def test_an_unknown_tool_is_a_not_found_failure():
    result = run([], [c("missing", {}, "n1")])[0]
    assert result.error_kind == "not_found"
    assert "Tool not found: missing" in result.error


def test_a_string_result_is_used_as_is_and_a_dict_result_is_json():
    @tool
    def text(name: str) -> str:
        """Greet."""
        return f"hi {name}"

    def data(name: str) -> dict:
        """Return a record."""
        return {"name": name}

    results = run([text, data], [c("text", {"name": "a"}, "t"), c("data", {"name": "b"}, "d")])
    assert results[0].content == "hi a"
    assert results[1].content == '{"name": "b"}'
    assert all(r.ok for r in results)


def test_a_raising_tool_is_a_tool_error_with_the_exception_text():
    @tool
    def boom(x: int) -> int:
        """Always fails."""
        raise ValueError("bad input")

    result = run([boom], [c("boom", {"x": 1}, "b1")])[0]
    assert (result.ok, result.error_kind) == (False, "tool_error")
    assert result.error == "ValueError: bad input"


def test_a_call_without_an_id_gets_one():
    @tool
    def ping() -> str:
        """Ping."""
        return "pong"

    result = run([ping], [{"name": "ping", "args": {}}])[0]
    assert result.tool_call_id.startswith("call_")


def test_a_failed_result_becomes_an_error_tool_message():
    result = ToolExecutionResult("x1", "t", {}, False, '{"error": "nope"}', error_kind="tool_error", error="nope")
    message = result.as_tool_message()
    assert message.status == "error" and message.tool_call_id == "x1"


# --- timeouts ----------------------------------------------------------------------------------------

def test_a_sync_tool_that_times_out_returns_without_waiting_for_it():
    @tool
    def slow(seconds: float) -> str:
        """Sleep."""
        time.sleep(seconds)
        return "done"

    started = time.monotonic()
    result = run([slow], [c("slow", {"seconds": 1.5}, "s1")], timeout_s=0.2)[0]
    assert result.error_kind == "timeout" and result.timed_out
    assert time.monotonic() - started < 1.2


def test_a_timeout_is_never_retried_even_for_an_idempotent_tool():
    attempts = []

    @idempotent
    @tool
    def slow(seconds: float) -> str:
        """Sleep."""
        attempts.append(1)
        time.sleep(seconds)
        return "done"

    tool_map = {"slow": slow}
    calls = [c("slow", {"seconds": 1.0}, "s2")]
    first = execute_tool_calls(step=1, tool_calls=calls, tool_map=tool_map, timeout_s=0.1)
    again = retry_failed_tool_calls(step=1, tool_calls=calls, results=first, tool_map=tool_map, timeout_s=0.1,
                                    attempts=2, retry_non_idempotent=True)
    assert again[0].attempts == 1 and len(attempts) == 1


async def test_an_async_coroutine_tool_is_cancelled_at_the_timeout():
    cancelled = []

    async def slow(seconds: float) -> str:
        """Sleep."""
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return "done"

    result = (await arun([slow], [c("slow", {"seconds": 5.0}, "a1")], timeout_s=0.1))[0]
    assert result.error_kind == "timeout"
    assert cancelled == [True]


# --- parallelism -------------------------------------------------------------------------------------

def _sleeper(name):
    @tool
    def sleeper(seconds: float) -> str:
        """Sleep."""
        time.sleep(seconds)
        return name
    sleeper.name = name
    return sleeper


def test_parallel_tools_overlap_and_keep_the_input_order():
    tools = [_sleeper("p1"), _sleeper("p2"), _sleeper("p3")]
    calls = [c("p1", {"seconds": 0.3}, "1"), c("p2", {"seconds": 0.3}, "2"), c("p3", {"seconds": 0.3}, "3")]
    started = time.monotonic()
    results = run(tools, calls, parallel=True, max_concurrency=3)
    assert time.monotonic() - started < 0.8
    assert [r.content for r in results] == ["p1", "p2", "p3"]


def test_max_concurrency_limits_the_parallel_tools():
    tools = [_sleeper("q1"), _sleeper("q2")]
    calls = [c("q1", {"seconds": 0.3}, "1"), c("q2", {"seconds": 0.3}, "2")]
    started = time.monotonic()
    run(tools, calls, parallel=True, max_concurrency=1)
    assert time.monotonic() - started >= 0.55


def test_parallel_off_runs_the_tools_one_after_another():
    tools = [_sleeper("r1"), _sleeper("r2")]
    calls = [c("r1", {"seconds": 0.3}, "1"), c("r2", {"seconds": 0.3}, "2")]
    started = time.monotonic()
    run(tools, calls, parallel=False)
    assert time.monotonic() - started >= 0.55


async def test_async_parallel_tools_overlap_and_keep_the_order():
    async def a(seconds: float) -> str:
        """Wait."""
        await asyncio.sleep(seconds)
        return "a"

    async def b(seconds: float) -> str:
        """Wait."""
        await asyncio.sleep(seconds)
        return "b"

    started = time.monotonic()
    results = await arun([a, b], [c("a", {"seconds": 0.3}, "1"), c("b", {"seconds": 0.3}, "2")], parallel=True)
    assert time.monotonic() - started < 0.55
    assert [r.content for r in results] == ["a", "b"]


# --- retries ------------------------------------------------------------------------------------------

def flaky_tool(name, fail_times):
    state = {"calls": 0}

    @tool
    def flaky(x: int) -> str:
        """Fails a few times."""
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise RuntimeError("temporary")
        return "ok"

    flaky.name = name
    return flaky, state


def test_an_idempotent_tool_is_retried_until_it_succeeds():
    tool_obj, state = flaky_tool("f1", fail_times=1)
    idempotent(tool_obj)
    tool_map = {"f1": tool_obj}
    calls = [c("f1", {"x": 1}, "r1")]
    first = execute_tool_calls(step=1, tool_calls=calls, tool_map=tool_map, timeout_s=5)
    results = retry_failed_tool_calls(step=1, tool_calls=calls, results=first, tool_map=tool_map, timeout_s=5,
                                      attempts=2)
    assert results[0].ok and results[0].content == "ok"
    assert results[0].attempts == 2 and state["calls"] == 2


def test_a_non_idempotent_tool_is_not_retried_unless_allowed():
    tool_obj, state = flaky_tool("f2", fail_times=5)
    tool_map = {"f2": tool_obj}
    calls = [c("f2", {"x": 1}, "r2")]
    first = execute_tool_calls(step=1, tool_calls=calls, tool_map=tool_map, timeout_s=5)
    kept = retry_failed_tool_calls(step=1, tool_calls=calls, results=first, tool_map=tool_map, timeout_s=5, attempts=2)
    assert kept[0].attempts == 1 and state["calls"] == 1
    allowed = retry_failed_tool_calls(step=1, tool_calls=calls, results=first, tool_map=tool_map, timeout_s=5,
                                      attempts=2, retry_non_idempotent=True)
    assert allowed[0].attempts == 3 and state["calls"] == 3


def test_a_missing_tool_is_retried_only_when_asked():
    calls = [c("ghost", {}, "g1")]
    first = execute_tool_calls(step=1, tool_calls=calls, tool_map={}, timeout_s=5)
    kept = retry_failed_tool_calls(step=1, tool_calls=calls, results=first, tool_map={}, timeout_s=5, attempts=1)
    assert kept[0].attempts == 1
    tried = retry_failed_tool_calls(step=1, tool_calls=calls, results=first, tool_map={}, timeout_s=5, attempts=1,
                                    retry_on_tool_not_found=True)
    assert tried[0].attempts == 2


def test_a_parse_failure_is_never_retried_even_with_every_flag_on():
    calls = [c("any", '{"x":', "bad")]
    first = execute_tool_calls(step=1, tool_calls=calls, tool_map={}, timeout_s=5)
    again = retry_failed_tool_calls(step=1, tool_calls=calls, results=first, tool_map={}, timeout_s=5, attempts=3,
                                    retry_on_tool_not_found=True, retry_non_idempotent=True)
    assert again[0].attempts == 1


def test_the_retry_backoff_is_applied_between_rounds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("llmswitch.agentic.execution.time.sleep", lambda seconds: sleeps.append(seconds))
    tool_obj, _ = flaky_tool("f3", fail_times=5)
    tool_map = {"f3": tool_obj}
    calls = [c("f3", {"x": 1}, "r3")]
    first = execute_tool_calls(step=1, tool_calls=calls, tool_map=tool_map, timeout_s=5)
    retry_failed_tool_calls(step=1, tool_calls=calls, results=first, tool_map=tool_map, timeout_s=5, attempts=2,
                            backoff_seconds=0.25, retry_non_idempotent=True)
    assert sleeps == [0.25, 0.25]


async def test_async_retries_an_idempotent_tool():
    state = {"calls": 0}

    async def flaky(x: int) -> str:
        """Fails once."""
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("temporary")
        return "ok"

    idempotent(flaky)
    tool_map = {"flaky": flaky}
    calls = [c("flaky", {"x": 1}, "af")]
    first = await aexecute_tool_calls(step=1, tool_calls=calls, tool_map=tool_map, timeout_s=5)
    again = await aretry_failed_tool_calls(step=1, tool_calls=calls, results=first, tool_map=tool_map, timeout_s=5,
                                           attempts=1)
    assert again[0].ok and again[0].attempts == 2


def test_idempotence_is_recorded_for_functions_and_tool_objects():
    def plain(x: int) -> int:
        """Plain."""
        return x

    assert not is_idempotent(plain)
    assert is_idempotent(idempotent(plain))

    @tool
    def object_tool(x: int) -> int:
        """Object."""
        return x

    assert not is_idempotent(object_tool)
    assert is_idempotent(idempotent(object_tool))


# --- events ------------------------------------------------------------------------------------------

def test_a_successful_call_emits_start_and_end_events():
    events = CollectingEventHandler()

    @tool
    def add(a: int) -> int:
        """Add."""
        return a + 1

    execute_tool_calls(step=2, tool_calls=[c("add", {"a": 1}, "e1")], tool_map={"add": add}, timeout_s=5,
                       handler=events)
    assert [e.event for e in events.events] == ["tool_start", "tool_end"]
    assert events.events[1].payload["result"] == 2 and events.events[1].step == 2


def test_a_failed_call_emits_an_error_event():
    events = CollectingEventHandler()
    execute_tool_calls(step=1, tool_calls=[c("ghost", {}, "e2")], tool_map={}, timeout_s=5, handler=events)
    # A refused call still gets a start and an error event, so event consumers see it.
    assert [e.event for e in events.events] == ["tool_start", "tool_error"]

    @tool
    def bad(x: int) -> int:
        """Bad."""
        raise ValueError("oops")

    execute_tool_calls(step=1, tool_calls=[c("bad", {"x": 1}, "e3")], tool_map={"bad": bad}, timeout_s=5,
                       handler=events)
    assert [e.event for e in events.events[2:]] == ["tool_start", "tool_error"]
    assert events.events[3].payload["error"] == "ValueError: oops"


def test_a_handler_that_raises_does_not_break_the_run(caplog):
    class Broken:
        def on_step_start(self, step, context):
            raise RuntimeError("handler bug")

        def on_tool_start(self, step, tool_name, args):
            raise RuntimeError("handler bug")

        def on_tool_end(self, step, tool_name, result, duration_s):
            raise RuntimeError("handler bug")

        def on_tool_error(self, step, tool_name, error, duration_s):
            raise RuntimeError("handler bug")

        def on_step_end(self, step, context):
            raise RuntimeError("handler bug")

    @tool
    def ok(x: int) -> int:
        """Fine."""
        return x

    with caplog.at_level(logging.ERROR, logger="llmswitch.agentic.execution"):
        results = execute_tool_calls(step=1, tool_calls=[c("ok", {"x": 3}, "h1")], tool_map={"ok": ok},
                                     timeout_s=5, handler=Broken())
    assert results[0].ok and results[0].content == "3"
    assert "Agent event handler on_tool_start raised" in caplog.text


def test_the_logging_handler_never_writes_argument_or_result_values(caplog):
    @tool
    def echo(secret: str) -> str:
        """Echo."""
        return f"got {secret}"

    with caplog.at_level(logging.INFO, logger="llmswitch.agentic"):
        execute_tool_calls(step=1, tool_calls=[c("echo", {"secret": "s3cr3t-value-123"}, "l1")],
                           tool_map={"echo": echo}, timeout_s=5, handler=LoggingEventHandler())
    assert "s3cr3t-value-123" not in caplog.text
    assert "tool_start" in caplog.text and "tool_end" in caplog.text


# --- the agent on tool failures ----------------------------------------------------------------------

def test_the_agent_retries_an_idempotent_tool_and_completes():
    tool_obj, state = flaky_tool("agent_flaky", fail_times=1)
    idempotent(tool_obj)
    model = Scripted(script=[call("agent_flaky", {"x": 1}, "af1"), AIMessage(content="It worked."), verdict(True)])
    runner = Agent(model, tools=[tool_obj], config=AgentConfig(tool_retry_attempts=1))
    result = runner.invoke("Try it.")
    assert result["status"] == "completed"
    assert result["steps"][0].tool_results[0].attempts == 2


def test_interrupt_on_tool_error_stops_the_run_with_a_reason():
    tool_obj, _ = flaky_tool("agent_bad", fail_times=9)
    model = Scripted(script=[call("agent_bad", {"x": 1}, "ab1")])
    result = Agent(model, tools=[tool_obj], config=AgentConfig(interrupt_on_tool_error=True)).invoke("x")
    assert result["status"] == "interrupted" and result["reason"] == "tool_error"
    assert len(model.seen) == 1


def test_escalation_stops_when_retries_are_exhausted_and_lists_the_failures():
    tool_obj, _ = flaky_tool("agent_stuck", fail_times=9)
    idempotent(tool_obj)
    model = Scripted(script=[call("agent_stuck", {"x": 1}, "as1")])
    config = AgentConfig(tool_retry_attempts=1, escalate_on_retry_exhausted=True)
    result = Agent(model, tools=[tool_obj], config=config).invoke("x")
    assert result["status"] == "stopped" and result["reason"] == "tool_retry_exhausted"
    assert result["failed_tools"] == [{"tool_name": "agent_stuck", "error": "RuntimeError: temporary", "attempts": 2}]


def test_without_interrupt_or_escalation_a_failed_tool_is_reported_and_the_run_goes_on():
    tool_obj, _ = flaky_tool("agent_soft", fail_times=9)
    model = Scripted(script=[call("agent_soft", {"x": 1}, "so1"), AIMessage(content="The tool failed."),
                             verdict(True)])
    result = Agent(model, tools=[tool_obj]).invoke("x")
    assert result["status"] == "completed"
    assert result["steps"][0].tool_results[0].ok is False


def test_the_agent_emits_step_and_tool_events_in_order():
    events = CollectingEventHandler()

    @tool
    def add(a: int) -> int:
        """Add."""
        return a + 1

    model = Scripted(script=[call("add", {"a": 1}, "ev1"), AIMessage(content="2"), verdict(True)])
    Agent(model, tools=[add], event_handler=events).invoke("x")
    assert [e.event for e in events.events] == [
        "step_start", "tool_start", "tool_end", "step_end", "step_start", "step_end"]


async def test_the_async_agent_stops_on_an_interrupting_tool_error():
    tool_obj, _ = flaky_tool("agent_async_bad", fail_times=9)
    model = Scripted(script=[call("agent_async_bad", {"x": 1}, "aa1")])
    result = await Agent(model, tools=[tool_obj], config=AgentConfig(interrupt_on_tool_error=True)).ainvoke("x")
    assert result["status"] == "interrupted"


# --- configuration -------------------------------------------------------------------------------------

@pytest.mark.parametrize("options", [
    {"tool_timeout_seconds": 0}, {"max_concurrency": 0}, {"tool_retry_attempts": -1},
    {"tool_retry_backoff_seconds": -1},
])
def test_invalid_tool_settings_are_rejected(options):
    with pytest.raises(ValueError):
        AgentConfig(**options)
