"""Tool isolation: a sync tool that runs past its limit is stopped when it runs in its own process."""
import asyncio
import threading
import time

import pytest

pytest.importorskip("langchain_core")

from llmswitch.agentic import AgentConfig  # noqa: E402
from llmswitch.agentic.execution import (  # noqa: E402
    aexecute_tool_calls,
    execute_tool_calls,
    process_isolation_available,
)

pytestmark = pytest.mark.skipif(not process_isolation_available(), reason="needs a platform with fork")


def call(name, args, call_id="c1"):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def writer_after_a_delay(marker):
    """A sync tool that sleeps past the limit, then leaves a file behind."""
    def tool(x: int) -> str:
        time.sleep(1.0)
        marker.write_text("written")
        return "done"
    return tool


def test_a_process_tool_past_its_limit_is_killed_and_its_side_effect_never_happens(tmp_path):
    marker = tmp_path / "side_effect.txt"
    results = execute_tool_calls(step=1, tool_calls=[call("slow", {"x": 1})],
                                 tool_map={"slow": writer_after_a_delay(marker)}, timeout_s=0.2,
                                 isolation="process")

    assert results[0].error_kind == "timeout"
    time.sleep(1.3)
    assert not marker.exists()


def test_a_thread_tool_past_its_limit_keeps_running_as_documented(tmp_path):
    marker = tmp_path / "side_effect.txt"
    results = execute_tool_calls(step=1, tool_calls=[call("slow", {"x": 1})],
                                 tool_map={"slow": writer_after_a_delay(marker)}, timeout_s=0.2)

    assert results[0].error_kind == "timeout"
    time.sleep(1.3)
    assert marker.exists()


def test_a_process_tool_returns_its_value():
    def add(a: int, b: int) -> int:
        return a + b

    result = execute_tool_calls(step=1, tool_calls=[call("add", {"a": 2, "b": 3})],
                                tool_map={"add": add}, timeout_s=5, isolation="process")[0]

    assert result.ok and result.content == "5"


def test_a_failure_inside_a_process_tool_is_reported_with_its_message():
    def broken(x: int) -> int:
        raise ValueError("oops")

    result = execute_tool_calls(step=1, tool_calls=[call("broken", {"x": 1})],
                                tool_map={"broken": broken}, timeout_s=5, isolation="process")[0]

    assert result.error_kind == "tool_error"
    assert result.error == "ValueError: oops"


def test_a_result_that_cannot_cross_the_process_boundary_is_a_tool_error():
    def unpicklable(x: int) -> object:
        return threading.Lock()

    result = execute_tool_calls(step=1, tool_calls=[call("bad", {"x": 1})],
                                tool_map={"bad": unpicklable}, timeout_s=5, isolation="process")[0]

    assert result.error_kind == "tool_error"


def test_a_process_tool_that_exits_without_a_result_is_a_tool_error():
    def exits(x: int) -> int:
        import os
        os._exit(3)

    result = execute_tool_calls(step=1, tool_calls=[call("exits", {"x": 1})],
                                tool_map={"exits": exits}, timeout_s=5, isolation="process")[0]

    assert result.error_kind == "tool_error"
    assert "without a result" in result.error


async def test_an_async_run_stops_a_sync_process_tool_at_its_limit(tmp_path):
    marker = tmp_path / "side_effect.txt"
    results = await aexecute_tool_calls(step=1, tool_calls=[call("slow", {"x": 1})],
                                        tool_map={"slow": writer_after_a_delay(marker)}, timeout_s=0.2,
                                        isolation="process")

    assert results[0].error_kind == "timeout"
    await asyncio.sleep(1.3)
    assert not marker.exists()


def test_the_agent_config_accepts_only_thread_or_process():
    assert AgentConfig(tool_isolation="process").tool_isolation == "process"
    with pytest.raises(ValueError, match="tool_isolation"):
        AgentConfig(tool_isolation="container")
