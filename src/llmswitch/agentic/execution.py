"""
llmswitch.agentic.execution

Running the tool calls the model asks for: argument parsing, per-tool timeouts, bounded parallelism,
retries for tools marked idempotent, and events.

Every failure has a kind, so retry decisions do not depend on the wording of an error message:

- ``parse``: the arguments are not a JSON object. Never retried.
- ``not_found``: no tool has that name. Retried only when ``retry_on_tool_not_found`` is set.
- ``timeout``: the tool did not finish in time. Never retried, since the tool may still be running
  or may already have acted.
- ``tool_error``: the tool raised. Retried only when the tool is idempotent, or when
  ``retry_non_idempotent`` is set.

Timeouts stop the wait, not the work. Python cannot stop a thread that is already running, so a sync
tool that times out keeps running in the background until it returns. An async tool written as a
coroutine is cancelled at its next await.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any, Literal

try:
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import BaseTool
except ImportError as exc:  # pragma: no cover - exercised by tests that hide the optional dependency
    raise ImportError(
        "The agentic features need the agentic extra. Install it with: "
        'pip install "llmswitch-sdk[agentic]"'
    ) from exc

from .events import AgentEventHandler, notify

logger = logging.getLogger(__name__)

ErrorKind = Literal["none", "parse", "not_found", "timeout", "tool_error"]
Tool = BaseTool | Callable[..., Any]

_IDEMPOTENT_KEY = "llmswitch_idempotent"


@dataclass
class ToolExecutionResult:
    """The outcome of one tool call."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    ok: bool
    content: str
    error_kind: ErrorKind = "none"
    error: str | None = None
    duration_s: float = 0.0
    attempts: int = 1

    @property
    def timed_out(self) -> bool:
        return self.error_kind == "timeout"

    def as_tool_message(self) -> ToolMessage:
        """The result as a tool message for the conversation."""
        return ToolMessage(content=self.content, tool_call_id=self.tool_call_id,
                           status="success" if self.ok else "error")


def idempotent(tool: Tool) -> Tool:
    """Mark a tool as safe to retry after an error, and return it.

    Only mark a tool idempotent when running it twice has the same effect as running it once,
    for example a read-only lookup. Works on plain functions and on tool objects.
    """
    if isinstance(tool, BaseTool):
        tool.metadata = {**(tool.metadata or {}), _IDEMPOTENT_KEY: True}
    else:
        setattr(tool, _IDEMPOTENT_KEY, True)
    return tool


def is_idempotent(tool: Tool | None) -> bool:
    """Whether a tool was marked with :func:`idempotent`."""
    if tool is None:
        return False
    if isinstance(tool, BaseTool):
        return bool((tool.metadata or {}).get(_IDEMPOTENT_KEY))
    return bool(getattr(tool, _IDEMPOTENT_KEY, False))


def parse_arguments(raw: Any) -> tuple[bool, dict[str, Any], str | None]:
    """Read the arguments of a tool call: a dict, ``None``, or a JSON object as text.

    Returns ``(ok, arguments, error)``.
    """
    if isinstance(raw, Mapping):
        return True, dict(raw), None
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        return True, {}, None
    if not isinstance(raw, str):
        return False, {}, f"Unsupported argument type: {type(raw).__name__}"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, {}, f"Malformed JSON arguments: {exc}"
    if not isinstance(value, dict):
        return False, {}, "Tool arguments must be a JSON object"
    return True, value, None


def _content(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _error_content(message: str) -> str:
    return json.dumps({"error": message})


def _call_id(call: Mapping[str, Any]) -> str:
    return call.get("id") or f"call_{uuid.uuid4().hex[:24]}"


def _failed(call_id: str, name: str, args: dict[str, Any], kind: ErrorKind, message: str,
            duration_s: float = 0.0) -> ToolExecutionResult:
    return ToolExecutionResult(call_id, name, args, False, _error_content(message),
                               error_kind=kind, error=message, duration_s=duration_s)


def _call_sync(tool: Tool, args: dict[str, Any]) -> Any:
    if isinstance(tool, BaseTool):
        return tool.invoke(args)
    result = tool(**args)
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        raise TypeError("This tool is async; run the agent with ainvoke.")
    return result


def _call_with_timeout(tool: Tool, args: dict[str, Any], timeout_s: float) -> Any:
    """Run a sync tool on its own thread and wait up to ``timeout_s`` for it."""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llmswitch-tool")
    try:
        future = pool.submit(copy_context().run, _call_sync, tool, args)
        return future.result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _execute_sync(call: Mapping[str, Any], tool_map: Mapping[str, Tool], timeout_s: float,
                  handler: AgentEventHandler | None, step: int) -> ToolExecutionResult:
    call_id, name = _call_id(call), call.get("name") or ""
    ok, args, error = parse_arguments(call.get("args"))
    if not ok:
        return _failed(call_id, name, {}, "parse", error or "Malformed arguments")
    tool = tool_map.get(name)
    if tool is None:
        return _failed(call_id, name, args, "not_found", f"Tool not found: {name}")
    notify(handler, "on_tool_start", step, name, args)
    started = time.monotonic()
    try:
        value = _call_with_timeout(tool, args, timeout_s)
    except FuturesTimeoutError:
        duration = time.monotonic() - started
        message = f"Tool '{name}' timed out after {timeout_s}s"
        notify(handler, "on_tool_error", step, name, message, duration)
        return _failed(call_id, name, args, "timeout", message, duration)
    except Exception as exc:  # noqa: BLE001 - any tool failure is reported back to the model
        duration = time.monotonic() - started
        message = f"{type(exc).__name__}: {exc}"
        notify(handler, "on_tool_error", step, name, message, duration)
        return _failed(call_id, name, args, "tool_error", message, duration)
    duration = time.monotonic() - started
    notify(handler, "on_tool_end", step, name, value, duration)
    return ToolExecutionResult(call_id, name, args, True, _content(value), duration_s=duration)


def execute_tool_calls(
    *,
    step: int,
    tool_calls: Sequence[Mapping[str, Any]],
    tool_map: Mapping[str, Tool],
    timeout_s: float,
    parallel: bool = True,
    max_concurrency: int = 4,
    handler: AgentEventHandler | None = None,
) -> list[ToolExecutionResult]:
    """Run tool calls, in parallel up to ``max_concurrency`` when ``parallel`` is set.

    Results come back in the order of ``tool_calls``, whatever order the tools finish in.
    """
    if not tool_calls:
        return []
    if not parallel or len(tool_calls) == 1:
        return [_execute_sync(call, tool_map, timeout_s, handler, step) for call in tool_calls]
    workers = max(1, min(max_concurrency, len(tool_calls)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llmswitch-agent") as pool:
        futures = [pool.submit(copy_context().run, _execute_sync, call, tool_map, timeout_s, handler, step)
                   for call in tool_calls]
        return [future.result() for future in futures]


async def _call_async(tool: Tool, args: dict[str, Any]) -> Any:
    if isinstance(tool, BaseTool):
        return await tool.ainvoke(args)
    if inspect.iscoroutinefunction(tool):
        return await tool(**args)
    # A sync callable runs on a thread, so it does not block the event loop. It cannot be cancelled once started.
    return await asyncio.to_thread(tool, **args)


async def _execute_async(call: Mapping[str, Any], tool_map: Mapping[str, Tool], timeout_s: float,
                         handler: AgentEventHandler | None, step: int,
                         semaphore: asyncio.Semaphore) -> ToolExecutionResult:
    call_id, name = _call_id(call), call.get("name") or ""
    ok, args, error = parse_arguments(call.get("args"))
    if not ok:
        return _failed(call_id, name, {}, "parse", error or "Malformed arguments")
    tool = tool_map.get(name)
    if tool is None:
        return _failed(call_id, name, args, "not_found", f"Tool not found: {name}")
    async with semaphore:
        notify(handler, "on_tool_start", step, name, args)
        started = time.monotonic()
        try:
            value = await asyncio.wait_for(_call_async(tool, args), timeout_s)
        except TimeoutError:
            duration = time.monotonic() - started
            message = f"Tool '{name}' timed out after {timeout_s}s"
            notify(handler, "on_tool_error", step, name, message, duration)
            return _failed(call_id, name, args, "timeout", message, duration)
        except Exception as exc:  # noqa: BLE001 - any tool failure is reported back to the model
            duration = time.monotonic() - started
            message = f"{type(exc).__name__}: {exc}"
            notify(handler, "on_tool_error", step, name, message, duration)
            return _failed(call_id, name, args, "tool_error", message, duration)
    duration = time.monotonic() - started
    notify(handler, "on_tool_end", step, name, value, duration)
    return ToolExecutionResult(call_id, name, args, True, _content(value), duration_s=duration)


async def aexecute_tool_calls(
    *,
    step: int,
    tool_calls: Sequence[Mapping[str, Any]],
    tool_map: Mapping[str, Tool],
    timeout_s: float,
    parallel: bool = True,
    max_concurrency: int = 4,
    handler: AgentEventHandler | None = None,
) -> list[ToolExecutionResult]:
    """Async :func:`execute_tool_calls`, bounded by a semaphore. Results keep the input order."""
    if not tool_calls:
        return []
    semaphore = asyncio.Semaphore(max(1, max_concurrency) if parallel else 1)
    return list(await asyncio.gather(*(
        _execute_async(call, tool_map, timeout_s, handler, step, semaphore) for call in tool_calls
    )))


def is_retryable(result: ToolExecutionResult, *, idempotent: bool, retry_on_tool_not_found: bool,
                 retry_non_idempotent: bool) -> bool:
    """Whether a failed call should be tried again, by its failure kind and the tool's idempotence."""
    if result.ok:
        return False
    if result.error_kind == "not_found":
        return retry_on_tool_not_found
    if result.error_kind != "tool_error":
        return False
    return idempotent or retry_non_idempotent


def _retry_indices(tool_calls: Sequence[Mapping[str, Any]], results: Sequence[ToolExecutionResult],
                   tool_map: Mapping[str, Tool], retry_on_tool_not_found: bool,
                   retry_non_idempotent: bool) -> list[int]:
    return [
        index for index, (call, result) in enumerate(zip(tool_calls, results, strict=True))
        if is_retryable(result, idempotent=is_idempotent(tool_map.get(call.get("name") or "")),
                        retry_on_tool_not_found=retry_on_tool_not_found,
                        retry_non_idempotent=retry_non_idempotent)
    ]


def retry_failed_tool_calls(
    *,
    step: int,
    tool_calls: Sequence[Mapping[str, Any]],
    results: Sequence[ToolExecutionResult],
    tool_map: Mapping[str, Tool],
    timeout_s: float,
    attempts: int,
    backoff_seconds: float = 0.0,
    retry_on_tool_not_found: bool = False,
    retry_non_idempotent: bool = False,
    parallel: bool = True,
    max_concurrency: int = 4,
    handler: AgentEventHandler | None = None,
) -> list[ToolExecutionResult]:
    """Retry the retryable failures, up to ``attempts`` extra rounds, sleeping ``backoff_seconds`` between them."""
    current = list(results)
    for _ in range(max(0, attempts)):
        indices = _retry_indices(tool_calls, current, tool_map, retry_on_tool_not_found, retry_non_idempotent)
        if not indices:
            break
        if backoff_seconds > 0:
            time.sleep(backoff_seconds)
        fresh = execute_tool_calls(
            step=step, tool_calls=[tool_calls[i] for i in indices], tool_map=tool_map, timeout_s=timeout_s,
            parallel=parallel, max_concurrency=max_concurrency, handler=handler,
        )
        for index, result in zip(indices, fresh, strict=True):
            result.attempts = current[index].attempts + 1
            current[index] = result
    return current


async def aretry_failed_tool_calls(
    *,
    step: int,
    tool_calls: Sequence[Mapping[str, Any]],
    results: Sequence[ToolExecutionResult],
    tool_map: Mapping[str, Tool],
    timeout_s: float,
    attempts: int,
    backoff_seconds: float = 0.0,
    retry_on_tool_not_found: bool = False,
    retry_non_idempotent: bool = False,
    parallel: bool = True,
    max_concurrency: int = 4,
    handler: AgentEventHandler | None = None,
) -> list[ToolExecutionResult]:
    """Async :func:`retry_failed_tool_calls`."""
    current = list(results)
    for _ in range(max(0, attempts)):
        indices = _retry_indices(tool_calls, current, tool_map, retry_on_tool_not_found, retry_non_idempotent)
        if not indices:
            break
        if backoff_seconds > 0:
            await asyncio.sleep(backoff_seconds)
        fresh = await aexecute_tool_calls(
            step=step, tool_calls=[tool_calls[i] for i in indices], tool_map=tool_map, timeout_s=timeout_s,
            parallel=parallel, max_concurrency=max_concurrency, handler=handler,
        )
        for index, result in zip(indices, fresh, strict=True):
            result.attempts = current[index].attempts + 1
            current[index] = result
    return current
